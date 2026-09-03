const config = require('../config/index.js')
const { logger } = require('./logger')

// 落到回退目标的入站模型名（给后续 dashboard 用），去重、封顶
const UNMAPPED_CAP = 100
const unmappedModels = new Set()

/**
 * 去掉客户端附加的尾部 [..] 后缀（Claude Code 会发 claude-opus-5[1m]）
 * @param {string} name - 原始模型名
 * @returns {string} 去后缀并 trim 后的名字
 */
const stripBracketSuffix = (name) => String(name || '').trim().replace(/\[[^\]]*\]$/, '').trim()

/**
 * 解析 MODEL_MAP：alias=target,alias2=target2,*=fallback
 * 无 '='、alias 或 target 为空的项静默丢弃；重复 alias 后者覆盖前者。永不抛错。
 * @param {string} raw - 环境变量原文
 * @returns {Object} 无原型对象 { alias: target }，'*' 键为通配回退
 */
const parseModelMap = (raw) => {
    const map = Object.create(null)
    for (const entry of String(raw || '').split(',')) {
        const eq = entry.indexOf('=')
        if (eq < 0) continue
        const alias = entry.slice(0, eq).trim()
        const target = entry.slice(eq + 1).trim()
        if (!alias || !target) continue
        map[alias] = target
    }
    return map
}

const lookup = (map, key) => (map && typeof map[key] === 'string' && map[key]) ? map[key] : null

const modelAliases = (model) => [model?.id, model?.name, model?.display_name, model?.upstream_id]
    .filter(Boolean)
    .map(value => String(value).trim().toLowerCase())

/**
 * 名字已经是上游模型，或其带后缀的变体（qwen3.8-max-thinking）时为 true。
 * 用「等于某个上游 id，或以 `<id>-` 开头」判断，不复制 chat-helpers 的后缀表。
 * @param {string} name - 已去掉 [..] 的模型名
 * @param {Array} models - 上游模型列表（/api/models 的 data）
 * @returns {boolean}
 */
const isKnownUpstreamModel = (name, models) => {
    const needle = String(name || '').trim().toLowerCase()
    if (!needle || !Array.isArray(models)) return false
    return models.some(model => modelAliases(model).some(alias => needle === alias || needle.startsWith(`${alias}-`)))
}

const firstModelByChatType = (models, chatType) => {
    if (!Array.isArray(models)) return null
    const matched = models.find(model => model?.info?.meta?.chat_type?.includes(chatType))
    return matched?.id || null
}

/**
 * 纯函数：精确项 > 上游已知 id 原样透传 > '*' 通配 > 上游第一个 t2t 模型 > 原样
 * @param {string} name - 入站模型名（可带 [..] 后缀）
 * @param {Object} map - parseModelMap 的结果
 * @param {Array} upstreamModels - 上游模型列表；空列表时无法判断已知 id，也没有动态回退
 * @returns {{ model: string, source: 'exact'|'known'|'wildcard'|'default'|'unchanged' }}
 */
const resolveModel = (name, map, upstreamModels = []) => {
    const raw = String(name || '')
    const stripped = stripBracketSuffix(raw)
    if (!stripped) return { model: raw, source: 'unchanged' }

    const exact = lookup(map, stripped)
    if (exact) return { model: exact, source: 'exact' }

    if (isKnownUpstreamModel(stripped, upstreamModels)) return { model: stripped, source: 'known' }

    const wildcard = lookup(map, '*')
    if (wildcard) return { model: wildcard, source: 'wildcard' }

    const fallback = firstModelByChatType(upstreamModels, 't2t')
    if (fallback) return { model: fallback, source: 'default' }

    return { model: raw, source: 'unchanged' }
}

/**
 * 记录一个落到回退目标的入站名
 * @param {string} name - 已去掉 [..] 的模型名
 * @returns {boolean} 本次是否新增（已存在或已封顶时 false）
 */
const recordUnmapped = (name) => {
    const key = String(name || '').trim()
    if (!key || unmappedModels.has(key) || unmappedModels.size >= UNMAPPED_CAP) return false
    unmappedModels.add(key)
    return true
}

const getUnmappedModels = () => Array.from(unmappedModels)

const resetUnmappedModels = () => unmappedModels.clear()

const loadUpstreamModels = async () => {
    try {
        // 调用时才 require：models-map.js 顶层加载 account.js，单元测试不能碰它。
        // 不在模块顶层解构 getLatestModels，让离线测试对它的 monkeypatch 仍然生效。
        const models = await require('../models/models-map.js').getLatestModels()
        return Array.isArray(models) ? models : []
    } catch (e) {
        return []
    }
}

/**
 * 两个请求构造器的入口都走这里：在 thinking / chat_type 判定之前把入站模型名换成 Qwen id。
 * 每次重新解析 config.modelMap（后续 dashboard 会在运行时改写它，不能缓存）。
 * @param {string} name - 请求体里的 model
 * @returns {Promise<string>} 映射后的模型名；空值原样返回
 */
const mapIncomingModel = async (name) => {
    if (typeof name !== 'string' || !name.trim()) return name

    const map = parseModelMap(config.modelMap)
    const stripped = stripBracketSuffix(name)
    // 精确命中不需要上游列表
    const upstreamModels = lookup(map, stripped) ? [] : await loadUpstreamModels()
    const { model, source } = resolveModel(name, map, upstreamModels)

    if ((source === 'wildcard' || source === 'default') && recordUnmapped(stripped)) {
        const via = source === 'wildcard'
            ? 'MODEL_MAP "*" entry'
            : 'first upstream t2t model; add a MODEL_MAP entry to pin it'
        logger.warn(`Model "${stripped}" has no MODEL_MAP entry, using "${model}" (${via})`, 'MODEL')
    }

    if (model !== name) {
        const origin = stripped === name ? '' : ` (from "${name}")`
        logger.info(`Model map: ${stripped} -> ${model} [${source}]${origin}`, 'MODEL')
    }

    return model
}

module.exports = {
    UNMAPPED_CAP,
    stripBracketSuffix,
    parseModelMap,
    isKnownUpstreamModel,
    resolveModel,
    recordUnmapped,
    getUnmappedModels,
    resetUnmappedModels,
    mapIncomingModel
}
