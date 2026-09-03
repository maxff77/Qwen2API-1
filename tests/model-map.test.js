// Pure tests for src/utils/model-map.js: no network, and account.js must never load
// (models-map.js is stubbed in require.cache before the resolver ever lazy-requires it).
const test = require('node:test')
const assert = require('node:assert/strict')
const Module = require('node:module')

const modelsMapPath = require.resolve('../src/models/models-map.js')
let upstreamModels = []
const modelsMapStub = new Module(modelsMapPath)
modelsMapStub.filename = modelsMapPath
modelsMapStub.loaded = true
modelsMapStub.exports = { getLatestModels: async () => upstreamModels }
require.cache[modelsMapPath] = modelsMapStub

const config = require('../src/config/index.js')
const { logger } = require('../src/utils/logger')
const {
  UNMAPPED_CAP,
  parseModelMap,
  resolveModel,
  recordUnmapped,
  getUnmappedModels,
  resetUnmappedModels,
  mapIncomingModel
} = require('../src/utils/model-map.js')

const T2T = (id) => ({ id, info: { meta: { chat_type: ['t2t', 'search'] } } })
const T2I = (id) => ({ id, info: { meta: { chat_type: ['t2i'] } } })
const UPSTREAM = [T2I('qwen-image'), T2T('qwen3.8-max'), T2T('qwen3-max')]

const captureLogs = () => {
  const lines = { warn: [], info: [] }
  const originalWarn = logger.warn
  const originalInfo = logger.info
  logger.warn = (message) => { lines.warn.push(String(message)) }
  logger.info = (message) => { lines.info.push(String(message)) }
  return { lines, restore: () => { logger.warn = originalWarn; logger.info = originalInfo } }
}

const withMap = async (raw, models, fn) => {
  const previous = config.modelMap
  config.modelMap = raw
  upstreamModels = models
  resetUnmappedModels()
  const logs = captureLogs()
  try {
    return await fn(logs.lines)
  } finally {
    logs.restore()
    config.modelMap = previous
    upstreamModels = []
    resetUnmappedModels()
  }
}

test('exact entry wins and carries the target suffix so thinking detection sees it', () => {
  const map = parseModelMap('claude-opus-5=qwen3.8-max-thinking')
  assert.deepEqual(resolveModel('claude-opus-5', map, UPSTREAM), { model: 'qwen3.8-max-thinking', source: 'exact' })
  assert.ok(resolveModel('claude-opus-5', map, UPSTREAM).model.includes('-thinking'))
})

test('trailing [..] suffix is stripped before the lookup', () => {
  const map = parseModelMap('claude-opus-5=qwen3.8-max-thinking')
  assert.deepEqual(resolveModel('claude-opus-5[1m]', map, UPSTREAM), resolveModel('claude-opus-5', map, UPSTREAM))
  assert.deepEqual(resolveModel('claude-opus-5[1m]', map, UPSTREAM), { model: 'qwen3.8-max-thinking', source: 'exact' })
})

test('* entry catches unlisted names and records them as unmapped', async () => {
  await withMap('*=qwen3.8-max', UPSTREAM, async (lines) => {
    assert.deepEqual(resolveModel('gpt-4o', parseModelMap(config.modelMap), UPSTREAM), { model: 'qwen3.8-max', source: 'wildcard' })
    assert.equal(await mapIncomingModel('gpt-4o'), 'qwen3.8-max')
    assert.deepEqual(getUnmappedModels(), ['gpt-4o'])
    assert.equal(lines.warn.length, 1)
    assert.match(lines.warn[0], /MODEL_MAP/)
    assert.match(lines.info[0], /gpt-4o -> qwen3\.8-max/)
    // second hit: same target, no second warn
    assert.equal(await mapIncomingModel('gpt-4o'), 'qwen3.8-max')
    assert.equal(lines.warn.length, 1)
  })
})

test('no map: first upstream t2t model, recorded, one warn naming MODEL_MAP', async () => {
  await withMap('', UPSTREAM, async (lines) => {
    assert.deepEqual(resolveModel('claude-sonnet-5', parseModelMap(''), UPSTREAM), { model: 'qwen3.8-max', source: 'default' })
    assert.equal(await mapIncomingModel('claude-sonnet-5'), 'qwen3.8-max')
    assert.equal(await mapIncomingModel('claude-sonnet-5[1m]'), 'qwen3.8-max')
    assert.deepEqual(getUnmappedModels(), ['claude-sonnet-5'])
    assert.equal(lines.warn.length, 1)
    assert.match(lines.warn[0], /MODEL_MAP/)
    assert.match(lines.info[0], /claude-sonnet-5 -> qwen3\.8-max/)
  })
  // empty upstream list: nothing to fall back to, name passes through unchanged (upstream error as today)
  await withMap('', [], async (lines) => {
    assert.deepEqual(resolveModel('claude-sonnet-5', parseModelMap(''), []), { model: 'claude-sonnet-5', source: 'unchanged' })
    assert.equal(await mapIncomingModel('claude-sonnet-5'), 'claude-sonnet-5')
    assert.deepEqual(getUnmappedModels(), [])
    assert.equal(lines.warn.length, 0)
    assert.equal(lines.info.length, 0)
  })
})

test('known Qwen id with no entry is untouched and not recorded, even with * set', async () => {
  for (const raw of ['', '*=qwen3-max', 'claude-opus-5=qwen3-max,*=qwen3-max']) {
    await withMap(raw, UPSTREAM, async (lines) => {
      const map = parseModelMap(raw)
      assert.deepEqual(resolveModel('qwen3.8-max-thinking', map, UPSTREAM), { model: 'qwen3.8-max-thinking', source: 'known' })
      assert.deepEqual(resolveModel('qwen3.8-max', map, UPSTREAM), { model: 'qwen3.8-max', source: 'known' })
      assert.deepEqual(resolveModel('qwen-image', map, UPSTREAM), { model: 'qwen-image', source: 'known' })
      assert.equal(await mapIncomingModel('qwen3.8-max-thinking'), 'qwen3.8-max-thinking')
      assert.deepEqual(getUnmappedModels(), [])
      assert.equal(lines.warn.length, 0)
      assert.equal(lines.info.length, 0)
    })
  }
})

test('malformed map keeps only the well-formed entry and never throws', () => {
  const map = parseModelMap('foo, =x, claude-a = qwen3.8-max')
  assert.deepEqual({ ...map }, { 'claude-a': 'qwen3.8-max' })
  assert.deepEqual({ ...parseModelMap('') }, {})
  assert.deepEqual({ ...parseModelMap(undefined) }, {})
  assert.deepEqual({ ...parseModelMap('a=1,a=2') }, { a: '2' })
  // prototype keys are not entries
  assert.deepEqual(resolveModel('constructor', parseModelMap(''), []), { model: 'constructor', source: 'unchanged' })
})

test('unmapped record is capped at 100 distinct names', () => {
  resetUnmappedModels()
  for (let i = 0; i < UNMAPPED_CAP; i += 1) assert.equal(recordUnmapped(`model-${i}`), true)
  assert.equal(recordUnmapped('model-0'), false)
  assert.equal(recordUnmapped('model-100'), false)
  assert.equal(getUnmappedModels().length, UNMAPPED_CAP)
  assert.ok(!getUnmappedModels().includes('model-100'))
  resetUnmappedModels()
})

test('empty or non-string model values pass through untouched', async () => {
  await withMap('*=qwen3-max', UPSTREAM, async () => {
    assert.equal(await mapIncomingModel(undefined), undefined)
    assert.equal(await mapIncomingModel(''), '')
    assert.equal(await mapIncomingModel('   '), '   ')
    assert.deepEqual(getUnmappedModels(), [])
  })
})

test('the resolver never loads account.js', () => {
  assert.equal(require.cache[require.resolve('../src/utils/account.js')], undefined)
})
