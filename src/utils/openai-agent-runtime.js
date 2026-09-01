const { isJson } = require('./tools.js')
const {
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator,
  containsOrphanProtocolResidue,
  ANSWER_PHASES
} = require('./tool-prompt.js')
const { consumeSSEStream, createUpstreamResponseFilter } = require('./sse.js')
const { createUpstreamDeltaNormalizer, createClientToolNamePredicate } = require('./chat-helpers.js')
const { assertNoUpstreamFailure } = require('./upstream-error.js')
const {
  parseAgentControlText,
  createAgentControlStreamParser,
  buildAgentRetryHint
} = require('./agent-turn.js')
const config = require('../config/index.js')
const { logger } = require('./logger.js')

const NON_RETRYABLE_FINISH_REASONS = new Set([
  'length',
  'max_tokens',
  'content_filter',
  'refusal'
])

const normalizeCreatedMetadata = (payload) => {
  const created = payload?.['response.created'] || payload?.response?.created
  if (!created || typeof created !== 'object') return null
  return {
    chatId: created.chat_id || created.chatId || null,
    parentId: created.parent_id || created.parentId || null,
    responseId: created.response_id || created.responseId || null,
    responseIndex: created.response_index ?? created.responseIndex ?? null
  }
}

const imageMarkdownFromDelta = (delta) => {
  const result = []
  for (const item of delta?.extra?.image_list || []) {
    if (item?.image) result.push(`![image](${item.image})`)
  }
  return result
}

/** 键排序后的规范 JSON：跨通道去重要把 `{"a":1,"b":2}` 与 `{"b": 2, "a": 1}` 判成同一份参数。 */
const canonicalJson = (value) => {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

/**
 * 本轮的工具调用登记簿：同名 + 规范 JSON 相同的第二个调用是跨通道的副本（文本解析器
 * 与原生累积器各自都能产出同一个调用），只保留先到的。
 * @returns {(call: Object) => boolean} true = 首次见到，可以发射
 */
const createToolCallLedger = () => {
  const seen = new Set()
  return (call) => {
    const args = call?.function?.arguments || '{}'
    let canonical
    try {
      canonical = canonicalJson(JSON.parse(args))
    } catch (_) {
      canonical = args
    }
    const key = `${call?.function?.name || ''}\u0000${canonical}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  }
}

/**
 * 原生 function_call 帧的喂入与关闭判定（OpenAI Agent 运行时 / chat.js 旧路径共用；
 * anthropic.js 有同形的私有副本）。完成证据读的是**原始** delta：归一化器对
 * role:function 返回 null（Defect A），不能从它那里拿。
 *
 * - 有 function_call → pushNativeSnapshot（分类在累积器里：无 function_id 且 answer phase
 *   才是客户端候选；think phase / 平台调用关闭时记 unknown_tool，走今天的 invalid_tool_call 重试）。
 * - role:function 且名字是客户端工具（与归一化器同一条谓词）→ closeByName：该调用的
 *   结果帧。无名帧与平台结果帧惰性。
 * - answer 帧 status finished / 非空 finish_reason → 回合结束，打开中的按 round_end 关闭。
 * 每次可能关闭之后都调一次 drain（幂等），关闭即发射。
 * @param {Object} accumulator - createNativeToolCallAccumulator 实例
 * @param {Object} delta - 原始上游 delta
 * @param {*} reportedFinishReason - choice 上报的 finish_reason
 * @param {{ isClientToolName: (name: unknown) => boolean, drain: () => void }} hooks
 */
const feedNativeFrame = (accumulator, delta, reportedFinishReason, { isClientToolName, drain }) => {
  const rawPhase = delta.phase
  if (Array.isArray(delta.tool_calls)) {
    accumulator.push(delta.tool_calls)
  } else if (delta.function_call) {
    accumulator.pushNativeSnapshot({
      name: delta.function_call.name,
      arguments: delta.function_call.arguments,
      phase: rawPhase,
      functionId: delta.function_id
    })
    drain()
  } else if (delta.role === 'function' && isClientToolName(delta.name)) {
    if (accumulator.closeByName(delta.name)) drain()
  }
  const answerFinished = delta.role !== 'function' && ANSWER_PHASES.has(rawPhase) && delta.status === 'finished'
  if ((reportedFinishReason !== undefined && reportedFinishReason !== null) || answerFinished) {
    if (accumulator.closeOpen('round_end')) drain()
  }
}

/**
 * 正文恢复帧：归一化后是 answer、内容非空、原始 role ≠ function、原始 phase ∈ ANSWER_PHASES。
 * 它关闭打开中的调用，也是早停的触发帧（批次已齐时）。
 */
const isProseResume = (delta, normalized, rawPhase) =>
  !!normalized && normalized.phase === 'answer' && !!normalized.content &&
  delta.role !== 'function' && ANSWER_PHASES.has(rawPhase)

/**
 * 早停条件：本轮打开过的客户端调用全部被各自的具名结果帧关闭，且至少一个过闸。
 * 平台调用两侧都不计。达不到就永不早停 —— 保护迟到的第三个并行调用。
 */
const nativeBatchComplete = (accumulator) => {
  const state = accumulator.batchState()
  return state.opened > 0 && state.opened === state.closedByResult && state.gated >= 1
}

/**
 * 完整消费一次 Qwen 上游 attempt。裸正文与工具调用始终留在门禁内；调用方可
 * 实时接收安全思考，以及已经进入 final/blocked 包装体的正式正文增量。
 * 每次调用都新建 parser/filter/accumulator，失败 attempt 不会污染下一次。
 */
const collectOpenAIAgentAttempt = async (upstreamResponse, options = {}) => {
  const hasTools = options.has_tools !== false
  const allowedToolNames = options.allowed_tool_names || []
  // clientToolNames：只有客户端声明过的工具名才算拦截证据（见 chat-helpers.js）。
  const normalizeDelta = createUpstreamDeltaNormalizer({ clientToolNames: allowedToolNames })
  const acceptUpstreamFrame = createUpstreamResponseFilter()
  const nativeTools = hasTools
    ? createNativeToolCallAccumulator({ allowedToolNames })
    : null
  const isClientToolName = createClientToolNamePredicate(allowedToolNames)
  // 本轮关闭即晋升的原生调用（takeCompleted 排出）。有了第一个之后，后续正文/思考是平台
  // "工具不存在"注入的回声，只丢不记；批次齐了就提前终止上游。
  const promotedNativeCalls = []
  let stopRequested = false
  const drainPromotedNativeCalls = () => {
    for (const call of nativeTools.takeCompleted()) {
      logger.warn(
        `OpenAI Agent 原生工具调用晋升为 tool_call：${call.function.name}（answer phase，无 function_id）`,
        'AGENT'
      )
      promotedNativeCalls.push(call)
    }
  }
  const reasoningStreamParser = typeof options.on_reasoning_delta === 'function'
    ? createToolCallStreamParser({ allowedToolNames })
    : null
  const controlStreamParser = typeof options.on_content_delta === 'function'
    ? createAgentControlStreamParser()
    : null
  const controlToolStreamParser = controlStreamParser
    ? createToolCallStreamParser({ allowedToolNames })
    : null
  let streamedVisibleText = ''
  // Texto rescatado de un <tool_call> que no parseó. No se emite aquí: el turn gate
  // todavía puede rechazar esta ronda, y emitirlo ahora lo duplicaría en cada intento
  // además de contar como "ya salió texto" en la guarda de :409.
  let recoveredContent = ''
  let recoveredReasoning = ''
  let streamedControlKind = null
  let controlToolParserFlushed = false

  const emitReasoningDelta = async (text) => {
    if (typeof options.on_reasoning_delta !== 'function') return
    await options.on_reasoning_delta(text || '', {
      attemptNumber: Math.max(1, Number(options.attempt_number) || 1)
    })
  }

  const emitContentDelta = async (text, kind) => {
    if (!text || typeof options.on_content_delta !== 'function') return
    streamedVisibleText += text
    streamedControlKind = kind || streamedControlKind
    await options.on_content_delta(text, {
      attemptNumber: Math.max(1, Number(options.attempt_number) || 1),
      kind: streamedControlKind
    })
  }

  const consumeControlStreamResult = async (result) => {
    if (!result || !controlToolStreamParser) return
    if (result.textDelta) {
      const parsed = controlToolStreamParser.push(result.textDelta)
      await emitContentDelta(parsed.textDelta, result.kind)
      recoveredContent += parsed.recoveredText
    }
    if (result.closed && !controlToolParserFlushed) {
      controlToolParserFlushed = true
      const parsed = controlToolStreamParser.flush()
      await emitContentDelta(parsed.textDelta, result.kind)
      recoveredContent += parsed.recoveredText
    }
  }

  const appendAnswer = async (text) => {
    if (!text) return
    answer += text
    if (controlStreamParser) {
      await consumeControlStreamResult(controlStreamParser.push(text))
    }
  }

  let reasoning = ''
  let answer = ''
  let answerStarted = false
  let webSearchInfo = null
  let upstreamFinishReason = null
  let acceptedResponseId = null
  const createdByResponseId = new Map()
  let primaryCreated = null
  let lastCreated = null
  const emittedImages = new Set()
  const pendingImages = []
  let totalTokens = {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0
  }

  const streamResult = await consumeSSEStream(upstreamResponse, async (frame) => {
    if (!frame.data || frame.data.trim() === '[DONE]') return
    const decoded = isJson(frame.data) ? JSON.parse(frame.data) : null
    if (decoded === null) return
    assertNoUpstreamFailure(decoded)

    const created = normalizeCreatedMetadata(decoded)
    if (created) {
      lastCreated = created
      if (created.responseId) createdByResponseId.set(created.responseId, created)
      const responseIndex = created.responseIndex === null || created.responseIndex === ''
        ? Number.NaN
        : Number(created.responseIndex)
      if (created.responseId && Number.isFinite(responseIndex) && responseIndex === 0) {
        primaryCreated = created
        acceptedResponseId = created.responseId
      }
    }

    if (!acceptUpstreamFrame(decoded)) return
    if (decoded.response_id) acceptedResponseId = decoded.response_id

    if (decoded.usage) {
      totalTokens = {
        prompt_tokens: decoded.usage.prompt_tokens || totalTokens.prompt_tokens,
        completion_tokens: decoded.usage.completion_tokens || totalTokens.completion_tokens,
        total_tokens: decoded.usage.total_tokens || totalTokens.total_tokens
      }
    }
    if (!Array.isArray(decoded.choices) || decoded.choices.length === 0) return

    const choice = decoded.choices[0]
    const reportedFinishReason = choice.finish_reason ?? choice.delta?.finish_reason
    if (reportedFinishReason !== undefined && reportedFinishReason !== null) {
      upstreamFinishReason = reportedFinishReason
    }

    const delta = choice.delta || {}
    const rawPhase = delta.phase
    if (nativeTools) {
      feedNativeFrame(nativeTools, delta, reportedFinishReason, {
        isClientToolName,
        drain: drainPromotedNativeCalls
      })
    }

    if (delta.name === 'web_search') {
      webSearchInfo = delta.extra?.web_search_info || webSearchInfo
    }

    const normalized = normalizeDelta(delta)
    const phase = normalized?.phase || null
    const images = imageMarkdownFromDelta(delta)
      .filter(item => !emittedImages.has(item) && !pendingImages.includes(item))
    if (images.length > 0) {
      if (phase === 'think' && !answerStarted) {
        pendingImages.push(...images)
      } else {
        const markdown = `${images.join('\n\n')}\n\n`
        await appendAnswer(markdown)
        images.forEach(item => emittedImages.add(item))
      }
    }

    if (!normalized) return
    if (nativeTools && isProseResume(delta, normalized, rawPhase)) {
      // 正文恢复关闭打开中的调用（过闸的随即晋升）。批次已齐 —— 每个客户端调用都被自己的
      // 结果帧关闭且至少一个过闸 —— 这一帧就是"工具不存在"叙述的开头：提前终止上游，
      // 内容丢弃。批次不齐则永不早停，照旧消费到底。
      if (nativeTools.closeOpen('boundary')) drainPromotedNativeCalls()
      if (nativeBatchComplete(nativeTools)) {
        stopRequested = true
        logger.warn('OpenAI Agent 原生工具批次已晋升，提前终止上游（用量按本地估算）', 'AGENT')
        return
      }
    }
    // 晋升之后的叙述（"工具不可用"）不进 answer/reasoning —— 调用前的正文已经在 answer 里了。
    if (promotedNativeCalls.length > 0) return
    if (normalized.phase === 'think') {
      reasoning += normalized.content
      if (reasoningStreamParser) {
        const streamed = reasoningStreamParser.push(normalized.content)
        await emitReasoningDelta(streamed.textDelta)
        recoveredReasoning += streamed.recoveredText
      }
      return
    }

    answerStarted = true
    if (pendingImages.length > 0) {
      await appendAnswer(`${pendingImages.join('\n\n')}\n\n`)
      pendingImages.forEach(item => emittedImages.add(item))
      pendingImages.length = 0
    }
    await appendAnswer(normalized.content)
  }, { shouldStop: () => stopRequested })

  if (reasoningStreamParser) {
    const streamed = reasoningStreamParser.flush()
    await emitReasoningDelta(streamed.textDelta)
    recoveredReasoning += streamed.recoveredText
  }
  if (controlStreamParser) {
    await consumeControlStreamResult(controlStreamParser.flush())
  }

  const textTools = hasTools
    ? parseToolCallsFromText(answer, { allowedToolNames })
    : { cleanedText: answer, toolCalls: [], errors: [] }
  const reasoningTools = hasTools && textTools.toolCalls.length === 0 && !textTools.cleanedText.trim()
    ? parseToolCallsFromText(reasoning, { allowedToolNames })
    : { cleanedText: reasoning, toolCalls: [], errors: [] }
  // 回合结束：打开中的原生调用按 round_end 关闭并排出，再 finalize() 单发结算 OpenAI
  // 形状的 tool_calls（原生的已排空，不会出来第二次）。
  let nativeToolCalls = []
  if (nativeTools) {
    nativeTools.closeOpen('round_end')
    drainPromotedNativeCalls()
    nativeToolCalls = [...promotedNativeCalls, ...nativeTools.finalize()]
  }
  // 部分 thinking 模型会把“整个可执行工具块”放进 think phase 后直接 EOF。
  // 仅当 thinking 除独立工具块外没有任何文字时才接纳，避免把推理中的示例或
  // 尚未决定执行的调用当成真实动作。
  const standaloneReasoningCalls = reasoningTools.toolCalls.length > 0 &&
    !reasoningTools.cleanedText.trim() &&
    reasoningTools.errors.length === 0
    ? reasoningTools.toolCalls
    : []
  // 原生在前（它先关闭），文本通道其次；同名同参数的后到副本按登记簿丢弃，再统一编号。
  const admitToolCall = createToolCallLedger()
  const toolCalls = [
    ...nativeToolCalls,
    ...(textTools.toolCalls.length > 0 ? textTools.toolCalls : standaloneReasoningCalls)
  ]
    .filter(call => {
      if (admitToolCall(call)) return true
      logger.warn(`OpenAI Agent 本轮重复的工具调用（${call.function.name}，跨通道同名同参数），丢弃后到的副本`, 'AGENT')
      return false
    })
    .map((call, index) => ({ ...call, index }))
  const toolErrors = [
    ...(textTools.errors || []),
    ...(textTools.toolCalls.length === 0 && !textTools.cleanedText.trim()
      ? (reasoningTools.errors || [])
      : []),
    ...(nativeTools?.getErrors?.() || [])
  ]
  const control = parseAgentControlText(textTools.cleanedText)
  const metadata = (acceptedResponseId && createdByResponseId.get(acceptedResponseId)) || primaryCreated || lastCreated || {
    chatId: null,
    parentId: null,
    responseId: acceptedResponseId
  }

  return {
    reasoning: standaloneReasoningCalls.length > 0 ? reasoningTools.cleanedText : reasoning,
    rawAnswer: answer,
    visibleText: control.text,
    controlKind: control.kind,
    streamedVisibleText,
    recoveredContent,
    recoveredReasoning,
    streamedControlKind,
    streamedControlState: controlStreamParser?.getState?.() || null,
    toolCalls,
    toolErrors,
    // 本轮过闸晋升的 Qwen 原生 function_call（已并入 toolCalls）。门禁凭它在 toolErrors
    // 否决与"正文不得与工具并存"之前接纳本轮。
    nativeToolCalls: promotedNativeCalls,
    // 平台拦截的现场证据：Defect A 丢弃的 role:function 帧的名字（去重、有上限）。
    // 门禁靠它识别"原生调用被平台吃掉、只剩叙述"的死亡回合。
    interceptedToolNames: normalizeDelta.interceptedToolNames,
    webSearchInfo,
    totalTokens,
    upstreamFinishReason,
    upstreamCompleted: streamResult.completed,
    upstreamEventCount: streamResult.eventCount,
    sawDone: streamResult.sawDone,
    metadata: {
      ...metadata,
      responseId: metadata?.responseId || acceptedResponseId || null
    }
  }
}

const requiresToolCall = (toolChoice) => {
  if (toolChoice === 'required') return true
  return !!(toolChoice && typeof toolChoice === 'object' && toolChoice.type === 'function' && toolChoice.function?.name)
}

const evaluateOpenAIAgentAttempt = (attempt, options = {}) => {
  const finishReason = attempt.upstreamFinishReason
  if (NON_RETRYABLE_FINISH_REASONS.has(finishReason)) {
    const normalized = finishReason === 'max_tokens' ? 'length' : finishReason
    return { accepted: true, finishReason: normalized, retryReason: null }
  }
  // 过闸的原生调用是结构化帧，比文本启发式更强的证据：有一个就接纳本轮 —— 排在
  // toolErrors 否决与"正文不得与工具并存"之前，不翻 agentTurnAllowProseWithTools。
  // 调用前的正文随 visibleText 交付；调用后的叙述在采集时就已丢弃。
  if ((attempt.nativeToolCalls?.length || 0) > 0) {
    return { accepted: true, finishReason: 'tool_calls', retryReason: null }
  }
  if (attempt.toolErrors.length > 0) {
    return { accepted: false, finishReason: null, retryReason: 'invalid_tool_call' }
  }
  if (attempt.toolCalls.length > 0) {
    if (!config.agentTurnAllowProseWithTools &&
        (attempt.controlKind !== 'empty' || attempt.visibleText.trim())) {
      return { accepted: false, finishReason: null, retryReason: 'invalid_tool_call' }
    }
    return { accepted: true, finishReason: 'tool_calls', retryReason: null }
  }
  if (requiresToolCall(options.tool_choice)) {
    return { accepted: false, finishReason: null, retryReason: 'required_tool' }
  }
  // 协议恢复防御（与 Anthropic 两个循环同族）。必须排在 final/blocked 接纳之前：
  // 事故正是以 <agent_final> 包着的失败叙述被当成合法完结交付出去的。
  // - intercepted：role:function 丢弃帧 = 平台吃掉了模型的原生调用，只剩叙述。
  // - malformed_protocol：方括号协议写坏（孤儿闭标记 / 开头裸负载）整段泄漏为
  //   可见正文。只是重试信号，泄漏的 JSON 永远不执行。
  // intercepted 在前——丢弃帧是更强的证据。protocol_recovery_used 表示共享的
  // 一次性恢复名额已用：跳过两个检查，让回合按原有规则交付（原样交付胜过死循环）。
  if (options.has_tools !== false && !options.protocol_recovery_used) {
    if ((attempt.interceptedToolNames?.length || 0) > 0) {
      return { accepted: false, finishReason: null, retryReason: 'intercepted' }
    }
    if (containsOrphanProtocolResidue(attempt.visibleText)) {
      return { accepted: false, finishReason: null, retryReason: 'malformed_protocol' }
    }
  }
  if (attempt.controlKind === 'final' || attempt.controlKind === 'blocked') {
    if (attempt.visibleText.trim()) {
      return { accepted: true, finishReason: 'stop', retryReason: null }
    }
    return { accepted: false, finishReason: null, retryReason: 'empty' }
  }
  if (attempt.controlKind === 'empty') {
    return { accepted: false, finishReason: null, retryReason: 'empty' }
  }
  if (attempt.controlKind === 'invalid_control') {
    return { accepted: false, finishReason: null, retryReason: 'invalid_control' }
  }
  if (config.agentTurnAcceptBareFinal && attempt.visibleText.trim()) {
    return { accepted: true, finishReason: 'stop', retryReason: null }
  }
  return { accepted: false, finishReason: null, retryReason: 'bare' }
}

const appendRetryHint = (requestBody, hint) => {
  const clone = requestBody && typeof requestBody === 'object'
    ? JSON.parse(JSON.stringify(requestBody))
    : {}
  const messages = Array.isArray(clone.messages) ? clone.messages : []
  if (messages.length === 0) {
    messages.push({ role: 'user', content: hint })
  } else {
    const last = messages[messages.length - 1]
    if (typeof last.content === 'string') {
      last.content = `${last.content}\n\n${hint}`
    } else if (Array.isArray(last.content)) {
      const textPart = last.content.find(part => part?.type === 'text')
      if (textPart) textPart.text = `${textPart.text || ''}\n\n${hint}`
      else last.content.unshift({ type: 'text', text: hint })
    } else {
      last.content = hint
    }
  }
  clone.messages = messages
  return clone
}

const exhaustedError = (attempt, retryReason) => {
  if (retryReason === 'empty' && !String(attempt?.reasoning || '').trim()) {
    return {
      status: 503,
      message: '上游连续返回空 Agent 回合，任务状态未被标记为完成',
      code: 'upstream_unavailable'
    }
  }
  const messages = {
    empty: '上游连续只返回思考内容，没有给出可执行工具调用或最终答复',
    bare: '上游连续返回未声明完成状态的文本，已阻止 Agent 将未完成任务误判为结束',
    invalid_control: '上游连续返回无效的 Agent 完成标记',
    invalid_tool_call: '上游连续返回残缺、非法或不存在的工具调用',
    required_tool: '上游连续违反 tool_choice，未返回要求的工具调用',
    intercepted: '上游的工具调用被平台拦截，重试后仍未恢复',
    malformed_protocol: '上游持续返回残缺的工具调用协议，未能恢复为可执行调用'
  }
  return {
    status: 429,
    message: messages[retryReason] || '上游未能生成有效的 Agent 回合',
    code: retryReason === 'invalid_tool_call' ? 'invalid_tool_call' : 'upstream_agent_turn_incomplete'
  }
}

/**
 * 执行严格 Agent 回合：每个 attempt 完全隔离；只有有效工具调用、显式完成/阻塞，
 * 或标准非重试终止原因才能提交给客户端。
 */
const runOpenAIAgentTurn = async (initialResponse, options = {}) => {
  const requestSender = options.sendChatRequest
  const maxAttempts = Math.min(
    6,
    Math.max(2, Number(options.agent_turn_max_attempts) || config.agentTurnMaxAttempts)
  )
  let currentResponse = initialResponse
  let lastAttempt = null
  let lastEvaluation = null
  let upstreamContext = { ...(options.upstream_context || {}) }
  const retryBaseBody = options.upstream_request_body || options.requestBody
  let attemptsMade = 0
  // 协议恢复重试（intercepted / malformed_protocol 共享）整个请求只允许一次。
  // 用过之后 evaluate 会跳过这两个检查，让第二次拦截/残缺按原有规则原样交付。
  let protocolRecoveryRetried = false

  const mergePresent = (base, extra) => {
    const merged = { ...base }
    for (const [key, value] of Object.entries(extra || {})) {
      if (value !== null && value !== undefined && value !== '') merged[key] = value
    }
    return merged
  }

  for (let attemptNumber = 1; attemptNumber <= maxAttempts; attemptNumber++) {
    attemptsMade = attemptNumber
    const attempt = await collectOpenAIAgentAttempt(currentResponse, {
      ...options,
      attempt_number: attemptNumber
    })
    const evaluation = evaluateOpenAIAgentAttempt(attempt, {
      ...options,
      protocol_recovery_used: protocolRecoveryRetried
    })
    lastAttempt = attempt
    lastEvaluation = evaluation
    upstreamContext = mergePresent(upstreamContext, attempt.metadata)

    if (evaluation.accepted) {
      // 恢复名额已用而本轮仍带拦截/残渣证据 = 第二次事故按原样交付。留一行日志，
      // 生产环境要能区分"提示被采纳、回合恢复"和"第二次、原样交付"。
      if (protocolRecoveryRetried && attempt.toolCalls.length === 0 &&
          ((attempt.interceptedToolNames?.length || 0) > 0 ||
            containsOrphanProtocolResidue(attempt.visibleText))) {
        const giveUpDrops = (attempt.interceptedToolNames?.length || 0) > 0
          ? ` (dropped: ${attempt.interceptedToolNames.join(', ')})`
          : ''
        logger.warn(
          `Agent 协议恢复重试已用完，第二次拦截/残缺协议按原样交付${giveUpDrops}`,
          'AGENT'
        )
      }
      // Solo ahora que la ronda quedó aceptada: si se hubiera emitido al vuelo, cada
      // intento rechazado habría dejado otra copia en el stream del cliente.
      if (attempt.recoveredReasoning && typeof options.on_reasoning_delta === 'function') {
        await options.on_reasoning_delta(attempt.recoveredReasoning, { attemptNumber })
      }
      if (attempt.recoveredContent && typeof options.on_content_delta === 'function') {
        await options.on_content_delta(attempt.recoveredContent, {
          attemptNumber,
          kind: attempt.streamedControlKind
        })
      }
      return {
        ok: true,
        attempt,
        finishReason: evaluation.finishReason,
        attempts: attemptNumber
      }
    }

    // 有丢弃帧时任何拒绝理由都带上名字：invalid_tool_call/required_tool 优先级更高
    // 时拦截会被盖住，这行日志是生产环境验证拦截确实发生的抓手。
    const dropSuffix = (attempt.interceptedToolNames?.length || 0) > 0
      ? `; dropped: ${attempt.interceptedToolNames.join(', ')}`
      : ''
    logger.warn(
      `Agent attempt ${attemptNumber}/${maxAttempts} 被回合门禁拒绝 (${evaluation.retryReason}${dropSuffix})`,
      'AGENT'
    )
    if (attempt.streamedVisibleText) {
      return {
        ok: false,
        error: {
          status: 422,
          message: '上游在已开始流式输出正式回复后返回了无效的 Agent 结束结构',
          code: 'upstream_agent_stream_invalidated'
        },
        attempt,
        attempts: attemptNumber
      }
    }
    if (attemptNumber >= maxAttempts || typeof requestSender !== 'function') break

    if (evaluation.retryReason === 'intercepted' || evaluation.retryReason === 'malformed_protocol') {
      protocolRecoveryRetried = true
    }
    let retryHint = buildAgentRetryHint(evaluation.retryReason)
    // 别的理由（invalid_tool_call/required_tool）盖住拦截时，提示词仍要把关键
    // 事实带上：调用没到客户端。不动优先级、不动名额。
    if (evaluation.retryReason !== 'intercepted' &&
        attempt.toolCalls.length === 0 &&
        (attempt.interceptedToolNames?.length || 0) > 0) {
      retryHint = `${retryHint}\n${buildAgentRetryHint('intercepted')}`
    }
    const retryBody = appendRetryHint(retryBaseBody, retryHint)
    const retryResponse = await requestSender(retryBody, {
      chatId: upstreamContext.chatId || null,
      parentId: upstreamContext.responseId || null,
      currentAccount: options.currentAccount || null,
      agentRetry: true
    })
    if (!retryResponse?.status || !retryResponse.response) {
      return {
        ok: false,
        error: {
          status: 502,
          message: retryResponse?.message || 'Agent 回合纠正请求失败',
          code: 'upstream_retry_failed'
        },
        attempt,
        attempts: attemptNumber
      }
    }
    currentResponse = retryResponse.response
    upstreamContext = mergePresent(upstreamContext, {
      chatId: retryResponse.chatId,
      currentAccount: retryResponse.currentAccount
    })
  }

  return {
    ok: false,
    error: exhaustedError(lastAttempt, lastEvaluation?.retryReason),
    attempt: lastAttempt,
    attempts: attemptsMade
  }
}

module.exports = {
  NON_RETRYABLE_FINISH_REASONS,
  normalizeCreatedMetadata,
  collectOpenAIAgentAttempt,
  evaluateOpenAIAgentAttempt,
  appendRetryHint,
  runOpenAIAgentTurn,
  // chat.js 旧路径共用的原生帧喂入
  feedNativeFrame
}
