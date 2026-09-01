// Promocion de los function_call NATIVOS de Qwen a tool_use (D2 + D3 del plan
// fix/native-toolcall-promotion). El modelo llama a las herramientas del cliente por la
// via nativa de la plataforma; upstream streamea `delta.function_call` con `arguments`
// como SNAPSHOT acumulativo (el final llega dos veces, sin function_id, phase "answer"),
// despues la plataforma inyecta `role:function "Tool X does not exists."` y el modelo
// narra 30-60 s que "no tiene herramientas". Aqui se pina que esas llamadas se vuelven
// tool_use al cerrarse, que la narracion jamas llega al cliente y que el upstream se
// corta en cuanto el lote esta completo (paridad call/result + primera prosa).
//
// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
// The config clamps this to [2, 6]; 3 keeps the cap-mutation tests short.
process.env.AGENT_TURN_MAX_ATTEMPTS = '3';

const test = require('node:test');
const { describe, it } = test;
const assert = require('node:assert/strict');
const { Readable } = require('node:stream');

// Los parches de require-cache van ANTES de requerir el controller (misma disciplina
// que anthropic-salvage-wiring.test.js): anthropic.js captura sendChatRequest por
// destructuring en su primer require. Sin red en tests: el fetch de modelos revienta
// y chat-helpers cae a sus fallbacks por nombre.
const modelsMap = require('../src/models/models-map.js');
modelsMap.getLatestModels = async () => { throw new Error('offline test: no model fetch'); };

const requestModule = require('../src/utils/request.js');
let e2eUpstreamFactory = null;
requestModule.sendChatRequest = async () => ({
  status: true,
  response: e2eUpstreamFactory(),
  currentAccount: null
});

const {
  handleAnthropicStream,
  handleAnthropicNonStream,
  handleAnthropicMessages
} = require('../src/controllers/anthropic.js');
const { logger } = require('../src/utils/logger.js');

test.after(() => {
  require('../src/utils/account.js').destroy();
});

/** Spy sobre logger.warn (el metodo REAL — logger.warning no existe en el singleton). */
const captureWarns = async (fn) => {
  const saved = logger.warn;
  const lines = [];
  logger.warn = (message) => { lines.push(String(message)); };
  try {
    await fn();
  } finally {
    logger.warn = saved;
  }
  return lines;
};

const createMockStreamResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  destroyed: false,
  set(headers) {
    Object.assign(this.headers, headers);
    return this;
  },
  status() {
    return this;
  },
  write(chunk) {
    this.output += String(chunk);
    return true;
  },
  end(chunk = '') {
    this.output += String(chunk);
    this.writableEnded = true;
  }
});

const createMockJsonResponse = () => ({
  statusCode: 200,
  body: null,
  headers: {},
  set(headers) {
    Object.assign(this.headers, headers);
    return this;
  },
  status(code) {
    this.statusCode = code;
    return this;
  },
  json(payload) {
    this.body = payload;
    return this;
  }
});

const answerFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'answer', content }, finish_reason: null }]
})}\n\n`;

const thinkFrame = (content) => `data: ${JSON.stringify({
  choices: [{ delta: { phase: 'think', content }, finish_reason: null }]
})}\n\n`;

const STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

/** One upstream turn from raw SSE frames, then a clean stop. */
const turnOf = (...frames) => () => Readable.from([...frames, STOP]);

const scriptedSender = (...turns) => {
  const queue = [...turns];
  const fn = async (body) => {
    fn.calls.push(body);
    const next = queue.shift();
    return next ? { status: true, response: next() } : { status: false };
  };
  fn.calls = [];
  return fn;
};

// ---- Fixtures byte-fieles a scratchpad/capture-foreign.txt (probe 2026-09-01, qwen3.8-max) ----
// Frame de llamada del cliente: role assistant, content '', phase answer, status typing,
// function_call {name, arguments}, extra.display_position answer, SIN function_id.
const nativeCallFrame = (name, snapshot) => `data: ${JSON.stringify({
  choices: [{
    delta: {
      role: 'assistant',
      content: '',
      phase: 'answer',
      status: 'typing',
      function_call: { name, arguments: snapshot },
      extra: { display_position: 'answer' }
    },
    finish_reason: null
  }]
})}\n\n`;

// Frame de herramienta de PLATAFORMA (code_interpreter): phase propia + function_id round_N_call_<hex>.
const platformCallFrame = (name, snapshot, id) => `data: ${JSON.stringify({
  choices: [{
    delta: {
      role: 'assistant',
      content: '',
      phase: name,
      status: 'typing',
      function_call: { name, arguments: snapshot },
      function_id: id,
      extra: { display_position: 'answer' }
    },
    finish_reason: null
  }]
})}\n\n`;

// Lookup de registry de la plataforma: role function, punto final incluido, name.
const notExistsFrame = (name) => `data: ${JSON.stringify({
  choices: [{
    delta: {
      role: 'function',
      content: `Tool ${name} does not exists.`,
      phase: 'answer',
      status: 'typing',
      name
    },
    finish_reason: null
  }]
})}\n\n`;

// Resultado de la herramienta de plataforma: status finished + extra.tool_result.
const platformResultFrame = (name, id, toolResult) => `data: ${JSON.stringify({
  choices: [{
    delta: {
      role: 'function',
      content: '',
      phase: name,
      status: 'finished',
      name,
      extra: {
        function_id: id.replace(/^round_\d+_/, ''),
        tool_result: toolResult,
        code_interpreter_info: toolResult,
        display_position: 'answer'
      },
      function_id: id
    },
    finish_reason: null
  }]
})}\n\n`;

// Frame #45 de la captura: fin de la respuesta (status finished, sin finish_reason).
const FINISHED_FRAME = `data: ${JSON.stringify({
  choices: [{ delta: { content: '', role: 'assistant', status: 'finished', phase: 'answer' }, finish_reason: null }]
})}\n\n`;

const SEND_MESSAGE_ARGS = '{"to": "riky", "message": "build is green"}';
const SEND_MESSAGE_PREFIXES = [
  '',
  '{"to": ',
  '{"to": "riky',
  '{"to": "riky"',
  '{"to": "riky", "message": ',
  '{"to": "riky", "message": "build is green',
  '{"to": "riky", "message": "build is green"'
];
const SEND_MESSAGE_SNAPSHOTS = [...SEND_MESSAGE_PREFIXES, SEND_MESSAGE_ARGS, SEND_MESSAGE_ARGS];

const BASH_ARGS = '{"command": "git status", "description": "Check git status on user\'s machine"}';
// Puntos de corte de los frames #11-#18; el final (#19, #20) llega DOS veces — estructural.
const BASH_PREFIXES = [
  '',
  '{"command": ',
  '{"command": "git status',
  '{"command": "git status"',
  '{"command": "git status", "description": "Check',
  '{"command": "git status", "description": "Check git status on user',
  '{"command": "git status", "description": "Check git status on user\'s machine',
  '{"command": "git status", "description": "Check git status on user\'s machine"'
];
const BASH_SNAPSHOTS = [...BASH_PREFIXES, BASH_ARGS, BASH_ARGS];
for (const prefix of SEND_MESSAGE_PREFIXES) assert.ok(SEND_MESSAGE_ARGS.startsWith(prefix), `cut point must be a prefix: ${prefix}`);
for (const prefix of BASH_PREFIXES) assert.ok(BASH_ARGS.startsWith(prefix), `cut point must be a prefix: ${prefix}`);

const LS_ARGS = '{"command": "ls"}';
const LS_SNAPSHOTS = ['', '{"command": ', LS_ARGS, LS_ARGS];

const CODE_INTERPRETER_ID = 'round_0_call_45542fe59a8346bf888dd458';
const CODE_INTERPRETER_SNAPSHOTS = ['', '{"code": "ls', '{"code": "ls -1 /tmp"}', '{"code": "ls -1 /tmp"}'];
const SANDBOX_RESULT = '```\nCount: 1\nFiles:\njail.log\n\n```';

const nativeTurn = (name, snapshots) => snapshots.map(snapshot => nativeCallFrame(name, snapshot));

// Narracion de la captura (#23-#42): NO matchea looksLikeUnexecutedToolAction.
const NARRATION_PIECES = [
  'The', ' required', ' tools `SendMessage`', ' and `Bash', '` are not available',
  ' in my current environment', '. I', ' only have access to', ' `code_interpreter',
  '`, `web_search', '`, `web_extractor', '`, and `web', '_search_image`. Therefore',
  ', I cannot send', ' a', ' message to teammate "', 'riky" or', ' run `git status',
  '` on your machine', '.'
];
const NARRATION_FRAMES = NARRATION_PIECES.map(answerFrame);
const NARRATION_MARKER = 'not available in my current environment';

// La captura completa del incidente: 9 SendMessage, 10 Bash, 2 result, narracion, finished.
const FOREIGN_TURN_FRAMES = [
  ...nativeTurn('SendMessage', SEND_MESSAGE_SNAPSHOTS),
  ...nativeTurn('Bash', BASH_SNAPSHOTS),
  notExistsFrame('SendMessage'),
  notExistsFrame('Bash'),
  ...NARRATION_FRAMES,
  FINISHED_FRAME
];

/**
 * Upstream que registra cada frame que el consumidor le PIDE. Se entrega el generador
 * crudo (consumeSSEStream solo necesita Symbol.asyncIterator): Readable.from
 * pre-cargaria hasta highWaterMark objetos y served[] mentiria.
 */
const recordingUpstream = (frames) => {
  const served = [];
  async function* gen() {
    for (const frame of frames) {
      served.push(frame);
      yield frame;
    }
  }
  return { served, stream: gen() };
};

const ALLOWED = ['SendMessage', 'Bash'];
const SCHEMAS = {
  SendMessage: {
    type: 'object',
    properties: { to: { type: 'string' }, message: { type: 'string' } },
    required: ['to', 'message']
  },
  Bash: {
    type: 'object',
    properties: {
      command: { type: 'string' },
      description: { type: 'string' },
      timeout: { type: 'number' }
    },
    required: ['command']
  }
};

const baseCtx = (sendRequest, overrides) => ({
  message_id: 'msg_native',
  model: 'qwen-test',
  hasTools: true,
  toolChoice: 'auto',
  allowedToolNames: ALLOWED,
  toolSchemas: SCHEMAS,
  requestBody: { messages: [] },
  sendRequest,
  ...overrides
});

const runStream = (upstream, sendRequest, overrides = {}) => {
  const res = createMockStreamResponse();
  return handleAnthropicStream(res, baseCtx(sendRequest, overrides), upstream()).then(() => res);
};

const runNonStream = (upstream, sendRequest, overrides = {}) => {
  const res = createMockJsonResponse();
  return handleAnthropicNonStream(res, baseCtx(sendRequest, overrides), upstream()).then(() => res);
};

/** Eventos Anthropic del wire, en orden. */
const eventsOf = (output) => output
  .split('\n\n')
  .filter(Boolean)
  .map(chunk => chunk.split('\n').find(line => line.startsWith('data: ')))
  .filter(Boolean)
  .map(line => JSON.parse(line.slice(6)));

/** Bloques tool_use reconstruidos (nombre + arguments concatenados por indice). */
const toolUsesOf = (output) => {
  const blocks = new Map();
  for (const event of eventsOf(output)) {
    if (event.type === 'content_block_start' && event.content_block?.type === 'tool_use') {
      blocks.set(event.index, { id: event.content_block.id, name: event.content_block.name, args: '' });
    }
    if (event.type === 'content_block_delta' && event.delta?.type === 'input_json_delta') {
      const block = blocks.get(event.index);
      if (block) block.args += event.delta.partial_json;
    }
  }
  return [...blocks.values()];
};

const toolUseNames = (output) => toolUsesOf(output).map(block => block.name);

const visibleTextOf = (output) => eventsOf(output)
  .filter(event => event.type === 'content_block_delta' && event.delta?.type === 'text_delta')
  .map(event => event.delta.text)
  .join('');

const thinkingTextOf = (output) => eventsOf(output)
  .filter(event => event.type === 'content_block_delta' && event.delta?.type === 'thinking_delta')
  .map(event => event.delta.thinking)
  .join('');

const stopReasonOf = (output) => eventsOf(output).find(event => event.type === 'message_delta')?.delta?.stop_reason;

const assertHeadlineWire = (res, sender) => {
  assert.equal(sender.calls.length, 0, 'native promotion must burn no retry');
  const uses = toolUsesOf(res.output);
  assert.deepEqual(uses.map(u => u.name), ['SendMessage', 'Bash']);
  assert.equal(uses[0].args, SEND_MESSAGE_ARGS, 'SendMessage arguments byte-exact (snapshot REPLACE, not +=)');
  assert.equal(uses[1].args, BASH_ARGS, 'Bash arguments byte-exact: the doubled final snapshot is ONE call');
  assert.ok(uses.every(u => /^call_[0-9a-f]{24}$/.test(u.id)), 'fresh ids, never a platform function_id');
  assert.equal(stopReasonOf(res.output), 'tool_use');
  assert.doesNotMatch(res.output, /"type":"error"/);
  assert.match(res.output, /"type":"message_stop"/);
};

describe('native function_call promotion (stream): the capture-foreign incident', () => {
  it('headline: two tool_use [SendMessage, Bash], args byte-exact, stop_reason tool_use, zero retries', async () => {
    const sender = scriptedSender();
    const res = await runStream(turnOf(...FOREIGN_TURN_FRAMES), sender);
    assertHeadlineWire(res, sender);
  });

  it('early stop: zero narration bytes on the wire AND the narration frames are never pulled from upstream', async () => {
    const sender = scriptedSender();
    const { served, stream } = recordingUpstream([...FOREIGN_TURN_FRAMES, STOP]);
    let res;
    const warns = await captureWarns(async () => {
      res = await runStream(() => stream, sender);
    });

    assertHeadlineWire(res, sender);
    assert.equal(res.output.includes(NARRATION_MARKER), false, 'zero narration bytes anywhere on the wire');
    assert.equal(visibleTextOf(res.output), '', 'no text block at all in a pure native round');
    // El corte es en la PRIMERA prosa tras la paridad: "The" se lee (es el frame de parada),
    // el resto de la narracion jamas se pide al upstream.
    const firstProseAt = FOREIGN_TURN_FRAMES.indexOf(NARRATION_FRAMES[0]);
    assert.equal(served.length, firstProseAt + 1, `upstream must stop on the first prose-resume frame, served ${served.length}`);
    assert.equal(served.some(frame => frame.includes(NARRATION_MARKER)), false, 'narration frames never pulled');
    assert.ok(
      warns.some(line => /提前终止上游/.test(line)),
      `expected one early-stop line, got:\n${warns.join('\n')}`
    );
    assert.ok(
      warns.filter(line => /原生工具调用晋升/.test(line)).length === 2,
      `expected one provenance line per promoted call, got:\n${warns.join('\n')}`
    );
    assert.equal(warns.some(line => line.includes('git status')), false, 'provenance never logs the arguments');
  });

  it('a Readable.from upstream that never sends [DONE] still settles cleanly (no error event, tool_use stop)', async () => {
    const sender = scriptedSender();
    const res = await runStream(() => Readable.from(FOREIGN_TURN_FRAMES), sender);
    assertHeadlineWire(res, sender);
  });

  it('usage fallback: an early-stopped round bills the local estimate, not ~0 output tokens', async () => {
    const sender = scriptedSender();
    const res = await runStream(turnOf(...FOREIGN_TURN_FRAMES), sender);
    const usage = eventsOf(res.output).find(event => event.type === 'message_delta')?.usage;
    assert.ok(usage && usage.output_tokens > 5, `emitted arguments must feed the estimate, got ${JSON.stringify(usage)}`);
  });

  it('chunk boundaries: the same turn split mid-JSON every 37 bytes yields the identical wire', async () => {
    const sender = scriptedSender();
    const raw = [...FOREIGN_TURN_FRAMES, STOP].join('');
    const chunks = [];
    for (let i = 0; i < raw.length; i += 37) chunks.push(raw.slice(i, i + 37));
    const res = await runStream(() => Readable.from(chunks), sender);
    assertHeadlineWire(res, sender);
    assert.equal(res.output.includes(NARRATION_MARKER), false);
  });

  it('prose BEFORE the native frames: prose delivered AND tool_use emitted (position gate does not apply to the native channel)', async () => {
    // Inverso deliberado del pin de texto en anthropic-toolcall-salvage.test.js:127-131
    // ("canonical call after prose ... suppressed"): un frame estructurado es evidencia
    // mas fuerte que la heuristica de posicion del canal de texto. La frase ademas
    // matchea looksLikeUnexecutedToolAction — con tool_use emitido ningun retry cabe.
    const sender = scriptedSender();
    const res = await runStream(
      turnOf(answerFrame('Let me check the repo first.'), ...FOREIGN_TURN_FRAMES),
      sender
    );
    assertHeadlineWire(res, sender);
    assert.equal(visibleTextOf(res.output), 'Let me check the repo first.');
    const proseAt = res.output.indexOf('Let me check the repo first.');
    const toolUseAt = res.output.indexOf('"type":"tool_use"');
    assert.ok(proseAt !== -1 && toolUseAt > proseAt, 'prose block precedes the tool_use blocks');
  });

  it('thinking after parity (prod 18:05 shape): the FIRST think frame after the result frames stops the upstream', async () => {
    // En produccion (2026-09-01 18:05:48) el modelo, tras la interceptacion, siguio PENSANDO 54s
    // antes de emitir prosa. Con tool_use ya en el cable, esperar la prosa es hacer esperar al
    // cliente esos 54s: la parada debe disparar en el primer frame con contenido, think o answer.
    const sender = scriptedSender();
    const thinkTail = [thinkFrame('The tools seem to be missing, let me reconsider...'), thinkFrame(' maybe I should report this.')];
    const frames = [
      ...nativeTurn('SendMessage', SEND_MESSAGE_SNAPSHOTS),
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      notExistsFrame('SendMessage'),
      notExistsFrame('Bash'),
      ...thinkTail,
      ...NARRATION_FRAMES,
      FINISHED_FRAME,
      STOP
    ];
    const { served, stream } = recordingUpstream(frames);
    const res = await runStream(() => stream, sender);

    assertHeadlineWire(res, sender);
    const firstThinkAt = frames.indexOf(thinkTail[0]);
    assert.equal(served.length, firstThinkAt + 1, `must stop on the first think frame after parity, served ${served.length}`);
    assert.equal(thinkingTextOf(res.output).includes('reconsider'), false, 'the post-parity thinking is discarded, not streamed');
    assert.equal(res.output.includes(NARRATION_MARKER), false);
  });

  it('thinking BEFORE parity does not stop: a think frame between the call frames and their results is not narration', async () => {
    const sender = scriptedSender();
    const frames = [
      ...nativeTurn('SendMessage', SEND_MESSAGE_SNAPSHOTS),
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      thinkFrame('waiting for the tools...'),
      notExistsFrame('SendMessage'),
      notExistsFrame('Bash'),
      ...NARRATION_FRAMES,
      FINISHED_FRAME,
      STOP
    ];
    const { served, stream } = recordingUpstream(frames);
    const res = await runStream(() => stream, sender);

    assertHeadlineWire(res, sender);
    const firstProseAt = frames.indexOf(NARRATION_FRAMES[0]);
    assert.equal(served.length, firstProseAt + 1, 'parity is only reached after both result frames; the stop waits for them');
  });

  it('post-tool-use suppression: result frames never arrive → no early stop, narration still stays off the wire', async () => {
    const sender = scriptedSender();
    const frames = [...nativeTurn('Bash', BASH_SNAPSHOTS), ...NARRATION_FRAMES, FINISHED_FRAME, STOP];
    const { served, stream } = recordingUpstream(frames);
    const res = await runStream(() => stream, sender);

    assert.equal(served.length, frames.length, 'no parity ⇒ no early stop ⇒ the whole turn is consumed');
    assert.deepEqual(toolUseNames(res.output), ['Bash']);
    assert.equal(toolUsesOf(res.output)[0].args, BASH_ARGS);
    assert.equal(res.output.includes(NARRATION_MARKER), false, 'narration after a tool_use is account-only');
    assert.equal(visibleTextOf(res.output), '');
    assert.equal(stopReasonOf(res.output), 'tool_use');
    assert.equal(sender.calls.length, 0);
  });
});

describe('native promotion (stream): same-name calls, duplicates and reopen', () => {
  it('Bash then Bash with distinct args → two tool_use, in order', async () => {
    const sender = scriptedSender();
    const res = await runStream(turnOf(
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      ...nativeTurn('Bash', LS_SNAPSHOTS),
      notExistsFrame('Bash'),
      notExistsFrame('Bash'),
      ...NARRATION_FRAMES,
      FINISHED_FRAME
    ), sender);

    assert.equal(sender.calls.length, 0);
    const uses = toolUsesOf(res.output);
    assert.deepEqual(uses.map(u => u.name), ['Bash', 'Bash']);
    assert.deepEqual(uses.map(u => u.args), [BASH_ARGS, LS_ARGS]);
    assert.equal(res.output.includes(NARRATION_MARKER), false);
  });

  it('a late byte-identical reopen after the result frame is dropped, not a second call', async () => {
    const sender = scriptedSender();
    const res = await runStream(turnOf(
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      notExistsFrame('Bash'),
      nativeCallFrame('Bash', BASH_ARGS),
      ...NARRATION_FRAMES,
      FINISHED_FRAME
    ), sender);

    assert.equal(sender.calls.length, 0);
    assert.deepEqual(toolUsesOf(res.output).map(u => u.args), [BASH_ARGS]);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('cross-channel dedupe: a text [TOOL CALL] Bash plus the native Bash with the same args → ONE tool_use', async () => {
    const textCall = '[TOOL CALL]{"name":"Bash","arguments":{"description":"Check git status on user\'s machine","command":"git status"}}[END TOOL CALL]';
    const sender = scriptedSender();
    let res;
    const warns = await captureWarns(async () => {
      res = await runStream(turnOf(
        answerFrame(textCall),
        ...nativeTurn('Bash', BASH_SNAPSHOTS),
        notExistsFrame('Bash'),
        ...NARRATION_FRAMES,
        FINISHED_FRAME
      ), sender);
    });

    assert.equal(sender.calls.length, 0);
    assert.deepEqual(toolUseNames(res.output), ['Bash'], 'the later duplicate must be dropped');
    assert.ok(warns.some(line => /重复/.test(line) && /Bash/.test(line)), `expected a dedupe warn, got:\n${warns.join('\n')}`);
    assert.equal(stopReasonOf(res.output), 'tool_use');
  });
});

describe('native promotion (stream): gates and platform tools keep today\'s contract', () => {
  it('allowedToolNames: [] → zero tool_use, exactly one (suppressed) retry — fail closed (R2 twin)', async () => {
    const sender = scriptedSender(turnOf(answerFrame('No tools were actually declared.')));
    const res = await runStream(turnOf(...FOREIGN_TURN_FRAMES), sender, { allowedToolNames: [], toolSchemas: {} });

    assert.equal(sender.calls.length, 1, 'must retry, never promote without a whitelist');
    assert.deepEqual(toolUseNames(res.output), [], 'promotion fired without a whitelist');
    assert.match(res.output, /"type":"message_stop"/);
  });

  it('name not in allowlist + not-exists frames + narration → one retry carrying the allowed names (D5 fallback)', async () => {
    const READ_CALL = '[TOOL CALL]{"name":"Read","arguments":{"path":"a.txt"}}[END TOOL CALL]';
    const sender = scriptedSender(turnOf(answerFrame(READ_CALL)));
    const res = await runStream(
      turnOf(...nativeTurn('Bash', BASH_SNAPSHOTS), notExistsFrame('Bash'), ...NARRATION_FRAMES, FINISHED_FRAME),
      sender,
      { allowedToolNames: ['Read'], toolSchemas: {} }
    );

    assert.equal(sender.calls.length, 1, 'exactly one tool_error retry');
    assert.match(JSON.stringify(sender.calls[0]), /Use ONLY these exact tool names: Read/);
    assert.deepEqual(toolUseNames(res.output), ['Read']);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('code_interpreter round (function_id, own phase, finished + tool_result): NO early stop, sandbox narration delivered, one tool_error retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame('Done.')));
    const frames = [
      ...CODE_INTERPRETER_SNAPSHOTS.map(s => platformCallFrame('code_interpreter', s, CODE_INTERPRETER_ID)),
      platformResultFrame('code_interpreter', CODE_INTERPRETER_ID, SANDBOX_RESULT),
      answerFrame('There'), answerFrame(' is **1**'), answerFrame(' file in `/tmp'), answerFrame('`:\n\n*'),
      answerFrame('   `jail'), answerFrame('.log`'),
      FINISHED_FRAME,
      STOP
    ];
    const { served, stream } = recordingUpstream(frames);
    const res = await runStream(() => stream, sender, { allowedToolNames: ['Bash'], toolSchemas: {} });

    assert.equal(served.length, frames.length, 'a platform call is not a client batch: never stop early');
    assert.deepEqual(toolUseNames(res.output), [], 'platform-own calls are never promoted');
    assert.match(visibleTextOf(res.output), /jail\.log/, 'the model\'s sandbox narration still reaches the client');
    assert.equal(sender.calls.length, 1, 'unknown_tool: code_interpreter → one suppressed tool_error retry');
    assert.match(JSON.stringify(sender.calls[0]), /Use ONLY these exact tool names: Bash/);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('think-phase native frames (no function_id) are evidence, not promotion → thought_tool_call retry', async () => {
    const thinkNative = (snapshot) => `data: ${JSON.stringify({
      choices: [{
        delta: {
          role: 'assistant', content: '', phase: 'think', status: 'typing',
          function_call: { name: 'Bash', arguments: snapshot },
          extra: { display_position: 'think' }
        },
        finish_reason: null
      }]
    })}\n\n`;
    const BASH_CALL = '[TOOL CALL]{"name":"Bash","arguments":{"command":"git status"}}[END TOOL CALL]';
    const sender = scriptedSender(turnOf(answerFrame(BASH_CALL)));
    const res = await runStream(
      turnOf(thinkNative(''), thinkNative(LS_ARGS), answerFrame('The status has been reviewed.')),
      sender
    );

    assert.equal(sender.calls.length, 1, 'exactly one protocol-recovery retry');
    assert.match(JSON.stringify(sender.calls[0]), /emitted inside your hidden reasoning/);
    assert.deepEqual(toolUseNames(res.output), ['Bash'], 'the retry\'s bracket call is the only tool_use');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });
});

describe('native promotion (stream): block hygiene and truncation', () => {
  it('a call closing inside an open thinking block: signature_delta + stop, then tool_use, then nothing', async () => {
    const sender = scriptedSender();
    const res = await runStream(turnOf(
      thinkFrame('Let me think about the repo state.'),
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      notExistsFrame('Bash'),
      thinkFrame('Hmm, the tool was rejected.'),
      ...NARRATION_FRAMES,
      FINISHED_FRAME
    ), sender);

    const events = eventsOf(res.output);
    const signatureAt = events.findIndex(e => e.type === 'content_block_delta' && e.delta?.type === 'signature_delta');
    const toolUseAt = events.findIndex(e => e.type === 'content_block_start' && e.content_block?.type === 'tool_use');
    assert.ok(signatureAt !== -1 && toolUseAt > signatureAt, 'thinking closes with signature_delta before the tool_use');
    assert.equal(events[signatureAt + 1].type, 'content_block_stop');
    const toolUseStopAt = events.findIndex((e, i) => i > toolUseAt && e.type === 'content_block_stop');
    const after = events.slice(toolUseStopAt + 1).map(e => e.type);
    assert.deepEqual(after, ['message_delta', 'message_stop'], `nothing after the tool_use block, got ${after.join(',')}`);
    assert.equal(thinkingTextOf(res.output), 'Let me think about the repo state.', 'post-tool-use thinking is account-only');
    assert.deepEqual(toolUseNames(res.output), ['Bash']);
  });

  it('truncated snapshot at EOF: no tool_use, bounded tool_error retries, explicit invalid_tool_call_error', async () => {
    const truncated = () => Readable.from([nativeCallFrame('Bash', '{"command": "git st')]);
    const sender = scriptedSender(truncated, truncated, truncated, truncated);
    const res = await runStream(truncated, sender);

    assert.equal(sender.calls.length, 2, 'AGENT_TURN_MAX_ATTEMPTS=3 ⇒ at most two retries, never a loop');
    assert.deepEqual(toolUseNames(res.output), []);
    assert.match(res.output, /"type":"error"/);
    assert.match(res.output, /truncated_native_call/);
  });

  it('truncated snapshot + finish_reason length: native-origin errors fire NO retry, no tool_use', async () => {
    const LENGTH_STOP = 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n';
    const sender = scriptedSender(turnOf(answerFrame('unused')));
    const res = await runStream(
      () => Readable.from([nativeCallFrame('Bash', '{"command": "git st'), LENGTH_STOP]),
      sender
    );

    assert.equal(sender.calls.length, 0, 'terminal finish: native-origin tool errors never retry');
    assert.deepEqual(toolUseNames(res.output), []);
    assert.match(res.output, /truncated_native_call/);
  });
});

describe('native promotion (non-stream twin)', () => {
  it('headline: content array = the two tool_use blocks, no 502, stop_reason tool_use, narration dropped', async () => {
    const sender = scriptedSender();
    const { served, stream } = recordingUpstream([...FOREIGN_TURN_FRAMES, STOP]);
    const res = await runNonStream(() => stream, sender);

    assert.equal(res.statusCode, 200, `expected delivery, got ${JSON.stringify(res.body?.error || null)}`);
    assert.equal(sender.calls.length, 0);
    const blocks = res.body?.content || [];
    assert.deepEqual(blocks.map(b => b.type), ['tool_use', 'tool_use']);
    assert.deepEqual(blocks.map(b => b.name), ['SendMessage', 'Bash']);
    assert.deepEqual(blocks[0].input, JSON.parse(SEND_MESSAGE_ARGS));
    assert.deepEqual(blocks[1].input, JSON.parse(BASH_ARGS));
    assert.equal(res.body.stop_reason, 'tool_use');
    assert.equal(JSON.stringify(res.body).includes(NARRATION_MARKER), false);
    const firstProseAt = FOREIGN_TURN_FRAMES.indexOf(NARRATION_FRAMES[0]);
    assert.equal(served.length, firstProseAt + 1, 'non-stream stops early too');
  });

  it('cross-channel dedupe replaces the concat: text Bash + native Bash same args → one tool_use', async () => {
    const textCall = '[TOOL CALL]{"name":"Bash","arguments":{"command":"git status","description":"Check git status on user\'s machine"}}[END TOOL CALL]';
    const sender = scriptedSender();
    const res = await runNonStream(turnOf(
      answerFrame(textCall),
      ...nativeTurn('Bash', BASH_SNAPSHOTS),
      notExistsFrame('Bash'),
      ...NARRATION_FRAMES,
      FINISHED_FRAME
    ), sender);

    assert.equal(res.statusCode, 200);
    const uses = (res.body?.content || []).filter(b => b.type === 'tool_use');
    assert.equal(uses.length, 1, 'ledger dedupe, not concat');
    assert.equal(uses[0].name, 'Bash');
  });

  it('allowedToolNames: [] → no tool_use, one retry (fail closed)', async () => {
    const sender = scriptedSender(turnOf(answerFrame('No tools were actually declared.')));
    const res = await runNonStream(turnOf(...FOREIGN_TURN_FRAMES), sender, { allowedToolNames: [], toolSchemas: {} });

    assert.equal(sender.calls.length, 1);
    assert.equal(res.statusCode, 200);
    assert.deepEqual((res.body?.content || []).filter(b => b.type === 'tool_use'), []);
  });
});

describe('production wiring: /v1/messages tools[].input_schema → toolSchemas → native gate (e2e)', () => {
  const SEND_MESSAGE_TOOL = {
    name: 'SendMessage',
    description: 'message a teammate',
    input_schema: SCHEMAS.SendMessage
  };
  const BASH_TOOL = { name: 'Bash', description: 'run a shell command', input_schema: SCHEMAS.Bash };

  it('non-stream request with real tools promotes both native calls through buildInternalRequest', async () => {
    e2eUpstreamFactory = () => Readable.from([...FOREIGN_TURN_FRAMES, STOP]);
    const req = {
      body: {
        model: 'qwen3-coder-plus',
        max_tokens: 512,
        stream: false,
        messages: [{ role: 'user', content: 'tell riky the build is green and check git status' }],
        tools: [SEND_MESSAGE_TOOL, BASH_TOOL]
      }
    };
    const res = createMockJsonResponse();
    await handleAnthropicMessages(req, res);

    assert.equal(res.statusCode, 200, `expected delivery, got ${JSON.stringify(res.body?.error || null)}`);
    const uses = (res.body?.content || []).filter(b => b.type === 'tool_use');
    assert.deepEqual(uses.map(u => u.name), ['SendMessage', 'Bash']);
    assert.equal(uses[1].input.command, 'git status');
    assert.equal(res.body.stop_reason, 'tool_use');
  });

  it('stream request: same promotion on the wire, zero narration bytes', async () => {
    e2eUpstreamFactory = () => Readable.from([...FOREIGN_TURN_FRAMES, STOP]);
    const req = {
      body: {
        model: 'qwen3-coder-plus',
        max_tokens: 512,
        stream: true,
        messages: [{ role: 'user', content: 'tell riky the build is green and check git status' }],
        tools: [SEND_MESSAGE_TOOL, BASH_TOOL]
      }
    };
    const res = createMockStreamResponse();
    await handleAnthropicMessages(req, res);

    assert.deepEqual(toolUseNames(res.output), ['SendMessage', 'Bash']);
    assert.equal(res.output.includes(NARRATION_MARKER), false);
    assert.equal(stopReasonOf(res.output), 'tool_use');
  });

  it('input_schema is load-bearing: a native call missing a required key is NOT promoted (schema_mismatch)', async () => {
    const missingCommand = [
      nativeCallFrame('Bash', ''),
      nativeCallFrame('Bash', '{"description": "no command here"}'),
      nativeCallFrame('Bash', '{"description": "no command here"}'),
      notExistsFrame('Bash'),
      FINISHED_FRAME,
      STOP
    ];
    e2eUpstreamFactory = () => Readable.from(missingCommand);
    const req = {
      body: {
        model: 'qwen3-coder-plus',
        max_tokens: 512,
        stream: false,
        messages: [{ role: 'user', content: 'check git status' }],
        tools: [BASH_TOOL]
      }
    };
    const res = createMockJsonResponse();
    await handleAnthropicMessages(req, res);

    const uses = (res.body?.content || []).filter(b => b.type === 'tool_use');
    assert.equal(uses.length, 0, 'a required key missing must never gate through');
    assert.equal(res.statusCode, 502);
    assert.equal(res.body?.error?.type, 'invalid_tool_call_error');
    assert.match(res.body?.error?.message || '', /schema_mismatch/);
  });
});
