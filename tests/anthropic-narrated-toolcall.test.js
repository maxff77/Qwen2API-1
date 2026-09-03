// Spec narrated-toolcall-and-inner-quote-repair (2026-09-02): las llamadas NARRADAS
// llegan al cliente. Reproduccion del incidente de la sesion de Claude Code del
// 2026-09-02 (5 llamadas por el canal de texto, solo Read#1 ejecutada): la puerta de
// POSICION del parser compartido tiraba cualquier `[TOOL CALL]…[END TOOL CALL]`
// precedido de prosa, un payload sin opener rechazado envenenaba el resto del lote, y
// la cadena de reparacion no sabia escapar comillas internas. Aqui se pina el wire
// Anthropic de punta a punta: tool_use tras prosa, el lote del incidente (fixture en
// disco, chunks de 9 bytes), y que ningun caso "sin retry" toque al sender.
//
// Harness: los mismos helpers de anthropic-native-toolcall.test.js (runStream,
// toolUsesOf, visibleTextOf), copiados — cada archivo de test corre en su proceso.
//
// Set before anything pulls in config/index.js, which snapshots env at load.
// node --test runs each file in its own process, so this cannot leak.
process.env.AGENT_TURN_MAX_ATTEMPTS = '3';

const test = require('node:test');
const { describe, it } = test;
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { Readable } = require('node:stream');

// Sin red en tests: mismos parches de require-cache que anthropic-native-toolcall.test.js.
const modelsMap = require('../src/models/models-map.js');
modelsMap.getLatestModels = async () => { throw new Error('offline test: no model fetch'); };
const requestModule = require('../src/utils/request.js');
requestModule.sendChatRequest = async () => ({ status: false });

const { handleAnthropicStream } = require('../src/controllers/anthropic.js');

test.after(() => {
  require('../src/utils/account.js').destroy();
});

const createMockStreamResponse = () => ({
  output: '',
  headers: {},
  writableEnded: false,
  destroyed: false,
  set(headers) { Object.assign(this.headers, headers); return this; },
  status() { return this; },
  write(chunk) { this.output += String(chunk); return true; },
  end(chunk = '') { this.output += String(chunk); this.writableEnded = true; }
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

/** El texto entero en frames de `chunk` bytes — la forma del incidente en el wire. */
const chunkedTurn = (text, chunk = 9) => () => {
  const frames = [];
  for (let i = 0; i < text.length; i += chunk) frames.push(answerFrame(text.slice(i, i + chunk)));
  return Readable.from([...frames, STOP]);
};

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

// Herramientas del fixture (spec, Code Map): Read requiere file_path, Bash requiere
// command (description opcional).
const ALLOWED = ['Read', 'Bash', 'Edit', 'Write', 'Glob', 'Grep'];
const SCHEMAS = {
  Read: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] },
  Bash: {
    type: 'object',
    properties: { command: { type: 'string' }, description: { type: 'string' } },
    required: ['command']
  },
  Edit: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] },
  Write: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] },
  Glob: { type: 'object', properties: { pattern: { type: 'string' } }, required: ['pattern'] },
  Grep: { type: 'object', properties: { pattern: { type: 'string' } }, required: ['pattern'] }
};

const baseCtx = (sendRequest, overrides) => ({
  message_id: 'msg_narrated',
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

/** Texto visible por bloque de texto (indice → texto), para afirmar sobre CADA bloque. */
const textBlocksOf = (output) => {
  const blocks = new Map();
  for (const event of eventsOf(output)) {
    if (event.type === 'content_block_start' && event.content_block?.type === 'text') {
      blocks.set(event.index, '');
    }
    if (event.type === 'content_block_delta' && event.delta?.type === 'text_delta') {
      blocks.set(event.index, (blocks.get(event.index) || '') + event.delta.text);
    }
  }
  return [...blocks.values()];
};

const visibleTextOf = (output) => eventsOf(output)
  .filter(event => event.type === 'content_block_delta' && event.delta?.type === 'text_delta')
  .map(event => event.delta.text)
  .join('');

const stopReasonOf = (output) => eventsOf(output).find(event => event.type === 'message_delta')?.delta?.stop_reason;

const FIXTURE = fs.readFileSync(
  path.join(__dirname, 'fixtures', 'incident-2026-09-02-narrated-batch.txt'),
  'utf8'
);
const EXPECTED_BASH_COMMANDS = [
  'cd "/work/payroll" && ls -la node_modules/.bin/tsc 2>/dev/null || echo "no tsc"',
  'cd "/work/payroll" && ls -la node_modules/.bin/ 2>/dev/null | head -20',
  'cd "/work/payroll" && cat package.json | head -30'
];

const GOOD_READ_CALL = '[TOOL CALL]{"name":"Read","arguments":{"file_path":"a"}}[END TOOL CALL]';

describe('narrated tool calls reach the client (spec 2026-09-02)', () => {
  it('AC2: prose + canonical call → text block with only the prose, then ONE tool_use, stop_reason tool_use, zero retries', async () => {
    const sender = scriptedSender(turnOf(answerFrame('retry would consume this')));
    const res = await runStream(chunkedTurn(
      'Let me check.\n\n[TOOL CALL]{"name":"Read","arguments":{"file_path":"package.json"}}\n[END TOOL CALL]'
    ), sender);

    assert.equal(sender.calls.length, 0, 'a narrated call must burn no retry');
    const uses = toolUsesOf(res.output);
    assert.deepEqual(uses.map(u => u.name), ['Read']);
    assert.deepEqual(JSON.parse(uses[0].args), { file_path: 'package.json' });
    assert.deepEqual(textBlocksOf(res.output).map(t => t.trim()), ['Let me check.'], 'exactly one text block, only the prose');
    const events = eventsOf(res.output);
    const textStart = events.findIndex(e => e.type === 'content_block_start' && e.content_block?.type === 'text');
    const toolStart = events.findIndex(e => e.type === 'content_block_start' && e.content_block?.type === 'tool_use');
    assert.ok(textStart !== -1 && toolStart > textStart, 'the tool_use block follows the prose block');
    assert.equal(stopReasonOf(res.output), 'tool_use');
    assert.doesNotMatch(res.output, /TOOL CALL/i, 'zero protocol bytes on the wire');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('AC1: the incident fixture in 9-byte chunks → 5 tool_use in order, exact Bash commands, no [END in any text block, no retry', async () => {
    assert.doesNotMatch(FIXTURE, /\/Users\//, 'the fixture must carry no personal paths');
    const sender = scriptedSender(turnOf(answerFrame('retry would consume this')));
    const res = await runStream(chunkedTurn(FIXTURE, 9), sender);

    assert.equal(sender.calls.length, 0, 'the batch must burn no retry');
    const uses = toolUsesOf(res.output);
    assert.deepEqual(uses.map(u => u.name), ['Read', 'Bash', 'Read', 'Bash', 'Bash']);
    assert.deepEqual(
      uses.filter(u => u.name === 'Bash').map(u => JSON.parse(u.args).command),
      EXPECTED_BASH_COMMANDS,
      'inner quotes must survive the repair byte-for-byte'
    );
    assert.deepEqual(
      uses.filter(u => u.name === 'Read').map(u => JSON.parse(u.args).file_path),
      ['/work/payroll/package.json', '/work/payroll/scripts/verify-story-1-5.ts']
    );
    for (const block of textBlocksOf(res.output)) {
      assert.doesNotMatch(block, /\[END/, 'no text block may carry a closer');
      assert.equal(block.trim(), '', 'the batch has no prose');
    }
    assert.doesNotMatch(res.output, /TOOL_?CALL/i, 'zero protocol bytes on the wire');
    assert.equal(stopReasonOf(res.output), 'tool_use');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('after-prose call that fails the semantic gate: no tool_use, prose delivered, span consumed, NO retry (sender never called)', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    // "Note:" no matchea looksLikeUnexecutedToolAction: la unica razon de retry posible
    // seria un error o un residuo, y el spec prohibe ambos para fallos tras prosa.
    const res = await runStream(chunkedTurn('Note:\n[TOOL CALL]{"name":"Bash","arguments":{}}[END TOOL CALL]'), sender);

    assert.equal(sender.calls.length, 0, 'an after-prose gate failure must never be coaxed into a retry');
    assert.deepEqual(toolUseNames(res.output), []);
    assert.equal(visibleTextOf(res.output).trim(), 'Note:');
    assert.doesNotMatch(res.output, /TOOL CALL/i, 'the span is consumed, never delivered');
    assert.equal(stopReasonOf(res.output), 'end_turn');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('after-prose opener-less payload failing the gate: payload stays visible, closer consumed, NO retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    const res = await runStream(chunkedTurn('Note:\n{"name":"Bash","arguments":{}}\n[END TOOL CALL]'), sender);

    assert.equal(sender.calls.length, 0, 'a visible orphan closer would fire malformed_protocol — it must be consumed');
    assert.deepEqual(toolUseNames(res.output), []);
    const visible = visibleTextOf(res.output);
    assert.match(visible, /"name":"Bash"/, 'the rejected payload may BE the answer: it is delivered');
    assert.doesNotMatch(res.output, /\[END/, 'the closer bytes are consumed');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('after-prose opener-less payload + closer that passes the gate (G5) → tool_use, prose delivered, no retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame('retry would consume this')));
    const res = await runStream(chunkedTurn('Reading:\n{"name":"Read","arguments":{"file_path":"a"}}\n[END TOOL CALL]'), sender);

    assert.equal(sender.calls.length, 0);
    assert.deepEqual(toolUseNames(res.output), ['Read']);
    assert.equal(visibleTextOf(res.output).trim(), 'Reading:');
    assert.equal(stopReasonOf(res.output), 'tool_use');
  });

  it('a first-position {"name":"X",…}\\n[END TOOL CALL] inside a THINK frame (X unknown) is unknown_tool evidence → thought_tool_call retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    const res = await runStream(turnOf(
      thinkFrame('{"name":"NotATool","arguments":{}}\n[END TOOL CALL]'),
      answerFrame('Done.')
    ), sender);

    assert.equal(sender.calls.length, 1, 'the leaked call in reasoning is evidence: exactly one retry');
    const hint = JSON.stringify(sender.calls[0]);
    assert.match(hint, /inside your hidden reasoning/, 'the retry reason must be thought_tool_call');
    assert.deepEqual(toolUseNames(res.output), ['Read'], 'the retry\'s call is forwarded');
    // El razonamiento se streamea en vivo como thinking_delta (comportamiento de hoy);
    // lo que no puede pasar es que el payload se ejecute o llegue como TEXTO.
    assert.doesNotMatch(visibleTextOf(res.output), /NotATool/, 'the leaked payload never reaches a text block');
    assert.doesNotMatch(res.output, /"name":"NotATool","input"/, 'the leaked payload is never promoted');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  // ── Pines del review loop 2 ──

  it('P8: narrated call + DOUBLED closer → one tool_use, prose delivered, no [END on the wire, NO retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame('retry would consume this')));
    const res = await runStream(chunkedTurn(
      'Let me check.\n\n[TOOL CALL]{"name":"Read","arguments":{"file_path":"a"}}[END TOOL CALL]\n[END TOOL CALL]'
    ), sender);

    assert.equal(sender.calls.length, 0, 'a doubled closer after a narrated call must burn no retry');
    assert.deepEqual(toolUseNames(res.output), ['Read']);
    assert.deepEqual(textBlocksOf(res.output).map(t => t.trim()), ['Let me check.']);
    assert.doesNotMatch(res.output, /\[END/, 'the duplicate closer is consumed, never streamed');
    assert.equal(stopReasonOf(res.output), 'tool_use');
  });

  it('P8: after-prose gate failure + DOUBLED closer → span and both closers consumed, prose delivered, NO retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    const res = await runStream(chunkedTurn('Note:\n[TOOL CALL]{"name":"Bash","arguments":{}}[END TOOL CALL]\n[END TOOL CALL]'), sender);

    assert.equal(sender.calls.length, 0);
    assert.deepEqual(toolUseNames(res.output), []);
    assert.equal(visibleTextOf(res.output).trim(), 'Note:');
    assert.doesNotMatch(res.output, /\[END/);
    assert.equal(stopReasonOf(res.output), 'end_turn');
  });

  it('P6: first-position hard rejection with a DOUBLED closer (leak-sample-#2 shape) → tool_error retry, and no [END byte ever reaches the wire', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    const res = await runStream(chunkedTurn('{"name":"NotATool","arguments":{}}\n[END TOOL CALL]\n[END TOOL CALL]'), sender);

    assert.equal(sender.calls.length, 1, 'unknown_tool at first position is tool_error evidence: one retry');
    assert.deepEqual(toolUseNames(res.output), ['Read'], 'the retry\'s call is forwarded');
    assert.doesNotMatch(res.output, /\[END/, 'the duplicate closer used to reach the wire as text');
    assert.doesNotMatch(res.output, /NotATool/);
    assert.doesNotMatch(res.output, /"type":"error"/);
  });

  it('P10: after-prose unbalanced payload cut at a closer → debris visible, closer consumed, NO malformed_protocol retry', async () => {
    const sender = scriptedSender(turnOf(answerFrame(GOOD_READ_CALL)));
    const res = await runStream(chunkedTurn('Note:\n{"name":"Bash","arguments":{"command":"echo {"}\n[END TOOL CALL]\nMore.'), sender);

    assert.equal(sender.calls.length, 0, 'a visible orphan closer would have fired malformed_protocol after prose');
    assert.deepEqual(toolUseNames(res.output), []);
    const visible = visibleTextOf(res.output);
    assert.match(visible, /^Note:\n/);
    assert.match(visible, /More\.$/);
    assert.doesNotMatch(res.output, /\[END/, 'the closer bytes are consumed');
    assert.equal(stopReasonOf(res.output), 'end_turn');
    assert.doesNotMatch(res.output, /"type":"error"/);
  });
});
