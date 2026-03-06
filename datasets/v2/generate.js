import fs from "fs";
import crypto from "crypto";

// ─── Config ───────────────────────────────────────────────────────────────────

const NV_API_KEY = process.env.NV_API_KEY
const MODE = process.argv[2];
const FILL_MODE = process.argv[3] === "--fill";

if (!MODE || !["keyword", "heartbeat"].includes(MODE)) {
  console.error("Usage: bun generator.js <keyword|heartbeat> [--fill]");
  process.exit(1);
}

const OUTPUT_FILE = FILL_MODE
  ? MODE === "heartbeat"
    ? "heartbeat_dataset_deduped.txt"
    : "keyword_dataset_deduped.txt"
  : MODE === "heartbeat"
    ? "heartbeat_dataset.txt"
    : "keyword_dataset.txt";

const MODEL = "openai/gpt-oss-120b";
const TARGET_TOTAL = 2500;
const MAX_BATCHES = 300;
const BASE_DELAY_MS = 1500;
const MAX_RETRIES = 5; // wait out rate limits, no fallback here

// ─── Targets ──────────────────────────────────────────────────────────────────

const TARGET =
  MODE === "heartbeat"
    ? {
        byMsgCount: { 2: 250, 3: 375, 4: 625, 5: 1250 },
        byAction: { communicate: 1125, send_media: 750, ignore: 625 },
      }
    : {
        byMsgCount: { 2: 250, 3: 500, 4: 750, 5: 1000 },
        byAction: { communicate: 1125, send_media: 750, ignore: 625 },
      };

const TARGET_MSG_DIST = {
  2: { MSG1: 0.3, MSG2: 0.7 },
  3: { MSG1: 0.2, MSG2: 0.3, MSG3: 0.5 },
  4: { MSG1: 0.05, MSG2: 0.15, MSG3: 0.3, MSG4: 0.5 },
  5: { MSG1: 0.05, MSG2: 0.1, MSG3: 0.15, MSG4: 0.2, MSG5: 0.5 },
};

// ─── Prompts ──────────────────────────────────────────────────────────────────

const KEYWORD_SYSTEM = `You generate WhatsApp AI router training data. Output ONLY raw blocks, zero extra text.

SETUP
- Router sees 2–5 WhatsApp messages, decides action for the LAST message
- ONLY the last message contains a keyword: ai, bot, assistant
- Previous messages must NOT contain these keywords

ACTIONS
communicate  → user is asking AI something or giving a command
send_media   → user explicitly asks for meme/gif/sticker/reaction
ignore       → keyword appears but AI is NOT actually being addressed

LANGUAGE MIX
50% Hinglish (Roman Hindi + English), 25% Roman Hindi, 25% Casual English
Natural typos ok. No formal Hindi.

FORMAT
MSG1: text
MSG2: text
...
OUT: action|target|content|confidence

FIELD RULES
- target: ALWAYS the last MSG label (MSG2, MSG3, MSG4, MSG5). If ignore → null
- content: send_media only → 2–4 word query e.g. "funny meme". All others → null
- confidence: communicate=0.80–0.95, send_media=0.85–0.98, ignore=0.75–0.90. Vary values.

HARD RULES
- ignore → target=null, content=null
- communicate → target=last MSG, content=null
- send_media → target=last MSG, content=<short query>
- ONLY last message may contain keyword
- All previous messages MUST NOT contain keyword

EXAMPLES
MSG1: kal exam hai padha hi nahi
MSG2: mera bhi same haal
MSG3: ai koi motivation de
OUT: send_media|MSG3|exam stress meme|0.88

MSG1: bro phone hang ho raha
MSG2: restart kiya?
MSG3: haan fir bhi slow
MSG4: ai kya karu
OUT: communicate|MSG4|null|0.91

MSG1: yaar tu bilkul ai jaisa baat karta hai
MSG2: koi feelings hi nahi tujhme
OUT: ignore|null|null|0.82

MSG1: bhai kya chal raha
MSG2: kuch nahi yaar timepass
MSG3: same here, bored af
MSG4: chalo kuch karte hain
MSG5: bot ek funny meme bhej
OUT: send_media|MSG5|funny bored meme|0.93

GENERATION RULES
- Each batch MUST contain all three action types
- Unique scenarios only: sarcasm, ambiguity, emotional context, multi-person chats, indirect refs
- Commands to AI = communicate. Explicit meme/gif/sticker requests only = send_media
- Output ONLY blocks separated by exactly one blank line. No numbering, no commentary.`;

const HEARTBEAT_SYSTEM = `You generate heartbeat training data for a WhatsApp AI router. Output ONLY raw blocks. Zero explanations, zero markdown.

PURPOSE
Router passively monitors 2–5 WhatsApp messages and decides whether/how to act. No guaranteed keyword. Infer action from pure context.

ACTIONS
communicate  → AI should reply (question, confusion, advice, emotional support, direct request)
send_media   → AI should send meme/gif/sticker (boredom, sarcasm, embarrassment, celebration, frustration)
ignore       → AI should stay silent (normal chat, AI mentioned casually, irrelevant topic)

KEYWORD RULE
Keywords (ai, bot, assistant) may appear in ANY message or NOT AT ALL. Decide from context only.

TARGET SELECTION
target = most contextually relevant MSG label. If ignore → null
Distribution:
5-msg: MSG5=50%, MSG4=20%, MSG3=15%, MSG2=10%, MSG1=5%
4-msg: MSG4=50%, MSG3=30%, MSG2=15%, MSG1=5%
3-msg: MSG3=50%, MSG2=30%, MSG1=20%
2-msg: MSG2=70%, MSG1=30%

LANGUAGE MIX
50% Hinglish, 25% Roman Hindi, 25% Casual English. Typos ok. No formal Hindi.

FORMAT
MSG1: text
MSG2: text
...
OUT: action|target|content|confidence

FIELD RULES
- target: MSG label or null (ignore)
- content: send_media only → 2–4 word query. All others → null
- confidence: communicate=0.80–0.95, send_media=0.85–0.98, ignore=0.75–0.90. Vary values.

HARD RULES
- ignore → target=null, content=null
- communicate → target=MSG?, content=null
- send_media → target=MSG?, content=<short query>

EXAMPLES
MSG1: bhai kal interview hai
MSG2: kuch prepare kiya?
MSG3: nahi yaar sab bhool gaya
MSG4: chill kar, tu smart hai
MSG5: mujhe lagta hai fail hounga
OUT: communicate|MSG5|null|0.87

MSG1: omg I just spilled coffee all over my laptop
MSG2: broooo noooo
MSG3: yes and my presentation was on it
OUT: send_media|MSG3|disaster reaction gif|0.91

MSG1: kya scene hai aaj
MSG2: nothing just chilling
MSG3: same bhai same
MSG4: weekend waste ho gaya
OUT: ignore|null|null|0.83

GENERATION RULES
- Each batch MUST have all three action types. Vary target MSG per distribution above.
- Unique scenarios: sarcasm, ambiguity, emotional context, multi-person, indirect refs, daily life
- Output ONLY blocks separated by exactly one blank line. No numbering, no commentary.`;

const SYSTEM_PROMPT = MODE === "heartbeat" ? HEARTBEAT_SYSTEM : KEYWORD_SYSTEM;
const USER_PROMPT_PREFIX = `Generate exactly the following sample counts:`;

function buildUserPrompt(needed, actionNeeded) {
  const counts = Object.entries(needed)
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `${k}-message: ${n}`)
    .join(", ");
  const actions = Object.entries(actionNeeded)
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `${k}: ${n}`)
    .join(", ");
  return `${USER_PROMPT_PREFIX}\nMessage counts: ${counts}\nAction counts required: ${actions}`;
}

// ─── API Call (primary only, patient retry on 429) ────────────────────────────

async function callGroq(userPrompt, retryCount = 0) {
  if (retryCount >= MAX_RETRIES)
    throw new Error(`Rate limited after ${MAX_RETRIES} retries`);

  const res = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${NV_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      reasoning_effort: "low",
      max_completion_tokens: 16000,
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (res.status === 429) {
    const wait = BASE_DELAY_MS * Math.pow(2, retryCount);
    console.warn(
      `[${MODEL}] Rate limited. Retry ${retryCount + 1}/${MAX_RETRIES} in ${wait}ms...`,
    );
    await sleep(wait);
    return callGroq(userPrompt, retryCount + 1);
  }
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);

  let full = "";
  let cachedTokens = 0;
  for await (const chunk of res.body) {
    for (const line of new TextDecoder().decode(chunk).split("\n")) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const d = t.slice(5).trim();
      if (d === "[DONE]") continue;
      try {
        const json = JSON.parse(d);
        const usage = json.usage ?? json.x_groq?.usage;
        if (usage?.prompt_tokens_details?.cached_tokens)
          cachedTokens = usage.prompt_tokens_details.cached_tokens;
        const token = json.choices?.[0]?.delta?.content;
        if (token) {
          process.stdout.write(token);
          full += token;
        }
      } catch {}
    }
  }
  if (cachedTokens > 0)
    console.log(`\n[cache] ${cachedTokens} tokens from cache`);
  return full;
}

// ─── Parser ───────────────────────────────────────────────────────────────────

const KEYWORD_RE = /\b(ai|bot|assistant|oye ai|ai bhai|ai sun|bot help)\b/i;

function parseSamples(text) {
  const samples = [];
  for (const block of text.split(/\n{2,}/)) {
    const lines = block
      .trim()
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    const msgLines = lines.filter((l) => /^MSG\d:/.test(l));
    const outLine = lines.find((l) => l.startsWith("OUT:"));
    if (msgLines.length < 2 || msgLines.length > 5 || !outLine) continue;

    const msgCount = msgLines.length;
    const parts = outLine.slice(4).trim().split("|");
    if (parts.length !== 4) continue;

    const [action, target, content, confStr] = parts.map((p) => p.trim());
    const confidence = parseFloat(confStr);

    if (!["communicate", "send_media", "ignore"].includes(action)) continue;
    const confMin =
      action === "communicate" ? 0.8 : action === "send_media" ? 0.85 : 0.75;
    const confMax =
      action === "communicate" ? 0.95 : action === "send_media" ? 0.98 : 0.9;
    if (isNaN(confidence) || confidence < confMin || confidence > confMax)
      continue;

    if (action === "ignore") {
      if (target !== "null" || content !== "null") continue;
    } else {
      if (!/^MSG[1-5]$/.test(target)) continue;
      const msgNum = parseInt(target.replace("MSG", ""));
      if (msgNum < 1 || msgNum > msgCount) continue;
      if (action === "communicate" && content !== "null") continue;
      if (
        action === "send_media" &&
        (!content ||
          content === "null" ||
          content.trim().split(/\s+/).length > 5)
      )
        continue;
    }

    if (MODE === "keyword") {
      if (action !== "ignore" && target !== `MSG${msgCount}`) continue;
      const lastMsg = msgLines[msgLines.length - 1].replace(/^MSG\d:\s*/, "");
      const prevMsgs = msgLines
        .slice(0, -1)
        .map((l) => l.replace(/^MSG\d:\s*/, ""));
      if (!KEYWORD_RE.test(lastMsg)) continue;
      if (prevMsgs.some((m) => KEYWORD_RE.test(m))) continue;
    }

    samples.push({ msgCount, action, target, text: block.trim() });
  }
  return samples;
}

// ─── Dedupe ───────────────────────────────────────────────────────────────────

const seen = new Set();

function isDuplicate(text) {
  const h1 = crypto.createHash("sha1").update(text).digest("hex");
  const h2 = crypto
    .createHash("sha1")
    .update(text.toLowerCase().replace(/\s+/g, " ").trim())
    .digest("hex");
  const msgContent = text
    .split("\n")
    .filter((l) => /^MSG\d:/.test(l.trim()))
    .map((l) =>
      l
        .replace(/^MSG\d:\s*/, "")
        .toLowerCase()
        .trim(),
    )
    .join("|");
  const h3 = crypto.createHash("sha1").update(msgContent).digest("hex");
  if (seen.has(h1) || seen.has(h2) || seen.has(h3)) return true;
  seen.add(h1);
  seen.add(h2);
  seen.add(h3);
  return false;
}

// ─── Distribution ─────────────────────────────────────────────────────────────

function initCounts() {
  return {
    byMsgCount: { 2: 0, 3: 0, 4: 0, 5: 0 },
    byAction: { communicate: 0, send_media: 0, ignore: 0 },
    byTarget: {
      2: { MSG1: 0, MSG2: 0 },
      3: { MSG1: 0, MSG2: 0, MSG3: 0 },
      4: { MSG1: 0, MSG2: 0, MSG3: 0, MSG4: 0 },
      5: { MSG1: 0, MSG2: 0, MSG3: 0, MSG4: 0, MSG5: 0 },
    },
  };
}

function canAccept(s, counts) {
  if (TARGET.byMsgCount[s.msgCount] - counts.byMsgCount[s.msgCount] <= 0)
    return false;
  if (TARGET.byAction[s.action] - counts.byAction[s.action] <= 0) return false;
  if (MODE === "heartbeat" && s.target !== "null") {
    const total = counts.byMsgCount[s.msgCount];
    if (total > 10) {
      const expected = TARGET_MSG_DIST[s.msgCount][s.target] ?? 0;
      const current = (counts.byTarget[s.msgCount][s.target] ?? 0) / total;
      if (current > expected * 1.5) return false;
    }
  }
  return true;
}

function getNeeded(counts) {
  const needed = {};
  let total = 0;
  for (const [k, t] of Object.entries(TARGET.byMsgCount)) {
    const n = t - counts.byMsgCount[k];
    if (n > 0) {
      needed[k] = n;
      total += n;
    }
  }
  return { needed, total };
}

function getActionNeeded(counts) {
  const needed = {};
  for (const [k, t] of Object.entries(TARGET.byAction)) {
    const n = t - counts.byAction[k];
    if (n > 0) needed[k] = n;
  }
  return needed;
}

function buildBatchNeeded(needed, actionNeeded, cap = 20) {
  const msgNeeded = Object.fromEntries(
    Object.entries(needed).map(([k, n]) => [k, Math.min(n, cap)]),
  );
  const totalBatch = Object.values(msgNeeded).reduce((a, b) => a + b, 0);
  const totalActions = Object.values(actionNeeded).reduce((a, b) => a + b, 0);
  const scaledActions = Object.fromEntries(
    Object.entries(actionNeeded).map(([k, n]) => [
      k,
      Math.max(1, Math.round((n / totalActions) * totalBatch)),
    ]),
  );
  return { msgNeeded, scaledActions };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function printProgress(n, counts) {
  console.log(
    `\nProgress: ${n}/${TARGET_TOTAL} (${((n / TARGET_TOTAL) * 100).toFixed(1)}%)`,
  );
  console.log("MsgCount:", counts.byMsgCount);
  console.log("Actions: ", counts.byAction);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(
    `\nMode: ${MODE.toUpperCase()}${FILL_MODE ? " (FILL)" : ""} | Model: ${MODEL}`,
  );
  console.log(`File: ${OUTPUT_FILE}\n`);

  const counts = initCounts();
  const collected = [];

  if (fs.existsSync(OUTPUT_FILE)) {
    for (const s of parseSamples(fs.readFileSync(OUTPUT_FILE, "utf8"))) {
      if (isDuplicate(s.text)) continue;
      collected.push(s.text);
      counts.byMsgCount[s.msgCount]++;
      counts.byAction[s.action]++;
      if (
        s.target !== "null" &&
        counts.byTarget[s.msgCount]?.[s.target] !== undefined
      )
        counts.byTarget[s.msgCount][s.target]++;
    }
    console.log(`Loaded: ${collected.length} existing samples`);
    printProgress(collected.length, counts);
  }

  if (collected.length > 0) fs.appendFileSync(OUTPUT_FILE, "\n");

  const fd = fs.openSync(OUTPUT_FILE, "a");
  let batch = 0;

  while (collected.length < TARGET_TOTAL && batch < MAX_BATCHES) {
    batch++;
    const { needed, total } = getNeeded(counts);
    if (total === 0) break;
    const actionNeeded = getActionNeeded(counts);
    const { msgNeeded, scaledActions } = buildBatchNeeded(needed, actionNeeded);

    console.log(`\n══ Batch ${batch} | Remaining: ${total} ══`);
    console.log(
      `Requesting → msgs: ${JSON.stringify(msgNeeded)} | actions: ${JSON.stringify(scaledActions)}`,
    );

    let raw;
    try {
      raw = await callGroq(buildUserPrompt(msgNeeded, scaledActions));
    } catch (err) {
      console.error("Failed:", err.message);
      await sleep(10000);
      continue;
    }

    const parsed = parseSamples(raw);
    console.log(`\n\nParsed: ${parsed.length} valid`);

    let added = 0;
    for (const s of parsed) {
      if (collected.length >= TARGET_TOTAL) break;
      if (isDuplicate(s.text) || !canAccept(s, counts)) continue;
      fs.writeSync(fd, (collected.length === 0 ? "" : "\n\n") + s.text);
      collected.push(s.text);
      counts.byMsgCount[s.msgCount]++;
      counts.byAction[s.action]++;
      if (
        s.target !== "null" &&
        counts.byTarget[s.msgCount]?.[s.target] !== undefined
      )
        counts.byTarget[s.msgCount][s.target]++;
      added++;
    }

    console.log(`Added: ${added}`);
    printProgress(collected.length, counts);
    await sleep(BASE_DELAY_MS);
  }

  fs.closeSync(fd);
  console.log("\n══ COMPLETE ══");
  console.log(`Saved ${collected.length}/${TARGET_TOTAL} → ${OUTPUT_FILE}`);
  console.log("Final MsgCount:", counts.byMsgCount);
  console.log("Final Actions: ", counts.byAction);
  if (collected.length < TARGET_TOTAL)
    console.warn(`Stopped early at batch ${batch}. Increase MAX_BATCHES.`);
}

main().catch(console.error);
