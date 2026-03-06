const fs = require("fs");

const OUTPUT_FILE = "conflict_samples.jsonl";

// ============================================================
// SCHEMA CONTRACT — IDENTICAL TO MAIN GENERATOR
// ============================================================
//
// communicate:
//   operation: "text" | "voice" | "reaction" | "question"
//   tone:      "friendly" | "sarcastic" | "professional" | "angry" | "romantic" | "excited"
//   target:    string (optional)
//   content:   string
//   NO .type, NO .format, NO .mood
//
// media_action:
//   operation: "play" | "search" | "generate" | "send"
//   type:      "music" | "image" | "sticker" | "meme" | "video"
//   query:     string
//   tone:      string (free-form — mood/style)
//   NO .format, NO .mood, NO .content
//
// retrieve_info:
//   operation: "search" | "lookup" | "analyze"
//   type:      "web" | "user" | "music" | "file"   ← "music" for song lookup (no "song" enum anywhere)
//   query:     string
//   format:    "summary" | "full" | "lyrics" | "detailed"
//   NO .tone, NO .content
//
// process_text:
//   operation: "translate" | "summarize" | "rewrite" | "compose"
//   content:   string
//   target:    string (language or recipient context)
//   format:    "short" | "detailed" | "bullet" | "formal"
//   tone:      ONLY on compose → "friendly"|"sarcastic"|"professional"|"angry"|"romantic"|"excited"
//   NO .type, NO .query
//
// manage_memory:
//   operation: "store" | "recall" | "forget"
//   content:   string
//   NO other fields
//
// ignore:
//   params: {}
//
// ============================================================

// ---------- UTILS ----------
function rand(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function maybe(p = 0.5) { return Math.random() < p; }
function lowConfidence() {
  const r = Math.random();
  if (r < 0.40) return +(0.82 + Math.random() * 0.07).toFixed(2); // 0.82–0.89
  if (r < 0.75) return +(0.74 + Math.random() * 0.08).toFixed(2); // 0.74–0.81
  return +(0.70 + Math.random() * 0.04).toFixed(2);                // 0.70–0.73
}

// ---------- VOCAB ----------
const names = [
  "rahul","neha","amit","john","sara","virat","rohit","priya","kabir","zara",
  "arjun","meera","tanya","ishaan",
];
const songs = [
  "Despacito","Shape of You","Tum Hi Ho","Believer","Kesariya","Anti-Hero",
  "Flowers","Blinding Lights","Sunflower","Levitating","Peaches","As It Was",
  "Cruel Summer","Raataan Lambiyan","Vampire",
];
const topics = [
  "cricket","AI","climate change","bitcoin","travel","gaming","fitness",
  "mental health","space","football","movies","cooking","stock market","web dev","anime",
];
const languages = [
  "Hindi","English","Spanish","French","German","Punjabi","Tamil","Japanese","Marathi",
];

// Strict enum (communicate + process_text compose)
const COMMUNICATE_TONES = ["friendly","sarcastic","professional","angry","romantic","excited"];
// Free-form (media_action only)
const MEDIA_TONES = [
  "neutral","neutral","neutral",
  "friendly","friendly",
  "romantic","romantic",
  "excited","excited",
  "angry","sarcastic","professional",
];

// ============================================================
// CONFLICT TYPE 1: communicate  ←→  process_text
// ============================================================

const ct1_communicate_wins = [
  (n)    => ({
    user: `send ${n} a nice birthday wish`,
    action: "communicate",
    params: { operation: "text", content: "Happy Birthday! 🎉 Hope your day is amazing!", target: n, tone: "friendly" },
    conflict: { with: "process_text", reason: "content is short & conventional; no real composition needed — direct send suffices" },
  }),
  (n)    => ({
    user: `tell ${n} i'm sorry`,
    action: "communicate",
    params: { operation: "text", content: "Hey, I'm really sorry about that 🙏", target: n, tone: "friendly" },
    conflict: { with: "process_text", reason: "apology is a single short message; compose would be overkill" },
  }),
  (n)    => ({
    user: `${n} ko bol dena i'll be late`,
    action: "communicate",
    params: { operation: "text", content: "Running late, be there soon!", target: n, tone: "friendly" },
    conflict: { with: "process_text", reason: "informing someone is a direct message action, not a text transformation" },
  }),
  (n)    => ({
    user: `wish ${n} good luck for the match`,
    action: "communicate",
    params: { operation: "text", content: "Good luck for the match! You've got this 💪", target: n, tone: "excited" },
    conflict: { with: "process_text", reason: "short contextual wish — direct communicate is better; process_text adds no value" },
  }),
  (n)    => ({
    user: `${n} ko romantic msg bhejo na`,
    action: "communicate",
    params: { operation: "text", content: "Thinking of you… always 💭❤️", target: n, tone: "romantic" },
    conflict: { with: "process_text", reason: "user wants delivery not authoring; communicate with romantic tone resolves it" },
  }),
];

const ct1_process_wins = [
  (n)    => ({
    user: `write a professional apology email to ${n} and send it`,
    action: "process_text",
    // tone omitted on compose here because format:"formal" already carries the signal
    params: { operation: "compose", content: "professional apology email", format: "formal" },
    conflict: { with: "communicate", reason: "explicit 'write' + 'professional' signals heavy composition; communicate is secondary" },
  }),
  (n)    => ({
    user: `ek accha sa birthday message likh aur ${n} ko bhejo`,
    action: "process_text",
    params: { operation: "compose", content: "birthday wish message", format: "detailed", tone: "friendly" },
    conflict: { with: "communicate", reason: "'likh' (write) is the primary ask; composition precedes sending" },
  }),
  ()     => ({
    user: `compose a leave application and send to my manager`,
    action: "process_text",
    params: { operation: "compose", content: "leave application", format: "formal" },
    conflict: { with: "communicate", reason: "formal document composition is the core task; sending is implicit follow-up" },
  }),
  (n)    => ({
    user: `draft a breakup message for ${n} that's not too harsh`,
    action: "process_text",
    params: { operation: "compose", content: "breakup message", format: "short" },
    conflict: { with: "communicate", reason: "user explicitly says draft — wants a crafted text, not an immediate send" },
  }),
  ()     => ({
    user: `write a cold email to a client in formal English`,
    action: "process_text",
    params: { operation: "compose", content: "cold outreach email", format: "formal" },
    conflict: { with: "communicate", reason: "target is unspecified + 'write' emphasis = compose; no concrete recipient to communicate to" },
  }),
];

// ============================================================
// CONFLICT TYPE 2: media_action  ←→  retrieve_info
// ============================================================

const ct2_retrieve_wins = [
  (song) => ({
    user: `${song} ki lyrics chahiye`,
    action: "retrieve_info",
    // type:"music" + operation:"lookup" + format:"lyrics" = song info/lyrics lookup
    params: { operation: "lookup", type: "music", query: song, format: "lyrics" },
    conflict: { with: "media_action", reason: "user wants text content (lyrics), not audio playback" },
  }),
  (song) => ({
    user: `who sang ${song}?`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: song, format: "summary" },
    conflict: { with: "media_action", reason: "question about metadata/artist — factual lookup, not play/generate" },
  }),
  (topic) => ({
    user: `find me a video about ${topic}`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "full" },
    conflict: { with: "media_action", reason: "'find a video about' is a web search intent; media_action would play/send local media" },
  }),
  (song) => ({
    user: `${song} ka matlab kya hai? uski story kya hai?`,
    action: "retrieve_info",
    params: { operation: "lookup", type: "music", query: song, format: "detailed" },
    conflict: { with: "media_action", reason: "asking for meaning/story = info retrieval, not media playback" },
  }),
  (topic) => ({
    user: `search ${topic} tutorial for me`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: `${topic} tutorial`, format: "detailed" },
    conflict: { with: "media_action", reason: "'tutorial' implies educational content search, not entertainment media action" },
  }),
];

const ct2_media_wins = [
  (song) => ({
    user: `${song} sunna hai yaar`,
    action: "media_action",
    // type:"music" — correct for play
    params: { operation: "play", type: "music", query: song },
    conflict: { with: "retrieve_info", reason: "'sunna hai' = want to listen = play; not a lyric or info request" },
  }),
  (topic) => ({
    user: `send a ${topic} meme yaar`,
    action: "media_action",
    params: { operation: "send", type: "meme", query: topic },
    conflict: { with: "retrieve_info", reason: "explicit send + meme = media_action; not an information query" },
  }),
  (topic) => ({
    user: `koi ${topic} funny video bhejo`,
    action: "media_action",
    // tone is free-form on media_action — "funny" is valid here
    params: { operation: "search", type: "video", query: topic, tone: "excited" },
    conflict: { with: "retrieve_info", reason: "user wants media delivered (video), not factual information about topic" },
  }),
  (topic) => ({
    user: `generate a ${topic} wallpaper for me`,
    action: "media_action",
    params: { operation: "generate", type: "image", query: topic },
    conflict: { with: "retrieve_info", reason: "'generate' clearly signals media creation, not information retrieval" },
  }),
  (song) => ({
    user: `play something like ${song}`,
    action: "media_action",
    params: { operation: "play", type: "music", query: song, tone: "neutral" },
    conflict: { with: "retrieve_info", reason: "similar-to request = playback recommendation; no text info needed" },
  }),
];

// ============================================================
// CONFLICT TYPE 3: process_text  ←→  retrieve_info
// ============================================================

const ct3_process_wins = [
  (topic) => ({
    user: `summarize the latest news on ${topic}`,
    action: "process_text",
    // summarize: no tone
    params: { operation: "summarize", content: `latest news on ${topic}`, format: "short" },
    conflict: { with: "retrieve_info", reason: "user asks for a condensed output = summarize; retrieve would return raw results" },
  }),
  (lang) => ({
    user: `translate this ${lang} article for me`,
    action: "process_text",
    // translate: target = destination language
    params: { operation: "translate", content: "provided article", target: "English" },
    conflict: { with: "retrieve_info", reason: "transformation of existing content = process_text; no new search needed" },
  }),
  ()     => ({
    user: `explain AI in simple words`,
    action: "process_text",
    params: { operation: "rewrite", content: "AI explanation", format: "short" },
    conflict: { with: "retrieve_info", reason: "'in simple words' requests a transformation/rewrite, not raw retrieval" },
  }),
  (topic) => ({
    user: `ELI5: ${topic}`,
    action: "process_text",
    params: { operation: "rewrite", content: topic, format: "short" },
    conflict: { with: "retrieve_info", reason: "ELI5 is an explicit simplification request = process_text rewrite" },
  }),
  ()     => ({
    user: `bullet points bana do is text ke`,
    action: "process_text",
    params: { operation: "rewrite", content: "provided text", format: "bullet" },
    conflict: { with: "retrieve_info", reason: "reformatting existing text = process_text; nothing to search for" },
  }),
];

const ct3_retrieve_wins = [
  (topic) => ({
    user: `tell me about ${topic}`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "summary" },
    conflict: { with: "process_text", reason: "'tell me about' = info retrieval; no text is given to process" },
  }),
  (topic) => ({
    user: `${topic} kya hota hai?`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "summary" },
    conflict: { with: "process_text", reason: "open factual question with no input text = retrieve, not process" },
  }),
  (topic) => ({
    user: `${topic} ke baare mein latest update do`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "detailed" },
    conflict: { with: "process_text", reason: "'latest update' implies live info fetch; no transformation needed" },
  }),
  (topic) => ({
    user: `give me a detailed breakdown of ${topic}`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "detailed" },
    conflict: { with: "process_text", reason: "no source text provided to process — query implies fresh retrieval" },
  }),
  (topic) => ({
    user: `research ${topic} for me`,
    action: "retrieve_info",
    params: { operation: "analyze", type: "web", query: topic, format: "full" },
    conflict: { with: "process_text", reason: "'research' = gather new info; process_text needs existing input text" },
  }),
];

// ============================================================
// CONFLICT TYPE 4: communicate  ←→  ignore
// ============================================================

const ct4_ignore_wins = [
  () => ({ user: `hmm`,            action: "ignore", params: {}, conflict: { with: "communicate", reason: "single filler word; no action, target, or intent can be inferred" } }),
  () => ({ user: `ok`,             action: "ignore", params: {}, conflict: { with: "communicate", reason: "acknowledgement with no further instruction; nothing to route" } }),
  () => ({ user: `wait wait wait`, action: "ignore", params: {}, conflict: { with: "communicate", reason: "user is pausing; no actionable request present" } }),
  () => ({ user: `nvm`,            action: "ignore", params: {}, conflict: { with: "communicate", reason: "explicit cancellation of intent; ignore is correct" } }),
  () => ({ user: `lol ok whatever`,action: "ignore", params: {}, conflict: { with: "communicate", reason: "conversational filler; ambiguous but no routable action" } }),
  () => ({ user: `k`,              action: "ignore", params: {}, conflict: { with: "communicate", reason: "single-letter ack; no intent signal strong enough to route" } }),
  () => ({ user: `haha bhai`,      action: "ignore", params: {}, conflict: { with: "communicate", reason: "casual laugh reaction; no send target or content requested" } }),
  () => ({ user: `bro 💀`,         action: "ignore", params: {}, conflict: { with: "communicate", reason: "emoji reaction to unspecified content; not enough context to act" } }),
];

const ct4_communicate_wins = [
  (n) => ({
    user: `${n} ko bolo lol`,
    action: "communicate",
    params: { operation: "text", content: "lol 😂", target: n, tone: "friendly" },
    conflict: { with: "ignore", reason: "despite casual content, explicit target + 'bolo' = clear send instruction" },
  }),
  (n) => ({
    user: `haha send this to ${n}`,
    action: "communicate",
    params: { operation: "text", content: "haha 😂", target: n, tone: "friendly" },
    conflict: { with: "ignore", reason: "explicit target + 'send' overrides apparent noise; actionable route exists" },
  }),
  (n) => ({
    user: `${n} ko ok bol do`,
    action: "communicate",
    params: { operation: "text", content: "Ok 👍", target: n, tone: "friendly" },
    conflict: { with: "ignore", reason: "named target + 'bol do' = delegation to communicate; not self-directed filler" },
  }),
];

// ============================================================
// CONFLICT TYPE 5: manage_memory  ←→  retrieve_info
// ============================================================

const ct5_memory_wins = [
  (fact) => ({
    user: `do you remember my food preference?`,
    action: "manage_memory",
    params: { operation: "recall", content: "food preference" },
    conflict: { with: "retrieve_info", reason: "'do you remember' targets personal stored memory, not an external web search" },
  }),
  (fact) => ({
    user: `what have I told you before?`,
    action: "manage_memory",
    params: { operation: "recall", content: "all stored user facts" },
    conflict: { with: "retrieve_info", reason: "introspective question about conversation history = memory recall" },
  }),
  (fact) => ({
    user: `mujhe kya pasand tha bata`,
    action: "manage_memory",
    params: { operation: "recall", content: "user preferences" },
    conflict: { with: "retrieve_info", reason: "first-person preference query = recall from memory, not web lookup" },
  }),
  (fact) => ({
    user: `save that I now prefer ${fact}`,
    action: "manage_memory",
    params: { operation: "store", content: `user prefers ${fact}` },
    conflict: { with: "retrieve_info", reason: "'save that' = explicit store command; nothing to search for" },
  }),
  (fact) => ({
    user: `forget everything I said about my diet`,
    action: "manage_memory",
    params: { operation: "forget", content: "diet preferences" },
    conflict: { with: "retrieve_info", reason: "deletion command on personal memory; retrieve_info has no forget operation" },
  }),
];

const ct5_retrieve_wins = [
  (topic) => ({
    user: `remember when ${topic} happened? what was it about?`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: topic, format: "detailed" },
    conflict: { with: "manage_memory", reason: "public event recall with 'what was it about' = web lookup, not personal memory" },
  }),
  (topic) => ({
    user: `tell me what happened with ${topic} last year`,
    action: "retrieve_info",
    params: { operation: "search", type: "web", query: `${topic} last year`, format: "summary" },
    conflict: { with: "manage_memory", reason: "historical public event query; memory store wouldn't have this data" },
  }),
];

// ============================================================
// CONFLICT TYPE 6: media_action  ←→  communicate
// ============================================================

const ct6_media_wins = [
  (n, song) => ({
    user: `${song} ${n} ko bhejo`,
    action: "media_action",
    // type:"music" for a song send — correct media_action type
    params: { operation: "send", type: "music", query: song },
    conflict: { with: "communicate", reason: "media object (song) is primary; sending it is a media delivery action" },
  }),
  (n, topic) => ({
    user: `${n} ko ek ${topic} meme bhejo`,
    action: "media_action",
    params: { operation: "send", type: "meme", query: topic },
    conflict: { with: "communicate", reason: "media type explicitly stated; this is a media delivery, not a text chat" },
  }),
  (n, topic) => ({
    user: `share that ${topic} gif with ${n}`,
    action: "media_action",
    // gif → closest valid type = image
    params: { operation: "send", type: "image", query: topic },
    conflict: { with: "communicate", reason: "gif/image share = media_action; communicate handles text/voice/reaction" },
  }),
];

const ct6_communicate_wins = [
  (n, song) => ({
    user: `tell ${n} about ${song}, they'll love it`,
    action: "communicate",
    params: { operation: "text", content: `You have to listen to ${song}, it's amazing!`, target: n, tone: "excited" },
    conflict: { with: "media_action", reason: "'tell about' = verbal recommendation message, not a media file transfer" },
  }),
  (n) => ({
    user: `react to ${n}'s message with 🔥`,
    action: "communicate",
    params: { operation: "reaction", content: "🔥", target: n, tone: "excited" },
    conflict: { with: "media_action", reason: "emoji reaction on a message = communicate reaction operation, not media send" },
  }),
];

// ============================================================
// CONFLICT TYPE 7: process_text  ←→  communicate
// ============================================================

const ct7_process_wins = [
  (n, lang) => ({
    user: `translate this to ${lang} and send to ${n}`,
    action: "process_text",
    // translate: target = language
    params: { operation: "translate", content: "provided text", target: lang },
    conflict: { with: "communicate", reason: "transformation (translate) must happen first; communicate is the downstream step" },
  }),
  (n) => ({
    user: `make this sound professional aur ${n} ko bhejo`,
    action: "process_text",
    // rewrite: no tone
    params: { operation: "rewrite", content: "provided text", format: "formal" },
    conflict: { with: "communicate", reason: "rewrite is the primary task; sending is implied but secondary" },
  }),
  () => ({
    user: `summarize this article and share it`,
    action: "process_text",
    params: { operation: "summarize", content: "provided article", format: "short" },
    conflict: { with: "communicate", reason: "summarization must precede sharing; process_text handles the transformation" },
  }),
];

// ============================================================
// BATCH BUILDERS
// ============================================================
function buildConflictSample(rawSample) {
  const sample = typeof rawSample === "function" ? rawSample() : rawSample;
  return {
    messages: [
      { role: "user", content: sample.user },
      {
        role: "assistant",
        content: JSON.stringify({
          action: sample.action,
          params: sample.params,
          confidence: lowConfidence(),
          conflict: sample.conflict,
        }),
      },
    ],
  };
}

function expandTemplates(templateList, argGenerators, perTemplate = 3) {
  const results = [];
  for (const tpl of templateList) {
    for (let i = 0; i < perTemplate; i++) {
      const args = argGenerators.map((gen) => gen());
      results.push(JSON.stringify(buildConflictSample(tpl(...args))));
    }
  }
  return results;
}

// ============================================================
// SCHEMA VALIDATOR (same rules as main generator)
// ============================================================
const VALID_ACTIONS = new Set(["communicate","media_action","retrieve_info","process_text","manage_memory","ignore"]);
const COMMUNICATE_OPS  = new Set(["text","voice","reaction","question"]);
const COMMUNICATE_TONE_SET = new Set(["friendly","sarcastic","professional","angry","romantic","excited"]);
const MEDIA_OPS   = new Set(["play","search","generate","send"]);
const MEDIA_TYPES = new Set(["music","image","sticker","meme","video"]);
const RETRIEVE_OPS    = new Set(["search","lookup","analyze"]);
const RETRIEVE_TYPES  = new Set(["web","user","music","file"]);
const RETRIEVE_FORMATS = new Set(["summary","full","lyrics","detailed"]);
const PROCESS_OPS  = new Set(["translate","summarize","rewrite","compose"]);
const PROCESS_FORMATS = new Set(["short","detailed","bullet","formal"]);
const MEMORY_OPS   = new Set(["store","recall","forget"]);

function validateSample(obj, lineIdx) {
  const inner = JSON.parse(obj.messages[1].content);
  const { action, params } = inner;
  const errors = [];

  if (!VALID_ACTIONS.has(action)) { errors.push(`Invalid action: ${action}`); return errors; }

  if (action === "communicate") {
    if (!COMMUNICATE_OPS.has(params.operation))
      errors.push(`communicate.operation invalid: ${params.operation}`);
    if (params.tone && !COMMUNICATE_TONE_SET.has(params.tone))
      errors.push(`communicate.tone invalid: ${params.tone}`);
    if (params.type !== undefined)
      errors.push(`communicate must NOT have .type`);
    if (params.format !== undefined)
      errors.push(`communicate must NOT have .format`);
    if (params.mood !== undefined)
      errors.push(`communicate must NOT have .mood`);
  }

  if (action === "media_action") {
    if (!MEDIA_OPS.has(params.operation))
      errors.push(`media_action.operation invalid: ${params.operation}`);
    if (!MEDIA_TYPES.has(params.type))
      errors.push(`media_action.type invalid: "${params.type}" — valid: music|image|sticker|meme|video`);
    if (params.format !== undefined)
      errors.push(`media_action must NOT have .format`);
    if (params.mood !== undefined)
      errors.push(`media_action must NOT have .mood (use .tone)`);
    if (params.content !== undefined)
      errors.push(`media_action must NOT have .content`);
  }

  if (action === "retrieve_info") {
    if (!RETRIEVE_OPS.has(params.operation))
      errors.push(`retrieve_info.operation invalid: ${params.operation}`);
    if (!RETRIEVE_TYPES.has(params.type))
      errors.push(`retrieve_info.type invalid: "${params.type}" — valid: web|user|song|file`);
    if (params.format && !RETRIEVE_FORMATS.has(params.format))
      errors.push(`retrieve_info.format invalid: ${params.format}`);
    if (params.tone !== undefined)
      errors.push(`retrieve_info must NOT have .tone`);
    if (params.content !== undefined)
      errors.push(`retrieve_info must NOT have .content`);
  }

  if (action === "process_text") {
    if (!PROCESS_OPS.has(params.operation))
      errors.push(`process_text.operation invalid: ${params.operation}`);
    if (params.format && !PROCESS_FORMATS.has(params.format))
      errors.push(`process_text.format invalid: ${params.format}`);
    if (params.tone) {
      if (!COMMUNICATE_TONE_SET.has(params.tone))
        errors.push(`process_text.tone invalid: ${params.tone}`);
      if (params.operation !== "compose")
        errors.push(`process_text.tone only valid on compose, got operation:"${params.operation}"`);
    }
    if (params.type !== undefined)
      errors.push(`process_text must NOT have .type`);
    if (params.query !== undefined)
      errors.push(`process_text must NOT have .query`);
  }

  if (action === "manage_memory") {
    if (!MEMORY_OPS.has(params.operation))
      errors.push(`manage_memory.operation invalid: ${params.operation}`);
    const allowedKeys = new Set(["operation","content"]);
    for (const k of Object.keys(params)) {
      if (!allowedKeys.has(k)) errors.push(`manage_memory must NOT have .${k}`);
    }
  }

  return errors;
}

// ============================================================
// MAIN
// ============================================================
function main() {
  const lines = [];

  const nameGen  = () => rand(names);
  const songGen  = () => rand(songs);
  const topicGen = () => rand(topics);
  const langGen  = () => rand(languages);
  const factGen  = () => rand([
    "vegan diet","morning runs","cold showers","no sugar","intermittent fasting",
    "keto","gym","meditation","no social media","early sleep",
  ]);

  // --- CT1 ---
  lines.push(...expandTemplates(ct1_communicate_wins, [nameGen, songGen], 4));
  lines.push(...expandTemplates(ct1_process_wins, [nameGen], 4));

  // --- CT2 ---
  lines.push(...expandTemplates(ct2_retrieve_wins, [songGen], 3));
  lines.push(...expandTemplates(ct2_media_wins, [songGen], 3));
  lines.push(...expandTemplates(ct2_retrieve_wins.slice(2, 5), [topicGen], 3));
  lines.push(...expandTemplates(ct2_media_wins.slice(2, 5), [topicGen], 3));

  // --- CT3 ---
  lines.push(...expandTemplates(ct3_process_wins, [topicGen], 3));
  lines.push(...expandTemplates(ct3_retrieve_wins, [topicGen], 3));
  lines.push(...expandTemplates(ct3_process_wins.slice(1, 3), [langGen], 3));

  // --- CT4 ---
  lines.push(...ct4_ignore_wins.map((fn) => JSON.stringify(buildConflictSample(fn()))));
  lines.push(...expandTemplates(ct4_communicate_wins, [nameGen], 4));

  // --- CT5 ---
  lines.push(...ct5_memory_wins.map((fn) => JSON.stringify(buildConflictSample(fn(factGen())))));
  lines.push(...expandTemplates(ct5_retrieve_wins, [topicGen], 3));

  // --- CT6 ---
  lines.push(...expandTemplates(ct6_media_wins, [nameGen, songGen], 3));
  lines.push(...expandTemplates(ct6_communicate_wins, [nameGen, songGen], 3));

  // --- CT7 ---
  lines.push(...expandTemplates(ct7_process_wins.slice(0, 2), [nameGen, langGen], 3));
  lines.push(...expandTemplates(ct7_process_wins.slice(2), [nameGen], 3));

  // --- HARD HAND-CRAFTED EDGE CASES (schema-verified) ---
  const handcrafted = [
    // "bhejo" = show me the lyrics (not play)
    {
      user: "Kesariya ke lyrics bhejo na",
      action: "retrieve_info",
      params: { operation: "lookup", type: "music", query: "Kesariya", format: "lyrics" },
      conflict: { with: "media_action", reason: "'lyrics bhejo' = text content request; 'bhejo' here means 'show me' not 'play'" },
    },
    // explain to X — composition beats communicate
    {
      user: "explain this concept to neha in simple terms",
      action: "process_text",
      params: { operation: "rewrite", content: "provided concept", format: "short" },
      conflict: { with: "communicate", reason: "simplification is the main task; neha is context, not primary action target" },
    },
    // "what I said" → personal memory, not web
    {
      user: "remember what I said about crypto?",
      action: "manage_memory",
      params: { operation: "recall", content: "crypto" },
      conflict: { with: "retrieve_info", reason: "'what I said' points to personal conversation history, not a web search" },
    },
    // note kar lo → store
    {
      user: "note kar lo: mujhe Mondays se nafrat hai",
      action: "manage_memory",
      params: { operation: "store", content: "user hates Mondays" },
      conflict: { with: "process_text", reason: "'note kar lo' is a storage command; no text transformation requested" },
    },
    // vague "something" → communicate (no specific media type)
    {
      user: "send rahul something funny",
      action: "communicate",
      params: { operation: "text", content: "😂 bhai ye sun!", target: "rahul", tone: "friendly" },
      conflict: { with: "media_action", reason: "vague 'something' with a named person = communicate; no specific media type stated" },
    },
    // translate a song → lyrics lookup (no user-provided text to transform)
    {
      user: "translate Tum Hi Ho for me",
      action: "retrieve_info",
      params: { operation: "lookup", type: "music", query: "Tum Hi Ho", format: "lyrics" },
      conflict: { with: "process_text", reason: "no user-provided text to translate; task is looking up existing lyrics+meaning" },
    },
    // write caption = compose (not communicate — no direct recipient)
    {
      user: "write a funny birthday caption for sara's post",
      action: "process_text",
      params: { operation: "compose", content: "funny birthday caption", format: "short", tone: "excited" },
      conflict: { with: "communicate", reason: "explicit 'write' for a public post caption = composition; no direct message to sara" },
    },
    // find sad songs = curate playable music
    {
      user: "find me some sad songs",
      action: "media_action",
      // type:"music", tone free-form "sad" — valid
      params: { operation: "search", type: "music", query: "sad songs", tone: "neutral" },
      conflict: { with: "retrieve_info", reason: "'find me songs' = curate playable music; retrieve_info returns text/articles, not a playlist" },
    },
    // summarize + remember → process wins (compose is primary)
    {
      user: "summarize this article and remember the key points",
      action: "process_text",
      params: { operation: "summarize", content: "provided article", format: "bullet" },
      conflict: { with: "manage_memory", reason: "summarization is the main action; memory storage is secondary and implicit" },
    },
    // what neha likes → personal memory (not web)
    {
      user: "tell me what neha likes",
      action: "manage_memory",
      params: { operation: "recall", content: "neha preferences" },
      conflict: { with: "retrieve_info", reason: "asking about a known contact's preferences implies personal memory, not a web query" },
    },
    // play something I like → media (memory is internal sub-step)
    {
      user: "play something I like",
      action: "media_action",
      params: { operation: "play", type: "music", query: "user preferred music", tone: "neutral" },
      conflict: { with: "manage_memory", reason: "final intent is playback; memory recall is an internal sub-step, not the routed action" },
    },
    // venting → ignore
    {
      user: "ugh this is so annoying",
      action: "ignore",
      params: {},
      conflict: { with: "communicate", reason: "venting with no target or instruction = ignore; nothing actionable to route" },
    },
    // search and send → retrieve first
    {
      user: "search for climate change facts and send them to amit",
      action: "retrieve_info",
      params: { operation: "search", type: "web", query: "climate change facts", format: "detailed" },
      conflict: { with: "communicate", reason: "search is the bottleneck action; communicate is implicit follow-up once results are found" },
    },
    // single question word → ignore
    {
      user: "kya?",
      action: "ignore",
      params: {},
      conflict: { with: "communicate", reason: "single-word confused response with no context; insufficient signal to route" },
    },
    // play my favourite → media (memory consulted internally)
    {
      user: "play my favourite song",
      action: "media_action",
      params: { operation: "play", type: "music", query: "user favourite song" },
      conflict: { with: "manage_memory", reason: "playback is the output action; memory is consulted internally to resolve 'favourite'" },
    },
    // "in one line" = rewrite/compress
    {
      user: "what's AI in one line?",
      action: "process_text",
      params: { operation: "rewrite", content: "AI definition", format: "short" },
      conflict: { with: "retrieve_info", reason: "'in one line' signals a condensed output format = process/rewrite, not raw retrieval" },
    },
    // write a tweet → compose
    {
      user: "write a tweet about today's cricket news",
      action: "process_text",
      params: { operation: "compose", content: "cricket news tweet", format: "short" },
      conflict: { with: "retrieve_info", reason: "writing a tweet = compose; retrieve_info might be a sub-step but compose is the primary route" },
    },
    // 3-way conflict → media wins (despacito is the object)
    {
      user: "despacito bhejo rahul ko aur usse bolo mood hai",
      action: "media_action",
      // type:"music" for a song send — "song" enum no longer exists anywhere
      params: { operation: "send", type: "music", query: "Despacito" },
      conflict: { with: "communicate", reason: "primary object is media (Despacito); 'bolo mood hai' is secondary text that follows the media send" },
    },
    // passive ack → ignore
    {
      user: "oh i see",
      action: "ignore",
      params: {},
      conflict: { with: "manage_memory", reason: "passive acknowledgement with no store/recall command; nothing to route" },
    },
    // incomplete translate → process with default
    {
      user: "translate karo",
      action: "process_text",
      params: { operation: "translate", content: "provided text", target: "English" },
      conflict: { with: "ignore", reason: "incomplete but has a clear operation ('translate'); default to English translation of context" },
    },
  ];

  for (const sample of handcrafted) {
    lines.push(JSON.stringify({
      messages: [
        { role: "user", content: sample.user },
        {
          role: "assistant",
          content: JSON.stringify({
            action: sample.action,
            params: sample.params,
            confidence: lowConfidence(),
            conflict: sample.conflict,
          }),
        },
      ],
    }));
  }

  // Shuffle
  for (let i = lines.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }

  // Validate every sample
  let violations = 0;
  const validatedLines = lines.map((l, idx) => {
    const obj = JSON.parse(l);
    const errors = validateSample(obj, idx);
    if (errors.length > 0) {
      violations++;
      console.error(`[LINE ${idx}] SCHEMA VIOLATION:`);
      errors.forEach((e) => console.error(`  → ${e}`));
      console.error(`  User: "${obj.messages[0].content}"`);
    }
    return l;
  });

  fs.writeFileSync(OUTPUT_FILE, validatedLines.join("\n"));

  console.log(`\n✅ Conflict dataset generated: ${OUTPUT_FILE}`);
  console.log(`📊 Total conflict samples: ${validatedLines.length}`);
  console.log(`🚨 Schema violations: ${violations}`);
  console.log(`\nConflict type breakdown:`);
  console.log(`  CT1 communicate ←→ process_text    : ${(ct1_communicate_wins.length + ct1_process_wins.length) * 4}`);
  console.log(`  CT2 media_action ←→ retrieve_info  : ~${(ct2_retrieve_wins.length + ct2_media_wins.length) * 3 * 2}`);
  console.log(`  CT3 process_text ←→ retrieve_info  : ~${(ct3_process_wins.length + ct3_retrieve_wins.length) * 3}`);
  console.log(`  CT4 communicate ←→ ignore          : ${ct4_ignore_wins.length + ct4_communicate_wins.length * 4}`);
  console.log(`  CT5 manage_memory ←→ retrieve_info : ${ct5_memory_wins.length + ct5_retrieve_wins.length * 3}`);
  console.log(`  CT6 media_action ←→ communicate    : ~${(ct6_media_wins.length + ct6_communicate_wins.length) * 3}`);
  console.log(`  CT7 process_text ←→ communicate    : ~${ct7_process_wins.length * 3}`);
  console.log(`  Hand-crafted edge cases            : ${handcrafted.length}`);
}

main();
