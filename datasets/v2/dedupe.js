import fs from "fs";
import crypto from "crypto";

// ─── Config ───────────────────────────────────────────────────────────────────

const MODE = process.argv[2];

if (!MODE || !["keyword", "heartbeat"].includes(MODE)) {
  console.error("Usage: bun dedupe.js <keyword|heartbeat>");
  process.exit(1);
}

const INPUT_FILE =
  MODE === "heartbeat" ? "heartbeat_dataset.txt" : "keyword_dataset.txt";
const OUTPUT_FILE = INPUT_FILE.replace(".txt", "_deduped.txt");
const REPORT_FILE = INPUT_FILE.replace(".txt", "_dedupe_report.txt");

// Jaccard similarity threshold — two samples scoring above this are considered duplicates.
// 0.85 = very aggressive (catches more near-dupes), 0.70 = lenient
// 0.80 is a good balance for short WhatsApp-style messages
const SIMILARITY_THRESHOLD = 0.8;

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
      if (action === "send_media") {
        if (!content || content === "null") continue;
        if (content.trim().split(/\s+/).length > 5) continue;
      }
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

    // Extract clean message texts (no MSG label, no OUT line)
    const msgTexts = msgLines.map((l) =>
      l
        .replace(/^MSG\d:\s*/, "")
        .toLowerCase()
        .trim(),
    );

    samples.push({ msgCount, action, target, text: block.trim(), msgTexts });
  }

  return samples;
}

// ─── Hash-based dedupe ────────────────────────────────────────────────────────

// Layer 1: byte-for-byte exact
function exactHash(text) {
  return crypto.createHash("sha1").update(text).digest("hex");
}

// Layer 2: same content, different whitespace/casing
function normalizedHash(text) {
  return crypto
    .createHash("sha1")
    .update(text.toLowerCase().replace(/\s+/g, " ").trim())
    .digest("hex");
}

// Layer 3: same MSG content, different OUT line
// FIX: only hash the message lines, explicitly exclude the OUT line
function msgContentHash(msgTexts) {
  return crypto.createHash("sha1").update(msgTexts.join("|")).digest("hex");
}

// ─── Semantic similarity (trigram Jaccard) ────────────────────────────────────
// Catches typo variants of the same conversation:
// "bhai kya kar raha" vs "bhai kya kr rha" → high similarity

function trigrams(str) {
  const s = str.replace(/\s+/g, " ").trim();
  const set = new Set();
  for (let i = 0; i <= s.length - 3; i++) {
    set.add(s.slice(i, i + 3));
  }
  return set;
}

function jaccardSimilarity(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 1;
  let intersection = 0;
  for (const item of setA) {
    if (setB.has(item)) intersection++;
  }
  return intersection / (setA.size + setB.size - intersection);
}

// Build trigram fingerprint from all message texts combined
function sampleFingerprint(msgTexts) {
  return trigrams(msgTexts.join(" "));
}

// ─── Main ─────────────────────────────────────────────────────────────────────

function main() {
  if (!fs.existsSync(INPUT_FILE)) {
    console.error(`File not found: ${INPUT_FILE}`);
    process.exit(1);
  }

  console.log(`\nDeduplicating: ${INPUT_FILE}`);
  console.log(`Mode: ${MODE.toUpperCase()}`);
  console.log(`Similarity threshold: ${SIMILARITY_THRESHOLD}\n`);

  const raw = fs.readFileSync(INPUT_FILE, "utf8");
  const all = parseSamples(raw);
  const totalBlocks = raw.split(/\n{2,}/).filter((b) => b.trim()).length;

  console.log(`Total blocks in file: ${totalBlocks}`);
  console.log(`Valid after parse:    ${all.length}`);

  const seenExact = new Set();
  const seenNormalized = new Set();
  const seenMsgContent = new Set();

  // For semantic check: store fingerprints of kept samples
  const keptFingerprints = [];

  const kept = [];
  let removedExact = 0;
  let removedNorm = 0;
  let removedMsgContent = 0;
  let removedSemantic = 0;

  for (const s of all) {
    // Layer 1: exact
    const h1 = exactHash(s.text);
    if (seenExact.has(h1)) {
      removedExact++;
      continue;
    }

    // Layer 2: normalized
    const h2 = normalizedHash(s.text);
    if (seenNormalized.has(h2)) {
      removedNorm++;
      continue;
    }

    // Layer 3: same messages, different OUT (uses msgTexts only, not OUT line)
    const h3 = msgContentHash(s.msgTexts);
    if (seenMsgContent.has(h3)) {
      removedMsgContent++;
      continue;
    }

    // Layer 4: semantic trigram similarity
    const fp = sampleFingerprint(s.msgTexts);
    let tooSimilar = false;
    for (const keptFp of keptFingerprints) {
      if (jaccardSimilarity(fp, keptFp) >= SIMILARITY_THRESHOLD) {
        tooSimilar = true;
        break;
      }
    }
    if (tooSimilar) {
      removedSemantic++;
      continue;
    }

    // Passed all layers — keep it
    seenExact.add(h1);
    seenNormalized.add(h2);
    seenMsgContent.add(h3);
    keptFingerprints.push(fp);
    kept.push(s);
  }

  // ─── Distribution stats ───────────────────────────────────────────────────

  const stats = {
    byMsgCount: { 2: 0, 3: 0, 4: 0, 5: 0 },
    byAction: { communicate: 0, send_media: 0, ignore: 0 },
  };
  for (const s of kept) {
    stats.byMsgCount[s.msgCount]++;
    stats.byAction[s.action]++;
  }

  // ─── Report ───────────────────────────────────────────────────────────────

  const totalRemoved =
    removedExact + removedNorm + removedMsgContent + removedSemantic;
  const removedInvalid = totalBlocks - all.length;

  const report = [
    `DEDUPE REPORT — ${MODE.toUpperCase()}`,
    `Generated: ${new Date().toISOString()}`,
    `Input file: ${INPUT_FILE}`,
    `Similarity threshold: ${SIMILARITY_THRESHOLD}`,
    ``,
    `── Input ──────────────────────────────`,
    `Total blocks in file:     ${totalBlocks}`,
    `Valid after parse:        ${all.length}`,
    `Invalid/malformed blocks: ${removedInvalid}`,
    ``,
    `── Duplicates Removed ─────────────────`,
    `Exact duplicates:         ${removedExact}`,
    `Whitespace/case variants: ${removedNorm}`,
    `Same messages, diff OUT:  ${removedMsgContent}`,
    `Near-duplicate (semantic):${removedSemantic}`,
    `Total removed:            ${totalRemoved}`,
    ``,
    `── Output ─────────────────────────────`,
    `Clean samples kept:       ${kept.length}`,
    ``,
    `── Distribution ───────────────────────`,
    `MsgCount:`,
    ...Object.entries(stats.byMsgCount).map(
      ([k, v]) =>
        `  ${k}-message: ${v} (${kept.length ? ((v / kept.length) * 100).toFixed(1) : 0}%)`,
    ),
    `Actions:`,
    ...Object.entries(stats.byAction).map(
      ([k, v]) =>
        `  ${k}: ${v} (${kept.length ? ((v / kept.length) * 100).toFixed(1) : 0}%)`,
    ),
  ].join("\n");

  console.log("\n" + report);

  fs.writeFileSync(OUTPUT_FILE, kept.map((s) => s.text).join("\n\n") + "\n");
  fs.writeFileSync(REPORT_FILE, report + "\n");

  console.log(`\n✓ Clean dataset → ${OUTPUT_FILE}`);
  console.log(`✓ Report        → ${REPORT_FILE}`);
}

main();
