import fs from "fs";
import crypto from "crypto";

// ─── Config ───────────────────────────────────────────────────────────────────

const DATASETS = [
  {
    name: "KEYWORD",
    file: "keyword_dataset_final.txt",
    mode: "keyword",
    target: {
      byMsgCount: { 2: 250, 3: 500, 4: 750, 5: 1000 },
      byAction: { communicate: 1125, send_media: 750, ignore: 625 },
    },
  },
  {
    name: "HEARTBEAT",
    file: "heartbeat_dataset_final.txt",
    mode: "heartbeat",
    target: {
      byMsgCount: { 2: 250, 3: 375, 4: 625, 5: 1250 },
      byAction: { communicate: 1125, send_media: 750, ignore: 625 },
    },
  },
];

const SIMILARITY_THRESHOLD = 0.8;

// ─── Parser ───────────────────────────────────────────────────────────────────

// ─── Parser ───────────────────────────────────────────────────────────────────

const KEYWORD_RE = /\b(ai|bot|assistant|oye ai|ai bhai|ai sun|bot help)\b/i;

function parseSamples(text, mode) {
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

    const [action, target, contentRaw, confStr] = parts.map((p) => p.trim());
    const confidence = parseFloat(confStr);

    if (!["communicate", "send_media", "ignore"].includes(action)) continue;

    const confMin =
      action === "communicate" ? 0.8 : action === "send_media" ? 0.85 : 0.75;
    const confMax =
      action === "communicate" ? 0.95 : action === "send_media" ? 0.98 : 0.9;

    if (isNaN(confidence) || confidence < confMin || confidence > confMax)
      continue;

    // ─── Action validation ─────────────────────────────

    let content = contentRaw;

    if (action === "ignore") {
      if (target !== "null") continue;
      content = "null";
    } else {
      if (!/^MSG[1-5]$/.test(target)) continue;

      const msgNum = parseInt(target.replace("MSG", ""));
      if (msgNum < 1 || msgNum > msgCount) continue;

      if (action === "communicate") {
        // NEW FORMAT: reasoning label
        if (!["low", "medium", "high"].includes(content)) continue;
      }

      if (action === "send_media") {
        if (!content || content === "null") continue;
        if (content.split(/\s+/).length > 5) continue;
      }
    }

    // ─── Keyword rules ─────────────────────────────

    if (mode === "keyword") {
      if (action !== "ignore" && target !== `MSG${msgCount}`) continue;

      const lastMsg = msgLines[msgLines.length - 1].replace(/^MSG\d:\s*/, "");
      const prevMsgs = msgLines
        .slice(0, -1)
        .map((l) => l.replace(/^MSG\d:\s*/, ""));

      if (!KEYWORD_RE.test(lastMsg)) continue;
      if (prevMsgs.some((m) => KEYWORD_RE.test(m))) continue;
    }

    const msgTexts = msgLines.map((l) =>
      l
        .replace(/^MSG\d:\s*/, "")
        .toLowerCase()
        .trim(),
    );

    samples.push({
      msgCount,
      action,
      target,
      confidence,
      content,
      text: block.trim(),
      msgTexts,
    });
  }

  return samples;
}

// ─── Dedupe check ─────────────────────────────────────────────────────────────

function trigrams(str) {
  const s = str.replace(/\s+/g, " ").trim();
  const set = new Set();
  for (let i = 0; i <= s.length - 3; i++) set.add(s.slice(i, i + 3));
  return set;
}

function jaccard(a, b) {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}

function countResidualDupes(samples) {
  const exact = new Set();
  const normalized = new Set();
  const msgContent = new Set();
  const fingerprints = [];

  let dupeExact = 0;
  let dupeNorm = 0;
  let dupeMsgHash = 0;
  let dupeSemantic = 0;

  for (const s of samples) {
    const h1 = crypto.createHash("sha1").update(s.text).digest("hex");
    if (exact.has(h1)) {
      dupeExact++;
      continue;
    }

    const h2 = crypto
      .createHash("sha1")
      .update(s.text.toLowerCase().replace(/\s+/g, " ").trim())
      .digest("hex");
    if (normalized.has(h2)) {
      dupeNorm++;
      continue;
    }

    const h3 = crypto
      .createHash("sha1")
      .update(s.msgTexts.join("|"))
      .digest("hex");
    if (msgContent.has(h3)) {
      dupeMsgHash++;
      continue;
    }

    const fp = trigrams(s.msgTexts.join(" "));
    let tooSimilar = false;
    for (const kfp of fingerprints) {
      if (jaccard(fp, kfp) >= SIMILARITY_THRESHOLD) {
        tooSimilar = true;
        break;
      }
    }
    if (tooSimilar) {
      dupeSemantic++;
      continue;
    }

    exact.add(h1);
    normalized.add(h2);
    msgContent.add(h3);
    fingerprints.push(fp);
  }

  return {
    dupeExact,
    dupeNorm,
    dupeMsgHash,
    dupeSemantic,
    total: dupeExact + dupeNorm + dupeMsgHash + dupeSemantic,
  };
}

// ─── Language detection ───────────────────────────────────────────────────────

const HINDI_WORDS =
  /\b(hai|hain|kya|nahi|bhai|yaar|kuch|aur|mera|tera|tum|hum|karo|bata|kal|aaj|abhi|phir|fir|woh|wo|toh|to|kyun|kaisa|kaise|kab|kahan|mat|bas|chal|sab|log|dost|pyaar|ghar|din|raat|time|baat|dil|man|agar|lekin|par|pe|se|ko|ka|ki|ke|ne|ye|yeh|waise|bilkul|ekdum|zyada|thoda|bahut|accha|acha|theek|theek|pakka|sahi)\b/i;
const ENGLISH_ONLY = /^[a-z0-9\s.,!?'"@#%&*()\-_]+$/i;

function detectLanguage(text) {
  const hasHindi = HINDI_WORDS.test(text);
  const isEnglish = ENGLISH_ONLY.test(text);
  if (hasHindi && !isEnglish) return "hinglish";
  if (hasHindi) return "hinglish";
  return "english";
}

function analyzeLanguageMix(samples) {
  const counts = { hinglish: 0, english: 0 };
  for (const s of samples) {
    const allText = s.msgTexts.join(" ");
    counts[detectLanguage(allText)]++;
  }
  return counts;
}

// ─── Confidence analysis ──────────────────────────────────────────────────────

function analyzeConfidence(samples) {
  const byAction = { communicate: [], send_media: [], ignore: [] };
  for (const s of samples) byAction[s.action].push(s.confidence);

  const stats = {};
  for (const [action, vals] of Object.entries(byAction)) {
    if (!vals.length) {
      stats[action] = null;
      continue;
    }
    const sorted = [...vals].sort((a, b) => a - b);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(
      vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length,
    );
    stats[action] = {
      mean: mean.toFixed(3),
      std: std.toFixed(3),
      min: sorted[0].toFixed(2),
      max: sorted[sorted.length - 1].toFixed(2),
      median: sorted[Math.floor(sorted.length / 2)].toFixed(2),
    };
  }
  return stats;
}

// ─── Content quality checks ───────────────────────────────────────────────────

function analyzeContentQuality(samples) {
  const issues = [];

  // send_media content word count distribution
  const mediaSamples = samples.filter((s) => s.action === "send_media");
  const mediaWordCounts = mediaSamples.map(
    (s) => (s.content || "").trim().split(/\s+/).length,
  );
  const avgMediaWords = mediaWordCounts.length
    ? (
        mediaWordCounts.reduce((a, b) => a + b, 0) / mediaWordCounts.length
      ).toFixed(1)
    : 0;

  // Average message lengths
  const avgMsgLen = (
    samples.reduce(
      (sum, s) =>
        sum + s.msgTexts.reduce((a, b) => a + b.length, 0) / s.msgTexts.length,
      0,
    ) / samples.length
  ).toFixed(1);

  // Samples with very short messages (< 5 chars avg) — low quality
  const tooShort = samples.filter(
    (s) => s.msgTexts.reduce((a, b) => a + b.length, 0) / s.msgTexts.length < 5,
  ).length;

  // Keyword coverage (keyword dataset only)
  const keywordHits = samples.filter(
    (s) =>
      s.msgTexts.length > 0 &&
      KEYWORD_RE.test(s.msgTexts[s.msgTexts.length - 1]),
  ).length;

  return { avgMediaWords, avgMsgLen, tooShort, keywordHits };
}

// ─── Distribution scoring ─────────────────────────────────────────────────────

// Returns 0-100 score based on how close actual distribution is to target
function distributionScore(actual, target, total) {
  const targetTotal = Object.values(target).reduce((a, b) => a + b, 0);
  let totalDiff = 0;
  let count = 0;
  for (const [k, t] of Object.entries(target)) {
    const targetPct = t / targetTotal;
    const actualPct = (actual[k] ?? 0) / total;
    totalDiff += Math.abs(targetPct - actualPct);
    count++;
  }
  const avgDiff = totalDiff / count;
  const score = Math.max(0, 100 - avgDiff * 500);
  return score.toFixed(1);
}

// ─── Main analysis ────────────────────────────────────────────────────────────

function analyzeDataset(ds) {
  const lines = [
    `${"═".repeat(60)}`,
    ` ANALYSIS: ${ds.name}`,
    ` File: ${ds.file}`,
    `${"═".repeat(60)}`,
  ];

  if (!fs.existsSync(ds.file)) {
    lines.push(`\n  ✗ File not found: ${ds.file}`);
    return lines.join("\n");
  }

  const raw = fs.readFileSync(ds.file, "utf8");
  const blocks = raw.split(/\n{2,}/).filter((b) => b.trim());
  const samples = parseSamples(raw, ds.mode);
  const total = samples.length;
  const target = ds.target;
  const targetTotal = Object.values(target.byMsgCount).reduce(
    (a, b) => a + b,
    0,
  );

  // ── Overview ───────────────────────────────────────────────────────────────
  lines.push(`\n── Overview ─────────────────────────────────────────`);
  lines.push(`  Total blocks in file:   ${blocks.length}`);
  lines.push(`  Valid parsed samples:   ${total}`);
  lines.push(`  Invalid/malformed:      ${blocks.length - total}`);
  lines.push(`  Target total:           ${targetTotal}`);
  lines.push(
    `  Completion:             ${((total / targetTotal) * 100).toFixed(1)}%`,
  );
  lines.push(
    `  Gap:                    ${Math.max(0, targetTotal - total)} samples short`,
  );

  // ── MsgCount distribution ──────────────────────────────────────────────────
  const msgCounts = { 2: 0, 3: 0, 4: 0, 5: 0 };
  for (const s of samples) msgCounts[s.msgCount]++;

  lines.push(`\n── Message Count Distribution ───────────────────────`);
  lines.push(
    `  ${"Count".padEnd(8)} ${"Actual".padEnd(8)} ${"Actual%".padEnd(10)} ${"Target%".padEnd(10)} ${"Gap".padEnd(8)} Status`,
  );
  lines.push(`  ${"─".repeat(52)}`);
  for (const [k, t] of Object.entries(target.byMsgCount)) {
    const actual = msgCounts[k] ?? 0;
    const actualPct = total ? ((actual / total) * 100).toFixed(1) : "0.0";
    const targetPct = ((t / targetTotal) * 100).toFixed(1);
    const gap = t - actual;
    const status = gap <= 0 ? "✓" : gap <= t * 0.1 ? "≈" : "✗";
    lines.push(
      `  ${(k + "-msg").padEnd(8)} ${String(actual).padEnd(8)} ${(actualPct + "%").padEnd(10)} ${(targetPct + "%").padEnd(10)} ${(gap > 0 ? "+" + gap : gap).toString().padEnd(8)} ${status}`,
    );
  }
  const msgScore = distributionScore(msgCounts, target.byMsgCount, total);
  lines.push(`  Distribution score: ${msgScore}/100`);

  // ── Action distribution ────────────────────────────────────────────────────
  const actionCounts = { communicate: 0, send_media: 0, ignore: 0 };
  for (const s of samples) actionCounts[s.action]++;

  lines.push(`\n── Action Distribution ──────────────────────────────`);
  lines.push(
    `  ${"Action".padEnd(14)} ${"Actual".padEnd(8)} ${"Actual%".padEnd(10)} ${"Target%".padEnd(10)} ${"Gap".padEnd(8)} Status`,
  );
  lines.push(`  ${"─".repeat(58)}`);
  for (const [k, t] of Object.entries(target.byAction)) {
    const actual = actionCounts[k] ?? 0;
    const actualPct = total ? ((actual / total) * 100).toFixed(1) : "0.0";
    const targetPct = ((t / targetTotal) * 100).toFixed(1);
    const gap = t - actual;
    const status = gap <= 0 ? "✓" : gap <= t * 0.1 ? "≈" : "✗";
    lines.push(
      `  ${k.padEnd(14)} ${String(actual).padEnd(8)} ${(actualPct + "%").padEnd(10)} ${(targetPct + "%").padEnd(10)} ${(gap > 0 ? "+" + gap : gap).toString().padEnd(8)} ${status}`,
    );
  }
  const actionScore = distributionScore(actionCounts, target.byAction, total);
  lines.push(`  Distribution score: ${actionScore}/100`);

  // ── Reasoning distribution (communicate only) ───────────────────────

  const reasoningCounts = { low: 0, medium: 0, high: 0 };

  for (const s of samples) {
    if (
      s.action === "communicate" &&
      reasoningCounts[s.content] !== undefined
    ) {
      reasoningCounts[s.content]++;
    }
  }

  const totalComm =
    reasoningCounts.low + reasoningCounts.medium + reasoningCounts.high;

  lines.push(`\n── Reasoning Distribution ───────────────────────────`);
  lines.push(
    `  low:    ${reasoningCounts.low} (${totalComm ? ((reasoningCounts.low / totalComm) * 100).toFixed(1) : 0}%)`,
  );
  lines.push(
    `  medium: ${reasoningCounts.medium} (${totalComm ? ((reasoningCounts.medium / totalComm) * 100).toFixed(1) : 0}%)`,
  );
  lines.push(
    `  high:   ${reasoningCounts.high} (${totalComm ? ((reasoningCounts.high / totalComm) * 100).toFixed(1) : 0}%)`,
  );

  // ── Confidence stats ───────────────────────────────────────────────────────
  const confStats = analyzeConfidence(samples);
  lines.push(`\n── Confidence Statistics ────────────────────────────`);
  lines.push(
    `  ${"Action".padEnd(14)} ${"Mean".padEnd(8)} ${"Std".padEnd(8)} ${"Min".padEnd(8)} ${"Median".padEnd(8)} Max`,
  );
  lines.push(`  ${"─".repeat(54)}`);
  for (const [action, stats] of Object.entries(confStats)) {
    if (!stats) continue;
    lines.push(
      `  ${action.padEnd(14)} ${stats.mean.padEnd(8)} ${stats.std.padEnd(8)} ${stats.min.padEnd(8)} ${stats.median.padEnd(8)} ${stats.max}`,
    );
  }

  // ── Language mix ───────────────────────────────────────────────────────────
  const langMix = analyzeLanguageMix(samples);
  lines.push(`\n── Language Mix ─────────────────────────────────────`);
  lines.push(
    `  Hinglish:  ${langMix.hinglish} (${total ? ((langMix.hinglish / total) * 100).toFixed(1) : 0}%)`,
  );
  lines.push(
    `  English:   ${langMix.english} (${total ? ((langMix.english / total) * 100).toFixed(1) : 0}%)`,
  );

  // ── Content quality ────────────────────────────────────────────────────────
  const quality = analyzeContentQuality(samples);
  lines.push(`\n── Content Quality ──────────────────────────────────`);
  lines.push(`  Avg message length (chars):  ${quality.avgMsgLen}`);
  lines.push(`  Avg send_media query words:  ${quality.avgMediaWords}`);
  lines.push(`  Very short messages (<5chr): ${quality.tooShort}`);
  if (ds.mode === "keyword") {
    lines.push(
      `  Keyword in last msg:         ${quality.keywordHits}/${total} (${total ? ((quality.keywordHits / total) * 100).toFixed(1) : 0}%)`,
    );
  }

  // ── Residual dupe check ────────────────────────────────────────────────────
  console.log(`  Running dedupe scan...`);
  const dupes = countResidualDupes(samples);
  lines.push(`\n── Residual Duplicate Check ─────────────────────────`);
  lines.push(`  Exact:              ${dupes.dupeExact}`);
  lines.push(`  Whitespace/case:    ${dupes.dupeNorm}`);
  lines.push(`  Same messages:      ${dupes.dupeMsgHash}`);
  lines.push(`  Near-duplicate:     ${dupes.dupeSemantic}`);
  lines.push(`  Total residual:     ${dupes.total}`);
  lines.push(`  Clean after rescan: ${total - dupes.total}`);

  // ── Overall health score ───────────────────────────────────────────────────
  const completionScore = Math.min(100, (total / targetTotal) * 100);
  const dupeScore = Math.max(0, 100 - (dupes.total / total) * 1000);
  const overallScore = (
    parseFloat(msgScore) * 0.3 +
    parseFloat(actionScore) * 0.3 +
    completionScore * 0.25 +
    dupeScore * 0.15
  ).toFixed(1);

  lines.push(`\n── Overall Health Score ─────────────────────────────`);
  lines.push(`  MsgCount distribution:  ${msgScore}/100    (weight 30%)`);
  lines.push(`  Action distribution:    ${actionScore}/100    (weight 30%)`);
  lines.push(
    `  Completion:             ${completionScore.toFixed(1)}/100    (weight 25%)`,
  );
  lines.push(
    `  Cleanliness:            ${dupeScore.toFixed(1)}/100    (weight 15%)`,
  );
  lines.push(`  ─────────────────────────────────────────`);
  lines.push(`  OVERALL SCORE:          ${overallScore}/100`);

  const grade =
    overallScore >= 90
      ? "A"
      : overallScore >= 75
        ? "B"
        : overallScore >= 60
          ? "C"
          : "D";
  lines.push(`  GRADE:                  ${grade}`);

  // ── Recommendations ────────────────────────────────────────────────────────
  const recs = [];
  if (total < targetTotal * 0.95)
    recs.push(`Generate ${targetTotal - total} more samples to reach target`);
  for (const [k, t] of Object.entries(target.byMsgCount)) {
    const gap = t - (msgCounts[k] ?? 0);
    if (gap > t * 0.1)
      recs.push(
        `${k}-message samples short by ${gap} — run msg${k}-heavy batch`,
      );
  }
  for (const [k, t] of Object.entries(target.byAction)) {
    const gap = t - (actionCounts[k] ?? 0);
    if (gap > t * 0.1)
      recs.push(`${k} action short by ${gap} — run ${k}-heavy batch`);
  }
  if (dupes.total > 0)
    recs.push(`${dupes.total} residual duplicates found — run dedupe.js again`);

  if (recs.length) {
    lines.push(`\n── Recommendations ──────────────────────────────────`);
    for (const r of recs) lines.push(`  → ${r}`);
  } else {
    lines.push(`\n  ✓ Dataset looks healthy. No major issues found.`);
  }

  return lines.join("\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

function main() {
  const mode = process.argv[2]; // optional: "keyword" | "heartbeat" | omit for both

  const targets = mode ? DATASETS.filter((d) => d.mode === mode) : DATASETS;

  if (mode && targets.length === 0) {
    console.error(
      `Unknown mode: ${mode}. Use keyword, heartbeat, or omit for both.`,
    );
    process.exit(1);
  }

  const timestamp = new Date().toISOString();
  const allLines = [`DATASET ANALYSIS REPORT`, `Generated: ${timestamp}`, ``];

  for (const ds of targets) {
    console.log(`\nAnalyzing ${ds.name}...`);
    allLines.push(analyzeDataset(ds));
    allLines.push("");
  }

  const report = allLines.join("\n");
  console.log("\n" + report);

  const outFile = mode ? `${mode}_analysis.txt` : "datasets_analysis.txt";
  fs.writeFileSync(outFile, report);
  console.log(`\n✓ Report saved → ${outFile}`);
}

main();
