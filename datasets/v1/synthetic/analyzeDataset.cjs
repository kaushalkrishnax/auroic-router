import fs from "fs";
import readline from "readline";

const FILE = "./dataset.jsonl";

const stats = {
  tools: {},
  operations: {},
  types: {},
  tones: {},
  formats: {},
  targets: {
    present: 0,
    missing: 0,
  },
  confidence: [],
  length: [],
  combos: {},
};

function inc(obj, key) {
  if (!key) return;
  obj[key] = (obj[key] || 0) + 1;
}

function comboKey(a, b, c) {
  return [a, b, c].filter(Boolean).join("|");
}

async function analyze() {
  const stream = fs.createReadStream(FILE);

  const rl = readline.createInterface({
    input: stream,
    crlfDelay: Infinity,
  });

  let total = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;

    const json = JSON.parse(line);
    const msg = json.messages?.[1]?.content || "";

    let data;

    try {
      data = JSON.parse(msg);
    } catch {
      continue;
    }

    total++;

    const tool = data.action;
    const params = data.params || {};

    inc(stats.tools, tool);

    if (params.operation) inc(stats.operations, params.operation);
    if (params.type) inc(stats.types, params.type);
    if (params.tone) inc(stats.tones, params.tone);
    if (params.format) inc(stats.formats, params.format);

    if (params.target) stats.targets.present++;
    else stats.targets.missing++;

    if (typeof data.confidence === "number") {
      stats.confidence.push(data.confidence);
    }

    // Approx token length
    const text = JSON.stringify(data);
    stats.length.push(text.length / 4);

    // Combination tracking
    const combo = comboKey(tool, params.operation, params.type);
    inc(stats.combos, combo);
  }

  printReport(total);
}

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function printTable(title, obj) {
  console.log(`\n=== ${title} ===`);
  Object.entries(obj)
    .sort((a, b) => b[1] - a[1])
    .forEach(([k, v]) => {
      console.log(`${k.padEnd(25)} : ${v}`);
    });
}

function printReport(total) {
  console.log("\n==============================");
  console.log("DATASET ANALYSIS REPORT");
  console.log("==============================");

  console.log(`Total Samples: ${total}`);

  printTable("Tool Distribution", stats.tools);
  printTable("Operation Distribution", stats.operations);
  printTable("Type Distribution", stats.types);
  printTable("Tone Distribution", stats.tones);
  printTable("Format Distribution", stats.formats);

  console.log("\n=== Target Presence ===");
  console.log("Present :", stats.targets.present);
  console.log("Missing :", stats.targets.missing);

  console.log("\n=== Confidence ===");
  console.log("Average :", avg(stats.confidence).toFixed(3));
  console.log("Min     :", Math.min(...stats.confidence));
  console.log("Max     :", Math.max(...stats.confidence));

  console.log("\n=== Token Length (approx) ===");
  console.log("Average tokens :", avg(stats.length).toFixed(1));

  printTable("Top Tool Combos", stats.combos);

  console.log("\n✅ Analysis Complete");
}

analyze();
