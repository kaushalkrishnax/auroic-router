#!/usr/bin/env node

import fs from "fs";
import readline from "readline";
import crypto from "crypto";

const INPUT = process.argv[2] || "dataset.jsonl";
const OUTPUT = process.argv[3] || "dataset_deduped.jsonl";

const seen = new Set();
const kept = [];
let total = 0;
let removed = 0;

/**
 * Normalize text aggressively
 */
function normalize(text = "") {
  return text
    .toLowerCase()
    .replace(/@\w+/g, "@user")     // normalize usernames
    .replace(/\d+/g, "0")          // normalize numbers
    .replace(/[^\w\s]/g, "")       // remove punctuation
    .replace(/\s+/g, " ")          // collapse spaces
    .trim();
}

/**
 * Create hash key for duplicate detection
 */
function createKey(obj) {
  try {
    const user = obj.messages?.find(m => m.role === "user")?.content || "";
    const assistant = obj.messages?.find(m => m.role === "assistant")?.content || "";

    const normUser = normalize(user);
    const normAssistant = normalize(assistant);

    const raw = normUser + "||" + normAssistant;

    return crypto.createHash("sha256").update(raw).digest("hex");
  } catch {
    return crypto.randomUUID();
  }
}

/**
 * Main
 */
async function run() {
  if (!fs.existsSync(INPUT)) {
    console.error("❌ File not found:", INPUT);
    process.exit(1);
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(INPUT),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line.trim()) continue;

    total++;

    try {
      const obj = JSON.parse(line);
      const key = createKey(obj);

      if (seen.has(key)) {
        removed++;
        continue;
      }

      seen.add(key);
      kept.push(line);

    } catch (err) {
      console.warn("⚠️ Skipping invalid JSON line");
    }
  }

  fs.writeFileSync(OUTPUT, kept.join("\n") + "\n");

  console.log("\n==============================");
  console.log("DEDUPLICATION REPORT");
  console.log("==============================");
  console.log("Input samples   :", total);
  console.log("Removed         :", removed);
  console.log("Final samples   :", kept.length);
  console.log("Output file     :", OUTPUT);
  console.log("==============================\n");
}

run();