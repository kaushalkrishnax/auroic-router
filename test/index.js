import express from "express";
import ollama from "ollama";

const app = express();
const PORT = 3000;
const MODEL_NAME = "auroic-router";
const TIMEOUT_MS = 8000;
const WAKE_WORDS = ["bot", "assistant", "jarvis", "ai", "chotu"];
const VALID_TYPES = [
  "text",
  "media",
  "react",
  "acknowledge",
  "translate",
  "ignore",
];

app.use(express.json());
app.use(express.static("public"));

function normalizeInput(text) {
  if (!text) return "";
  const pattern = new RegExp(`\\b(${WAKE_WORDS.join("|")})\\b`, "gi");
  return text.replace(pattern, "").trim();
}

function normalizeMessages(messages) {
  const msgs = messages.slice(-5);
  while (msgs.length < 5) msgs.unshift("...");
  return msgs;
}

function parseDecision(text) {
  if (!text) throw new Error("Empty response");

  const line = text.replace(/^R:\s*/i, "").trim();
  const parts = Object.fromEntries(
    line.split("|").map((p) => {
      const [k, ...v] = p.trim().split("=");
      return [k.trim().toLowerCase(), v.join("=").trim()];
    }),
  );

  const type = parts.type;
  if (!VALID_TYPES.includes(type)) throw new Error(`Invalid TYPE: "${type}"`);

  return {
    type,
    target: parts.target === "null" ? null : (parts.target ?? null),
    effort: parts.effort === "null" ? null : (parts.effort ?? null),
    title: parts.title?.trim() || null,
  };
}

async function runRouter(messages) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const prompt = normalizeMessages(messages)
      .map((m, i) => `M${i + 1}: ${normalizeInput(m.trim())}`)
      .join("\n");

    const res = await ollama.chat({
      model: MODEL_NAME,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      signal: controller.signal,
    });

    console.log(`FULL: ${res.message.content}`);

    const raw =
      res.message.content
        .split("\n")
        .find((l) => l.trim().startsWith("R:"))
        ?.trim() ?? "";
    return { raw, decision: parseDecision(raw) };
  } finally {
    clearTimeout(timeout);
  }
}

function fallbackDecision() {
  return { type: "ignore", target: null, effort: null, title: null };
}

app.post("/route", async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages array required" });
  }

  try {
    const { raw, decision } = await runRouter(messages);
    res.json({ raw, decision, fallback: false });
  } catch (err) {
    console.error("Router error:", err.message);
    res.json({
      raw: null,
      decision: fallbackDecision(),
      fallback: true,
      error: err.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Router running on http://localhost:${PORT} → ${MODEL_NAME}`);
});
