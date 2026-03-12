import express from "express";
import ollama from "ollama";

const app = express();
const PORT = 3000;
const MODEL_NAME = "auroic-router-0.6b";
const TIMEOUT_MS = 12000;
const VALID_TYPES = ["text", "media", "react", "ignore"];
const VALID_EFFORTS = ["low", "medium", "high"];

const SYSTEM_PROMPT =
  "You are the Auroic Router. Given history messages H1-H5 and candidate messages C1-C3, output exactly one routing decision.";

app.use(express.json());
app.use(express.static("public"));

// Input builder
// Accepts { history: string[], candidates: string[] }
// or legacy { messages: string[] } for the lab UI (treats all as candidates)
function buildPrompt({ history = [], candidates = [] }) {
  const hist = [...history];
  while (hist.length < 5) hist.unshift("...");
  const hist5 = hist.slice(-5);

  const cands = [...candidates];
  while (cands.length < 3) cands.push("...");
  const cands3 = cands.slice(0, 3);

  const hLines = hist5
    .map((m, i) => `H${i + 1}: ${m.trim() || "..."}`)
    .join("\n");
  const cLines = cands3
    .map((m, i) => `C${i + 1}: ${m.trim() || "..."}`)
    .join("\n");
  return `${hLines}\n${cLines}`;
}

// Response parser
function parseDecision(text) {
  if (!text) throw new Error("Empty response");

  const line = text
    .split("\n")
    .find((l) => l.trim().startsWith("R:"))
    ?.trim();

  if (!line) throw new Error("No R: line in response");

  const parts = Object.fromEntries(
    line
      .replace(/^R:\s*/i, "")
      .split("|")
      .map((p) => {
        const [k, ...v] = p.trim().split("=");
        return [k.trim().toLowerCase(), v.join("=").trim()];
      }),
  );

  const type = parts.type?.toLowerCase();
  if (!VALID_TYPES.includes(type)) throw new Error(`Invalid TYPE: "${type}"`);

  const effort =
    type === "text" && VALID_EFFORTS.includes(parts.effort?.toLowerCase())
      ? parts.effort.toLowerCase()
      : null;

  const title =
    type === "react" || type === "media" ? parts.title?.trim() || null : null;

  const target =
    type !== "ignore" && parts.target ? parts.target.toUpperCase() : null;

  return { type, target, effort, title };
}

// Router
async function runRouter({ history, candidates }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const prompt = buildPrompt({ history, candidates });

    const res = await ollama.chat({
      model: MODEL_NAME,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      stream: false,
      signal: controller.signal,
      options: {
        temperature: 0.3,
        top_p: 0.95,
        top_k: 20,
        repeat_penalty: 1.1,
        stop: ["<|im_end|>", "<|im_start|>"],
      },
      think: false,
    });

    const full = res.message.content;
    console.log(`[ROUTER] ${full.replace(/\n/g, " ").slice(0, 120)}`);

    const raw =
      full
        .split("\n")
        .find((l) => l.trim().startsWith("R:"))
        ?.trim() ?? "";

    return { raw, decision: parseDecision(full) };
  } finally {
    clearTimeout(timeout);
  }
}

function fallbackDecision() {
  return { type: "ignore", target: null, effort: null, title: null };
}

// Routes

// Full structured call — production use
// POST /route { history: string[], candidates: string[] }
app.post("/route", async (req, res) => {
  const { history, candidates } = req.body;

  if (!candidates || !Array.isArray(candidates) || candidates.length === 0) {
    return res.status(400).json({ error: "candidates array required" });
  }

  try {
    const { raw, decision } = await runRouter({
      history: history || [],
      candidates,
    });
    res.json({ raw, decision, fallback: false });
  } catch (err) {
    console.error("[ROUTER ERROR]", err.message);
    res.json({
      raw: null,
      decision: fallbackDecision(),
      fallback: true,
      error: err.message,
    });
  }
});

// Lab UI call — all messages treated as window, last 3 become candidates
// POST /route/lab { messages: string[] }
app.post("/route/lab", async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages array required" });
  }

  const history = messages.slice(0, -3).slice(-5);
  const candidates = messages.slice(-3);

  try {
    const { raw, decision } = await runRouter({ history, candidates });
    res.json({ raw, decision, fallback: false });
  } catch (err) {
    console.error("[ROUTER ERROR]", err.message);
    res.json({
      raw: null,
      decision: fallbackDecision(),
      fallback: true,
      error: err.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Auroic Router server → http://localhost:${PORT}`);
  console.log(`Model: ${MODEL_NAME}`);
});
