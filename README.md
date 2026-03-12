# Auroic Router — Research Repository

> Training data, generation pipelines, and experiment history for the Auroic Router — a compact edge-deployable chat routing model for conversational AI systems.

- **Main Project** → [github.com/kaushalkrishnax/auroic](https://github.com/kaushalkrishnax/auroic)
- **Fine Tuned Model** → [huggingface.co/kaushalkrishnax/auroic-router-0.6b](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)

---

## What is the Auroic Router?

The Auroic Router is a small language model trained to read a window of recent group chat messages and decide what action a conversational AI should take next — reply with text, send a gif, react with an emoji, or ignore noise entirely.

It is not a chatbot. It does not generate responses. It makes one decision per window, fast, on CPU.

```
H1: bhai sun
H2: kya hua bata
H3: okay
H4: arre
H5: haan bol
C1: ...
C2: ...
C3: bhai parents divorce ho raha hai adjust karna mushkil hai kya karoon

→ R: TYPE=text | TARGET=C3 | EFFORT=high
```

The router sits at the front of a larger pipeline, directing traffic to the right handler so downstream models do not waste compute on irrelevant messages.

---

## Current Model — v4

| Property             | Value                        |
| -------------------- | ---------------------------- |
| Base                 | Qwen3-0.6B                   |
| Total parameters     | 616M                         |
| Trainable parameters | 20.2M (LoRA r=32)            |
| Quantization         | Q8_0 GGUF                    |
| Training samples     | 9,300                        |
| Fine-tuning          | Unsloth, 2 epochs, LoRA r=32 |
| Final training loss  | 0.667                        |
| Deployment           | Ollama / llama.cpp           |
| Cold start           | ~6s on i3-6100U CPU          |
| Warm inference       | ~3s loaded in memory         |

**→ [Download and use the model on HuggingFace](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)**

---

## Output Format — v4

```
R: TYPE=text   | TARGET=C2 | EFFORT=low|medium|high
R: TYPE=react  | TARGET=C1 | TITLE=🔥
R: TYPE=media  | TARGET=C3 | TITLE=rain cozy vibes
R: TYPE=ignore
```

| Type     | When                                                        | Fields                          |
| -------- | ----------------------------------------------------------- | ------------------------------- |
| `text`   | Help request, advice, emotional support, technical question | `EFFORT=low/medium/high`        |
| `react`  | Notable moment deserving an emoji reaction                  | `TITLE=<emoji>`                 |
| `media`  | Situation implying a gif, meme, or sticker                  | `TITLE=<2-3 word search query>` |
| `ignore` | Pure noise, spam, keyboard smash, laughter chains           | none                            |

---

## Input Format — v4

```
H1: <message>   ← oldest processed message
H2: <message>
H3: <message>
H4: <message>
H5: <message>   ← most recent processed message
C1: <message>   ← unprocessed candidate (use ... for filler)
C2: <message>
C3: <message>   ← newest unprocessed candidate
```

Always provide exactly 5 history messages and 3 candidates. Use `...` for empty slots.

The router is **stateless** — every call must be a fresh context with no conversation history passed between calls.

---

## Repository Structure

```
datasets/
├── v1/                         ← Proof of concept (deprecated)
│   ├── llm/                    ← LLM-generated batches, 24 action types
│   └── synthetic/              ← Conflict sample generator + output
│
├── v2/                         ← Strategy expansion (deprecated)
│   ├── generate.js
│   ├── analyze.js
│   ├── dedupe.js
│   ├── heartbeat/
│   ├── keyword/
│   └── combined/
│
├── v3/                         ← Previous production dataset (deprecated)
│   ├── generate_dataset.py
│   ├── compact_to_jsonl.py
│   ├── nim_annotator.py
│   ├── analyze_dataset.py
│   ├── dataset_v3.txt
│   ├── dataset_v3.jsonl
│   ├── dataset_v3_thinks.json
│   └── eval_benchmark_300.jsonl
│
└── v4/                         ← Current, production dataset
    ├── generate_dataset_v4.py  ← Main generator (9,300 samples)
    ├── nim_annotator_v4.py     ← NVIDIA NIM think-block annotation pipeline
    ├── analyze_dataset_v4.py   ← Distribution analysis + format validation
    ├── dataset_v4.jsonl        ← 9,300 base samples, no think blocks
    ├── dataset_v4_training.jsonl ← 9,300 samples with think blocks, training-ready
    ├── dataset_v4_thinks.json  ← Think block annotations only, no decisions
    ├── annotation_manifest_v4.jsonl ← Per-sample annotation records
    └── dataset_v4.txt          ← Compact human-readable review format
```

---

## Model + Dataset Evolution

### v1 — Proof of Concept

**Model:** Qwen2.5-0.5B-Instruct (~502M) · **Dataset:** ~3,000 samples across 24 action categories

Two generation strategies combined — LLM-prompted conversation batches and a synthetic conflict sample generator. The model showed early promise and proved the routing concept was viable. But 125 samples per category across 24 categories is not enough for a sub-1B model to generalize. It memorized patterns rather than learning intent.

**What worked:** Proved a small model can learn routing behavior.
**What failed:** 24 categories × 125 samples = severe overfitting, not enough signal per class.

---

### v2 — Task Simplification

**Model:** Qwen3.5-0.8B multimodal (~820M) · **Dataset:** ~5,000+ samples, heartbeat + keyword strategies

Simplified the problem to two concerns: **heartbeat** (window-level conversation state) and **keyword** (direct bot mention). Also explored variable window sizes from 2 to 5 messages. Introduced a proper generation pipeline with semantic deduplication.

Used Qwen3.5-0.8B, a multimodal model, with plans to add image recognition to the same router. Failed because the architecture was fundamentally different from the Qwen3 family and behaved poorly on pure text routing.

**What worked:** Heartbeat + keyword framing was a good simplification. Dedup pipeline was solid.
**What failed:** Wrong model family. Multimodal base model not suited for pure text routing.

---

### v3 — Scale + Implicit Intent

**Model:** Qwen3-0.6B-Instruct (~600M) · **Dataset:** 7,000 base / 20,297 augmented

Complete redesign of both model and dataset strategy. Switched to implicit intent — the model must infer action from what happened, not from what was asked:

| Old — explicit            | New — implicit                                            |
| ------------------------- | --------------------------------------------------------- |
| `"bhai funny meme bhej"`  | `"teacher ne galti se dating app screen share kar di 😂"` |
| `"sad sticker bhej yaar"` | `"bahut down feel ho raha hai aaj kuch accha nahi"`       |
| `"send gaming gif"`       | `"valorant mein ace maar diya just now insane tha"`       |

Think blocks annotated by `qwen3-next-80b-a3b-instruct` on NVIDIA NIM. 6 action types including acknowledge and translate.

**What worked:** Implicit intent framing. Qwen3 architecture. Think-block annotation pipeline.
**What failed:** Flat M1-M5 window with no distinction between context and candidates. acknowledge/translate types added noise with too few samples. LoRA r=8 too lean.

---

### v4 — H/C Window Architecture (Current)

**Model:** Qwen3-0.6B · **Dataset:** 9,300 samples

The biggest architectural change across all versions. Replaced the flat 5-message window with a structured **History + Candidates** format — 5 processed history messages (H1-H5) provide context, 1-3 unprocessed candidates (C1-C3) are what the router actually decides on. Filler `...` slots allow sparse windows without confusing the model.

**Why this matters:** In a real group chat, the bot sees a stream of messages. The router needs to know which messages are new (candidates) vs already processed (history). The old M1-M5 format treated all 5 messages equally — the router had no way to know which message to target. The H/C split solves this at the architecture level.

**4 action types only:** Dropped acknowledge and translate. Acknowledge was ambiguous — most were text or react. Translate was too rare in real Indian GC data to be worth the training signal dilution. Cleaner taxonomy = better generalization.

**@BOT window specialization:** 1,500 of 9,300 samples are dedicated @BOT mention windows — explicit bot mentions always force a non-ignore decision and are always hard tier. Separate deduplication pool prevents contamination with normal samples.

**Filler windows:** 800 samples with 1-2 real candidates and `...` fillers — teaches the model to work with sparse candidate sets, which happens when the trigger fires on fewer than 3 new messages.

**Think-block tiers:** Three tiers instead of uniform annotation — hard (39.3%) gets full reasoning, medium (32.8%) gets short 1-2 sentence reasoning, easy (27.8%) gets no think block and outputs the decision directly. Matches real inference behavior where obvious cases should skip thinking entirely.

**Dataset distribution (v4, 9,300 samples):**

| Type   | Count | %     |
| ------ | ----- | ----- |
| text   | 4,042 | 43.5% |
| ignore | 1,942 | 20.9% |
| react  | 1,688 | 18.2% |
| media  | 1,628 | 17.5% |

| Split          | Count |
| -------------- | ----- |
| Normal windows | 7,000 |
| @BOT windows   | 1,500 |
| Filler windows | 800   |

Language distribution: 58.8% Hinglish, 23% English, 18.2% mixed

Think blocks annotated by `qwen3-next-80b-a3b-instruct` on NVIDIA NIM.

**Training config:**

```python
SOURCE_MODEL = "unsloth/Qwen3-0.6B"
r            = 32
lora_alpha   = 32
epochs       = 2
batch_size   = 16   # 2 per device × 8 grad accum
learning_rate = 2e-4
max_seq_len  = 2048
```

Loss curve:

```
step 10   → 4.41
step 30   → 1.13
step 40   → 0.52
step 50   → 0.32
step 582  → 0.47  (epoch 1 end)
step 1164 → 0.667 (epoch 2 final)
```

---

## Evolution Summary

```
v1 (Qwen2.5-0.5B)       v2 (Qwen3.5-0.8B)       v3 (Qwen3-0.6B)         v4 (Qwen3-0.6B)
├─ 24 action types       ├─ 2 concerns             ├─ 6 action types         ├─ 4 action types
├─ 3K samples            ├─ Variable window        ├─ 7K + 20K augmented     ├─ 9.3K samples
├─ Explicit intent       ├─ Heartbeat/keyword      ├─ Implicit intent        ├─ H/C window split
├─ Proved concept        ├─ Wrong architecture     ├─ Think annotation       ├─ @BOT specialization
├─ LLM + synthetic gen   ├─ Good dedup pipeline    ├─ Flat M1-M5 format      ├─ Tiered think blocks
└─ Q4_K_M GGUF           └─ Q4_K_M GGUF            └─ Q8_0 GGUF r=8          └─ Q8_0 GGUF r=32
```

Four attempts, two complete failures, one working model iterated into a better one. Each version solved what the previous got wrong.

---

## License

Apache-2.0 — inherits from Qwen3 base model license.

---

## Author

**Kaushal Krishna**

- GitHub: [kaushalkrishnax](https://github.com/kaushalkrishnax)
- Main project: [github.com/kaushalkrishnax/auroic](https://github.com/kaushalkrishnax/auroic)
- Model: [huggingface.co/kaushalkrishnax/auroic-router-0.6b](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)
