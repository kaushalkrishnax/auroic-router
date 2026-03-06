# Auroic Router — Research Repository

> Training data, generation pipelines, and experiment history for the Auroic Router — a compact edge-deployable chat routing model for conversational AI systems.

- **Main Project** → [github.com/kaushalkrishnax/auroic](https://github.com/kaushalkrishnax/auroic)
- **Fine Tuned Model** → [huggingface.co/kaushalkrishnax/auroic-router-0.6b](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)

---

## What is the Auroic Router?

The Auroic Router is a small language model trained to read a window of 5 group chat messages and decide what action a conversational AI should take next — reply with text, send media, react with an emoji, acknowledge a status, translate a foreign message, or ignore noise entirely.

It is not a chatbot. It does not generate responses. It makes one decision per window, fast, on CPU.

```
[5 chat messages] → Router → R: TYPE=text | TARGET=M3 | EFFORT=medium | TITLE=text:exam_stress
```

The router sits at the front of a larger pipeline, directing traffic to the right handler so downstream models do not waste compute on irrelevant messages.

---

## Current Model — v3

| Property            | Value                            |
| ------------------- | -------------------------------- |
| Base                | Qwen3-0.6B-Instruct              |
| Parameters          | ~600M total, 5M trainable (LoRA) |
| Quantization        | Q8_0 GGUF                        |
| Training samples    | 7,000 with think-block reasoning |
| Fine-tuning         | Unsloth, 2 epochs, LoRA r=8      |
| Final training loss | ~0.45                            |
| Deployment          | Ollama / llama.cpp               |
| Cold start          | ~6s on i3-6100U CPU              |
| Warm inference      | ~3–4s loaded in memory           |

**→ [Download and use the model on HuggingFace](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)**

---

## Output Format

```
R: TYPE=<text|media|react|acknowledge|translate|ignore> | TARGET=<M1-M5|null> | EFFORT=<low|medium|high|null> | TITLE=<canonical_title>
```

Examples:

```
R: TYPE=text      | TARGET=M2 | EFFORT=medium | TITLE=text:exam_stress
R: TYPE=media     | TARGET=M5 | EFFORT=low    | TITLE=media:funny_moment
R: TYPE=react     | TARGET=M4 | EFFORT=high   | TITLE=react:placement_win
R: TYPE=translate | TARGET=M1 | EFFORT=medium | TITLE=translate:japanese_text
R: TYPE=ignore    | TARGET=null | EFFORT=null | TITLE=ignore
```

---

## Action Types

| Type          | When                                          | Example trigger                          |
| ------------- | --------------------------------------------- | ---------------------------------------- |
| `text`        | Help request, advice, emotional support       | `"sar dard ho raha hai kya karoon"`      |
| `media`       | Situational context implying gif/sticker/meme | `"boss gir gaya meeting mein 😂"`        |
| `react`       | Notable event deserving emoji reaction        | `"campus placement ho gayi Google mein"` |
| `acknowledge` | Status update, no question asked              | `"ghar pahunch gaya safely"`             |
| `translate`   | Foreign script present, others confused       | `"감사합니다 정말 도움이 많이 됐어요"`   |
| `ignore`      | Noise, spam, keyboard smash, laughter chains  | `"😂😂😂 lmaooo 💀💀"`                   |

---

## Inference

Requires exactly 5 messages. Empty fillers are fine and better than fewer messages:

```
M1: ...
M2: ...
M3: bhai placement test kal hai
M4: kuch nahi pada abhi
M5: 6 ghante bache hain 😭
```

Recommended Modelfile settings:

```
PARAMETER temperature    0.3
PARAMETER top_p          0.9
PARAMETER top_k          20
PARAMETER repeat_penalty 1.1
```

Use `temperature 0.6` if slight output variation is acceptable — closer to Qwen3's official thinking mode recommendation but less deterministic for routing.

---

## Repository Structure

```
datasets/
├── v1/                        ← Foundation (deprecated)
│   ├── llm/                   ← LLM-generated batches, 24 action types
│   └── synthetic/             ← Conflict sample generator + output
│
├── v2/                        ← Strategy expansion (deprecated)
│   ├── generate.js
│   ├── analyze.js
│   ├── dedupe.js
│   ├── heartbeat/             ← Heartbeat strategy dataset
│   ├── keyword/               ← Keyword detection dataset
│   └── combined/              ← Merged + deduplicated output
│
└── v3/                        ← Current, production dataset
    ├── generate_dataset.py    ← Main generator (7,000 samples, compact .txt)
    ├── compact_to_jsonl.py    ← Converter + think-block injector
    ├── nim_annotator.py       ← NVIDIA NIM annotation pipeline
    ├── analyze_dataset.py     ← Distribution analysis + validation
    ├── bak-scripts/           ← Previous script iterations
    ├── dataset_v3.txt         ← 7,000 base samples, compact format
    ├── dataset_v3.jsonl       ← 7,000 samples with think blocks, training-ready
    ├── dataset_v3_thinks.json ← 7,000 think blocks indexed by sample
    ├── dataset_v3_augmented.txt
    ├── dataset_v3_augmented.jsonl
    └── eval_benchmark_300.jsonl
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

Simplified the problem to two concerns: **heartbeat** (window-level conversation state) and **keyword** (direct bot mention — should the router respond at all?). Also explored variable window sizes from 2 to 5 messages. Introduced a proper generation pipeline with semantic deduplication.

Used Qwen3.5-0.8B, a multimodal `image-text-to-text` model, with plans to add image recognition to the same router. Failed because the architecture was fundamentally different from the Qwen3 family, behaved poorly on pure text routing, and the base (non-instruct) variant gave no usable output without heavy prompting.

**What worked:** Heartbeat + keyword framing was a good simplification. Dedup pipeline was solid.
**What failed:** Wrong model family. Multimodal base model not suited for pure text routing.

---

### v3 — Scale + Implicit Intent (Current)

**Model:** Qwen3-0.6B-Instruct (~600M) · **Dataset:** 7,000 base / 20,297 augmented

Complete redesign of both model and dataset strategy. Qwen3-0.6B sits at the sweet spot between the older Qwen2.5 family and the experimental Qwen3.5 — native thinking mode, strong ChatML instruction following, and Hinglish + multilingual capability out of the box.

**Implicit intent** — the biggest change. Previous versions relied on explicit requests. v3 uses situational context — the model must infer action from what happened, not from what was asked:

| Old — explicit            | New — implicit                                            |
| ------------------------- | --------------------------------------------------------- |
| `"bhai funny meme bhej"`  | `"teacher ne galti se dating app screen share kar di 😂"` |
| `"sad sticker bhej yaar"` | `"bahut down feel ho raha hai aaj kuch accha nahi"`       |
| `"send gaming gif"`       | `"valorant mein ace maar diya just now insane tha"`       |
| `"help karo please"`      | `"agar fail hua na..."`                                   |

**Think-block annotation** — every base sample was annotated with a `<think>` reasoning block by `qwen3-next-80b-a3b-instruct` (80B total, ~3B active via sparse MoE) on NVIDIA NIM free tier. 875 batches of 8, ~1–2 hours with rate-limit-aware auto-resume. Each block quotes the signal phrase and justifies the decision:

```
M2 says 'stress ho raha yaar' and M4 repeats 'same bro',
but M5's 'agar fail hua na' is a real cry for help —
needs text advice, not just react or ignore.
```

**Context Permutation Augmentation** — the 20,297 augmented samples are message-order shuffles of the base 7,000, training the model to find signal regardless of position. Augmented samples are decision-only (no think blocks) to reinforce output format without contradicting reasoning patterns. Included for future fine-tuning of larger 2B–4B Qwen models where 90–95% accuracy is expected.

**Dataset distribution (base 7,000):**

| Type        | Count | %     |
| ----------- | ----- | ----- |
| text        | 2,604 | 37.2% |
| react       | 1,133 | 16.2% |
| media       | 1,083 | 15.5% |
| acknowledge | 957   | 13.7% |
| translate   | 742   | 10.6% |
| ignore      | 481   | 6.9%  |

**Training config:**

```python
SOURCE_MODEL  = "unsloth/Qwen3-0.6B"
r             = 8
lora_alpha    = 16
epochs        = 2
batch_size    = 8        # 2 per device × 4 grad accum
learning_rate = 2e-4
max_seq_len   = 512
```

Loss curve:

```
step 10   → 2.96
step 50   → 1.06
step 200  → 0.80
step 875  → 0.52  (epoch 1 end)
step 1750 → 0.45  (epoch 2 final)
```

---

## Evolution Summary

```
v1 (Qwen2.5-0.5B)           v2 (Qwen3.5-0.8B)           v3 (Qwen3-0.6B)
├─ 24 action categories      ├─ 2 concerns only           ├─ 6 action types
├─ 3K samples, overfit       ├─ Variable window (2-5)     ├─ 7K samples + thinks
├─ Proved concept viable     ├─ Wrong architecture        ├─ Implicit intent
├─ LLM + synthetic gen       ├─ Heartbeat + keyword       ├─ NIM think annotation
└─ Q4_K_M GGUF               └─ Q4_K_M GGUF               └─ Q8_0 GGUF (recommended)
```

Three attempts, two failures, one working model. The failures were not wasted — v1 proved the concept, v2 found the right task framing. v3 combined both lessons with the right architecture.

No structured git commit history exists for this repository — development moved fast and was focused on shipping a working router for the main Auroic project. This README serves as the complete record.

---

## License

Apache-2.0 — inherits from Qwen3 base model license.

---

## Author

**Kaushal Krishna**
- GitHub: [github.com/kaushalkrishnax](https://github.com/kaushalkrishnax)
- Main project: [github.com/kaushalkrishnax/auroic](https://github.com/kaushalkrishnax/auroic)
- Model: [huggingface.co/kaushalkrishnax/auroic-router-0.6b](https://huggingface.co/kaushalkrishnax/auroic-router-0.6b)
