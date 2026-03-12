#!/usr/bin/env python3
"""
NIM Think Block Annotator v4
==============================
Reads annotation_manifest_v4.jsonl, annotates with think blocks via NVIDIA NIM,
produces dataset_v4_thinks.json for compact_to_jsonl.py

Three prompt templates based on think_tier:
  hard   → full think block (2-3 sentences, detailed reasoning)
  medium → short think block (1-2 sentences, key signal only)
  easy   → skip entirely (no annotation needed, model outputs directly)

Usage:
    python nim_annotator_v4.py
    python nim_annotator_v4.py --manifest annotation_manifest_v4.jsonl --output thinks_v4.json --batch-size 8
"""

import json, re, time, sys, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NV_API_KEY"),
)

MODEL = "qwen/qwen3-next-80b-a3b-instruct"
BATCH_SIZE = 8
SLEEP_BETWEEN = 1.1
DAILY_REQUEST_LIMIT = 950

# ─── Input/Output format reference ───────────────────────────────────────────
#
# Input format (H5+C3):
#   H1: <history msg>
#   H2: <history msg>
#   H3: <history msg>
#   H4: <history msg>
#   H5: <history msg>
#   C1: <candidate or ...>
#   C2: <candidate or ...>
#   C3: <candidate or ...>
#   /think
#
# Output format (v4, no nulls):
#   R: TYPE=text   | TARGET=C2 | EFFORT=low/medium/high
#   R: TYPE=react  | TARGET=C1 | TITLE=🔥
#   R: TYPE=media  | TARGET=C3 | TITLE=crying laughing
#   R: TYPE=ignore
#
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — one per think tier
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_HARD = """You are annotating reasoning traces for training a small Qwen3 0.6B group chat router.

The router reads a group chat window: H1-H5 are history messages (already processed context), C1-C3 are candidate messages (what to evaluate now). Some candidates may be '...' which means filler/empty slot. Some messages contain @BOT which is an explicit bot mention.

The router must decide:
  text   → someone needs help, advice, or a question answered. OUTPUT has TARGET + EFFORT
  react  → a moment deserves an emoji reaction. OUTPUT has TARGET + TITLE (the emoji)
  media  → a situation calls for a gif/sticker/meme. OUTPUT has TARGET + TITLE (search query)
  ignore → all candidates are noise, spam, laughter, keyboard smash, or nothing actionable

Your job: write a FULL think block showing HOW a smart reader scans the window and arrives at the decision.

Output format — one B line per block in order, no line breaks inside quotes:
B1: "think text here"
B2: "think text here"
...

Think block rules (HARD tier — full reasoning):
- 2-3 short sentences, under 90 tokens total
- First: scan H1-H5 quickly, note the conversational context
- Second: identify which candidate (C1/C2/C3) has the real signal and why
- Third: explain why this action type and not another one
- Quote 3-5 words from the signal candidate as evidence
- Mention if fillers ('...') are present and were skipped
- For @BOT mentions: note the explicit mention and why ignore is off the table
- Use natural thinking language, not robotic templates

Good examples:

For TYPE=text:
B1: "H1-H5 is casual chatter with no real asks. C2 stands out — 'bhai ye python error samajh nahi aa raha' is a clear technical help request. Text fits, not react which would ignore the actual problem entirely."

For TYPE=react:
B1: "History is filler noise. C1 drops news — 'placement ho gayi campus se finally done' is pure celebration. React with emoji fits perfectly, text would feel over-formal for a win moment like this."

For TYPE=media:
B1: "Context is relaxed chat. C3 describes a funny situation — 'teacher ne galti se dating app screen share kar di' calls for a reaction gif. Media fits, text would over-explain what is already funny."

For TYPE=ignore:
B1: "All candidates are noise — C1 is keyboard smash, C2 is '...', C3 is emoji spam. History is also pure laughter chain. Nothing actionable anywhere. Ignore is the only right call."

For @BOT with text:
B1: "C1 has an explicit @BOT mention — 'bhai ye api 404 de rahi hai baaki sab theek hai'. Direct ask, ignore is not an option here. High effort because API debugging needs step by step help."

For filler window:
B1: "Sparse window — C2 and C3 are '...' fillers. Only C1 has real content: 'bahut down feel ho raha hai' — emotional low. Media comfort gif fits, single candidate but signal is clear."

Rules:
- Output EXACTLY one B line per input block — B1 for first, B2 for second, etc
- Each B line must be a single line with no internal line breaks
- NEVER skip a block — write your best reasoning even if uncertain
- Read intent through typos, Hinglish, slang, emoji — surface noise is normal
- No preamble, no explanation after the B lines, output only the B lines"""


SYSTEM_MEDIUM = """You are annotating short reasoning traces for training a small Qwen3 0.6B group chat router.

The router reads H1-H5 (history context) and C1-C3 (candidate messages, some may be '...' fillers). It decides: text, react, media, or ignore.

Your job: write a SHORT think block — just the key signal and decision. No long explanation needed.

Output format — one B line per block, no line breaks inside quotes:
B1: "think text here"
B2: "think text here"
...

Think block rules (MEDIUM tier — brief reasoning):
- 1-2 short sentences ONLY, under 50 tokens
- Identify the signal candidate and quote 2-4 words from it
- State why the action type fits in one line
- Skip history analysis unless directly relevant

Good examples:

For TYPE=text:
B1: "C2 says 'kya karun breakup ho gayi' — emotional support needed. Text fits."

For TYPE=react:
B1: "C3 announces 'placement ho gayi finally' — celebration moment. React with 🥳."

For TYPE=media:
B1: "C1 describes a funny moment — 'teacher dating app screen share'. Media gif fits."

For TYPE=ignore:
B1: "All candidates are laughter noise or '...' fillers. Nothing to route, ignore."

For @BOT:
B1: "C1 has @BOT mention with a direct question. Cannot ignore, text response needed."

For filler window (... slots):
B1: "C1 is the only real message — 'bahut stressed hai'. Text support needed."

Rules:
- EXACTLY one B line per block, single line, no internal breaks
- NEVER skip a block
- Output only B lines, nothing else"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def is_truncated(text):
    t = text.strip()
    if not t:
        return True
    if t.endswith(("—", ":", ",")):
        return True
    if len(t.split()) < 6:
        return True
    return False


def load_manifest(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_batch(batch):
    """Format a batch of manifest records into one NIM prompt."""
    parts = []
    for i, rec in enumerate(batch):
        parts.append(f"Block B{i+1}:\n{rec['input']}\nDecision: {rec['output']}")
    return "\n\n".join(parts)


def parse_response(response_text, batch_size):
    """
    Robust parser — handles quoted, single-quoted, and unquoted B lines.
    Returns dict: {local_index: think_text}
    """
    result = {}

    # Strategy 1: double-quoted
    quoted = re.findall(
        r'^B(\d+):\s*"((?:[^"\\]|\\.)+)"',
        response_text,
        re.MULTILINE
    )
    for num_str, text in quoted:
        num = int(num_str) - 1
        if 0 <= num < batch_size:
            result[num] = text.strip()

    if len(result) == batch_size:
        return result

    # Strategy 2: single-quoted
    single = re.findall(
        r"^B(\d+):\s*'((?:[^'\\]|\\.)+)'",
        response_text,
        re.MULTILINE
    )
    for num_str, text in single:
        num = int(num_str) - 1
        if 0 <= num < batch_size and num not in result:
            result[num] = text.strip()

    if len(result) == batch_size:
        return result

    # Strategy 3: unquoted — grab everything after B1: until next B line
    lines = response_text.split("\n")
    current_num = None
    current_text = []

    for line in lines:
        b_match = re.match(r"^B(\d+):\s*(.*)", line)
        if b_match:
            if current_num is not None and current_num not in result:
                combined = " ".join(current_text).strip().strip("\"'")
                if combined and len(combined.split()) >= 5:
                    result[current_num] = combined
            current_num = int(b_match.group(1)) - 1
            current_text = [b_match.group(2).strip().strip("\"'")]
        elif current_num is not None and line.strip():
            current_text.append(line.strip().strip("\"'"))

    # save last
    if current_num is not None and current_num not in result:
        combined = " ".join(current_text).strip().strip("\"'")
        if combined and len(combined.split()) >= 5:
            result[current_num] = combined

    return result


def call_nim(batch, system_prompt):
    user_prompt = format_batch(batch)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=900,
            top_p=0.8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""
        return parse_response(raw, len(batch)), raw
    except Exception as e:
        print(f"   ⚠️  API error: {e}")
        return {}, ""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANNOTATOR
# ═══════════════════════════════════════════════════════════════════════════════

def annotate(manifest_path, output_path, batch_size=BATCH_SIZE, resume=True):
    thinks = {}

    # Load existing progress
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            thinks = json.load(f)

        # Remove truncated entries for retry
        truncated = [k for k, v in thinks.items() if is_truncated(v)]
        if truncated:
            print(f"   ⚠️  {len(truncated)} truncated entries removed for retry")
            for k in truncated:
                del thinks[k]

        print(f"   Resuming — {len(thinks)} valid thinks already done")

    all_records = load_manifest(manifest_path)
    total = len(all_records)

    # Separate by tier
    hard_records   = [r for r in all_records if r["think_tier"] == "hard"]
    medium_records = [r for r in all_records if r["think_tier"] == "medium"]
    easy_records   = [r for r in all_records if r["think_tier"] == "easy"]

    print(f"   Manifest: {total} records")
    print(f"   hard (full think):   {len(hard_records)}")
    print(f"   medium (short):      {len(medium_records)}")
    print(f"   easy (skip):         {len(easy_records)} — no annotation needed")
    print()

    requests_made = 0
    failed_ids = []

    def process_tier(records, system_prompt, tier_name):
        nonlocal requests_made

        pending = [r for r in records if r["id"] not in thinks]
        if not pending:
            print(f"   [{tier_name}] All done already, skipping")
            return

        print(f"   [{tier_name}] Processing {len(pending)} pending records...")
        batches = [pending[i:i+batch_size] for i in range(0, len(pending), batch_size)]

        total_hits = 0
        for b_num, batch in enumerate(batches):
            if requests_made >= DAILY_REQUEST_LIMIT:
                print(f"\n⚠️  Daily limit reached at [{tier_name}] batch {b_num}. Saving progress.")
                return

            result, raw = call_nim(batch, system_prompt)
            requests_made += 1

            hits = 0
            for local_idx, rec in enumerate(batch):
                if local_idx in result:
                    text = result[local_idx]
                    if not is_truncated(text):
                        thinks[rec["id"]] = text
                        hits += 1
                    else:
                        failed_ids.append(rec["id"])
                else:
                    failed_ids.append(rec["id"])

            total_hits += hits
            running_rate = total_hits / ((b_num + 1) * batch_size) * 100

            print(
                f"   [{tier_name}] Batch {b_num+1}/{len(batches)} "
                f"{hits}/{len(batch)} ✓  "
                f"[total {len(thinks)}/{total} | rate {running_rate:.0f}%]"
            )

            if hits <= 1 and len(batch) >= 4:
                print(f"   ⚠️  Low hits — raw snippet: {raw[:200]!r}")

            # Save after every batch
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(thinks, f, ensure_ascii=False, indent=2)

            time.sleep(SLEEP_BETWEEN)

    # Process hard tier first — most important
    process_tier(hard_records, SYSTEM_HARD, "hard")

    # Process medium tier
    process_tier(medium_records, SYSTEM_MEDIUM, "medium")

    # Easy tier — stamp empty string as marker (no annotation needed)
    easy_pending = [r for r in easy_records if r["id"] not in thinks]
    if easy_pending:
        print(f"\n   [easy] Marking {len(easy_pending)} records as no-think...")
        for rec in easy_pending:
            thinks[rec["id"]] = ""  # empty = no think block
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thinks, f, ensure_ascii=False, indent=2)
        print(f"   [easy] Done")

    # Retry individual failures
    all_annotatable = hard_records + medium_records
    unique_failed = list(set(
        fid for fid in failed_ids if fid not in thinks or is_truncated(thinks.get(fid, ""))
    ))

    if unique_failed:
        print(f"\n🔁 Retrying {len(unique_failed)} failed blocks individually...")
        id_to_record = {r["id"]: r for r in all_annotatable}
        recovered = 0

        for fid in unique_failed:
            if requests_made >= DAILY_REQUEST_LIMIT:
                print("   Daily limit hit during retry. Stopping.")
                break

            rec = id_to_record.get(fid)
            if not rec:
                continue

            # Use appropriate system prompt based on tier
            sys_prompt = SYSTEM_HARD if rec["think_tier"] == "hard" else SYSTEM_MEDIUM
            result, _ = call_nim([rec], sys_prompt)
            requests_made += 1

            if 0 in result and not is_truncated(result[0]):
                thinks[fid] = result[0]
                recovered += 1
                print(f"   ✓ recovered {fid}")
            else:
                print(f"   ✗ failed again {fid}")

            time.sleep(SLEEP_BETWEEN)

        print(f"   Recovered {recovered}/{len(unique_failed)}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thinks, f, ensure_ascii=False, indent=2)

    # Final stats
    hard_done   = sum(1 for r in hard_records   if r["id"] in thinks and thinks[r["id"]])
    medium_done = sum(1 for r in medium_records if r["id"] in thinks and thinks[r["id"]])
    easy_done   = sum(1 for r in easy_records   if r["id"] in thinks)
    total_done  = hard_done + medium_done + easy_done

    print(f"\n{'='*55}")
    print(f"✅ Annotation complete")
    print(f"   hard   coverage: {hard_done}/{len(hard_records)} ({hard_done/max(len(hard_records),1)*100:.1f}%)")
    print(f"   medium coverage: {medium_done}/{len(medium_records)} ({medium_done/max(len(medium_records),1)*100:.1f}%)")
    print(f"   easy   coverage: {easy_done}/{len(easy_records)} ({easy_done/max(len(easy_records),1)*100:.1f}%)")
    print(f"   total:           {total_done}/{total} ({total_done/max(total,1)*100:.1f}%)")
    print(f"   Requests made:   {requests_made}")
    print(f"   Output:          {output_path}")
    if total_done < total:
        missing = total - total_done
        print(f"   ⚠️  {missing} missing — run again to fill gaps (auto-resumes)")


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL BUILDER — merge thinks into training-ready JSONL
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_jsonl(manifest_path, thinks_path, base_jsonl_path, output_path):
    """
    Merge think blocks into the base JSONL to produce final training JSONL.

    For hard/medium: insert <think>...</think> before the R: decision line
    For easy: keep output as-is (no think block)
    """
    with open(thinks_path, "r", encoding="utf-8") as f:
        thinks = json.load(f)

    records = load_manifest(manifest_path)
    id_to_tier  = {r["id"]: r["think_tier"] for r in records}
    id_to_think = thinks

    with open(base_jsonl_path, "r", encoding="utf-8") as f:
        base_samples = [json.loads(l) for l in f if l.strip()]

    output_samples = []
    missing_thinks = 0

    import hashlib
    for s in base_samples:
        inp = s["messages"][1]["content"]
        sid = hashlib.md5(inp.encode()).hexdigest()[:12]

        tier = id_to_tier.get(sid, "easy")
        think_text = id_to_think.get(sid, "")

        if tier in ("hard", "medium") and not think_text:
            missing_thinks += 1

        # Build assistant content
        decision = s["messages"][2]["content"]
        if think_text and think_text.strip():
            assistant_content = f"<think>\n{think_text.strip()}\n</think>\n{decision}"
        else:
            assistant_content = decision

        output_samples.append({
            "type": "chatml",
            "messages": [
                s["messages"][0],
                s["messages"][1],
                {"role": "assistant", "content": assistant_content},
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in output_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Training JSONL built: {len(output_samples)} samples → {output_path}")
    if missing_thinks:
        print(f"   ⚠️  {missing_thinks} hard/medium samples missing think blocks — run annotator first")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    manifest_path  = "annotation_manifest_v4.jsonl"
    output_path    = "dataset_v4_thinks.json"
    batch_size     = BATCH_SIZE
    build_mode     = False
    base_jsonl     = "dataset_v4.jsonl"
    final_out      = "dataset_v4_training.jsonl"

    if "--manifest" in args:
        manifest_path = args[args.index("--manifest") + 1]
    if "--output" in args:
        output_path = args[args.index("--output") + 1]
    if "--batch-size" in args:
        batch_size = int(args[args.index("--batch-size") + 1])
    if "--build" in args:
        build_mode = True
    if "--base-jsonl" in args:
        base_jsonl = args[args.index("--base-jsonl") + 1]
    if "--final-out" in args:
        final_out = args[args.index("--final-out") + 1]

    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        print(f"   Run generate_dataset_v4.py first to produce annotation_manifest_v4.jsonl")
        sys.exit(1)

    if build_mode:
        # Merge thinks into final training JSONL
        if not os.path.exists(output_path):
            print(f"❌ Thinks file not found: {output_path}")
            print(f"   Run annotator first (without --build) to generate thinks")
            sys.exit(1)
        if not os.path.exists(base_jsonl):
            print(f"❌ Base JSONL not found: {base_jsonl}")
            sys.exit(1)
        print(f"🔨 Building training JSONL...")
        build_training_jsonl(manifest_path, output_path, base_jsonl, final_out)
        return

    print(f"🚀 NIM Annotator v4 — {MODEL}")
    print(f"   Manifest:   {manifest_path}")
    print(f"   Output:     {output_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Daily cap:  {DAILY_REQUEST_LIMIT} requests")
    print()

    annotate(manifest_path, output_path, batch_size, resume=True)

    print(f"\nNext step when annotation is complete:")
    print(f"   python nim_annotator_v4.py --build")
    print(f"   → produces {final_out} (training-ready)")


if __name__ == "__main__":
    main()