#!/usr/bin/env python3
"""
NIM Think Block Annotator — Qwen3-Next-80B-A3B-Instruct
=========================================================
Reads dataset_v3.txt, annotates with think blocks via NVIDIA NIM,
produces dataset_v3_thinks.json for compact_to_jsonl.py --thinks

Usage:
    python nim_annotator.py dataset_v3.txt
    python nim_annotator.py dataset_v3.txt --output thinks.json --batch-size 5
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
BATCH_SIZE = 5
SLEEP_BETWEEN = 1.1
DAILY_REQUEST_LIMIT = 950

SYSTEM = """You are annotating reasoning traces for training a tiny Qwen3 0.6B chat router model.

The router reads 5 group chat messages and decides: what action to take, which message triggered it, how much effort, and what the topic is.

Your job: for each block of messages + routing decision, write a short think block that shows HOW a smart reader would arrive at that decision by scanning the messages naturally.

Output format — one quoted string per block in order:
B1: "think text here"
B2: "think text here"
...and so on for ALL blocks

Think block style:
- Write like you are quickly scanning a chat and thinking out loud
- 2-3 short sentences, under 80 tokens total
- Mention which message caught your attention and why
- Quote 3-5 words from the signal message as evidence
- Explain why this action and not a different one
- Use natural language, not robotic templates

Examples of good think blocks:

For media:
B1: "Scanning messages — M5 says 'send cooking gif something looks delicious', clear media request. No help needed, just wants a gif sent. Media fits, not text which would miss the point entirely."

For text/advice:
B1: "M4 stands out — 'sar dard ho raha hai kya karoon remedy batao', asking for a remedy. Person wants actual advice, not just a reaction or acknowledgment. Text response needed."

For translate:
B1: "M1 is Japanese script — '頑張ってください応援していますよ', others are asking what it means. Intent is to understand the foreign text. Translate fits, nothing else addresses the language barrier."

For acknowledge:
B1: "M5 is a simple status drop — 'reached hospital safely waiting outside for doc'. No question, no request, just informing. Acknowledge fits, text would be overkill for a status update."

For react:
B1: "M5 shares good news — 'result aa gaya marks acche aaye yaar finally'. Pure celebration moment, no advice needed. React with emoji fits better than text which would feel too formal."

For ignore:
B1: "All messages are noise — spam links, keyboard smash, or pure laughter chains. No real intent anywhere. Ignore fits, nothing to route."

Rules:
- Output EXACTLY one B line per input block — B1 for first, B2 for second, etc
- Each B line must be a single line — no line breaks inside the quoted string
- NEVER skip a block — if unsure, write your best guess
- Messages have typos, Hinglish, slang, emoji — read intent not surface quality
- No preamble, no explanation, output only the B lines nothing else"""


def is_truncated(think_text):
    text = think_text.strip()
    if not text:
        return True
    if text.endswith("—") or text.endswith(":") or text.endswith(","):
        return True
    if len(text.split()) < 8:
        return True
    return False


def parse_compact_blocks(path):
    blocks = []
    current = []
    index = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current:
                    block = _extract_block(current, index)
                    if block:
                        blocks.append(block)
                        index += 1
                    current = []
            else:
                current.append(line)

    if current:
        block = _extract_block(current, index)
        if block:
            blocks.append(block)

    return blocks


def _extract_block(lines, index):
    msgs = [l for l in lines if re.match(r"^M[1-5]: ", l)]
    dec = next((l for l in lines if l.startswith("R: ")), None)
    if len(msgs) != 5 or not dec:
        return None
    return (index, "\n".join(msgs), dec)


def format_batch(batch):
    parts = []
    for i, (idx, msgs, dec) in enumerate(batch):
        parts.append(f"Block B{i+1}:\n{msgs}\nDecision: {dec}")
    return "\n\n".join(parts)


def parse_response(response_text, batch_size):
    """
    Robust parser — handles:
    - B1: "quoted text"
    - B1: unquoted text
    - B1: 'single quoted'
    - multiline responses where think spills to next line
    """
    result = {}

    # Strategy 1: quoted strings — most reliable
    # matches B1: "..." including text with internal punctuation
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
        return result  # perfect parse, done

    # Strategy 2: single quoted
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

    # Strategy 3: unquoted — grab everything after B1: until next B line or end
    # split response into lines, find B markers
    lines = response_text.split('\n')
    current_num = None
    current_text = []

    for line in lines:
        b_match = re.match(r'^B(\d+):\s*(.*)', line)
        if b_match:
            # save previous
            if current_num is not None and current_num not in result:
                combined = ' '.join(current_text).strip().strip('"\'')
                if combined and len(combined.split()) >= 5:
                    result[current_num] = combined
            current_num = int(b_match.group(1)) - 1
            current_text = [b_match.group(2).strip().strip('"\'')]
        elif current_num is not None and line.strip():
            # continuation line — append if not a new B marker
            current_text.append(line.strip().strip('"\''))

    # save last
    if current_num is not None and current_num not in result:
        combined = ' '.join(current_text).strip().strip('"\'')
        if combined and len(combined.split()) >= 5:
            result[current_num] = combined

    return result


def call_nim(batch):
    user_prompt = format_batch(batch)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=800,      # bumped from 600 — gives room for all 5 blocks
            top_p=0.8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""
        return parse_response(raw, len(batch)), raw

    except Exception as e:
        print(f"   ⚠️  API error: {e}")
        return {}, ""


def annotate(txt_path, output_path, batch_size=BATCH_SIZE, resume=True):
    thinks = {}

    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            thinks = json.load(f)

        # auto-detect and remove truncated entries for re-annotation
        truncated = [k for k, v in thinks.items() if is_truncated(v)]
        if truncated:
            print(f"   ⚠️  {len(truncated)} truncated entries removed for retry")
            for k in truncated:
                del thinks[k]

        print(f"   Resuming — {len(thinks)} valid thinks already done")

    all_blocks = parse_compact_blocks(txt_path)
    total = len(all_blocks)
    print(f"   Total blocks: {total}")

    pending = [(idx, msgs, dec) for idx, msgs, dec in all_blocks
               if str(idx) not in thinks]
    print(f"   Pending: {len(pending)}")

    batches = [pending[i:i+batch_size]
               for i in range(0, len(pending), batch_size)]
    print(f"   Batches: {len(batches)} × {batch_size}")
    print()

    requests_made = 0
    failed_indices = []
    total_hits = 0

    for b_num, batch in enumerate(batches):
        if requests_made >= DAILY_REQUEST_LIMIT:
            print(f"\n⚠️  Daily limit reached at batch {b_num}. Saving progress.")
            break

        result, raw = call_nim(batch)
        requests_made += 1

        hits = 0
        for local_idx, (global_idx, _, _) in enumerate(batch):
            if local_idx in result:
                text = result[local_idx]
                if not is_truncated(text):
                    thinks[str(global_idx)] = text
                    hits += 1
                else:
                    failed_indices.append(global_idx)
            else:
                failed_indices.append(global_idx)

        total_hits += hits
        running_rate = total_hits / (requests_made * batch_size) * 100

        print(
            f"   Batch {b_num+1}/{len(batches)} "
            f"{hits}/{len(batch)} ✓  "
            f"[total {len(thinks)}/{total} | parse rate {running_rate:.0f}%]"
        )

        # debug: print raw if very low hit rate
        if hits <= 1 and len(batch) >= 4:
            print(f"   ⚠️  Low hits — raw snippet: {raw[:200]!r}")

        # save after every batch
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thinks, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_BETWEEN)

    # retry failed individually
    if failed_indices:
        unique_failed = list(set(
            idx for idx in failed_indices if str(idx) not in thinks
        ))
        print(f"\n🔁 Retrying {len(unique_failed)} failed blocks individually...")

        retry_blocks = [b for b in all_blocks if b[0] in unique_failed]
        recovered = 0

        for block in retry_blocks:
            if requests_made >= DAILY_REQUEST_LIMIT:
                print("   Daily limit hit during retry. Stopping.")
                break

            result, _ = call_nim([block])
            requests_made += 1

            if 0 in result and not is_truncated(result[0]):
                thinks[str(block[0])] = result[0]
                recovered += 1
                print(f"   ✓ recovered {block[0]}")
            else:
                print(f"   ✗ failed again {block[0]}")

            time.sleep(SLEEP_BETWEEN)

        print(f"   Recovered {recovered}/{len(unique_failed)}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thinks, f, ensure_ascii=False, indent=2)

    # final stats
    coverage = len(thinks) / total * 100
    print(f"\n{'='*50}")
    print(f"✅ {len(thinks)}/{total} think blocks ({coverage:.1f}% coverage)")
    print(f"   Requests made: {requests_made}")
    print(f"   Output: {output_path}")
    if len(thinks) < total:
        missing = total - len(thinks)
        print(f"   ⚠️  {missing} missing — run again tomorrow to fill gaps")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python nim_annotator.py <dataset.txt> [--output thinks.json] [--batch-size 5]")
        sys.exit(1)

    txt_path = args[0]
    output_path = "dataset_v3_thinks.json"
    batch_size = BATCH_SIZE

    if "--output" in args:
        idx = args.index("--output")
        output_path = args[idx + 1]

    if "--batch-size" in args:
        idx = args.index("--batch-size")
        batch_size = int(args[idx + 1])

    if not os.path.exists(txt_path):
        print(f"❌ File not found: {txt_path}")
        sys.exit(1)

    print(f"🚀 NIM Annotator — {MODEL}")
    print(f"   Input: {txt_path}")
    print(f"   Output: {output_path}")
    print(f"   Batch size: {batch_size}")
    print()

    annotate(txt_path, output_path, batch_size)


if __name__ == "__main__":
    main()