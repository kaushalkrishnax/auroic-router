#!/usr/bin/env python3
"""
Compact .txt → Training .jsonl Converter
==========================================
Reads compact dataset format and produces ChatML JSONL for SFT training.
Supports optional think-block injection via --thinks flag.

Usage:
    python compact_to_jsonl.py dataset_v3.txt
    python compact_to_jsonl.py dataset_v3.txt dataset_v3_augmented.txt
    python compact_to_jsonl.py dataset_v3.txt --thinks dataset_v3_thinks.json
"""

import json, sys, os, re

# ── Must match modelfile SYSTEM exactly ──
SYSTEM = "You are the Auroic Router. Given 5 chat messages, output exactly one routing decision in this format: R: TYPE=<text|media|react|acknowledge|translate|ignore> | TARGET=<M1|M2|M3|M4|M5|null> | EFFORT=<low|medium|high|null> | TITLE=<canonical_title>"

VALID_TYPES   = {"text", "media", "react", "acknowledge", "translate", "ignore"}
VALID_TARGETS = {"M1", "M2", "M3", "M4", "M5", "null"}
VALID_EFFORTS = {"low", "medium", "high", "null"}


# ═══════════════════════════════════════════════════════════
# THINK BLOCK CLEANER
# ═══════════════════════════════════════════════════════════

def clean_think(text):
    """Clean think block text — remove Bxx references, fix whitespace."""
    if not text:
        return text

    # remove "same as B13" / "similar to B2" / "as in B5" / "like B3"
    text = re.sub(r',?\s*same as B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s*similar to B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s*as in B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s*like B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s*same logic as B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r',?\s*see B\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(same as B\d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(see B\d+\)', '', text, flags=re.IGNORECASE)

    # fix trailing comma before period
    text = re.sub(r',\s*\.', '.', text)
    # fix double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # fix " — ." artifacts
    text = re.sub(r'—\s*\.', '.', text)
    # strip
    text = text.strip()

    return text


def is_truncated(text):
    """Detect think blocks that got cut off mid-sentence."""
    if not text:
        return True
    text = text.strip()
    if text.endswith(("—", ":", ",", "—\n")):
        return True
    if len(text.split()) < 8:
        return True
    return False


# ═══════════════════════════════════════════════════════════
# COMPACT PARSER
# ═══════════════════════════════════════════════════════════

def parse_compact(path):
    """Parse compact .txt format into list of (user_content, decision) tuples."""
    samples = []
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current_lines:
                    sample = _parse_block(current_lines)
                    if sample:
                        samples.append(sample)
                    current_lines = []
                continue
            current_lines.append(line)

    # handle last block with no trailing blank line
    if current_lines:
        sample = _parse_block(current_lines)
        if sample:
            samples.append(sample)

    return samples


def _parse_block(lines):
    """Parse one compact block into (user_content, decision_line)."""
    msg_lines = []
    decision = None

    for line in lines:
        if line.startswith("R: "):
            decision = line
        elif re.match(r"^M[1-5]: ", line):
            msg_lines.append(line)

    if len(msg_lines) != 5 or decision is None:
        return None

    user_content = "\n".join(msg_lines) + "\n/think"
    return (user_content, decision)


# ═══════════════════════════════════════════════════════════
# CONVERTER
# ═══════════════════════════════════════════════════════════

def convert(txt_path, thinks=None):
    """Convert compact .txt to .jsonl, optionally merging think blocks."""
    samples = parse_compact(txt_path)
    if not samples:
        print(f"❌ No samples parsed from {txt_path}")
        return

    # load think blocks
    think_map = {}
    if thinks and os.path.exists(thinks):
        with open(thinks, "r", encoding="utf-8") as f:
            raw_thinks = json.load(f)

        # clean all think blocks on load
        cleaned = 0
        skipped = 0
        for k, v in raw_thinks.items():
            v_clean = clean_think(v)
            if is_truncated(v_clean):
                skipped += 1
                continue
            think_map[k] = v_clean
            if v_clean != v:
                cleaned += 1

        print(f"   Loaded {len(think_map)} think blocks from {thinks}")
        if cleaned:
            print(f"   Cleaned {cleaned} think blocks (Bxx refs removed)")
        if skipped:
            print(f"   ⚠️  Skipped {skipped} truncated think blocks")

    # build jsonl
    jsonl_path = txt_path.rsplit(".", 1)[0] + ".jsonl"
    think_injected = 0
    think_missing = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, (user_content, decision) in enumerate(samples):
            if str(i) in think_map:
                thinking = think_map[str(i)]
                assistant_content = f"<think>\n{thinking}\n</think>\n{decision}"
                think_injected += 1
            else:
                assistant_content = decision
                think_missing += 1

            entry = {
                "type": "chatml",
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ {txt_path} → {jsonl_path} ({len(samples)} samples)")
    print(f"   Think blocks injected : {think_injected}")
    if think_missing:
        print(f"   ⚠️  Missing think blocks: {think_missing} (decision only)")

    _verify(jsonl_path, len(samples))
    return jsonl_path


# ═══════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════

def _verify(jsonl_path, expected_count):
    """Validate every JSONL entry — format, fields, values."""
    errors = []
    count = 0
    think_count = 0
    type_counter = {}
    target_counter = {}
    effort_counter = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            count += 1
            try:
                d = json.loads(line)
                msgs = d.get("messages", [])

                # role checks
                if len(msgs) != 3:
                    errors.append(f"#{i}: expected 3 messages got {len(msgs)}")
                    continue
                for role_idx, expected_role in enumerate(["system", "user", "assistant"]):
                    if msgs[role_idx]["role"] != expected_role:
                        errors.append(f"#{i}: role[{role_idx}] should be {expected_role}")

                # system prompt check
                if msgs[0]["content"] != SYSTEM:
                    errors.append(f"#{i}: system prompt mismatch")

                # user content checks
                user = msgs[1]["content"]
                m_lines = [l for l in user.split("\n") if re.match(r"^M[1-5]: ", l)]
                if len(m_lines) != 5:
                    errors.append(f"#{i}: expected 5 M lines got {len(m_lines)}")
                if not user.endswith("/think"):
                    errors.append(f"#{i}: missing /think in user content")

                # assistant content checks
                assistant = msgs[2]["content"]
                r_lines = [l for l in assistant.split("\n") if l.startswith("R: ")]
                if not r_lines:
                    errors.append(f"#{i}: no R: line in assistant")
                    continue

                # parse R: fields
                r_line = r_lines[0]
                parts = dict(p.split("=", 1) for p in r_line.replace("R: ", "").split(" | "))
                typ    = parts.get("TYPE", "")
                target = parts.get("TARGET", "")
                effort = parts.get("EFFORT", "")
                title  = parts.get("TITLE", "")

                if typ not in VALID_TYPES:
                    errors.append(f"#{i}: invalid TYPE={typ}")
                if target not in VALID_TARGETS:
                    errors.append(f"#{i}: invalid TARGET={target}")
                if effort not in VALID_EFFORTS:
                    errors.append(f"#{i}: invalid EFFORT={effort}")
                if not title:
                    errors.append(f"#{i}: empty TITLE")

                # ignore rules
                if typ == "ignore":
                    if target != "null":
                        errors.append(f"#{i}: ignore must have TARGET=null")
                    if effort != "null":
                        errors.append(f"#{i}: ignore must have EFFORT=null")
                else:
                    if effort == "null":
                        errors.append(f"#{i}: non-ignore must have effort set")
                    if target == "null":
                        errors.append(f"#{i}: non-ignore must have target set")

                # think block presence
                if "<think>" in assistant:
                    think_count += 1

                # counters
                type_counter[typ]    = type_counter.get(typ, 0) + 1
                target_counter[target] = target_counter.get(target, 0) + 1
                effort_counter[effort] = effort_counter.get(effort, 0) + 1

            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"#{i}: parse error — {e}")

    if count != expected_count:
        errors.append(f"Count mismatch: expected {expected_count} got {count}")

    # report
    if errors:
        print(f"   ❌ {len(errors)} errors:")
        for e in errors[:15]:
            print(f"      {e}")
    else:
        print(f"   ✅ All {count} entries valid")

    print(f"   Think blocks present : {think_count}/{count} ({think_count/count*100:.1f}%)")
    print(f"   TYPE  : { {k: type_counter[k] for k in sorted(type_counter)} }")
    print(f"   TARGET: { {k: target_counter[k] for k in sorted(target_counter)} }")
    print(f"   EFFORT: { {k: effort_counter[k] for k in sorted(effort_counter)} }")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python compact_to_jsonl.py <file.txt> [file2.txt ...] [--thinks thinks.json]")
        sys.exit(1)

    thinks = None
    if "--thinks" in args:
        idx = args.index("--thinks")
        if idx + 1 < len(args):
            thinks = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("❌ --thinks requires a path argument")
            sys.exit(1)

    for path in args:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        convert(path, thinks=thinks)


if __name__ == "__main__":
    main()