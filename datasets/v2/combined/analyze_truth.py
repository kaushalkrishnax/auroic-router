import sys
import re
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
import math

# -----------------------------
# Config
# -----------------------------

HINDI_WORDS = re.compile(
    r"\b(hai|hain|kya|nahi|bhai|yaar|kuch|aur|mera|tera|tum|hum|karo|bata|kal|aaj|abhi|phir|fir|woh|wo|toh|to|kyun|kaisa|kaise|kab|kahan|mat|bas|chal|sab|log|dost|pyaar|ghar|din|raat|baat|dil|agar|lekin|par|pe|se|ko|ka|ki|ke|ne|ye|yeh|zyada|thoda|bahut|accha|acha|theek|sahi)\b",
    re.I,
)

TOKENIZER = re.compile(r"\w+")

SIMILARITY_THRESHOLD = 0.85


# -----------------------------
# Parsing
# -----------------------------

def parse_dataset(text):
    blocks = []
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue

        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        msg_lines = [l for l in lines if l.startswith("MSG")]
        out_line = next((l for l in lines if l.startswith("OUT:")), None)

        if not out_line:
            continue

        parts = out_line[4:].strip().split("|")
        if len(parts) != 4:
            continue

        action, target, content, conf = [p.strip() for p in parts]

        try:
            conf = float(conf)
        except:
            continue

        messages = []
        for l in msg_lines:
            _, msg = l.split(":", 1)
            messages.append(msg.strip())

        blocks.append({
            "messages": messages,
            "action": action,
            "target": target,
            "content": content,
            "confidence": conf,
            "raw": raw
        })

    return blocks


# -----------------------------
# Helpers
# -----------------------------

def detect_language(text):
    if HINDI_WORDS.search(text):
        return "hinglish"
    return "english"

def tokenize(text):
    return TOKENIZER.findall(text.lower())

def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0
    ent = 0
    for v in counter.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent

def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1
    return inter / union


# -----------------------------
# Analyzer
# -----------------------------

def analyze(blocks):
    total = len(blocks)
    print(f"\n════════════════════════════════")
    print(f" TOTAL SAMPLES: {total}")
    print(f"════════════════════════════════\n")

    action_counts = Counter()
    reasoning_counts = Counter()
    target_counts = Counter()
    lang_counts = Counter()
    confidence_values = []
    token_counter = Counter()
    bigram_counter = Counter()
    msg_position_bias = Counter()
    repeated_hash = set()
    duplicate_count = 0

    all_message_hashes = []
    short_msgs = 0

    for b in blocks:
        action_counts[b["action"]] += 1
        confidence_values.append(b["confidence"])

        if b["action"] == "communicate":
            reasoning_counts[b["content"]] += 1

        target_counts[b["target"]] += 1

        joined = " ".join(b["messages"])
        lang_counts[detect_language(joined)] += 1

        tokens = tokenize(joined)
        token_counter.update(tokens)

        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            bigram_counter[bigram] += 1

        for i, msg in enumerate(b["messages"]):
            msg_position_bias[i + 1] += 1

            if len(msg.strip()) <= 3:
                short_msgs += 1

        h = hashlib.sha1(joined.encode()).hexdigest()
        if h in repeated_hash:
            duplicate_count += 1
        repeated_hash.add(h)

        all_message_hashes.append(set(tokens))

    # -----------------------------
    # Print Basic Stats
    # -----------------------------

    print("ACTION DISTRIBUTION:")
    for k, v in action_counts.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print("\nREASONING DISTRIBUTION:")
    for k, v in reasoning_counts.items():
        print(f"  {k}: {v} ({v/sum(reasoning_counts.values())*100:.1f}%)")

    print("\nTARGET BIAS:")
    for k, v in target_counts.items():
        print(f"  {k}: {v}")

    print("\nLANGUAGE MIX:")
    for k, v in lang_counts.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print("\nCONFIDENCE:")
    print(f"  mean: {sum(confidence_values)/len(confidence_values):.3f}")
    print(f"  min: {min(confidence_values):.2f}")
    print(f"  max: {max(confidence_values):.2f}")

    # -----------------------------
    # Diversity Metrics
    # -----------------------------

    vocab_size = len(token_counter)
    total_tokens = sum(token_counter.values())

    print("\nLEXICAL DIVERSITY:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Type-token ratio: {vocab_size/total_tokens:.3f}")
    print(f"  Token entropy: {entropy(token_counter):.3f}")
    print(f"  Bigram entropy: {entropy(bigram_counter):.3f}")

    # -----------------------------
    # Repetition
    # -----------------------------

    print("\nREPETITION CHECK:")
    print(f"  Exact duplicates: {duplicate_count}")

    high_similarity = 0
    for a, b in combinations(all_message_hashes[:500], 2):  # limit for speed
        if jaccard(a, b) > SIMILARITY_THRESHOLD:
            high_similarity += 1

    print(f"  High semantic similarity pairs (sampled): {high_similarity}")

    # -----------------------------
    # Quality Flags
    # -----------------------------

    print("\nQUALITY FLAGS:")

    if short_msgs > total * 0.2:
        print("  ⚠ Too many ultra-short messages")

    if action_counts["ignore"] > total * 0.4:
        print("  ⚠ Ignore class too dominant")

    if reasoning_counts["medium"] > sum(reasoning_counts.values()) * 0.6:
        print("  ⚠ Medium reasoning heavily dominant")

    if entropy(token_counter) < 6:
        print("  ⚠ Low lexical entropy — repetitive vocabulary")

    if duplicate_count > 0:
        print("  ⚠ Exact duplicate conversations exist")

    print("\n════════════════════════════════")
    print(" END OF REPORT")
    print("════════════════════════════════\n")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_truth.py <dataset.txt>")
        sys.exit(1)

    file = sys.argv[1]
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = parse_dataset(text)
    analyze(blocks)
