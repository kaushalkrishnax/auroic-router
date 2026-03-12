import json
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

try:
    from datasketch import MinHash
    _HAS_MINHASH = True
except ImportError:
    _HAS_MINHASH = False
    print("⚠️  datasketch not found, semantic dedup check disabled")

DATASET_PATH = "dataset_v4.jsonl"

# ─── v4 output format (no nulls) ─────────────────────────────────────────────
#   R: TYPE=text   | TARGET=C2 | EFFORT=low/medium/high
#   R: TYPE=react  | TARGET=C1 | TITLE=🔥
#   R: TYPE=media  | TARGET=C3 | TITLE=crying laughing
#   R: TYPE=ignore
# ─────────────────────────────────────────────────────────────────────────────

# ─── Patterns ────────────────────────────────────────────────────────────────

token_pattern = re.compile(r"[a-zA-Z\u0900-\u097F]+")  # latin + devanagari

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001FA00-\U0001FA9F"
    "\U0001FAA0-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

filler_pattern = re.compile(r"^\.\.\.$")  # exactly "..."

def tokenize(text):
    return token_pattern.findall(text.lower())

def parse_decision(decision_line):
    """
    Parse v4 decision line. Returns dict with TYPE + available fields.
    Handles all 4 formats:
      R: TYPE=text   | TARGET=C2 | EFFORT=low
      R: TYPE=react  | TARGET=C1 | TITLE=🔥
      R: TYPE=media  | TARGET=C3 | TITLE=crying laughing
      R: TYPE=ignore
    """
    # Strip think block if present
    if "<think>" in decision_line:
        decision_line = re.sub(r"<think>.*?</think>\n?", "", decision_line, flags=re.DOTALL)
    decision_line = decision_line.strip()

    # Find the R: line
    r_line = next((l for l in decision_line.split("\n") if l.startswith("R: ")), decision_line)

    parts = {}
    for segment in r_line.replace("R: ", "").split(" | "):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()

    return parts

def parse_window(user_content):
    """
    Parse H1-H5 history and C1-C3 candidates from v4 input.
    Returns (history_list, candidates_list)
    """
    history = []
    candidates = []
    for line in user_content.split("\n"):
        line = line.strip()
        if not line or line == "/think":
            continue
        if re.match(r"^H\d+:", line):
            history.append(line.split(":", 1)[1].strip())
        elif re.match(r"^C\d+:", line):
            candidates.append(line.split(":", 1)[1].strip())
    return history, candidates

# ─── Load ─────────────────────────────────────────────────────────────────────

samples = []
with open(DATASET_PATH, "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Loaded samples: {len(samples)}")

# ─── Containers ───────────────────────────────────────────────────────────────

type_counter    = Counter()
target_counter  = Counter()
effort_counter  = Counter()
title_counter   = Counter()

# Window stats
history_msg_lengths  = []
candidate_msg_lengths = []
filler_counts        = []   # how many "..." per sample
emoji_counts_history = []
emoji_counts_cands   = []
vocab                = Counter()

# Target quality
target_tokens        = []
context_tokens       = []
weak_signal          = 0   # target candidate < 3 tokens

# Special bucket tracking (from _meta if present)
bot_count    = 0
filler_count = 0

# Think tier tracking
think_tier_counter = Counter()

# Semantic dedup
duplicates   = 0
minhashes    = []

# Validation errors
errors = []

# ─── Iterate ──────────────────────────────────────────────────────────────────

for i, s in enumerate(tqdm(samples)):
    user    = s["messages"][1]["content"]
    raw_dec = s["messages"][2]["content"]

    # Meta
    meta = s.get("_meta", {})
    if meta.get("is_bot"):   bot_count += 1
    if meta.get("is_filler"): filler_count += 1
    think_tier_counter[meta.get("think_tier", "unknown")] += 1

    # Parse decision
    parts = parse_decision(raw_dec)
    TYPE   = parts.get("TYPE",   "")
    TARGET = parts.get("TARGET", "")
    EFFORT = parts.get("EFFORT", "")
    TITLE  = parts.get("TITLE",  "")

    type_counter[TYPE]     += 1
    target_counter[TARGET] += 1
    if EFFORT:
        effort_counter[EFFORT] += 1
    if TITLE:
        title_counter[TITLE]   += 1

    # Validate format
    if TYPE == "text":
        if not TARGET.startswith("C"):
            errors.append(f"#{i} text bad target: {TARGET}")
        if EFFORT not in ("low","medium","high"):
            errors.append(f"#{i} text bad effort: {EFFORT!r}")
        if "TITLE" in parts:
            errors.append(f"#{i} text has unexpected TITLE field")
    elif TYPE == "react":
        if not TARGET.startswith("C"):
            errors.append(f"#{i} react bad target: {TARGET}")
        if not TITLE:
            errors.append(f"#{i} react missing TITLE")
        if "EFFORT" in parts:
            errors.append(f"#{i} react has unexpected EFFORT field")
    elif TYPE == "media":
        if not TARGET.startswith("C"):
            errors.append(f"#{i} media bad target: {TARGET}")
        if not TITLE:
            errors.append(f"#{i} media missing TITLE")
        if "EFFORT" in parts:
            errors.append(f"#{i} media has unexpected EFFORT field")
        elif TITLE:
            word_count = len(TITLE.split())
            if word_count > 3:
                errors.append(f"#{i} media TITLE > 3 words: {TITLE!r}")
    elif TYPE == "ignore":
        if len(parts) > 1:
            errors.append(f"#{i} ignore has extra fields: {parts}")
    elif TYPE:
        errors.append(f"#{i} invalid TYPE: {TYPE!r}")
    else:
        errors.append(f"#{i} missing TYPE in: {raw_dec[:60]}")

    # Parse window
    history, candidates = parse_window(user)

    # History stats
    for msg in history:
        toks = tokenize(msg)
        vocab.update(toks)
        history_msg_lengths.append(len(toks))
        emoji_counts_history.append(len(emoji_pattern.findall(msg)))

    # Candidate stats
    n_fillers = 0
    for msg in candidates:
        if filler_pattern.match(msg):
            n_fillers += 1
            candidate_msg_lengths.append(0)
            emoji_counts_cands.append(0)
        else:
            toks = tokenize(msg)
            vocab.update(toks)
            candidate_msg_lengths.append(len(toks))
            emoji_counts_cands.append(len(emoji_pattern.findall(msg)))
    filler_counts.append(n_fillers)

    # Target signal quality
    if TARGET.startswith("C"):
        try:
            idx = int(TARGET[1]) - 1
            if 0 <= idx < len(candidates):
                tgt_msg = candidates[idx]
                if not filler_pattern.match(tgt_msg):
                    tgt_toks = tokenize(tgt_msg)
                    target_tokens.extend(tgt_toks)
                    ctx = " ".join(
                        c for j, c in enumerate(candidates + history)
                        if j != idx and not filler_pattern.match(c)
                    )
                    context_tokens.extend(tokenize(ctx))
                    if len(tgt_toks) < 3:
                        weak_signal += 1
        except (ValueError, IndexError):
            pass

    # Semantic dedup via MinHash
    if _HAS_MINHASH:
        all_text = " ".join(
            m for m in candidates if not filler_pattern.match(m)
        )
        tokens_set = set(tokenize(all_text))
        if tokens_set:
            mh = MinHash(num_perm=64)
            for t in tokens_set:
                mh.update(t.encode("utf8"))
            for prev in minhashes[-1000:]:  # check last 1000 for speed
                if mh.jaccard(prev) > 0.85:
                    duplicates += 1
                    break
            minhashes.append(mh)

# ─── Compute stats ────────────────────────────────────────────────────────────

vocab_size   = len(vocab)
total_tokens = sum(vocab.values())
probs        = np.array(list(vocab.values())) / total_tokens
entropy      = -(probs * np.log2(probs + 1e-12)).sum()

total = len(samples)

# ─── Report ───────────────────────────────────────────────────────────────────

def bar(count, total, width=30):
    filled = int(width * count / max(total, 1))
    return "█" * filled + "░" * (width - filled)

print("\n" + "="*55)
print("   AUROIC ROUTER v4 — DATASET ANALYSIS REPORT")
print("="*55)

# ── TYPE ──────────────────────────────────────────────
print("\n▸ TYPE distribution")
for t in ["text","react","media","ignore"]:
    c = type_counter.get(t, 0)
    pct = c / total * 100
    print(f"  {t:8s} {c:6d}  ({pct:5.1f}%)  {bar(c, total, 25)}")
unknown_types = {k: v for k,v in type_counter.items() if k not in ("text","react","media","ignore")}
if unknown_types:
    print(f"  ⚠️  Unknown types: {unknown_types}")

# ── TARGET ────────────────────────────────────────────
print("\n▸ TARGET distribution (non-ignore)")
valid_targets = {k: v for k,v in target_counter.items() if k.startswith("C")}
total_targeted = sum(valid_targets.values())
for t in ["C1","C2","C3"]:
    c = valid_targets.get(t, 0)
    pct = c / max(total_targeted, 1) * 100
    print(f"  {t}  {c:6d}  ({pct:5.1f}%)  {bar(c, total_targeted, 25)}")

# ── EFFORT ────────────────────────────────────────────
print("\n▸ EFFORT distribution (text only)")
effort_total = sum(effort_counter.values())
for e in ["low","medium","high"]:
    c = effort_counter.get(e, 0)
    pct = c / max(effort_total, 1) * 100
    print(f"  {e:8s} {c:6d}  ({pct:5.1f}%)")

# ── FILLER SLOTS ──────────────────────────────────────
print("\n▸ Filler slot distribution (candidates with '...')")
fc = Counter(filler_counts)
for n in sorted(fc.keys()):
    label = f"{n} filler{'s' if n!=1 else ''}"
    print(f"  {label:12s}: {fc[n]:6d} samples ({fc[n]/total*100:.1f}%)")

# ── SPECIAL BUCKETS ───────────────────────────────────
print("\n▸ Special buckets")
print(f"  @BOT windows:    {bot_count:6d} ({bot_count/total*100:.1f}%)")
print(f"  Filler windows:  {filler_count:6d} ({filler_count/total*100:.1f}%)")

# ── THINK TIERS ───────────────────────────────────────
print("\n▸ Think tier distribution")
for tier in ["hard","medium","easy","unknown"]:
    c = think_tier_counter.get(tier, 0)
    if c:
        print(f"  {tier:8s} {c:6d} ({c/total*100:.1f}%)")

# ── MEDIA TITLES ──────────────────────────────────────
print("\n▸ Media TITLE word count distribution")
media_titles = [t for t in title_counter.elements() if not any(ord(c) > 127 for c in t)]
if media_titles:
    wc = Counter(len(t.split()) for t in media_titles)
    for w in sorted(wc.keys()):
        print(f"  {w} word{'s' if w!=1 else '':1s}: {wc[w]:5d}")

# ── REACT TITLES ──────────────────────────────────────
print("\n▸ React TITLE (emoji) — top 15")
emoji_titles = [(t, c) for t, c in title_counter.most_common() if any(ord(ch) > 127 for ch in t)]
for t, c in emoji_titles[:15]:
    print(f"  {t}  {c:5d}")

# ── VOCAB ─────────────────────────────────────────────
print(f"\n▸ Vocabulary")
print(f"  Unique tokens:    {vocab_size:7d}")
print(f"  Total tokens:     {total_tokens:7d}")
print(f"  Vocabulary entropy: {entropy:.2f} bits")
print(f"  Top 20 tokens:")
for tok, cnt in vocab.most_common(20):
    print(f"    {tok:20s} {cnt:6d}")

# ── MESSAGE LENGTH ────────────────────────────────────
print(f"\n▸ Message length (tokens)")
print(f"  History msgs — mean: {np.mean(history_msg_lengths):.1f}  std: {np.std(history_msg_lengths):.1f}  max: {max(history_msg_lengths)}")
real_cand_lengths = [l for l in candidate_msg_lengths if l > 0]
if real_cand_lengths:
    print(f"  Candidates    — mean: {np.mean(real_cand_lengths):.1f}  std: {np.std(real_cand_lengths):.1f}  max: {max(real_cand_lengths)}")

# ── EMOJI ─────────────────────────────────────────────
print(f"\n▸ Emoji density")
print(f"  History msgs avg:   {np.mean(emoji_counts_history):.2f} emoji/msg")
real_cand_emoji = [e for i, e in enumerate(emoji_counts_cands) if candidate_msg_lengths[i] > 0]
if real_cand_emoji:
    print(f"  Candidate msgs avg: {np.mean(real_cand_emoji):.2f} emoji/msg")

# ── TARGET QUALITY ────────────────────────────────────
print(f"\n▸ Target signal quality")
print(f"  Weak signal targets (< 3 tokens): {weak_signal}")
print(f"  Weak signal rate:                 {weak_signal/max(total_targeted,1)*100:.1f}%")
if target_tokens:
    tgt_vocab  = Counter(target_tokens)
    ctx_vocab  = Counter(context_tokens)
    overlap    = sum((min(tgt_vocab[t], ctx_vocab[t]) for t in tgt_vocab), 0)
    total_tgt  = sum(tgt_vocab.values())
    print(f"  Target vs context token overlap:  {overlap/max(total_tgt,1)*100:.1f}%")

# ── DEDUP ─────────────────────────────────────────────
if _HAS_MINHASH:
    print(f"\n▸ Semantic similarity")
    print(f"  Near-duplicates (Jaccard > 0.85): {duplicates}")
    print(f"  Duplicate rate:                   {duplicates/total*100:.1f}%")

# ── VALIDATION ────────────────────────────────────────
print(f"\n▸ Format validation")
if errors:
    print(f"  ❌ {len(errors)} errors found:")
    for e in errors[:20]:
        print(f"    {e}")
    if len(errors) > 20:
        print(f"    ... and {len(errors)-20} more")
else:
    print(f"  ✅ All {total} samples pass format validation")

print("\n" + "="*55 + "\n")