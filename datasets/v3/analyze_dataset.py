import json
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasketch import MinHash
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer

DATASET_PATH = "dataset_v3.jsonl"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

decision_pattern = re.compile(
    r"TYPE=(.*?) \| TARGET=(.*?) \| EFFORT=(.*?) \| TITLE=(.*)"
)

token_pattern = re.compile(r"[a-zA-Z]+")

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

def tokenize(text):
    return token_pattern.findall(text.lower())

def parse_messages(user_content):
    lines = user_content.split("\n")
    msgs = []
    for line in lines:
        msgs.append(line.split(":",1)[1].strip())
    return msgs

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

samples = []

with open(DATASET_PATH, "r", encoding="utf8") as f:
    for line in f:
        samples.append(json.loads(line))

print("Loaded samples:", len(samples))

# --------------------------------------------------
# Containers
# --------------------------------------------------

type_counter = Counter()
target_counter = Counter()
effort_counter = Counter()

titles = []
vocab = Counter()
msg_lengths = []
emoji_counts = []

target_tokens = []
context_tokens = []

duplicates = 0
minhashes = []

title_leaks = 0
weak_signal = 0

# --------------------------------------------------
# Iterate samples
# --------------------------------------------------

for s in tqdm(samples):

    user = s["messages"][1]["content"]
    decision = s["messages"][2]["content"]

    m = decision_pattern.search(decision)

    TYPE = m.group(1)
    TARGET = m.group(2)
    EFFORT = m.group(3)
    TITLE = m.group(4)

    type_counter[TYPE]+=1
    target_counter[TARGET]+=1
    effort_counter[EFFORT]+=1
    titles.append(TITLE)

    msgs = parse_messages(user)

    # token stats
    for msg in msgs:
        toks = tokenize(msg)
        vocab.update(toks)
        msg_lengths.append(len(toks))
        emoji_counts.append(len(emoji_pattern.findall(msg)))

    # target analysis
    if TARGET != "null":
        idx = int(TARGET[1])-1
        tgt = msgs[idx]
        ctx = " ".join([m for i,m in enumerate(msgs) if i!=idx])

        target_tokens.extend(tokenize(tgt))
        context_tokens.extend(tokenize(ctx))

        # weak signal detection
        if len(tokenize(tgt)) < 3:
            weak_signal+=1

        # title leakage
        if TITLE != "null":
            if TITLE.lower() in ctx.lower():
                title_leaks+=1

    # semantic duplicate detection
    text = " ".join(msgs)
    tokens = set(tokenize(text))

    mh = MinHash(num_perm=64)
    for t in tokens:
        mh.update(t.encode("utf8"))

    for prev in minhashes:
        if mh.jaccard(prev) > 0.85:
            duplicates+=1
            break

    minhashes.append(mh)

# --------------------------------------------------
# Vocabulary stats
# --------------------------------------------------

vocab_size = len(vocab)
total_tokens = sum(vocab.values())

# entropy
probs = np.array(list(vocab.values()))/total_tokens
entropy = -(probs*np.log2(probs)).sum()

# --------------------------------------------------
# Title analysis
# --------------------------------------------------

title_counter = Counter(titles)
unique_titles = len(title_counter)

# --------------------------------------------------
# Print report
# --------------------------------------------------

print("\n================ DATASET REPORT ================\n")

print("TYPE distribution")
for k,v in type_counter.items():
    print(k,":",v)

print("\nTARGET distribution")
for k,v in target_counter.items():
    print(k,":",v)

print("\nEFFORT distribution")
for k,v in effort_counter.items():
    print(k,":",v)

print("\nVocabulary size:",vocab_size)
print("Total tokens:",total_tokens)
print("Vocabulary entropy:",round(entropy,2))

print("\nAverage message length:",np.mean(msg_lengths))
print("Std message length:",np.std(msg_lengths))

print("\nEmoji per message avg:",np.mean(emoji_counts))

print("\nUnique titles:",unique_titles)
print("Top titles:")
for t,c in title_counter.most_common(10):
    print(t,c)

print("\nWeak target signals:",weak_signal)
print("Title leakage cases:",title_leaks)

print("\nSemantic duplicates:",duplicates)

print("\nTarget token count:",len(target_tokens))
print("Context token count:",len(context_tokens))

print("\n================ END REPORT ================\n")