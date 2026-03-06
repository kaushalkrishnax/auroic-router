import json
import os
import random

KEYWORD_FILE = "keyword_dataset_final.txt"
HEARTBEAT_FILE = "heartbeat_dataset_final.txt"
OUTPUT_FILE = "combined_dataset.jsonl"


def parse_block(block):
    lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
    
    msg_lines = []
    out_line = None

    for line in lines:
        if line.startswith("MSG"):
            msg_lines.append(line)
        elif line.startswith("OUT:"):
            out_line = line[4:].strip()

    if not msg_lines or not out_line:
        return None

    return {
        "user": "\n".join(msg_lines),
        "assistant": out_line
    }


def load_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b for b in raw.split("\n\n") if b.strip()]
    parsed = []

    for block in blocks:
        item = parse_block(block)
        if item:
            parsed.append(item)

    return parsed


def main():
    print("Loading datasets...")

    keyword_data = load_file(KEYWORD_FILE)
    heartbeat_data = load_file(HEARTBEAT_FILE)

    print(f"Keyword samples: {len(keyword_data)}")
    print(f"Heartbeat samples: {len(heartbeat_data)}")

    combined = keyword_data + heartbeat_data

    print(f"Total before shuffle: {len(combined)}")

    random.shuffle(combined)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in combined:
            entry = {
                "messages": [
                    {"role": "user", "content": sample["user"]},
                    {"role": "assistant", "content": sample["assistant"]}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total entries: {len(combined)}")


if __name__ == "__main__":
    main()