"""Clean pseudo-labeled data and build final combined manifest."""
import json, os

BASE = os.path.expanduser("~/wanjiku-tts/data/manifests")

def is_repetitive(text):
    words = text.split()
    return len(words) > 5 and len(set(words)) < len(words) * 0.3

def load_and_clean(path, remove_english=False, remove_repetitive=True):
    eng = set("the is are was were and or but for with this that have has from they you will can not what when how which".split())
    kept, removed = [], 0
    with open(path) as f:
        for l in f:
            item = json.loads(l)
            txt = item.get("text", "")
            if remove_repetitive and is_repetitive(txt):
                removed += 1; continue
            if remove_english:
                w = txt.lower().split()
                if len(w) > 5 and sum(1 for x in w if x in eng) > len(w) * 0.3:
                    removed += 1; continue
            kept.append(item)
    return kept, removed

# Clean pseudo-labeled datasets
print("=== Cleaning ===")
for name, remove_eng in [("grn_pseudo", True), ("radio_pseudo", True), ("course_pseudo", False)]:
    path = f"{BASE}/{name}/train.jsonl"
    kept, removed = load_and_clean(path, remove_english=remove_eng)
    with open(path, "w") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {name}: removed {removed}, kept {len(kept)}")

# Mukuyu — only remove repetitive
path = f"{BASE}/mukuyu_pseudo/train.jsonl"
kept, removed = load_and_clean(path, remove_english=False)
with open(path, "w") as f:
    for item in kept:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"  mukuyu_pseudo: removed {removed}, kept {len(kept)}")

# Build final combined manifest
print("\n=== Building combined manifest ===")
audio_sources = [
    ("digigreen", f"{BASE}/digigreen/train.jsonl"),
    ("bible", f"{BASE}/bible/train.jsonl"),
    ("waxal", f"{BASE}/waxal/train.jsonl"),
    ("grn_pseudo", f"{BASE}/grn_pseudo/train.jsonl"),
    ("radio_pseudo", f"{BASE}/radio_pseudo/train.jsonl"),
    ("course_pseudo", f"{BASE}/course_pseudo/train.jsonl"),
    ("mukuyu_pseudo", f"{BASE}/mukuyu_pseudo/train.jsonl"),
]

out_dir = f"{BASE}/combined_v2"
os.makedirs(out_dir, exist_ok=True)

all_items = []
for name, path in audio_sources:
    with open(path) as f:
        items = [json.loads(l) for l in f]
    all_items.extend(items)
    print(f"  {name}: {len(items)}")

with open(f"{out_dir}/train.jsonl", "w") as f:
    for item in all_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Copy dev/test from original combined (DigiGreen + Bible + WAXAL)
for split in ["dev", "test"]:
    src = f"{BASE}/combined/{split}.jsonl"
    if os.path.exists(src):
        os.system(f"cp {src} {out_dir}/{split}.jsonl")
        with open(src) as f:
            n = sum(1 for _ in f)
        print(f"  {split}: {n}")

print(f"\n  COMBINED V2 TRAIN: {len(all_items)} audio samples")

# Count text tasks
text_total = 0
for name in ["denoising", "translation", "kik_swahili", "classification"]:
    with open(f"{BASE}/text_tasks/{name}.jsonl") as f:
        n = sum(1 for _ in f)
    text_total += n
print(f"  TEXT TASKS: {text_total} samples")
print(f"  GRAND TOTAL: {len(all_items) + text_total} training samples")
