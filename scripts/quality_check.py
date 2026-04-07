"""Quality check all training data."""
import json, os, random
random.seed(42)

manifests = [
    ("digigreen", "data/manifests/digigreen/train.jsonl"),
    ("bible", "data/manifests/bible/train.jsonl"),
    ("waxal", "data/manifests/waxal/train.jsonl"),
    ("grn_pseudo", "data/manifests/grn_pseudo/train.jsonl"),
    ("radio_pseudo", "data/manifests/radio_pseudo/train.jsonl"),
    ("course_pseudo", "data/manifests/course_pseudo/train.jsonl"),
    ("mukuyu_pseudo", "data/manifests/mukuyu_pseudo/train.jsonl"),
]

eng = set("the is are was were and or but for with this that have has from they you will can not what when how which".split())
total_s = total_h = 0

print("AUDIO DATA QUALITY CHECK")
print("=" * 60)
for name, path in manifests:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        items = [json.loads(l) for l in f]
    durs = [i.get("duration", 0) for i in items if "duration" in i]
    hrs = sum(durs) / 3600 if durs else 0
    empty = rep = eng_h = 0
    for i in items:
        txt = i.get("text", "")
        if len(txt.strip()) < 3:
            empty += 1
        w = txt.split()
        if len(w) > 5 and len(set(w)) < len(w) * 0.3:
            rep += 1
        wl = txt.lower().split()
        if len(wl) > 5 and sum(1 for x in wl if x in eng) > len(wl) * 0.3:
            eng_h += 1
    avg_w = sum(len(i.get("text", "").split()) for i in items) / max(len(items), 1)
    missing = 0
    if items and "audio_path" in items[0]:
        for i in random.sample(items, min(50, len(items))):
            if not os.path.exists(i["audio_path"]):
                missing += 1
    total_s += len(items)
    total_h += hrs
    print(f"  {name}: {len(items)} samples, {hrs:.1f}h")
    print(f"    avg_words={avg_w:.0f} empty={empty} rep={rep} eng_heavy={eng_h} missing={missing}/50")
    s = random.choice(items)
    print(f"    >> {s.get('text', '')[:100]}")

print(f"\n  TOTAL AUDIO: {total_s} samples, {total_h:.1f}h")

print("\nTEXT TASK QUALITY CHECK")
print("=" * 60)
total_t = 0
for name in ["denoising", "translation", "kik_swahili", "classification"]:
    path = f"data/manifests/text_tasks/{name}.jsonl"
    with open(path) as f:
        items = [json.loads(l) for l in f]
    total_t += len(items)
    valid = sum(1 for i in items if "messages" in i and len(i["messages"]) == 2)
    print(f"  {name}: {len(items)} ({valid} valid)")

print(f"\n  TOTAL TEXT: {total_t}")
print(f"\nGRAND TOTAL: {total_s + total_t} samples ({total_h:.1f}h audio)")
