"""Prepare all text data into task-specific training formats for Gemma 4."""
import json, os, random, re, sys
sys.path.insert(0, os.path.expanduser("~/wanjiku-tts/scripts"))
from normalize_text import normalize as normalize_kikuyu

random.seed(42)
BASE = os.path.expanduser("~/wanjiku-tts/data")
OUT = os.path.expanduser("~/wanjiku-tts/data/manifests/text_tasks")
os.makedirs(OUT, exist_ok=True)

def msg(user_text, assistant_text):
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
    ]}

all_tasks = []

# === 1. TRANSLATION: Kikuyu ↔ English ===
print("=== Translation pairs ===")
trans = []

# FLORES parallel
for suffix in ["kik_eng", "eng_kik"]:
    path = f"{BASE}/text_datasets/flores_{suffix}_parallel.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                kik = normalize_kikuyu(d.get("kikuyu", d.get("kik", "")))
                eng = d.get("english", d.get("eng", ""))
                if kik and eng:
                    trans.append(msg(f"Translate this Kikuyu to English: {kik}", eng))
                    trans.append(msg(f"Translate this English to Kikuyu: {eng}", kik))

# CGIAR parallel
path = f"{BASE}/text_datasets/cgiar_kikuyu_english.jsonl"
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            kik = normalize_kikuyu(d.get("kikuyu", ""))
            eng = d.get("english", "")
            if kik and eng:
                trans.append(msg(f"Translate this Kikuyu to English: {kik}", eng))
                trans.append(msg(f"Translate this English to Kikuyu: {eng}", kik))

random.shuffle(trans)
print(f"  {len(trans)} translation pairs")

# === 2. DENOISING: Teach Kikuyu spelling/grammar ===
print("=== Denoising samples ===")
denoise = []

def add_noise(text):
    chars = list(text)
    n = max(1, len(chars) // 10)
    for _ in range(n):
        op = random.choice(["sub", "dup", "del"])
        pos = random.randint(0, len(chars) - 1)
        if op == "sub" and chars[pos].isalpha():
            chars[pos] = random.choice("abcdeghikmnortuw")
        elif op == "dup":
            chars.insert(pos, chars[pos])
        elif op == "del" and len(chars) > 5:
            chars.pop(pos)
    return "".join(chars)

# Collect all clean Kikuyu text
all_text = []

# Wikipedia
wiki_path = f"{BASE}/text_datasets/wiki_kikuyu/wiki_final.txt"
if os.path.exists(wiki_path):
    with open(wiki_path) as f:
        all_text.extend([normalize_kikuyu(l.strip()) for l in f if len(l.strip()) > 30])

# Bible
bible_dir = f"{BASE}/bible_audio/text/release/USX_1"
if os.path.exists(bible_dir):
    import glob
    for usfm in glob.glob(f"{bible_dir}/*.usfm"):
        with open(usfm) as f:
            for line in f:
                line = re.sub(r"\\[a-z]+\s*\d*\s*", "", line).strip()
                if len(line) > 20:
                    all_text.append(normalize_kikuyu(line))

# FLORES
for name in ["flores200_kikuyu.jsonl", "sib200_kikuyu.jsonl"]:
    path = f"{BASE}/text_datasets/{name}"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                t = d.get("text", d.get("sentence", ""))
                if t:
                    all_text.append(normalize_kikuyu(t))

# Leipzig
for name in ["leipzig_text/kik_community_2017-sentences.txt"]:
    path = f"{BASE}/{name}"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    all_text.append(normalize_kikuyu(parts[1]))

# Bloom + Storybook
for name in ["bloom_kikuyu.jsonl", "africanstorybook_kikuyu.jsonl"]:
    path = f"{BASE}/text_datasets/{name}"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                t = d.get("text", "")
                if t:
                    all_text.append(normalize_kikuyu(t))

# Deduplicate
all_text = list(set(t for t in all_text if len(t) > 20))
print(f"  Unique Kikuyu sentences: {len(all_text)}")

# Generate denoising pairs (cap at 15K)
for t in random.sample(all_text, min(15000, len(all_text))):
    noisy = add_noise(t)
    denoise.append(msg(f"Correct this noisy Kikuyu text: {noisy}", t))

print(f"  {len(denoise)} denoising pairs")

# === 3. KIKUYU-SWAHILI ===
print("=== Kikuyu-Swahili pairs ===")
kik_sw = []
path = f"{BASE}/text_datasets/kik_sw_parallel.jsonl"
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            kik = normalize_kikuyu(d.get("kikuyu", d.get("kik", "")))
            sw = d.get("swahili", d.get("sw", ""))
            if kik and sw:
                kik_sw.append(msg(f"Translate this Kikuyu to Swahili: {kik}", sw))

print(f"  {len(kik_sw)} Kikuyu-Swahili pairs")

# === 4. CLASSIFICATION (SIB-200) ===
print("=== Classification samples ===")
classify = []
path = f"{BASE}/text_datasets/sib200_kikuyu.jsonl"
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = normalize_kikuyu(d.get("text", d.get("sentence", "")))
            cat = d.get("category", d.get("label", ""))
            if t and cat:
                classify.append(msg(f"Classify this Kikuyu text into a topic: {t}", cat))

print(f"  {len(classify)} classification samples")

# === Save all ===
tasks = {
    "translation": trans,
    "denoising": denoise,
    "kik_swahili": kik_sw,
    "classification": classify,
}

total = 0
for name, data in tasks.items():
    path = f"{OUT}/{name}.jsonl"
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    total += len(data)
    print(f"  Saved {name}: {len(data)} samples")

print(f"\nTotal text-task samples: {total}")
print(f"Output: {OUT}/")
