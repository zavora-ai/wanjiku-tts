"""Prepare DigiGreen Kikuyu ASR training manifest.

Matches WAV files to CSV transcripts, normalizes text, validates audio,
filters bad pairs, and splits into train/dev/test.
"""
import os, json, csv, random
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(__file__))
from normalize_text import normalize

AUDIO_DIR = os.path.expanduser("~/wanjiku-tts/data/digigreen/audio/KikuyuASR/dg_16")
CSV_PATH = os.path.expanduser("~/wanjiku-tts/data/digigreen/digital_green_recordings.csv")
OUT_DIR = os.path.expanduser("~/wanjiku-tts/data/manifests/digigreen")
MAX_DURATION = 30.0  # Gemma 4 E2B audio limit
MIN_DURATION = 0.5
MIN_TEXT_LEN = 3

def get_duration(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip())
    except:
        return 0.0

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load CSV ($ separated: path | transcript)
    print("Loading CSV...")
    csv_map = {}
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f, delimiter="$")
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            orig_path, transcript = row[0], row[1]
            fname = os.path.basename(orig_path)
            csv_map[fname] = transcript
    print(f"  CSV entries: {len(csv_map)}")

    # Match WAVs to transcripts
    print("Matching WAVs to transcripts...")
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"  WAV files: {len(wav_files)}")

    matched = []
    unmatched_wav = 0
    for fname in wav_files:
        if fname in csv_map:
            matched.append((fname, csv_map[fname]))
        else:
            unmatched_wav += 1
    print(f"  Matched: {len(matched)}, Unmatched WAVs: {unmatched_wav}")

    # Validate and normalize
    print("Validating and normalizing...")
    valid = []
    skipped = {"short": 0, "long": 0, "empty_text": 0, "bad_audio": 0}
    for i, (fname, transcript) in enumerate(matched):
        if i % 5000 == 0:
            print(f"  Processing {i}/{len(matched)}...")

        # Normalize text
        text = normalize(transcript.strip(), expand_numbers=True)
        if len(text) < MIN_TEXT_LEN:
            skipped["empty_text"] += 1
            continue

        # Check audio duration
        audio_path = os.path.join(AUDIO_DIR, fname)
        dur = get_duration(audio_path)
        if dur < MIN_DURATION:
            skipped["short"] += 1
            continue
        if dur > MAX_DURATION:
            skipped["long"] += 1
            continue
        if dur == 0.0:
            skipped["bad_audio"] += 1
            continue

        valid.append({
            "audio_path": audio_path,
            "text": text,
            "duration": round(dur, 2),
            "source": "digigreen",
        })

    print(f"  Valid: {len(valid)}, Skipped: {skipped}")
    total_hours = sum(v["duration"] for v in valid) / 3600
    print(f"  Total duration: {total_hours:.1f}h")

    # Split: 90/5/5
    random.seed(42)
    random.shuffle(valid)
    n = len(valid)
    n_dev = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    splits = {
        "test": valid[:n_test],
        "dev": valid[n_test:n_test + n_dev],
        "train": valid[n_test + n_dev:],
    }

    for split_name, items in splits.items():
        out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        hours = sum(it["duration"] for it in items) / 3600
        print(f"  {split_name}: {len(items)} samples, {hours:.1f}h → {out_path}")

    print("Done!")

if __name__ == "__main__":
    main()
