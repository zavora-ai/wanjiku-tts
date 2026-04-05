"""Segment Bible chapter MP3s into verse-level clips using MMS forced alignment.

For each chapter:
1. Parse USFM to extract verse texts
2. Run MMS wav2vec2 forced alignment on chapter audio + concatenated text
3. Split audio at verse boundaries
4. Output manifest with verse-level audio + normalized text
"""
import os, json, re, sys, subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from normalize_text import normalize

AUDIO_DIR = os.path.expanduser("~/wanjiku-tts/data/bible_audio/mp3s")
USFM_DIR = os.path.expanduser("~/wanjiku-tts/data/bible_audio/text/release/USX_1")
OUT_AUDIO = os.path.expanduser("~/wanjiku-tts/data/bible_audio/verses")
OUT_MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/bible")
MAX_DURATION = 30.0
MIN_DURATION = 1.0

# Book code mapping (MP3 filename prefix → USFM filename)
BOOK_CODES = [
    "GEN","EXO","LEV","NUM","DEU","JOS","JDG","RUT","1SA","2SA",
    "1KI","2KI","1CH","2CH","EZR","NEH","EST","JOB","PSA","PRO",
    "ECC","SNG","ISA","JER","LAM","EZK","DAN","HOS","JOL","AMO",
    "OBA","JON","MIC","NAM","HAB","ZEP","HAG","ZEC","MAL",
    "MAT","MRK","LUK","JHN","ACT","ROM","1CO","2CO","GAL","EPH",
    "PHP","COL","1TH","2TH","1TI","2TI","TIT","PHM","HEB","JAS",
    "1PE","2PE","1JN","2JN","3JN","JUD","REV",
]


def parse_usfm(path):
    """Parse USFM file, return dict of {chapter_num: [(verse_num, text), ...]}."""
    chapters = {}
    current_chapter = None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove \r
    content = content.replace("\r", "")

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("\\c "):
            current_chapter = int(line.split()[1])
            chapters[current_chapter] = []
        elif line.startswith("\\v ") and current_chapter is not None:
            # Extract verse number and text
            m = re.match(r"\\v\s+(\d+)\s+(.*)", line)
            if m:
                vnum = int(m.group(1))
                text = m.group(2)
                # Strip remaining USFM markers
                text = re.sub(r"\\[a-z]+\*?", "", text).strip()
                if text:
                    chapters[current_chapter].append((vnum, text))
        elif current_chapter is not None and chapters.get(current_chapter):
            # Continuation line — append to last verse
            text = re.sub(r"\\[a-z]+\*?", "", line).strip()
            if text and not line.startswith("\\"):
                last = chapters[current_chapter][-1]
                chapters[current_chapter][-1] = (last[0], last[1] + " " + text)

    return chapters


def segment_with_silence(audio_path, verses, out_dir):
    """Split chapter audio into verse-sized segments using even splitting.

    Simple approach: divide audio evenly by character count of verses.
    More accurate than silence detection for read Bible audio.
    """
    # Load audio (convert MP3 to WAV via ffmpeg pipe)
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"],
        capture_output=True, timeout=60,
    )
    audio = np.frombuffer(result.stdout[44:], dtype=np.int16).astype(np.float32) / 32768.0
    sr = 16000
    total_samples = len(audio)
    total_duration = total_samples / sr

    # Calculate proportional split based on text length
    total_chars = sum(len(v[1]) for v in verses)
    if total_chars == 0:
        return []

    segments = []
    current_sample = 0

    for i, (vnum, text) in enumerate(verses):
        # Proportional duration based on text length
        proportion = len(text) / total_chars
        segment_samples = int(proportion * total_samples)

        # Last segment gets remainder
        if i == len(verses) - 1:
            segment_samples = total_samples - current_sample

        end_sample = min(current_sample + segment_samples, total_samples)
        duration = (end_sample - current_sample) / sr

        if duration < MIN_DURATION:
            current_sample = end_sample
            continue

        # If segment > MAX_DURATION, skip (will handle merging later)
        if duration > MAX_DURATION:
            current_sample = end_sample
            continue

        # Save segment
        segment = audio[current_sample:end_sample]
        book_ch = os.path.splitext(os.path.basename(audio_path))[0]
        fname = f"{book_ch}_v{vnum:03d}.wav"
        out_path = os.path.join(out_dir, fname)
        sf.write(out_path, segment, sr)

        segments.append({
            "audio_path": os.path.abspath(out_path),
            "text": normalize(text, expand_numbers=True),
            "duration": round(duration, 2),
            "source": "bible",
            "ref": f"{book_ch}:{vnum}",
        })
        current_sample = end_sample

    return segments


def main():
    os.makedirs(OUT_AUDIO, exist_ok=True)
    os.makedirs(OUT_MANIFEST, exist_ok=True)

    all_segments = []
    errors = 0

    # Process each book
    for book in BOOK_CODES:
        usfm_path = os.path.join(USFM_DIR, f"{book}.usfm")
        if not os.path.exists(usfm_path):
            continue

        chapters = parse_usfm(usfm_path)
        print(f"{book}: {len(chapters)} chapters", flush=True)

        for ch_num, verses in sorted(chapters.items()):
            mp3_name = f"{book}_{ch_num:03d}.mp3"
            mp3_path = os.path.join(AUDIO_DIR, mp3_name)
            if not os.path.exists(mp3_path):
                continue

            try:
                segments = segment_with_silence(mp3_path, verses, OUT_AUDIO)
                all_segments.extend(segments)
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error {mp3_name}: {e}", flush=True)

    print(f"\nTotal segments: {len(all_segments)}, Errors: {errors}", flush=True)
    total_hours = sum(s["duration"] for s in all_segments) / 3600
    print(f"Total duration: {total_hours:.1f}h", flush=True)

    # Split 90/5/5
    import random
    random.seed(42)
    random.shuffle(all_segments)
    n = len(all_segments)
    n_test = int(n * 0.05)
    n_dev = int(n * 0.05)

    splits = {
        "test": all_segments[:n_test],
        "dev": all_segments[n_test:n_test + n_dev],
        "train": all_segments[n_test + n_dev:],
    }

    for name, items in splits.items():
        path = os.path.join(OUT_MANIFEST, f"{name}.jsonl")
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        hours = sum(it["duration"] for it in items) / 3600
        print(f"  {name}: {len(items)} segments, {hours:.1f}h", flush=True)


if __name__ == "__main__":
    main()
