"""Segment radio using Silero VAD + label with DeKUT Whisper."""
import os, json, glob, time, subprocess, torch, soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

RADIO_DIRS = [
    ("kameme", os.path.expanduser("~/wanjiku-tts/data/radio_raw")),
    ("inooro", os.path.expanduser("~/wanjiku-tts/data/radio_raw_inooro")),
    ("gukena", os.path.expanduser("~/wanjiku-tts/data/radio_raw_gukena")),
]
SEG_DIR = os.path.expanduser("~/wanjiku-tts/data/radio_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/radio_pseudo/train.jsonl")
os.makedirs(SEG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
MIN_DUR, MAX_DUR = 5.0, 25.0

# Load VAD
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
get_speech_ts = utils[0]
print("VAD loaded")

def vad_segment(wav_path, station):
    """Use Silero VAD to find speech, merge/split to 5-25s, extract clips."""
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    # Convert to 16kHz mono
    tmp = f"/tmp/{basename}_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp], capture_output=True)
    audio, sr = sf.read(tmp)
    wav = torch.tensor(audio).float()
    timestamps = get_speech_ts(wav, vad_model, sampling_rate=16000,
        min_speech_duration_ms=2000, min_silence_duration_ms=500)

    # Merge short + split long into 5-25s segments
    segments = []
    for t in timestamps:
        s, e = t["start"] / 16000, t["end"] / 16000
        while e - s > MAX_DUR:
            segments.append((s, s + MAX_DUR))
            s += MAX_DUR
        if e - s >= MIN_DUR:
            segments.append((s, e))

    # Extract clips from original (better quality than tmp)
    out_paths = []
    for j, (s, e) in enumerate(segments):
        out = os.path.join(SEG_DIR, f"{basename}_{j:04d}.wav")
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-ss", str(s), "-to", str(e),
            "-ar", "16000", "-ac", "1", "-f", "wav", out], capture_output=True)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            out_paths.append(out)
    os.remove(tmp)
    return out_paths

# Step 1: Segment — 1 file per station
print("=== Step 1: VAD Segmentation ===")
all_wavs = []
for station, dir_path in RADIO_DIRS:
    files = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
    for f in files:
        if os.path.getsize(f) > 100_000_000:
            print(f"  {station}: {os.path.basename(f)}...", end=" ", flush=True)
            segs = vad_segment(f, station)
            print(f"{len(segs)} segments")
            all_wavs.extend(segs)
            break
print(f"Total: {len(all_wavs)} segments")

# Step 2: DeKUT labeling
print("\n=== Step 2: DeKUT labeling ===")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "DeKUT-DSAIL/tunuh-whisper-base-kikuyu-v1", use_safetensors=True).to("cuda").eval()
processor = AutoProcessor.from_pretrained("openai/whisper-base")
print("Model loaded")

results = []
t0 = time.time()
for i, wav_path in enumerate(all_wavs):
    audio, sr = sf.read(wav_path)
    dur = len(audio) / sr
    try:
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(inputs.input_features.to("cuda"), max_new_tokens=256)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except:
        hyp = ""
    # Filter: skip repetitive/garbage output
    words = hyp.split()
    if hyp and len(words) > 2 and len(set(words)) > len(words) * 0.3:
        results.append({"audio_path": wav_path, "text": hyp, "source": "radio_pseudo", "duration": round(dur, 2)})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_wavs)} done, {len(results)} kept, {time.time()-t0:.0f}s")

with open(MANIFEST, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone: {len(results)}/{len(all_wavs)} kept in {time.time()-t0:.0f}s")
print(f"Manifest: {MANIFEST}")
for r in results[:5]:
    print(f"  ({r['duration']}s) {r['text'][:80]}")
