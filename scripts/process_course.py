"""Process all 21 Kikuyu course videos: VAD segment + DeKUT label."""
import os, json, glob, time, subprocess, torch, soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SRC_DIR = os.path.expanduser("~/wanjiku-tts/data/kikuyu_course")
SEG_DIR = os.path.expanduser("~/wanjiku-tts/data/course_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/course_pseudo/train.jsonl")
os.makedirs(SEG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
MIN_DUR, MAX_DUR = 5.0, 25.0

vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
get_speech_ts = utils[0]
print("VAD loaded")

# Step 1: Segment all videos
print("=== Step 1: VAD Segmentation ===")
all_wavs = []
for mp4 in sorted(glob.glob(os.path.join(SRC_DIR, "*.mp4"))):
    basename = os.path.splitext(os.path.basename(mp4))[0].replace(" ", "_")
    tmp = f"/tmp/{basename}_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ar", "16000", "-ac", "1", "-f", "wav", tmp], capture_output=True)
    audio, sr = sf.read(tmp)
    wav = torch.tensor(audio).float()
    timestamps = get_speech_ts(wav, vad_model, sampling_rate=16000,
        min_speech_duration_ms=2000, min_silence_duration_ms=500)
    segments = []
    for t in timestamps:
        s, e = t["start"] / 16000, t["end"] / 16000
        while e - s > MAX_DUR:
            segments.append((s, s + MAX_DUR))
            s += MAX_DUR
        if e - s >= MIN_DUR:
            segments.append((s, e))
    for j, (s, e) in enumerate(segments):
        out = os.path.join(SEG_DIR, f"{basename}_{j:04d}.wav")
        subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ss", str(s), "-to", str(e),
            "-ar", "16000", "-ac", "1", "-f", "wav", out], capture_output=True)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            all_wavs.append(out)
    os.remove(tmp)
    print(f"  {basename}: {len(segments)} segments")
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
    if hyp and len(hyp) > 3:
        results.append({"audio_path": wav_path, "text": hyp, "source": "course_bilingual", "duration": round(dur, 2)})
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_wavs)} done, {len(results)} kept, {time.time()-t0:.0f}s")

with open(MANIFEST, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

total_hrs = sum(r["duration"] for r in results) / 3600
print(f"\nDone: {len(results)}/{len(all_wavs)} kept ({total_hrs:.1f}h)")
print(f"Manifest: {MANIFEST}")
