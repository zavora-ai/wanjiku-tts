"""Segment GRN audio into 5-25s clips, then pseudo-label with Paza Phi-4."""
import os, json, time, subprocess, glob, torch, soundfile as sf, numpy as np

GRN_DIR = os.path.expanduser("~/wanjiku-tts/data/grn_kikuyu")
OUT_DIR = os.path.expanduser("~/wanjiku-tts/data/grn_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/grn_pseudo/train.jsonl")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

TARGET_MIN, TARGET_MAX = 5.0, 25.0
SILENCE_THRESH = -30  # dB (stricter = only real silences)
SILENCE_MIN = 0.8  # seconds (longer = fewer splits)

def segment_file(mp3_path):
    basename = os.path.splitext(os.path.basename(mp3_path))[0]
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", mp3_path]).strip())

    # Detect silences
    result = subprocess.run(
        ["ffmpeg", "-i", mp3_path, "-af", f"silencedetect=noise={SILENCE_THRESH}dB:d={SILENCE_MIN}",
         "-f", "null", "-"], capture_output=True, text=True)

    # Parse silence midpoints as split candidates
    splits = []
    s_start = None
    for line in result.stderr.split("\n"):
        if "silence_start" in line:
            try: s_start = float(line.split("silence_start: ")[1].split()[0])
            except: pass
        elif "silence_end" in line and s_start is not None:
            try:
                s_end = float(line.split("silence_end: ")[1].split()[0])
                splits.append((s_start + s_end) / 2)  # split at midpoint of silence
                s_start = None
            except: pass

    # Build segments by merging short regions to hit TARGET_MIN
    boundaries = [0.0] + splits + [dur]
    segments = []
    seg_start = 0.0
    for b in boundaries[1:]:
        seg_dur = b - seg_start
        if seg_dur >= TARGET_MIN:
            # If too long, hard-split at TARGET_MAX
            while b - seg_start > TARGET_MAX:
                segments.append((seg_start, seg_start + TARGET_MAX))
                seg_start += TARGET_MAX
            if b - seg_start >= TARGET_MIN:
                segments.append((seg_start, b))
            elif b - seg_start >= 2.0 and segments:
                # Merge short tail into previous
                prev = segments.pop()
                segments.append((prev[0], b))
            seg_start = b

    # Extract WAVs
    out_paths = []
    for j, (start, end) in enumerate(segments):
        out_path = os.path.join(OUT_DIR, f"{basename}_{j:04d}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path, "-ss", str(start), "-to", str(end),
            "-ar", "16000", "-ac", "1", "-f", "wav", out_path
        ], capture_output=True)
        if os.path.exists(out_path) and sf.info(out_path).duration >= 2.0:
            out_paths.append(out_path)
    return out_paths

# Step 1: Segment
print("=== Step 1: Segmenting GRN audio ===")
mp3s = sorted(glob.glob(os.path.join(GRN_DIR, "*.mp3")))
all_wavs = []
for mp3 in mp3s:
    print(f"  {os.path.basename(mp3)}...", end=" ", flush=True)
    wavs = segment_file(mp3)
    print(f"{len(wavs)} segments")
    all_wavs.extend(wavs)

durs = [sf.info(w).duration for w in all_wavs]
print(f"\nTotal: {len(all_wavs)} segments, {sum(durs)/3600:.1f}h")
print(f"Duration range: {min(durs):.1f}s - {max(durs):.1f}s, median: {sorted(durs)[len(durs)//2]:.1f}s")

# Step 2: Paza Phi-4 inference
print("\n=== Step 2: Paza Phi-4 inference ===")
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL = "microsoft/paza-Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda", torch_dtype=torch.bfloat16,
    trust_remote_code=True, _attn_implementation="eager").eval()
gen_config = GenerationConfig.from_pretrained(MODEL)
prompt = "<|user|><|audio_1|>Transcribe the audio to Kikuyu.<|end|><|assistant|>"
print("Model loaded")

results = []
for i, wav_path in enumerate(all_wavs):
    audio, sr = sf.read(wav_path)
    dur = len(audio) / sr
    try:
        inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, generation_config=gen_config)
        out = out[:, inputs["input_ids"].shape[1]:]
        hyp = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    except Exception as e:
        hyp = ""
        print(f"  [{i}] ERROR: {e}")

    if hyp and len(hyp) > 3:
        results.append({"audio_path": wav_path, "text": hyp, "source": "grn_pseudo", "duration": round(dur, 2)})

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(all_wavs)} done, {len(results)} kept")

with open(MANIFEST, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n=== Done ===")
print(f"Segments: {len(all_wavs)}, Transcribed: {len(results)}")
print(f"Manifest: {MANIFEST}")
for r in results[:5]:
    print(f"  [{os.path.basename(r['audio_path'])}] ({r['duration']}s) {r['text'][:100]}")
