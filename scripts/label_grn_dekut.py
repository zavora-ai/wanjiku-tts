"""Pseudo-label GRN segments with DeKUT Whisper (CPU, fast)."""
import os, json, glob, time, torch, soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SEG_DIR = os.path.expanduser("~/wanjiku-tts/data/grn_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/grn_pseudo/train.jsonl")
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

wavs = sorted(glob.glob(os.path.join(SEG_DIR, "*.wav")))
print(f"Segments: {len(wavs)}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "DeKUT-DSAIL/tunuh-whisper-base-kikuyu-v1", use_safetensors=True).to("cuda").eval()
processor = AutoProcessor.from_pretrained("openai/whisper-base")
print("Model loaded")

results = []
t0 = time.time()
for i, wav_path in enumerate(wavs):
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
        results.append({"audio_path": wav_path, "text": hyp, "source": "grn_pseudo", "duration": round(dur, 2)})

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(wavs)} done, {len(results)} kept, {time.time()-t0:.0f}s")

with open(MANIFEST, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone: {len(results)}/{len(wavs)} transcribed in {time.time()-t0:.0f}s")
print(f"Manifest: {MANIFEST}")
for r in results[:3]:
    print(f"  {os.path.basename(r['audio_path'])}: {r['text'][:80]}")
