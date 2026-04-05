"""Transcribe all Kikuyu course lessons using Whisper large-v3-turbo."""
import torch, soundfile as sf, subprocess, os, json
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pathlib import Path

proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo", torch_dtype=torch.float32
).to("cuda")

course_dir = Path(os.path.expanduser("~/wanjiku-tts/data/kikuyu_course"))
out_dir = Path(os.path.expanduser("~/wanjiku-tts/data/kikuyu_course_transcripts"))
out_dir.mkdir(exist_ok=True)

for mp4 in sorted(course_dir.glob("*.mp4")):
    name = mp4.stem
    wav = f"/tmp/{name}.wav"
    subprocess.run(["ffmpeg", "-i", str(mp4), "-ar", "16000", "-ac", "1", "-y", wav], capture_output=True)
    audio, sr = sf.read(wav, dtype="float32")

    print(f"\n=== {name} ({len(audio)/sr/60:.1f} min) ===", flush=True)

    segments = []
    chunk_size = 30 * sr
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        inputs = proc(chunk, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, language="en")
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        start = i // sr
        end = min(start + 30, len(audio) // sr)
        segments.append({"start": start, "end": end, "text": text})
        print(f"  [{start:4d}s] {text[:100]}", flush=True)

    with open(out_dir / f"{name}.json", "w") as f:
        json.dump({"file": name, "duration": len(audio)/sr, "segments": segments}, f, indent=2, ensure_ascii=False)
    os.remove(wav)

print("\nDone!", flush=True)
