"""Run Paza Phi-4 on existing GRN segments to generate pseudo-labels."""
import os, json, time, glob, torch, soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

SEG_DIR = os.path.expanduser("~/wanjiku-tts/data/grn_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/grn_pseudo/train.jsonl")
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

wavs = sorted(glob.glob(os.path.join(SEG_DIR, "*.wav")))
print(f"Segments: {len(wavs)}")

MODEL = "microsoft/paza-Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda", torch_dtype=torch.bfloat16,
    trust_remote_code=True, _attn_implementation="eager").eval()
gen_config = GenerationConfig.from_pretrained(MODEL)
prompt = "<|user|><|audio_1|>Transcribe the audio to Kikuyu.<|end|><|assistant|>"
print("Model loaded")

results = []
t0 = time.time()
for i, wav_path in enumerate(wavs):
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

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(wavs)} done, {len(results)} kept, {time.time()-t0:.0f}s")

with open(MANIFEST, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone: {len(results)}/{len(wavs)} transcribed, saved to {MANIFEST}")
