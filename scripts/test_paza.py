"""Quick test of Microsoft Paza Phi-4 ASR on our Kikuyu audio samples (GPU)."""
import torch, json, random, os, time, soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL = "microsoft/paza-Phi-4-multimodal-instruct"
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/combined/test.jsonl")
N_SAMPLES = 10

with open(MANIFEST) as f:
    lines = [json.loads(l) for l in f]

# Pick samples from each source
by_src = {}
for l in lines:
    by_src.setdefault(l.get("source","?"), []).append(l)
samples = []
random.seed(42)
for src, items in by_src.items():
    samples.extend(random.sample(items, min(3, len(items))))
samples = samples[:N_SAMPLES]

print(f"Loading Paza model on GPU (bfloat16)...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda", torch_dtype=torch.bfloat16,
    trust_remote_code=True, _attn_implementation="eager",
).eval()
gen_config = GenerationConfig.from_pretrained(MODEL)
print(f"Model loaded in {time.time()-t0:.0f}s")

prompt_tpl = "<|user|><|audio_1|>Transcribe the audio to Kikuyu.<|end|><|assistant|>"

for i, s in enumerate(samples):
    ref = s["text"]
    source = s.get("source", "?")
    print(f"\n--- Sample {i+1} [{source}] ---")
    print(f"REF: {ref}")

    audio_data, sr = sf.read(s["audio_path"])
    t1 = time.time()
    inputs = processor(text=prompt_tpl, audios=[(audio_data, sr)], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, generation_config=gen_config)
    out = out[:, inputs["input_ids"].shape[1]:]
    hyp = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"HYP: {hyp}")
    print(f"Time: {time.time()-t1:.1f}s")

print("\nDone.")
