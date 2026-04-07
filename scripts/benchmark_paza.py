"""Benchmark Paza Phi-4 on DigiGreen test set — compute WER/CER."""
import json, os, time, torch, soundfile as sf, re
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/digigreen/test.jsonl")
MODEL = "microsoft/paza-Phi-4-multimodal-instruct"
OUT = os.path.expanduser("~/wanjiku-tts/models/paza_digigreen_benchmark.jsonl")

# Diacritic normalization: macron→tilde (Paza outputs macrons)
MACRON_TO_TILDE = str.maketrans("īūĪŪ", "ĩũĨŨ")
def normalize(t):
    return re.sub(r'\s+', ' ', t.translate(MACRON_TO_TILDE).strip().lower())

with open(MANIFEST) as f:
    samples = [json.loads(l) for l in f]
print(f"Samples: {len(samples)}")

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda", torch_dtype=torch.bfloat16,
    trust_remote_code=True, _attn_implementation="eager").eval()
gen_config = GenerationConfig.from_pretrained(MODEL)
prompt = "<|user|><|audio_1|>Transcribe the audio to Kikuyu.<|end|><|assistant|>"
print("Model loaded")

# WER/CER computation
def wer_cer(ref, hyp):
    ref_w, hyp_w = ref.split(), hyp.split()
    ref_c, hyp_c = list(ref), list(hyp)
    def edit_dist(a, b):
        d = list(range(len(b)+1))
        for i in range(1, len(a)+1):
            nd = [i] + [0]*len(b)
            for j in range(1, len(b)+1):
                nd[j] = min(nd[j-1]+1, d[j]+1, d[j-1]+(0 if a[i-1]==b[j-1] else 1))
            d = nd
        return d[-1]
    w = edit_dist(ref_w, hyp_w) / max(len(ref_w), 1)
    c = edit_dist(ref_c, hyp_c) / max(len(ref_c), 1)
    return w, c

total_wer, total_cer, total_w, total_c = 0, 0, 0, 0
results = []
t0 = time.time()

for i, s in enumerate(samples):
    audio, sr = sf.read(s["audio_path"])
    try:
        inputs = processor(text=prompt, audios=[(audio, sr)], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, generation_config=gen_config)
        out = out[:, inputs["input_ids"].shape[1]:]
        hyp = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    except:
        hyp = ""

    ref_n = normalize(s["text"])
    hyp_n = normalize(hyp)
    w, c = wer_cer(ref_n, hyp_n)
    ref_words = len(ref_n.split())
    ref_chars = len(ref_n)
    total_wer += w * ref_words
    total_cer += c * ref_chars
    total_w += ref_words
    total_c += ref_chars
    results.append({"ref": s["text"], "hyp": hyp, "wer": round(w,4), "cer": round(c,4)})

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        avg_wer = total_wer / max(total_w, 1)
        avg_cer = total_cer / max(total_c, 1)
        print(f"  {i+1}/{len(samples)} | WER: {avg_wer:.3f} | CER: {avg_cer:.3f} | {elapsed:.0f}s")

# Final
avg_wer = total_wer / max(total_w, 1)
avg_cer = total_cer / max(total_c, 1)
print(f"\n=== FINAL (DigiGreen test, {len(samples)} samples) ===")
print(f"WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
print(f"CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
print(f"Time: {time.time()-t0:.0f}s")

with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Save summary
import datetime
summary = {
    "model": MODEL,
    "dataset": "DigiGreen Kikuyu ASR (test split)",
    "samples": len(samples),
    "wer": round(avg_wer, 4),
    "cer": round(avg_cer, 4),
    "wer_pct": f"{avg_wer*100:.2f}%",
    "cer_pct": f"{avg_cer*100:.2f}%",
    "total_time_sec": round(time.time()-t0, 1),
    "avg_time_per_sample": round((time.time()-t0)/len(samples), 2),
    "date": datetime.datetime.now().isoformat(),
    "notes": "Macron→tilde normalization applied. Eager attention (no flash-attn). BF16.",
    "pazabench_comparable": True,
}
summary_path = OUT.replace(".jsonl", "_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Results saved: {OUT}")
print(f"Summary saved: {summary_path}")
