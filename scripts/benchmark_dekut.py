"""Benchmark DeKUT Whisper on DigiGreen test set — compare with Paza."""
import json, os, time, re, torch, soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/digigreen/test.jsonl")
OUT = os.path.expanduser("~/wanjiku-tts/models/dekut_digigreen_benchmark.jsonl")

MACRON_TO_TILDE = str.maketrans("īūĪŪ", "ĩũĨŨ")
def normalize(t):
    return re.sub(r'\s+', ' ', t.translate(MACRON_TO_TILDE).strip().lower())

def wer_cer(ref, hyp):
    def ed(a, b):
        d = list(range(len(b)+1))
        for i in range(1, len(a)+1):
            nd = [i] + [0]*len(b)
            for j in range(1, len(b)+1):
                nd[j] = min(nd[j-1]+1, d[j]+1, d[j-1]+(0 if a[i-1]==b[j-1] else 1))
            d = nd
        return d[-1]
    return ed(ref.split(), hyp.split()) / max(len(ref.split()), 1), ed(list(ref), list(hyp)) / max(len(ref), 1)

with open(MANIFEST) as f:
    samples = [json.loads(l) for l in f]
print(f"Samples: {len(samples)}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "DeKUT-DSAIL/tunuh-whisper-base-kikuyu-v1", use_safetensors=True).to("cuda").eval()
processor = AutoProcessor.from_pretrained("openai/whisper-base")
print("Model loaded")

total_wer = total_cer = total_w = total_c = 0
results = []
t0 = time.time()

for i, s in enumerate(samples):
    audio, sr = sf.read(s["audio_path"])
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(inputs.input_features.to("cuda"), max_new_tokens=256)
    hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    ref_n, hyp_n = normalize(s["text"]), normalize(hyp)
    w, c = wer_cer(ref_n, hyp_n)
    rw, rc = len(ref_n.split()), len(ref_n)
    total_wer += w * rw; total_cer += c * rc
    total_w += rw; total_c += rc
    results.append({"ref": s["text"], "hyp": hyp, "wer": round(w, 4), "cer": round(c, 4)})

    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(samples)} | WER: {total_wer/total_w:.3f} | CER: {total_cer/total_c:.3f} | {time.time()-t0:.0f}s")

wer_final = total_wer / max(total_w, 1)
cer_final = total_cer / max(total_c, 1)
print(f"\n=== DeKUT on DigiGreen test ({len(samples)} samples) ===")
print(f"WER: {wer_final:.4f} ({wer_final*100:.2f}%)")
print(f"CER: {cer_final:.4f} ({cer_final*100:.2f}%)")
print(f"Time: {time.time()-t0:.0f}s ({(time.time()-t0)/len(samples):.2f}s/sample)")
print(f"\nPaza baseline: WER 15.28%, CER 7.44%")

with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved: {OUT}")
