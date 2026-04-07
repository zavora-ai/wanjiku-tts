"""Test all 3 Paza models on same samples for comparison."""
import json, random, os, time, torch, soundfile as sf, numpy as np

MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/combined/test.jsonl")

with open(MANIFEST) as f:
    lines = [json.loads(l) for l in f]

# Pick 2 samples per source
by_src = {}
for l in lines:
    by_src.setdefault(l.get("source","?"), []).append(l)
samples = []
random.seed(42)
for src in ["digigreen", "waxal", "bible"]:
    if src in by_src:
        samples.extend(random.sample(by_src[src], min(2, len(by_src[src]))))

print(f"Testing {len(samples)} samples\n")
for i, s in enumerate(samples):
    print(f"[{i}] [{s.get('source')}] REF: {s['text'][:80]}...")

# --- Model 1: Paza Whisper ---
print("\n" + "="*60)
print("MODEL: paza-whisper-large-v3-turbo")
print("="*60)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
t0 = time.time()
wmodel = AutoModelForSpeechSeq2Seq.from_pretrained(
    "microsoft/paza-whisper-large-v3-turbo", torch_dtype=torch.bfloat16, device_map="cuda")
wproc = AutoProcessor.from_pretrained("microsoft/paza-whisper-large-v3-turbo")
pipe = pipeline("automatic-speech-recognition", model=wmodel, tokenizer=wproc.tokenizer,
    feature_extractor=wproc.feature_extractor, torch_dtype=torch.bfloat16, device_map="cuda",
    generate_kwargs={"task": "transcribe"})
print(f"Loaded in {time.time()-t0:.0f}s")

for i, s in enumerate(samples):
    audio, sr = sf.read(s["audio_path"])
    t1 = time.time()
    result = pipe({"raw": audio, "sampling_rate": sr})
    print(f"\n[{i}] [{s.get('source')}]")
    print(f"  REF: {s['text']}")
    print(f"  HYP: {result['text']}")
    print(f"  Time: {time.time()-t1:.1f}s")

del wmodel, wproc, pipe
torch.cuda.empty_cache()

# --- Model 2: Paza MMS ---
print("\n" + "="*60)
print("MODEL: paza-mms-1b-all")
print("="*60)
from transformers import Wav2Vec2ForCTC, AutoProcessor as AP2
t0 = time.time()
mproc = AP2.from_pretrained("microsoft/paza-mms-1b-all")
mmodel = Wav2Vec2ForCTC.from_pretrained("microsoft/paza-mms-1b-all").to("cuda")
mmodel.eval()
# Load Kikuyu adapter
try:
    mproc.tokenizer.set_target_lang("kik")
    mmodel.load_adapter("kik")
    print("Loaded kik adapter")
except Exception as e:
    print(f"Adapter load: {e}")
print(f"Loaded in {time.time()-t0:.0f}s")

for i, s in enumerate(samples):
    audio, sr = sf.read(s["audio_path"])
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr))
    t1 = time.time()
    inputs = mproc(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = mmodel(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    hyp = mproc.batch_decode(ids)[0]
    print(f"\n[{i}] [{s.get('source')}]")
    print(f"  REF: {s['text']}")
    print(f"  HYP: {hyp}")
    print(f"  Time: {time.time()-t1:.1f}s")

print("\nDone.")
