"""Segment radio audio with speaker diarization using speechbrain (CPU)."""
import os, json, glob, subprocess, numpy as np, soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

RADIO_DIRS = {
    "kameme": os.path.expanduser("~/wanjiku-tts/data/radio_raw"),
    "inooro": os.path.expanduser("~/wanjiku-tts/data/radio_raw_inooro"),
    "gukena": os.path.expanduser("~/wanjiku-tts/data/radio_raw_gukena"),
}
OUT_DIR = os.path.expanduser("~/wanjiku-tts/data/radio_segments")
MANIFEST = os.path.expanduser("~/wanjiku-tts/data/manifests/radio_segments/segments.jsonl")
MIN_DUR, MAX_DUR = 3.0, 25.0
SILENCE_THRESH, SILENCE_MIN = -30, 0.8
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

# Load speaker encoder on CPU
print("Loading speaker encoder...")
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
print("Loaded")

import torch

def get_segments_from_silence(wav_path):
    """Silence-based segmentation, merge to TARGET range."""
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", wav_path]).strip())
    result = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af", f"silencedetect=noise={SILENCE_THRESH}dB:d={SILENCE_MIN}",
         "-f", "null", "-"], capture_output=True, text=True)
    splits = []
    s_start = None
    for line in result.stderr.split("\n"):
        if "silence_start" in line:
            try: s_start = float(line.split("silence_start: ")[1].split()[0])
            except: pass
        elif "silence_end" in line and s_start is not None:
            try:
                s_end = float(line.split("silence_end: ")[1].split()[0])
                splits.append((s_start + s_end) / 2)
                s_start = None
            except: pass
    boundaries = [0.0] + splits + [dur]
    segments = []
    seg_start = 0.0
    for b in boundaries[1:]:
        if b - seg_start >= 5.0:
            while b - seg_start > MAX_DUR:
                segments.append((seg_start, seg_start + MAX_DUR))
                seg_start += MAX_DUR
            if b - seg_start >= MIN_DUR:
                segments.append((seg_start, b))
            seg_start = b
    return segments

def get_speaker_embedding(wav_path):
    """Get speaker embedding from a WAV file."""
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
    signal = torch.tensor(audio).unsqueeze(0).float()
    with torch.no_grad():
        emb = spk_model.encode_batch(signal)
    return emb.squeeze().numpy()

def process_file(wav_path, station):
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"  Segmenting {basename}...", flush=True)
    segments = get_segments_from_silence(wav_path)
    print(f"    {len(segments)} segments found")

    # Extract WAVs and get embeddings
    seg_paths, embeddings = [], []
    for j, (start, end) in enumerate(segments):
        out_path = os.path.join(OUT_DIR, f"{basename}_{j:04d}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path, "-ss", str(start), "-to", str(end),
            "-ar", "16000", "-ac", "1", "-f", "wav", out_path
        ], capture_output=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            try:
                emb = get_speaker_embedding(out_path)
                seg_paths.append((out_path, round(end - start, 2)))
                embeddings.append(emb)
            except:
                pass
        if (j + 1) % 50 == 0:
            print(f"    {j+1}/{len(segments)} extracted", flush=True)

    # Cluster speakers
    if len(embeddings) < 2:
        return []
    X = np.array(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.7, metric="cosine", linkage="average")
    labels = clustering.fit_predict(X)
    n_speakers = len(set(labels))
    print(f"    {n_speakers} speakers detected")

    # Rename files with speaker labels and build results
    results = []
    for (path, dur), label in zip(seg_paths, labels):
        speaker = f"speaker_{label}"
        new_path = path.replace(".wav", f"_{speaker}.wav")
        os.rename(path, new_path)
        results.append({"path": new_path, "speaker": speaker, "duration": dur, "station": station})

    # Speaker stats
    speakers = {}
    for r in results:
        speakers[r["speaker"]] = speakers.get(r["speaker"], 0) + r["duration"]
    for spk, dur in sorted(speakers.items(), key=lambda x: -x[1]):
        print(f"    {spk}: {dur:.0f}s ({dur/60:.1f}min)")
    return results

# Process one file per station
all_segments = []
for station, dir_path in RADIO_DIRS.items():
    wavs = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
    for wav in wavs:
        if os.path.getsize(wav) > 100_000_000:
            print(f"\n=== {station}: {os.path.basename(wav)} ===")
            segs = process_file(wav, station)
            all_segments.extend(segs)
            break

with open(MANIFEST, "w") as f:
    for s in all_segments:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"\nTotal: {len(all_segments)} segments saved to {MANIFEST}")
