"""Clean, segment, and normalize radio broadcast audio recordings."""
import argparse
import json
from pathlib import Path

import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import soundfile as sf
import yaml


def load_audio(path, target_sr):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        import torch
        waveform = torch.tensor(data).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        data = waveform.squeeze().numpy()
    return data, target_sr


def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)


def normalize_loudness(audio, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def segment_by_silence(audio, sr, min_dur=3.0, max_dur=15.0, silence_thresh=0.01, min_silence=0.3):
    """Split audio on silence regions."""
    segments = []
    min_samples = int(min_dur * sr)
    max_samples = int(max_dur * sr)
    min_sil_samples = int(min_silence * sr)

    is_silent = np.abs(audio) < silence_thresh
    # Find silence boundaries
    boundaries = [0]
    silent_run = 0
    for i, s in enumerate(is_silent):
        if s:
            silent_run += 1
        else:
            if silent_run >= min_sil_samples:
                boundaries.append(i)
            silent_run = 0
    boundaries.append(len(audio))

    # Merge into segments within duration limits
    start = boundaries[0]
    for b in boundaries[1:]:
        length = b - start
        if length >= max_samples:
            segments.append((start, start + max_samples))
            start = start + max_samples
        elif length >= min_samples and b == boundaries[-1]:
            segments.append((start, b))
        elif b == boundaries[-1] and length >= min_samples:
            segments.append((start, b))
        elif length >= min_samples:
            # Check if next boundary would exceed max
            segments.append((start, b))
            start = b

    return [(s, e) for s, e in segments if (e - s) >= min_samples]


def main():
    parser = argparse.ArgumentParser(description="Clean and segment audio")
    parser.add_argument("--input", required=True, help="Input directory with raw audio")
    parser.add_argument("--output", required=True, help="Output directory for clean segments")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    acfg = cfg["audio"]
    sr = acfg["target_sample_rate"]
    manifest = []
    idx = 0

    for audio_file in sorted(in_dir.glob("*.wav")):
        print(f"Processing {audio_file.name}...")
        audio, sr = load_audio(audio_file, sr)
        audio = reduce_noise(audio, sr)
        audio = normalize_loudness(audio, sr, acfg["target_lufs"])

        segments = segment_by_silence(audio, sr, acfg["min_duration"], acfg["max_duration"])
        for start, end in segments:
            seg = audio[start:end]
            name = f"KAM_{idx:05d}.wav"
            sf.write(out_dir / name, seg, sr)
            manifest.append({"audio": name, "duration": round(len(seg) / sr, 2), "text": ""})
            idx += 1

    manifest_path = Path(cfg["data"]["transcripts_dir"]) / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Produced {idx} segments → {out_dir}")
    print(f"Manifest → {manifest_path} (transcriptions need to be filled in)")


if __name__ == "__main__":
    main()
