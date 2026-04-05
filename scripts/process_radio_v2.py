"""
Radio audio processing pipeline v2.
Uses SpeechBrain (diarization) + Meta MMS (Kikuyu ASR) + noisereduce.

Usage:
    python scripts/process_radio_v2.py --input data/radio_raw --output data/radio_processed
    python scripts/process_radio_v2.py --input data/radio_raw --output data/radio_processed --stage transcribe
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


STAGES = ["diarize", "filter", "clean", "transcribe", "normalize"]


def run_diarize(input_dir, work_dir, window_sec=3, hop_sec=1, threshold=0.7):
    """Stage 1: Speaker diarization using SpeechBrain ECAPA-TDNN."""
    from speechbrain.inference import SpeakerRecognition
    from sklearn.cluster import AgglomerativeClustering

    encoder = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    results = {}

    for wav in sorted(input_dir.glob("*.wav")):
        print(f"  [diarize] {wav.name}")
        audio, sr = torchaudio.load(wav)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000

        window = window_sec * sr
        step = hop_sec * sr
        embeddings, times = [], []

        for start in range(0, audio.shape[1] - window, step):
            chunk = audio[:, start:start + window]
            emb = encoder.encode_batch(chunk).squeeze().cpu().detach().numpy()
            embeddings.append(emb)
            times.append(start / sr)

        if len(embeddings) < 2:
            continue

        embeddings = np.array(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

        # Convert frame labels to speaker segments
        segments = []
        prev_label, seg_start = labels[0], times[0]
        for t, label in zip(times[1:], labels[1:]):
            if label != prev_label:
                segments.append({"start": round(seg_start, 2), "end": round(t, 2), "speaker": int(prev_label)})
                seg_start = t
                prev_label = label
        segments.append({"start": round(seg_start, 2), "end": round(times[-1] + window_sec, 2), "speaker": int(prev_label)})

        n_spk = len(set(labels))
        print(f"    {len(segments)} segments, {n_spk} speakers")
        results[wav.name] = segments

    out = work_dir / "diarized.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")


def run_filter(input_dir, work_dir, min_dur=3.0, max_dur=30.0):
    """Stage 2: Keep single-speaker segments of suitable duration."""
    with open(work_dir / "diarized.json") as f:
        diarized = json.load(f)

    filtered = {}
    total_in, total_out = 0, 0

    for fname, segments in diarized.items():
        kept = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            total_in += dur
            if dur < min_dur:
                continue
            # Split long segments
            if dur > max_dur:
                t = seg["start"]
                while t + min_dur <= seg["end"]:
                    end = min(t + max_dur, seg["end"])
                    kept.append({"start": round(t, 2), "end": round(end, 2), "speaker": seg["speaker"]})
                    total_out += end - t
                    t = end
            else:
                kept.append(seg)
                total_out += dur
        filtered[fname] = kept
        print(f"  [filter] {fname}: {len(segments)} -> {len(kept)} segments")

    print(f"  Total: {total_in/3600:.1f}h -> {total_out/3600:.1f}h")
    with open(work_dir / "filtered.json", "w") as f:
        json.dump(filtered, f, indent=2)


def run_clean(input_dir, work_dir, clips_dir):
    """Stage 3: Extract and clean segments with noisereduce."""
    import noisereduce as nr

    with open(work_dir / "filtered.json") as f:
        filtered = json.load(f)

    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    idx = 0

    for fname, segments in filtered.items():
        wav_path = input_dir / fname
        audio, sr = sf.read(wav_path, dtype="float32")

        for seg in segments:
            start = int(seg["start"] * sr)
            end = int(seg["end"] * sr)
            clip = audio[start:end]

            # Noise reduction
            clip = nr.reduce_noise(y=clip, sr=sr, prop_decrease=0.6)

            # SNR check
            power = np.mean(clip ** 2)
            frames = [clip[i:i+int(0.025*sr)] for i in range(0, len(clip) - int(0.025*sr), int(0.025*sr))]
            noise = np.mean(sorted([np.mean(f**2) for f in frames])[:max(1, len(frames)//10)]) + 1e-10
            snr = 10 * np.log10(power / noise)

            if snr < 15:
                continue

            name = f"RADIO_{idx:06d}.wav"
            sf.write(clips_dir / name, clip, sr)
            manifest.append({
                "audio": name, "speaker": f"speaker_{seg['speaker']}",
                "duration": round(float(len(clip) / sr), 2), "snr": round(float(snr), 1),
                "source": fname, "text": "",
            })
            idx += 1

    with open(work_dir / "clips_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [clean] {idx} clips extracted to {clips_dir}")


def run_transcribe(clips_dir, work_dir, output_dir):
    """Stage 4: Transcribe with Whisper-base-kikuyu."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained("MaryWambo/whisper-base-kikuyu4")
    model = WhisperForConditionalGeneration.from_pretrained("MaryWambo/whisper-base-kikuyu4")
    if torch.cuda.is_available():
        model = model.to("cuda")

    with open(work_dir / "clips_manifest.json") as f:
        manifest = json.load(f)

    for entry in manifest:
        audio, sr = sf.read(clips_dir / entry["audio"], dtype="float32")
        if sr != 16000:
            audio = torchaudio.functional.resample(
                torch.tensor(audio).unsqueeze(0), sr, 16000
            ).squeeze().numpy()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256)
        entry["text"] = processor.batch_decode(ids, skip_special_tokens=True)[0]

    # Write final manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "radio_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  [transcribe] {len(manifest)} clips -> {manifest_path}")


def run_normalize(clips_dir, output_dir, target_lufs=-23.0):
    """Stage 5: Loudness normalize to -23 LUFS, resample to 24kHz."""
    import pyloudnorm as pyln

    manifest_path = output_dir / "radio_manifest.jsonl"
    entries = [json.loads(l) for l in open(manifest_path)]
    out_audio = output_dir / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)

    final = []
    for entry in entries:
        audio, sr = sf.read(clips_dir / entry["audio"], dtype="float32")

        # Resample to 24kHz
        if sr != 24000:
            audio = torchaudio.functional.resample(
                torch.tensor(audio).unsqueeze(0), sr, 24000
            ).squeeze().numpy()
            sr = 24000

        # Normalize loudness
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if not np.isinf(loudness):
            audio = pyln.normalize.loudness(audio, loudness, target_lufs)

        sf.write(out_audio / entry["audio"], audio, sr)
        entry["duration"] = round(len(audio) / sr, 2)
        final.append(entry)

    with open(output_dir / "radio_manifest.jsonl", "w") as f:
        for entry in final:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = sum(e["duration"] for e in final)
    print(f"  [normalize] {len(final)} clips, {total/3600:.1f}h total -> {out_audio}")


def main():
    parser = argparse.ArgumentParser(description="Process radio audio for TTS")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/radio_processed")
    parser.add_argument("--stage", choices=STAGES, help="Start from this stage")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    work_dir = output_dir / "work"
    clips_dir = output_dir / "clips"

    start = STAGES.index(args.stage) if args.stage else 0

    if start <= 0:
        print("=== Stage 1: Diarization ===")
        run_diarize(input_dir, work_dir)
    if start <= 1:
        print("=== Stage 2: Filter ===")
        run_filter(input_dir, work_dir)
    if start <= 2:
        print("=== Stage 3: Clean ===")
        run_clean(input_dir, work_dir, clips_dir)
    if start <= 3:
        print("=== Stage 4: Transcribe ===")
        run_transcribe(clips_dir, work_dir, output_dir)
    if start <= 4:
        print("=== Stage 5: Normalize ===")
        run_normalize(clips_dir, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
