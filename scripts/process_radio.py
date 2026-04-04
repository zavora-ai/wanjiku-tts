"""
Radio audio processing pipeline.
Processes raw radio recordings into clean, labeled, single-speaker Kikuyu clips.

Usage:
    python scripts/process_radio.py --input data/radio_raw --output data/radio_processed --config configs/config.yaml
    python scripts/process_radio.py --input data/radio_raw --stage demucs      # Run only one stage
    python scripts/process_radio.py --input data/radio_raw --stage classify    # Run from classification onwards
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Stage 1: Music/Speech Separation ──

def run_demucs(input_dir, output_dir):
    """Separate vocals from music/background using Demucs."""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(input_dir.glob("*.wav"))
    print(f"[demucs] Processing {len(wavs)} files...")
    for wav in wavs:
        print(f"  {wav.name}")
        subprocess.run([
            "python3", "-m", "demucs", "--two-stems", "vocals",
            "-o", str(output_dir), "-n", "htdemucs",
            str(wav),
        ], check=True)
    print(f"[demucs] Done. Vocals in {output_dir}/htdemucs/*/vocals.wav")


# ── Stage 2: Voice Activity Detection ──

def run_vad(vocals_dir, output_path, min_speech=1.0, min_silence=0.5):
    """Detect speech segments using Silero VAD."""
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
    get_speech_ts = utils[0]

    all_segments = {}
    for wav_path in sorted(vocals_dir.rglob("vocals.wav")):
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != 16000:
            import torchaudio
            audio_t = torch.tensor(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio = audio_t.squeeze().numpy()

        speech_ts = get_speech_ts(
            torch.tensor(audio), model,
            min_speech_duration_ms=int(min_speech * 1000),
            min_silence_duration_ms=int(min_silence * 1000),
            sampling_rate=16000,
        )
        segments = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in speech_ts]
        key = wav_path.parent.name
        all_segments[key] = segments
        print(f"  [vad] {key}: {len(segments)} segments, {sum(s['end']-s['start'] for s in segments)/3600:.1f}h speech")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_segments, f, indent=2)
    print(f"[vad] Saved to {output_path}")


# ── Stage 3: Speaker Diarization ──

def run_diarization(vocals_dir, vad_path, output_path):
    """Diarize speakers using pyannote.audio."""
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    with open(vad_path) as f:
        vad_segments = json.load(f)

    results = {}
    for key, segments in vad_segments.items():
        wav_path = vocals_dir / key / "vocals.wav"
        if not wav_path.exists():
            continue
        print(f"  [diarize] {key}...")
        diarization = pipeline(str(wav_path))
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        results[key] = turns
        speakers = set(t["speaker"] for t in turns)
        print(f"    {len(turns)} turns, {len(speakers)} speakers")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[diarize] Saved to {output_path}")


# ── Stage 4: Content Classification ──

def classify_segment(audio, sr):
    """Classify a speech segment by content type and quality."""
    duration = len(audio) / sr

    # SNR estimation (simple: signal power vs noise floor)
    signal_power = np.mean(audio ** 2)
    # Estimate noise from quietest 10% of frames
    frame_size = int(0.025 * sr)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio) - frame_size, frame_size)]
    frame_powers = sorted([np.mean(f ** 2) for f in frames])
    noise_power = np.mean(frame_powers[:max(1, len(frame_powers) // 10)]) + 1e-10
    snr = 10 * np.log10(signal_power / noise_power)

    # Spectral flatness (music vs speech indicator)
    from scipy.fft import rfft
    spectrum = np.abs(rfft(audio[:sr]))  # first second
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arith_mean = np.mean(spectrum) + 1e-10
    spectral_flatness = geo_mean / arith_mean

    # Heuristic classification
    content_type = "speech"
    if spectral_flatness > 0.3:
        content_type = "music"
    elif duration < 8 and snr > 30:
        content_type = "jingle"  # short, high production quality
    elif snr < 15:
        content_type = "caller"  # phone quality
    elif duration < 60:
        content_type = "monologue"
    else:
        content_type = "news"  # long single-speaker = likely news/story

    return {"type": content_type, "snr": round(snr, 1), "spectral_flatness": round(spectral_flatness, 3), "duration": round(duration, 2)}


def run_classification(vocals_dir, diarize_path, output_path):
    """Classify content type for each diarized segment."""
    with open(diarize_path) as f:
        diarized = json.load(f)

    results = {}
    for key, turns in diarized.items():
        wav_path = vocals_dir / key / "vocals.wav"
        if not wav_path.exists():
            continue
        audio, sr = sf.read(wav_path, dtype="float32")
        classified = []
        for turn in turns:
            start_sample = int(turn["start"] * sr)
            end_sample = int(turn["end"] * sr)
            segment_audio = audio[start_sample:end_sample]
            if len(segment_audio) < sr:  # skip < 1s
                continue
            info = classify_segment(segment_audio, sr)
            info.update(turn)
            classified.append(info)
        results[key] = classified
        types = {}
        for c in classified:
            types[c["type"]] = types.get(c["type"], 0) + 1
        print(f"  [classify] {key}: {types}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[classify] Saved to {output_path}")


# ── Stage 5: Language Detection + Whisper Transcription ──

def run_transcription(vocals_dir, classified_path, output_dir, min_snr=20, target_types=None):
    """Transcribe clean Kikuyu segments with Whisper."""
    import whisper
    model = whisper.load_model("large-v3")

    if target_types is None:
        target_types = {"news", "story", "monologue", "speech"}

    with open(classified_path) as f:
        classified = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    idx = 0

    for key, segments in classified.items():
        wav_path = vocals_dir / key / "vocals.wav"
        if not wav_path.exists():
            continue
        audio, sr = sf.read(wav_path, dtype="float32")

        for seg in segments:
            if seg["type"] not in target_types:
                continue
            if seg["snr"] < min_snr:
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = audio[start_sample:end_sample]

            # Whisper transcription + language detection
            result = model.transcribe(
                segment_audio, language=None, fp16=torch.cuda.is_available()
            )
            lang = result.get("language", "unknown")
            if lang not in ("ki", "sw", "en"):  # Kikuyu, Swahili, English
                continue

            # Save clip
            clip_name = f"RADIO_{idx:06d}.wav"
            sf.write(output_dir / clip_name, segment_audio, sr)
            manifest.append({
                "audio": clip_name,
                "text": result["text"].strip(),
                "speaker": seg["speaker"],
                "type": seg["type"],
                "language": lang,
                "duration": seg["duration"],
                "snr": seg["snr"],
                "source": key,
            })
            idx += 1

    manifest_path = output_dir.parent / "transcripts" / "radio_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[transcribe] {idx} clips saved to {output_dir}")
    print(f"[transcribe] Manifest: {manifest_path}")


# ── Stage 6: Loudness Normalization ──

def run_normalize(clip_dir, target_lufs=-23.0):
    """Normalize all clips to consistent loudness."""
    import pyloudnorm as pyln
    clips = sorted(clip_dir.glob("*.wav"))
    print(f"[normalize] Processing {len(clips)} clips...")
    meter = pyln.Meter(24000)
    for clip in clips:
        audio, sr = sf.read(clip, dtype="float32")
        loudness = meter.integrated_loudness(audio)
        if np.isinf(loudness):
            continue
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        sf.write(clip, audio, sr)
    print(f"[normalize] Done.")


# ── Main ──

STAGES = ["demucs", "vad", "diarize", "classify", "transcribe", "normalize"]

def main():
    parser = argparse.ArgumentParser(description="Process radio audio for TTS training")
    parser.add_argument("--input", required=True, help="Input directory with raw WAV files")
    parser.add_argument("--output", default="data/radio_processed", help="Output directory")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=STAGES, help="Run only this stage (and onwards)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    work_dir = output_dir / "work"

    demucs_dir = work_dir / "demucs"
    vad_path = work_dir / "vad_segments.json"
    diarize_path = work_dir / "diarized.json"
    classify_path = work_dir / "classified.json"
    clips_dir = output_dir / "clips"

    start_idx = STAGES.index(args.stage) if args.stage else 0

    if start_idx <= 0:
        run_demucs(input_dir, demucs_dir)
    if start_idx <= 1:
        run_vad(demucs_dir / "htdemucs", vad_path)
    if start_idx <= 2:
        run_diarization(demucs_dir / "htdemucs", vad_path, diarize_path)
    if start_idx <= 3:
        run_classification(demucs_dir / "htdemucs", diarize_path, classify_path)
    if start_idx <= 4:
        run_transcription(demucs_dir / "htdemucs", classify_path, clips_dir)
    if start_idx <= 5:
        run_normalize(clips_dir)

    print("\n=== Pipeline Complete ===")
    print(f"Clean clips: {clips_dir}")
    print(f"Manifest: {output_dir}/transcripts/radio_manifest.jsonl")


if __name__ == "__main__":
    main()
