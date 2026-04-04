# Wanjiku Kikuyu TTS

Kikuyu Text-to-Speech system fine-tuned on Kameme FM broadcast voices, built on Qwen3-TTS.

## Quick Start

```bash
# Clone and setup
cd wanjiku-tts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download WAXAL Kikuyu dataset
python scripts/download_waxal.py

# Record Kameme FM stream (records for 1 hour by default)
python scripts/record_stream.py --duration 3600

# Clean and segment audio
python scripts/clean_audio.py --input data/kameme_raw --output data/kameme_clean

# Normalize transcription text
python scripts/normalize_text.py --input data/transcripts/raw.jsonl --output data/transcripts/manifest.jsonl

# Fine-tune
python scripts/finetune.py --config configs/config.yaml
```

## Prerequisites

- Python 3.10+
- FFmpeg installed (`brew install ffmpeg` on macOS)
- NVIDIA GPU with 40GB+ VRAM for training (A100 recommended)
- ~100GB disk space for datasets

## Project Structure

```
├── SPEC.md              # Full project specification
├── configs/config.yaml  # Training and inference config
├── data/
│   ├── waxal_kikuyu/    # Google WAXAL Kikuyu subset
│   ├── kameme_raw/      # Raw radio recordings
│   ├── kameme_clean/    # Processed utterance clips
│   └── transcripts/     # JSONL manifests
├── scripts/
│   ├── download_waxal.py
│   ├── record_stream.py
│   ├── clean_audio.py
│   ├── normalize_text.py
│   └── finetune.py
└── models/checkpoints/
```

## Usage

```python
from wanjiku_tts import WanjikuTTS

tts = WanjikuTTS(model_path="models/checkpoints/kameme-v1")
tts.synthesize("Ũhoro wa mũthenya", output="speech.wav")

# Voice cloning with 3-second reference
tts.synthesize("Ũhoro wa mũthenya", reference_audio="presenter.wav", output="speech.wav")
```

## Data Sources

- [Google WAXAL](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/introducing-waxal-a-new-open-dataset-for-african-speech-technology/) — Open Kikuyu speech dataset
- Kameme FM live stream — Broadcast voice recordings

## License

Apache 2.0 (matches Qwen3-TTS license)
