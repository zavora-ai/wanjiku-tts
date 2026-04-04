# Wanjiku Kikuyu TTS — Project Specification

## 1. Overview

Build a production-quality Kikuyu Text-to-Speech system fine-tuned on Kameme FM broadcast voices. The system will generate natural Kikuyu speech that captures the distinctive tone, cadence, and code-switching patterns of Kameme FM presenters.

### 1.1 Goals

- Generate intelligible, natural-sounding Kikuyu speech
- Support voice cloning of specific Kameme FM presenters (3-second reference)
- Handle Kikuyu–Swahili–English code-switching common in Kenyan broadcasting
- Produce broadcast-quality audio (24kHz+, mono, normalized)
- Provide a simple API/CLI for text-to-speech inference

### 1.2 Non-Goals (v1)

- Real-time streaming synthesis (batch inference is fine for v1)
- Automatic Kikuyu text generation or translation
- Music or sound effects generation
- Multi-speaker conversation synthesis

---

## 2. Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Text Input  │────▶│ Text Normalizer │──▶│  Qwen3-TTS      │──▶ Audio Output
│  (Kikuyu)    │     │ (Kikuyu rules)  │   │  (fine-tuned)   │    (.wav 24kHz)
└─────────────┘     └──────────────┘     └─────────────────┘
                                                │
                                          ┌─────┴──────┐
                                          │ Speaker     │
                                          │ Embedding   │
                                          │ (x-vector)  │
                                          └────────────┘
```

### 2.1 Base Model

**Qwen3-TTS-1.7B-Base** (Apache 2.0)

- Dual-track autoregressive architecture
- X-vector speaker embeddings for voice cloning
- 3-second reference audio voice cloning
- Native support: Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic
- Kikuyu: NOT natively supported → requires fine-tuning

### 2.2 Why Qwen3-TTS

| Feature | Benefit for this project |
|---|---|
| 3-sec voice cloning | Clone specific Kameme presenters with minimal audio |
| Apache 2.0 license | Commercial use for Kameme |
| Voice design mode | Create new Kikuyu voices from text descriptions |
| Streaming support | Future real-time deployment |
| 1.7B parameters | Trainable on a single A100/H100 GPU |

---

## 3. Data Strategy

### 3.1 Primary Dataset: Google WAXAL (Kikuyu subset)

- **Source**: Google Research WAXAL open dataset (released Feb 2026)
- **Content**: ~1,250 hours transcribed Kikuyu speech + ~20 hours studio-quality recordings
- **Format**: WAV audio + text transcriptions
- **License**: Open (check specific WAXAL license terms)
- **Purpose**: Teach the model Kikuyu phonology, prosody, and vocabulary

**Download script**: `scripts/download_waxal.py`

### 3.2 Secondary Dataset: Kameme FM Recordings

- **Source**: Live stream at `https://radio.or.ke/kameme/` + archived segments
- **Target volume**: 50–100 hours of clean, transcribed presenter speech
- **Collection method**: Stream recording → speech separation → segmentation → transcription
- **Purpose**: Fine-tune for Kameme broadcast voice style

**Collection pipeline**:
1. `scripts/record_stream.py` — Capture raw audio from the live stream
2. `scripts/clean_audio.py` — Remove music/jingles, separate speech, segment into utterances
3. Manual/semi-automatic transcription of segments

### 3.3 Data Processing Requirements

| Step | Tool | Output |
|---|---|---|
| Stream capture | ffmpeg via record_stream.py | Raw WAV/MP3 files |
| Music/speech separation | Demucs (htdemucs) | Isolated vocal tracks |
| Voice Activity Detection | Silero VAD | Utterance boundaries |
| Segmentation | pydub | 5–15 second clips |
| Noise reduction | noisereduce | Clean speech clips |
| Loudness normalization | pyloudnorm (EBU R128) | -23 LUFS normalized audio |
| Transcription bootstrap | Whisper large-v3 / Google Chirp | Draft transcriptions |
| Transcription review | Manual | Verified transcriptions |

### 3.4 Data Format

Each training sample:
```
data/
├── kameme_clean/
│   ├── KAM_0001.wav          # 24kHz mono WAV, 5-15s
│   ├── KAM_0002.wav
│   └── ...
└── transcripts/
    └── manifest.jsonl         # {"audio": "KAM_0001.wav", "text": "...", "speaker": "presenter_1", "duration": 8.3}
```

### 3.5 Text Normalization Rules (Kikuyu-specific)

The `scripts/normalize_text.py` handles:

1. **Number expansion**: `1000` → `ngiri ĩmwe` (one thousand)
2. **Abbreviation expansion**: `KBC` → `Kei Bii Cii`
3. **Loanword handling**: Keep English/Swahili loanwords as-is (common in Kameme broadcasts)
4. **Tone diacritics**: Preserve Kikuyu tone marks (ĩ, ũ) — critical for correct pronunciation
5. **Punctuation normalization**: Standardize quotes, dashes, ellipses
6. **Currency**: `Ksh 500` → `ciringĩ magana matano`

---

## 4. Training Plan

### 4.1 Phase 1: Language Adaptation (WAXAL)

Fine-tune Qwen3-TTS-1.7B-Base on the WAXAL Kikuyu dataset to learn:
- Kikuyu phoneme inventory (including prenasalized stops: mb, nd, ng, nj)
- Tonal patterns (Kikuyu has two tones: high and low)
- Prosodic patterns specific to Kikuyu

**Hyperparameters (starting point)**:
```yaml
learning_rate: 1e-5
batch_size: 8
gradient_accumulation_steps: 4
warmup_steps: 500
max_steps: 50000
fp16: true
```

**Hardware**: 1x NVIDIA A100 80GB (or equivalent)
**Estimated time**: ~24–48 hours

### 4.2 Phase 2: Voice Style Adaptation (Kameme)

Fine-tune the Phase 1 checkpoint on Kameme FM data to learn:
- Broadcast speech patterns (pacing, emphasis)
- Kameme-specific code-switching patterns
- Individual presenter voice characteristics

**Hyperparameters**:
```yaml
learning_rate: 5e-6          # Lower LR to preserve Kikuyu knowledge
batch_size: 4
max_steps: 10000
```

**Estimated time**: ~6–12 hours

### 4.3 Phase 3: Voice Cloning Setup

Extract x-vector speaker embeddings from target Kameme presenters:
- Requires 3+ seconds of clean reference audio per presenter
- Store embeddings for inference-time voice selection

---

## 5. Evaluation

### 5.1 Objective Metrics

| Metric | Target | Tool |
|---|---|---|
| MOS (Mean Opinion Score) | ≥ 3.5/5.0 | Human evaluation panel |
| Speaker similarity (cosine) | ≥ 0.85 | Resemblyzer |
| Character Error Rate (ASR roundtrip) | ≤ 15% | Whisper transcribe → compare |
| PESQ | ≥ 3.0 | pesq library |

### 5.2 Subjective Evaluation

- Panel of 5+ native Kikuyu speakers
- Rate: naturalness, intelligibility, speaker similarity, tone accuracy
- A/B test against existing Kikuyu TTS models (BrianMwangi/African-Kikuyu-TTS, gateremark/kikuyu-tts-v1)

### 5.3 Test Set

- 200 held-out Kikuyu sentences (diverse topics)
- 50 code-switched sentences (Kikuyu + English/Swahili)
- 20 sentences with numbers, dates, currency
- 10 long-form passages (news bulletin style)

---

## 6. Inference API

### 6.1 CLI Usage

```bash
# Basic synthesis
python -m wanjiku_tts --text "Ũhoro wa mũthenya" --output speech.wav

# With voice cloning
python -m wanjiku_tts --text "Ũhoro wa mũthenya" --reference presenter.wav --output speech.wav

# With voice description
python -m wanjiku_tts --text "Ũhoro wa mũthenya" --voice-desc "warm male Kikuyu broadcaster, authoritative" --output speech.wav
```

### 6.2 Python API

```python
from wanjiku_tts import WanjikuTTS

tts = WanjikuTTS(model_path="models/checkpoints/kameme-v1")
tts.synthesize("Ũhoro wa mũthenya", output="speech.wav")
tts.synthesize("Ũhoro wa mũthenya", reference_audio="presenter.wav", output="speech.wav")
```

---

## 7. Deployment

### 7.1 v1 Target

- Local inference on GPU machine (A100/H100 or consumer GPU with quantization)
- Batch processing of text scripts → audio files
- CLI + Python API

### 7.2 Future (v2)

- FastAPI server with REST endpoint
- Streaming WebSocket synthesis
- Quantized model (INT8/INT4) for consumer GPU deployment
- Mobile-optimized model via ONNX export

---

## 8. Project Structure

```
wanjiku-tts/
├── SPEC.md                    # This document
├── README.md                  # Setup and usage guide
├── requirements.txt           # Python dependencies
├── configs/
│   └── config.yaml            # Training and inference configuration
├── data/
│   ├── waxal_kikuyu/          # Google WAXAL Kikuyu subset
│   ├── kameme_raw/            # Raw radio recordings
│   ├── kameme_clean/          # Processed utterance clips
│   └── transcripts/           # JSONL manifests
├── scripts/
│   ├── download_waxal.py      # Download and extract WAXAL dataset
│   ├── record_stream.py       # Capture Kameme FM live stream
│   ├── clean_audio.py         # Audio cleaning pipeline
│   ├── normalize_text.py      # Kikuyu text normalization
│   └── finetune.py            # Fine-tuning script
└── models/
    └── checkpoints/           # Saved model checkpoints
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---|---|---|
| qwen-tts | ≥0.1.0 | Base TTS model and inference |
| transformers | ≥4.40 | Model loading and training |
| torch | ≥2.2 | Deep learning framework |
| torchaudio | ≥2.2 | Audio processing |
| demucs | ≥4.0 | Music/speech separation |
| pydub | ≥0.25 | Audio segmentation |
| noisereduce | ≥3.0 | Noise reduction |
| pyloudnorm | ≥0.1 | EBU R128 loudness normalization |
| silero-vad | latest | Voice activity detection |
| datasets | ≥2.18 | HuggingFace dataset loading |
| ffmpeg-python | ≥0.2 | FFmpeg wrapper for stream capture |
| soundfile | ≥0.12 | Audio I/O |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Kikuyu tone errors after fine-tuning | Unintelligible speech | Encode tone diacritics in training text; evaluate with native speakers early |
| Insufficient Kameme data quality | Poor voice style transfer | Aggressive audio cleaning; prioritize quality over quantity |
| Qwen3-TTS architecture not adaptable to Kikuyu | Project failure | Fallback: use VITS or Meta MMS as alternative base model |
| Code-switching breaks synthesis | Garbled mixed-language output | Train on code-switched examples explicitly; use language tags in input |
| Kameme FM licensing issues | Cannot use recordings | Get written permission from Kameme; they commissioned this project so should be fine |
| GPU compute costs | Budget overrun | Start with 0.6B model for prototyping; use A100 spot instances |

---

## 11. Timeline

| Week | Milestone |
|---|---|
| 1–2 | Data pipeline: download WAXAL, set up stream recording, begin Kameme capture |
| 3–4 | Audio cleaning and transcription of Kameme segments |
| 5–6 | Phase 1 fine-tuning on WAXAL Kikuyu |
| 7–8 | Phase 2 fine-tuning on Kameme data |
| 9 | Evaluation with native speaker panel |
| 10 | Bug fixes, optimization, delivery |

---

## 12. Success Criteria

1. MOS score ≥ 3.5 from native Kikuyu speaker panel
2. Successful voice cloning of at least 2 Kameme presenters
3. Handles code-switched Kikuyu/English/Swahili sentences without breaking
4. Generates broadcast-quality audio (24kHz, clean, normalized)
5. End-to-end latency < 10 seconds for a 30-word sentence (batch mode)
