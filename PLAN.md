# Wanjiku TTS & ASR — Project Plan

## System Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          WANJIKU SPEECH SYSTEMS                              │
│                                                                              │
│  ┌─── ASR (Speech → Text) ──────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐  │   │
│  │  │  Kikuyu  │──▶│  Gemma 4 E2B │──▶│    Text      │──▶│ Kikuyu   │  │   │
│  │  │  Audio   │   │  (LoRA fine- │   │  Normalizer  │   │ Text     │  │   │
│  │  │  Input   │   │   tuned)     │   │  (post-proc) │   │ Output   │  │   │
│  │  └──────────┘   └──────────────┘   └──────────────┘   └──────────┘  │   │
│  │                                                                       │   │
│  │  Model: google/gemma-4-E2B (2.3B, native audio, Apache 2.0)         │   │
│  │  Training: Unsloth LoRA on ~150h paired Kikuyu data                  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─── TTS (Text → Speech) ──────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐  │   │
│  │  │  Kikuyu  │──▶│    Text      │──▶│  Qwen3-TTS   │──▶│  24kHz   │  │   │
│  │  │  Text    │   │  Normalizer  │   │  (fine-tuned) │   │  WAV     │  │   │
│  │  └──────────┘   └──────────────┘   └──────┬───────┘   └──────────┘  │   │
│  │                                           │                          │   │
│  │                                    ┌──────┴───────┐                  │   │
│  │                                    │   Speaker     │                  │   │
│  │                                    │   Embedding   │                  │   │
│  │                                    │  (ref audio)  │                  │   │
│  │                                    └──────────────┘                  │   │
│  │                                                                       │   │
│  │  Model: Qwen3-TTS-12Hz-1.7B-Base (1.7B, voice cloning, Apache 2.0) │   │
│  │  Modes: voice cloning (3s ref) │ preset speakers                     │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─── Shared Components ────────────────────────────────────────────────┐   │
│  │  Text Normalizer: diacritic unification (4 variants → tilde)         │   │
│  │  G2P: rule-based Kikuyu grapheme-to-phoneme (see RESEARCH.md)        │   │
│  │  Data Pipeline: process_radio_v2.py (diarize → denoise → ASR → norm) │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
                    ┌─── Paired Data (ready) ───────────────────────────┐
                    │                                                    │
DigiGreen (62h)  ───┤                                                    │
WAXAL (10.3h)    ───┤──▶ Text Normalizer ──▶ Training Manifests ──┬──▶ ASR Training
Bible aligned    ───┤                                              │    (Gemma 4 E2B)
                    │                                              │
                    └──────────────────────────────────────────────┤
                                                                   │
                    ┌─── Unpaired Audio ───────────────────────────┤
                    │                                              │
Radio (36h+)     ───┤                                              │
GRN (16h)        ───┤──▶ ASR (bootstrap) ──▶ New Paired Data ─────┘
Course (5h)      ───┤
                    └──────────────────────────────────────────────┐
                                                                   │
                    ┌─── TTS Training ─────────────────────────────┤
                    │                                              │
Bible audio      ───┤──▶ Forced Alignment ──▶ Verse Segments ──▶ TTS Training
(113h, 1 speaker)   │                                              (Qwen3-TTS)
                    │                                              │
DigiGreen        ───┤──▶ Multi-speaker voice cloning data ────────┘
                    └──────────────────────────────────────────────┐
                                                                   │
                    ┌─── Iteration Loop ───────────────────────────┘
                    │
                    │  ASR transcribes audio ──▶ more TTS training data
                    │  TTS synthesizes text  ──▶ data augmentation for ASR
                    └──────────────────────────────────────────────────
```

### AWS Infrastructure

```
Local Machine (macOS)                  AWS (us-east-1)
┌──────────────────┐            ┌──────────────────────────────┐
│ Data capture      │──upload──▶│  S3: wanjiku-tts-            │
│ Audio monitoring  │           │       971994957690/          │
│ Evaluation        │           │  ├── data/waxal_kikuyu/      │
│ afplay samples    │           │  ├── data/digigreen/         │
└──────────────────┘            │  ├── data/bible_audio/       │
                                │  ├── data/radio_raw*/        │
                                │  ├── data/grn_kikuyu/        │
                                │  ├── data/text_datasets/     │
                                │  └── models/checkpoints/     │
                                └─────────┬────────────────────┘
                                          │
                                ┌─────────▼────────────────────┐
                                │  EC2 g5.2xlarge              │
                                │  (A10G 24GB GPU)             │
                                │  i-076aaa7423b670d2a         │
                                │  IP: 13.217.62.47            │
                                │  200GB gp3 (73% used)        │
                                │                              │
                                │  torch 2.10, CUDA 12.8       │
                                │  unsloth 2026.4.2            │
                                │  transformers 5.5.0          │
                                │                              │
                                │  - Train ASR (Gemma 4 E2B)   │
                                │  - Train TTS (Qwen3-TTS)     │
                                │  - Process audio pipelines   │
                                │  - Push checkpoints to S3    │
                                └──────────────────────────────┘
```

---

## Phase 1 — Data Preparation (Weeks 1–2)

### 1.1 Environment Setup
- [x] Set up Python 3.10+ venv, install dependencies
- [x] Install ffmpeg on EC2 instance
- [x] Configure AWS CLI with credentials
- [x] Create S3 bucket (`s3://wanjiku-tts-971994957690`)
- [x] Launch EC2 g5.2xlarge (A10G 24GB), attach IAM role for S3 access
- [x] Verify Qwen3-TTS-12Hz-1.7B-Base loads and runs (4.2GB VRAM, voice cloning works)
- [x] Verify Gemma 4 E2B loads for zero-shot ASR

### 1.2 Data Collection (Complete)
- [x] Download Google WAXAL Kikuyu TTS (10.3h, 2,026 clips, 8 speakers)
- [x] Download DigiGreen Kikuyu ASR (62h, 25,379 WAVs + transcripts)
- [x] Download Biblica Kikuyu Bible audio (113h, 1,189 MP3s + USFM text)
- [x] Download GRN 5fish Kikuyu programs (15.9h, 15 MP3s)
- [x] Record radio streams: Kameme FM (~20h), Inooro FM (~16h), Gukena FM (~1h)
- [x] Download Kikuyu course videos (4.8h, 21 MP4s)
- [x] Download text datasets: SIB-200, FLORES-200, Bloom, Leipzig, African Storybook
- [x] Download translation pairs: CGIAR, FLORES parallel, Kikuyu-Kiswahili
- [x] Download UCLA Phonetics, Lomax recording, Kikuyu Gospel songs

### 1.3 Text Normalizer
- [ ] Build diacritic unification: macron (ī,ū) → tilde (ĩ,ũ), Greek/breve (ῖ,ῦ,ŭ,ȋ) → tilde
- [ ] Handle bare text detection (flag missing diacritics for review)
- [ ] Kikuyu number expansion rules (digits → Kikuyu words)
- [ ] Punctuation standardization
- [ ] Loanword handling (English/Swahili in Kikuyu text)
- [ ] Test on samples from each dataset (WAXAL, DigiGreen, Bible, radio)
- [ ] Run normalizer on all transcript sets

### 1.4 DigiGreen Processing
- [ ] Match 25,379 WAV files to 26,483 CSV transcripts (by filename)
- [ ] Normalize transcripts through text normalizer
- [ ] Validate audio (duration, sample rate, silence detection)
- [ ] Filter bad pairs (empty audio, mismatched transcripts)
- [ ] Split: 90% train / 5% dev / 5% test
- [ ] Output: manifest JSONL (path, text, duration, speaker_id)

### 1.5 WAXAL Processing
- [ ] Normalize transcripts (macron → tilde)
- [ ] Validate and verify existing splits
- [ ] Output: manifest JSONL

### 1.6 Bible Forced Alignment
- [ ] Convert USFM → plain text per chapter (strip markers, keep verse boundaries)
- [ ] Forced alignment using MMS wav2vec2 to segment chapter MP3s → verse clips
- [ ] Expected yield: ~8,000–10,000 verse segments from 1,189 chapters
- [ ] Quality filter: remove segments with poor alignment scores
- [ ] Output: manifest JSONL

### 1.7 Radio Processing
- [ ] Run process_radio_v2.py on all recordings:
  - SpeechBrain diarization → speaker segments
  - Noise reduction (noisereduce)
  - ASR transcription (MaryWambo/whisper-base-kikuyu4 initially)
  - Loudness normalization (pyloudnorm, EBU R128 -23 LUFS)
- [ ] Output: manifest JSONL (lower confidence — ASR-generated transcripts)

### 1.8 S3 Backup
- [ ] Sync all processed data to s3://wanjiku-tts-971994957690/

### Milestone: ✅ ~150h paired audio+text in normalized manifests, backed up to S3

---

## Phase 2 — ASR: Gemma 4 E2B Fine-tune (Weeks 3–4)

### 2.1 Training Data
- [ ] Combine: DigiGreen (62h) + WAXAL (10h) + Bible aligned (~80h) = ~150h
- [ ] Format for Gemma 4: audio input + prompt → normalized Kikuyu text
- [ ] Prompt: `"Transcribe this Kikuyu speech accurately."` + audio
- [ ] Create stratified train/dev/test splits (by source)

### 2.2 Fine-tune with Unsloth
- [ ] Load `unsloth/gemma-4-E2B-it` (instruction-tuned, audio-capable)
- [ ] LoRA config: r=16, alpha=32, target audio+text layers
- [ ] Training: ~3 epochs, batch size tuned for A10G 24GB
- [ ] Checkpoint every 1000 steps to S3
- [ ] Monitor loss, generate sample transcriptions during training

### 2.3 Evaluation
- [ ] WER on DigiGreen test set (agriculture domain)
- [ ] WER on WAXAL test set (image descriptions)
- [ ] WER on Bible test set (formal/religious)
- [ ] Compare against baselines:
  - MaryWambo/whisper-base-kikuyu4
  - Gemma 4 E2B zero-shot
  - microsoft/paza-whisper-large-v3-turbo

### 2.4 Bootstrap More Data
- [ ] Use trained ASR to transcribe:
  - Radio recordings (~36h) → new paired data
  - GRN programs (16h) → new paired data
  - Kikuyu course audio (5h) → new paired data
- [ ] Human review sample for quality
- [ ] Add to training set and retrain

### Milestone: ✅ ASR with WER <20% on DigiGreen test, <15% on Bible test

---

## Phase 3 — TTS: Qwen3-TTS Fine-tune (Weeks 5–6)

### 3.1 Training Data
- [ ] Primary: Bible audio (single speaker, studio quality, ~80h after alignment)
- [ ] Secondary: DigiGreen (multi-speaker, for voice diversity)
- [ ] All text normalized through Phase 1.3 normalizer
- [ ] Format for Qwen3-TTS: text + reference audio → synthesized speech

### 3.2 Phase A — Single Speaker (Bible Voice)
- [ ] Fine-tune on Bible speaker data
  - LR: 1e-5, batch: 8, grad accum: 4, warmup: 500
  - Estimated: 12–24 hours on A10G
- [ ] Generate samples every 5,000 steps — listen for Kikuyu quality
- [ ] Select best checkpoint based on validation loss

### 3.3 Phase B — Multi-speaker Voice Cloning
- [ ] Add DigiGreen speakers for voice cloning diversity
- [ ] Fine-tune voice cloning: given reference audio + text → speech in that voice
- [ ] Target: clone any Kikuyu speaker from 10s of reference audio
- [ ] Test with radio presenter voices

### 3.4 Evaluation
- [ ] MOS listening tests (naturalness, target ≥3.5/5)
- [ ] Intelligibility: synthesize → ASR → compare to input (round-trip WER <25%)
- [ ] Speaker similarity: cosine similarity on embeddings (target ≥0.7)
- [ ] Compare against: facebook/mms-tts-kik, gateremark/kikuyu-tts-v1

### Milestone: ✅ TTS with MOS ≥3.5, voice cloning working, intelligible Kikuyu output

---

## Phase 4 — Iteration & Delivery (Week 7+)

### 4.1 ASR↔TTS Data Loop
- [ ] ASR transcribes remaining audio → more TTS training data
- [ ] TTS synthesizes diverse text → data augmentation for ASR
- [ ] Retrain both models on expanded data

### 4.2 G2P Integration
- [ ] Build rule-based Kikuyu G2P (rules in RESEARCH.md)
- [ ] Integrate into TTS pipeline for pronunciation accuracy
- [ ] Use G2P for ASR decoder constraints

### 4.3 Packaging
- [ ] Build CLI: `python -m wanjiku --asr audio.wav` / `python -m wanjiku --tts "text"`
- [ ] Build Python API (`WanjikuASR`, `WanjikuTTS` classes)
- [ ] Quantize models (GGUF/AWQ) for inference efficiency
- [ ] FastAPI endpoints for serving

### 4.4 Documentation & Release
- [ ] Update README with usage instructions
- [ ] Model cards: training data, metrics, limitations
- [ ] Tag release v1.0 on GitHub
- [ ] Push models to HuggingFace (if public release)

### Milestone: ✅ v1.0 released — ASR + TTS serving Kikuyu speech

---

## Key Dependencies

```
Data Collection (done) ──▶ Text Normalizer ──▶ Data Processing ──▶ S3 Upload
                                                                       │
                                                    ┌──────────────────┘
                                                    ▼
                                          ┌── ASR Training (Gemma 4 E2B)
                                          │         │
                                          │         ▼
                                          │   ASR Bootstrap (transcribe radio/GRN)
                                          │         │
                                          │         ▼
                                          │   ASR Retrain (expanded data)
                                          │
Bible Forced Alignment ──────────────────▶├── TTS Training (Qwen3-TTS)
                                          │         │
                                          │         ▼
                                          │   TTS Multi-speaker
                                          │
                                          └──▶ Iteration Loop ──▶ Packaging ──▶ v1.0
```

## Critical Path

**Text normalizer (3 days)** → **DigiGreen + WAXAL processing (2 days)** → **Bible alignment (3 days)** → **ASR training (3 days)** → **ASR eval + bootstrap (3 days)** → **TTS training (3 days)** → **TTS eval (2 days)** → **Packaging (2 days)** = **~3 weeks minimum**

Bible forced alignment can run in parallel with DigiGreen/WAXAL processing. ASR bootstrap feeds into TTS training data.

---

## Cost Estimate (AWS)

| Resource | Spec | Hours | $/hr | Total |
|----------|------|-------|------|-------|
| EC2 g5.2xlarge | ASR training (Gemma 4) | 48 | ~$1.21 | ~$58 |
| EC2 g5.2xlarge | TTS training (Qwen3) | 36 | ~$1.21 | ~$44 |
| EC2 g5.2xlarge | Data processing/alignment | 24 | ~$1.21 | ~$29 |
| EC2 g5.2xlarge | Iteration/eval/bootstrap | 40 | ~$1.21 | ~$48 |
| EC2 g5.2xlarge | Recording (already running) | ~200 | ~$1.21 | ~$242 |
| S3 storage | ~20GB | 720 (month) | ~$0.023/GB | ~$0.50 |
| **Total estimate** | | | | **~$422** |

> For current pricing, check the [AWS Pricing Calculator](https://calculator.aws/).

---

## Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| ASR WER (DigiGreen test) | <20% | ~45% (Gemma 4 zero-shot) |
| ASR WER (Bible test) | <15% | N/A |
| TTS MOS (naturalness) | ≥3.5/5 | ~2.0 (mms-tts-kik) |
| TTS intelligibility (round-trip WER) | <25% | N/A |
| Voice cloning similarity | ≥0.7 cosine | N/A |
| Inference latency (ASR) | <2x real-time | N/A |
| Inference latency (TTS) | <3x real-time | N/A |

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Kikuyu tones rendered incorrectly in TTS | High | High | Tone learned implicitly from audio codec (confirmed by arxiv 2403.16865); early native speaker eval |
| R2 | Gemma 4 E2B LoRA fails (like previous PEFT attempt) | Medium | High | Unsloth has native Gemma 4 support; fallback to Paza Whisper fine-tune |
| R3 | Orthographic inconsistency across datasets | High | Medium | Text normalizer (Phase 1.3) unifies 4 diacritic variants |
| R4 | Bible forced alignment quality too low | Medium | Medium | Use MMS wav2vec2 + quality filtering; fallback to chapter-level training |
| R5 | Insufficient clean radio audio | Medium | Low | Already have 36h+; DigiGreen+Bible provide enough paired data without radio |
| R6 | GPU spot instance interruption | Medium | Low | Checkpoint every 1000 steps to S3; auto-resume |
| R7 | Code-switching breaks ASR/TTS | High | Medium | Include code-switched examples in training; language tags in prompts |
| R8 | A10G 24GB insufficient for Gemma 4 training | Low | High | Unsloth 4-bit quantization; gradient checkpointing; reduce batch size |
| R9 | MCAA1-MSU dataset access never granted | Medium | Medium | Already have 150h+ without it; nice-to-have, not blocking |

---

*Last updated: 2026-04-05*
