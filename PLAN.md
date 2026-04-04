# Wanjiku TTS — Project Plan

## Design Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        WANJIKU TTS SYSTEM                            │
│                                                                      │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐  │
│  │  Kikuyu  │──▶│    Text      │──▶│  Qwen3-TTS   │──▶│  24kHz   │  │
│  │  Text    │   │  Normalizer  │   │  (fine-tuned) │   │  WAV     │  │
│  └─────────┘   └──────────────┘   └──────┬───────┘   └──────────┘  │
│                                          │                           │
│                                   ┌──────┴───────┐                   │
│                                   │   Speaker     │                   │
│                                   │   Embedding   │                   │
│                                   │  (x-vector)   │                   │
│                                   └──────────────┘                   │
│                                                                      │
│  Modes: voice cloning (3s ref) │ voice design │ preset speakers      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
WAXAL Kikuyu Dataset (1,250h)  ──┐
                                  ├──▶ Phase 1: Language Adaptation ──▶ Kikuyu checkpoint
Radio Broadcast Recordings (50h) ─┘                                          │
                                                                             ▼
                                       Phase 2: Voice Style Fine-tune ──▶ Final Model
                                                                             │
                                       Phase 3: Speaker Embedding    ──▶ Voice Cloning
                                                Extraction                   Ready
```

### AWS Infrastructure

```
Local Machine                          AWS
┌──────────────┐                ┌─────────────────────┐
│ Data capture  │───upload──▶   │  S3: wanjiku-tts/    │
│ Audio cleaning│               │  ├── data/            │
│ Transcription │               │  ├── models/          │
└──────────────┘                │  └── logs/            │
                                └─────────┬───────────┘
                                          │
                                ┌─────────▼───────────┐
                                │  EC2 GPU Instance    │
                                │  (p3.2xlarge V100    │
                                │   or p4d A100)       │
                                │                      │
                                │  - Pull data from S3 │
                                │  - Train model       │
                                │  - Push checkpoints  │
                                └──────────────────────┘
```

---

## Phase 1 — Data Collection & Preparation (Weeks 1–4)

### 1.1 Environment Setup
- [ ] Set up Python 3.10+ venv, install dependencies
- [ ] Install ffmpeg locally
- [ ] Configure AWS CLI with credentials
- [ ] Create S3 bucket for project data and model artifacts
- [ ] Verify GPU instance availability in target region (check quotas for p3/p4d)

### 1.2 WAXAL Dataset
- [ ] Download Google WAXAL Kikuyu subset via HuggingFace
- [ ] Inspect dataset: count hours, check audio quality, review transcription format
- [ ] Convert to training format (24kHz mono WAV + JSONL manifest)
- [ ] Upload processed WAXAL data to S3
- [ ] Split: 95% train / 5% validation

### 1.3 Radio Broadcast Collection
- [ ] Configure stream URL in config.yaml
- [ ] Record initial 20 hours of broadcast audio (multiple sessions across different shows)
- [ ] Run speech separation (Demucs) to isolate vocals from music/jingles
- [ ] Run VAD + segmentation into 5–15 second utterance clips
- [ ] Run noise reduction and loudness normalization (EBU R128, -23 LUFS)
- [ ] Bootstrap transcriptions using Whisper large-v3
- [ ] Manual review and correction of transcriptions (native Kikuyu speaker required)
- [ ] Label speaker IDs for each segment
- [ ] Continue recording to reach 50+ hours target
- [ ] Upload cleaned broadcast data to S3

### 1.4 Text Normalization
- [ ] Validate Kikuyu number expansion rules with native speaker
- [ ] Add missing abbreviation expansions common in broadcasts
- [ ] Handle Kikuyu tone diacritics (ĩ, ũ) — verify preservation through pipeline
- [ ] Test normalizer on 100 sample sentences, fix edge cases
- [ ] Run normalizer on full transcript set

### Milestone: ✅ 1,250h+ WAXAL + 50h+ broadcast data, cleaned, transcribed, on S3

---

## Phase 2 — Model Training (Weeks 5–8)

### 2.1 Training Infrastructure
- [ ] Launch EC2 GPU instance (p3.2xlarge minimum, p4d.24xlarge preferred)
- [ ] Install training dependencies on instance (torch, qwen-tts, transformers)
- [ ] Pull training data from S3 to instance local storage
- [ ] Verify Qwen3-TTS-12Hz-1.7B-Base loads and runs inference on English (sanity check)
- [ ] Set up checkpoint saving to S3 (every 1000 steps)

### 2.2 Phase 1 Training — Kikuyu Language Adaptation
- [ ] Fine-tune on WAXAL Kikuyu data
  - LR: 1e-5, batch: 8, grad accum: 4, warmup: 500, max steps: 50,000
  - Estimated: 24–48 hours on A100
- [ ] Monitor training loss, check for divergence
- [ ] Generate sample outputs every 5,000 steps — listen for Kikuyu phoneme quality
- [ ] Select best checkpoint based on validation loss
- [ ] Quick listening test: 10 Kikuyu sentences, check intelligibility

### 2.3 Phase 2 Training — Broadcast Voice Style
- [ ] Fine-tune Phase 1 checkpoint on broadcast data
  - LR: 5e-6 (lower to preserve Kikuyu knowledge), batch: 4, max steps: 10,000
  - Estimated: 6–12 hours
- [ ] Generate samples — check for broadcast voice style, pacing, emphasis
- [ ] Test code-switched sentences (Kikuyu + English + Swahili)
- [ ] Select best checkpoint
- [ ] Save final model to S3

### 2.4 Voice Cloning Setup
- [ ] Select 2–3 target presenter voices from broadcast data
- [ ] Extract 3+ second clean reference clips per presenter
- [ ] Generate x-vector speaker embeddings
- [ ] Test voice cloning: synthesize same text with different presenter embeddings
- [ ] Store embeddings alongside model checkpoint

### Milestone: ✅ Fine-tuned model producing intelligible Kikuyu speech with broadcast voice style

---

## Phase 3 — Evaluation & Iteration (Weeks 9–10)

### 3.1 Objective Evaluation
- [ ] ASR roundtrip test: synthesize → Whisper transcribe → compute CER (target: ≤15%)
- [ ] Speaker similarity: cosine similarity on embeddings (target: ≥0.85)
- [ ] PESQ audio quality score (target: ≥3.0)
- [ ] Test on held-out set:
  - 200 general Kikuyu sentences
  - 50 code-switched sentences
  - 20 sentences with numbers/dates/currency
  - 10 long-form news bulletin passages

### 3.2 Subjective Evaluation
- [ ] Recruit 5+ native Kikuyu speakers for evaluation panel
- [ ] MOS test: rate naturalness 1–5 (target: ≥3.5)
- [ ] Intelligibility test: can listeners understand the content?
- [ ] Tone accuracy test: are Kikuyu tones correct?
- [ ] Speaker similarity test: does cloned voice sound like the presenter?
- [ ] A/B test against existing models (BrianMwangi/African-Kikuyu-TTS, gateremark/kikuyu-tts-v1)

### 3.3 Iteration
- [ ] Identify failure modes from evaluation (tone errors, code-switching breaks, etc.)
- [ ] Augment training data to address weak areas
- [ ] Re-train if needed (partial fine-tuning on problem areas)
- [ ] Re-evaluate until targets met

### Milestone: ✅ MOS ≥ 3.5, CER ≤ 15%, voice cloning working for 2+ presenters

---

## Phase 4 — Packaging & Delivery (Week 10)

### 4.1 Inference Package
- [ ] Build CLI interface (`python -m wanjiku_tts --text "..." --output speech.wav`)
- [ ] Build Python API (`WanjikuTTS` class with `synthesize()` method)
- [ ] Support voice cloning mode (pass reference audio)
- [ ] Support voice design mode (pass text description)
- [ ] Add model quantization option (INT8) for smaller GPU deployment

### 4.2 Documentation
- [ ] Update README with final usage instructions
- [ ] Document model card: training data, performance metrics, limitations
- [ ] Document known limitations (which Kikuyu constructs work poorly, etc.)
- [ ] Write deployment guide for AWS inference

### 4.3 Delivery
- [ ] Push final model checkpoint to S3 / HuggingFace (if public release)
- [ ] Tag release v1.0 on GitHub
- [ ] Demo: generate sample audio clips for stakeholder review
- [ ] Handoff documentation

### Milestone: ✅ v1.0 released, demo audio delivered

---

## Task Summary

| Phase | Tasks | Duration | Blockers |
|-------|-------|----------|----------|
| 1 — Data | 24 tasks | Weeks 1–4 | Native Kikuyu speaker for transcription review |
| 2 — Training | 16 tasks | Weeks 5–8 | AWS GPU quota approval, Phase 1 data ready |
| 3 — Evaluation | 12 tasks | Weeks 9–10 | Evaluation panel recruitment |
| 4 — Delivery | 9 tasks | Week 10 | Evaluation targets met |
| **Total** | **61 tasks** | **10 weeks** | |

---

## Key Dependencies

```
Environment Setup ──▶ WAXAL Download ──▶ Data Processing ──▶ S3 Upload
                                                                 │
Radio Recording ──▶ Audio Cleaning ──▶ Transcription ──▶ S3 Upload
                                                                 │
                                              ┌──────────────────┘
                                              ▼
                                    GPU Instance Setup
                                              │
                                              ▼
                                    Phase 1 Training (WAXAL)
                                              │
                                              ▼
                                    Phase 2 Training (Broadcast)
                                              │
                                              ▼
                                    Voice Cloning Setup
                                              │
                                              ▼
                                    Evaluation ──▶ Iteration Loop
                                              │
                                              ▼
                                    Packaging & Delivery
```

---

## Critical Path

The longest sequential chain determines the minimum project duration:

**Radio recording (2 weeks)** → **Transcription review (2 weeks)** → **Phase 1 training (1 week)** → **Phase 2 training (1 week)** → **Evaluation (1 week)** → **Delivery (1 week)** = **8 weeks minimum**

The WAXAL download and processing can happen in parallel with radio recording. Transcription review is the most likely bottleneck — it requires a native Kikuyu speaker and is labor-intensive.

---

## Cost Estimate (AWS)

| Resource | Spec | Hours | $/hr | Total |
|----------|------|-------|------|-------|
| EC2 p3.2xlarge (V100) | Phase 1 training | 48 | ~$3.06 | ~$147 |
| EC2 p3.2xlarge (V100) | Phase 2 training | 12 | ~$3.06 | ~$37 |
| EC2 p3.2xlarge (V100) | Iteration/eval | 20 | ~$3.06 | ~$61 |
| S3 storage | ~500GB | 720 (month) | ~$0.023/GB | ~$12 |
| Data transfer | uploads/downloads | — | — | ~$10 |
| **Total estimate** | | | | **~$267** |

Use spot instances to cut EC2 costs by ~60–70%. Budget $100–150 with spots.

> For current pricing, check the [AWS Pricing Calculator](https://calculator.aws/).

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|------|-----------|--------|------------|-------|
| R1 | Kikuyu tones rendered incorrectly | High | High | Encode diacritics in training text; early native speaker eval at step 5k | ML lead |
| R2 | Insufficient clean broadcast audio | Medium | High | Prioritize quality over quantity; 20h clean > 100h noisy | Data lead |
| R3 | Qwen3-TTS won't adapt to Kikuyu | Low | Critical | Fallback to VITS or Meta MMS base model | ML lead |
| R4 | Code-switching breaks synthesis | High | Medium | Explicit code-switched training examples; language tags | ML lead |
| R5 | GPU spot instance interruption | Medium | Low | Checkpoint every 1000 steps to S3; auto-resume | Infra |
| R6 | Transcription bottleneck delays project | High | Medium | Start recording early; use Whisper bootstrap to speed review | Data lead |
| R7 | WAXAL dataset format incompatible | Low | Medium | Inspect dataset early (Week 1); write conversion script | Data lead |
