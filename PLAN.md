# Wanjiku TTS & ASR — Build Plan

Two production-quality Kikuyu speech systems: ASR (speech-to-text) and TTS (text-to-speech).

## Models

| System | Base Model | Params | Method | VRAM |
|--------|-----------|--------|--------|------|
| **ASR** | Gemma 4 E2B (google/gemma-4-E2B) | 2.3B | Unsloth LoRA fine-tune | ~12GB |
| **TTS** | Qwen3-TTS-12Hz-1.7B-Base | 1.7B | Voice cloning fine-tune | ~8GB |

Infrastructure: EC2 g5.2xlarge (A10G 24GB), us-east-1.

## Data Summary

**Paired audio + text (ready for training):**
| Dataset | Hours | Pairs | Notes |
|---------|-------|-------|-------|
| DigiGreen Kikuyu ASR | ~62h | 25,379 | Agriculture queries, multi-speaker |
| Google WAXAL kik_tts | 10.3h | 2,026 | Image descriptions, 8 speakers |

**Audio with text (needs forced alignment):**
| Dataset | Hours | Notes |
|---------|-------|-------|
| Biblica Kikuyu Bible | ~113h | Single speaker, studio quality, USFM verse text |
| GRN 5fish programs | 15.9h | Narrated Bible stories |

**Audio only (needs ASR transcription):**
| Dataset | Hours | Notes |
|---------|-------|-------|
| Radio (3 stations) | ~36h+ | Broadcast, multi-speaker, noisy |
| Kikuyu course videos | ~4.8h | Bilingual English→Kikuyu |
| Kikuyu Gospel songs | 2.0h | Singing — low priority |

**Text only (for normalization, prompts, evaluation):**
- FLORES-200: 2,009 sentences
- SIB-200: 1,004 sentences
- Bloom Library: 10 stories
- Leipzig: 815 sentences
- Bible USFM: full Kikuyu Bible
- Translation pairs: ~6K (Kikuyu↔English, Kikuyu↔Swahili)

---

## Phase 1: Data Preparation
*Goal: Clean, normalize, and align all data into training-ready format.*

### 1.1 Text Normalizer
Build a Kikuyu text normalizer that handles:
- Diacritic unification: macron (ī,ū) → tilde (ĩ,ũ), Greek/breve (ῖ,ῦ,ŭ,ȋ) → tilde
- Bare text detection (flag text missing diacritics for manual review)
- Number normalization (digits → Kikuyu words)
- Punctuation standardization
- Loanword handling (English/Swahili words in Kikuyu text)

### 1.2 DigiGreen Processing
- Match 25,379 WAV files to 26,483 CSV transcripts (by filename)
- Normalize transcripts through text normalizer
- Validate audio (check duration, sample rate, silence)
- Filter bad pairs (empty audio, mismatched transcripts)
- Split: 90% train / 5% dev / 5% test
- Output: manifest JSONL (path, text, duration, speaker_id)

### 1.3 WAXAL Processing
- Already paired — normalize transcripts (macron → tilde)
- Validate and split
- Output: manifest JSONL

### 1.4 Bible Forced Alignment
- Convert USFM → plain text per chapter (strip verse markers, keep verse boundaries)
- Forced alignment using MMS wav2vec2 (facebook/mms-1b) to segment chapter MP3s into verse-level clips
- Each verse becomes one training sample: audio clip + normalized text
- Expected yield: ~8,000-10,000 verse segments from 1,189 chapters
- Output: manifest JSONL

### 1.5 Radio Processing
- Run process_radio_v2.py pipeline on all recordings:
  - SpeechBrain diarization → speaker segments
  - Noise reduction
  - ASR transcription (initially with MaryWambo/whisper-base-kikuyu4, later with our Gemma 4 ASR)
  - Loudness normalization
- Output: manifest JSONL (lower confidence — ASR-generated transcripts)

### 1.6 S3 Backup
- Sync all processed data to s3://wanjiku-tts-971994957690/

---

## Phase 2: ASR — Gemma 4 E2B Fine-tune
*Goal: Best-in-class Kikuyu speech recognition.*

### 2.1 Training Data Preparation
- Combine DigiGreen (62h) + WAXAL (10h) + Bible aligned (est. 80h usable) = ~150h paired data
- Format for Gemma 4: audio input + instruction prompt → text output
- Prompt template: `"Transcribe this Kikuyu speech accurately."` + audio → normalized Kikuyu text
- Create train/dev/test splits (stratified by source)

### 2.2 Fine-tune with Unsloth
- Load `unsloth/gemma-4-E2B-it` (instruction-tuned variant)
- LoRA config: r=16, alpha=32, target_modules for audio+text layers
- Training: ~3 epochs on 150h data
- Batch size tuned to fit A10G 24GB
- Evaluate on held-out test set (WER metric)

### 2.3 Evaluation
- WER on DigiGreen test set (agriculture domain)
- WER on WAXAL test set (image descriptions)
- WER on Bible test set (formal/religious)
- Compare against: MaryWambo/whisper-base-kikuyu4, Gemma 4 zero-shot

### 2.4 Bootstrap More Data
- Use trained ASR to transcribe:
  - Radio recordings (~36h) → new paired data
  - GRN programs (16h) → new paired data
  - Kikuyu course audio (5h) → new paired data
- Human review sample for quality
- Add to training set and retrain (Phase 2.5)

---

## Phase 3: TTS — Qwen3-TTS Fine-tune
*Goal: Natural-sounding Kikuyu voice synthesis with voice cloning.*

### 3.1 Training Data Preparation
- Primary: Bible audio (single speaker, studio quality, ~80h after alignment)
  - This is ideal for TTS: consistent voice, clean audio, diverse vocabulary
- Secondary: DigiGreen (multi-speaker, for speaker diversity)
- Format for Qwen3-TTS: text + reference audio → synthesized speech
- All text normalized through Phase 1.1 normalizer

### 3.2 Fine-tune Qwen3-TTS
- Load `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")`
- Fine-tune on Bible speaker first (single-speaker TTS)
- Training config from qwen-tts documentation
- Evaluate: MOS (Mean Opinion Score), intelligibility, speaker similarity

### 3.3 Multi-speaker Extension
- Add DigiGreen speakers for voice cloning diversity
- Fine-tune voice cloning capability: given reference audio + text → speech in that voice
- Target: clone any Kikuyu speaker from 10s of reference audio

### 3.4 Evaluation
- MOS listening tests (naturalness)
- Intelligibility: synthesize text → run through ASR → compare to input (round-trip WER)
- Speaker similarity: compare cloned voice to reference
- Compare against: facebook/mms-tts-kik, gateremark/kikuyu-tts-v1

---

## Phase 4: Iteration & Optimization
*Goal: Virtuous cycle between ASR and TTS.*

### 4.1 ASR→TTS Data Loop
- Use ASR to transcribe all remaining audio → more TTS training data
- Use TTS to synthesize diverse text → data augmentation for ASR
- Retrain both models on expanded data

### 4.2 G2P Integration
- Build rule-based Kikuyu G2P (phonemic orthography, rules defined in RESEARCH.md)
- Integrate into TTS pipeline for pronunciation accuracy
- Use G2P for ASR language model / decoder constraints

### 4.3 Deployment Preparation
- Quantize models (GGUF/AWQ) for inference efficiency
- Build API endpoints (FastAPI)
- Latency optimization for real-time use

---

## Execution Order

```
Week 1:  Phase 1.1 (normalizer) + 1.2 (DigiGreen) + 1.3 (WAXAL)
Week 2:  Phase 1.4 (Bible alignment) + 1.5 (radio pipeline)
Week 3:  Phase 2.1-2.2 (ASR training)
Week 4:  Phase 2.3-2.4 (ASR eval + bootstrap)
Week 5:  Phase 3.1-3.2 (TTS training)
Week 6:  Phase 3.3-3.4 (TTS multi-speaker + eval)
Week 7+: Phase 4 (iteration)
```

## Cost Estimate

- EC2 g5.2xlarge: ~$1.21/hr × ~12hr/day × 42 days = ~$610
- S3 storage: ~$5/month
- Total: ~$620

## Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| ASR WER (DigiGreen test) | <20% | ~45% (Gemma 4 zero-shot) |
| ASR WER (Bible test) | <15% | N/A |
| TTS MOS | >3.5/5 | ~2.0 (mms-tts-kik) |
| TTS intelligibility (round-trip WER) | <25% | N/A |
| Voice cloning similarity | >0.7 cosine | N/A |

---

*Last updated: 2026-04-05*
