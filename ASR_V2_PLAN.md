# ASR v2 Improvement Plan

Based on research findings from the v1 training run. Execute after v1 evaluation.

## Key Reference

**Akera et al. (2025)** — "How much speech data is necessary for ASR in African languages? An evaluation of data scaling in Kinyarwanda and Kikuyu" (arXiv:2510.07221)

- Trained Whisper on **270h of Kikuyu** data
- 50h → WER <13%, 200h → WER <10%
- **38.6% of high-error cases caused by noisy ground truth transcriptions**
- Data quality > data quantity
- Released models and data

**Microsoft Paza** — Fine-tuned Whisper-large-v3-turbo on Kikuyu + 5 other Kenyan languages (microsoft/paza-whisper-large-v3-turbo). Trained on MCAA1-MSU data (270h Kikuyu). Available as a teacher model.

## v1 Baseline (current run)

- Model: Gemma 4 E2B, LoRA r=32
- Data: 44K audio (DigiGreen 23K + Bible 20K + WAXAL 1.2K) + 8.8K text denoising
- Training: 14K steps, ~1 epoch
- Known issues: Bible alignment is proportional (rough), no data quality filtering

## Improvement Areas (ordered by expected impact)

### 1. Fix Bible Alignment Quality
**Why:** Akera et al. found noisy transcripts are the #1 error source (38.6%). Our Bible data uses proportional character-count splitting (CV=0.36), meaning many verse segments have misaligned text.

**How:**
- Use MMS wav2vec2 forced alignment (facebook/mms-1b) to get word-level timestamps
- Segment at verse boundaries using aligned timestamps
- Score each segment by alignment confidence
- Filter out low-confidence segments (bottom 20%)

**Expected impact:** Could fix the largest single source of training noise.

**Papers:**
- Akera et al. (2510.07221) — data quality > quantity
- BibleTTS (Meyer et al. 2022) — used MFA for alignment, describes methodology

### 2. Pseudo-labeling Unlabeled Audio
**Why:** We have 56h of untranscribed audio (GRN 16h, radio 36h, course 5h). Using v1 to transcribe it and adding high-confidence segments expands training data.

**How:**
- Run v1 model on GRN and radio audio
- Score each transcription by model confidence (log probability)
- Keep top 70% by confidence
- Human-review a random 5% sample to estimate quality
- Add to training set

**Expected impact:** 10-30% WER reduction per iteration (literature consensus).

**Papers:**
- ReHear (2602.18721) — iterative pseudo-label refinement with audio LLMs
- "Empowering Low-Resource Language ASR via Large-Scale Pseudo Labeling" (2408.14026)
- "A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data" (2506.11130)

### 3. Cross-lingual Transfer from Swahili
**Why:** Swahili is a related Bantu language with abundant ASR data. Shared phonological features can help the model generalize.

**How:**
- Add 5-10h of Swahili ASR data (from Common Voice or KenSpeech) to training mix
- Use as ~5% of training data (don't overwhelm Kikuyu)
- Alternatively: initialize from Paza model weights instead of base Gemma 4

**Expected impact:** Up to 17% relative WER reduction in zero-shot (Gupta et al. 2024).

**Papers:**
- Gupta et al. (2410.13445) — cross-lingual transfer for low-resource ASR
- Microsoft Paza — already trained on Swahili+Kikuyu

### 4. Iterative LoRA Training (Focus-Feedback-Fix)
**Why:** Single-pass training may not optimally learn all aspects. Iterative approach focuses on failure modes.

**How:**
- After v1 eval, identify high-error categories (e.g., numbers, names, code-switching)
- Create focused training batches for weak areas
- Fine-tune v1 checkpoint on these focused batches
- Re-evaluate

**Expected impact:** Targeted improvement on specific failure modes.

**Papers:**
- "Iterative LoRA Training through Focus-Feedback-Fix for Multilingual Speech Recognition" (2507.08477)

### 5. TTS Data Augmentation
**Why:** Once TTS is trained, we can synthesize audio from our 34K text corpus, creating unlimited paired data.

**How:**
- Train Qwen3-TTS on Bible audio (Phase 3 of main plan)
- Generate synthetic audio for FLORES, SIB-200, Leipzig, and other text
- Add synthetic audio to ASR training (1:2 synthetic-to-real ratio per literature)

**Expected impact:** 6.5-25% relative WER improvement (varies by language).

**Papers:**
- "Synthetic Voice Data for ASR in African Languages" (2507.17578) — tested on Chichewa and Dholuo
- "Frustratingly Easy Data Augmentation for Low-Resource ASR" (2509.15373)
- "A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data" (2506.11130)

### 6. Distillation from Paza
**Why:** Microsoft's Paza model was trained on 270h of Kikuyu (MCAA1-MSU data we don't have access to). We can use it as a teacher.

**How:**
- Run Paza on our audio to generate teacher transcriptions
- Train our model to match Paza's output (knowledge distillation)
- Or: use Paza transcriptions as additional pseudo-labels

**Expected impact:** Access to knowledge from 270h of data we can't directly use.

**Papers:**
- "Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM" (2506.11089)

## Execution Order

```
v1 finishes → Evaluate (WER on 3 test sets, listen to samples)
                    │
                    ├─ If WER > 40%: Focus on data quality fixes (Step 1)
                    ├─ If WER 20-40%: Do Steps 1 + 2 together
                    └─ If WER < 20%: Do Steps 2 + 3, start TTS
                    │
                    v
              v2 training run
                    │
                    v
              Evaluate v2 → Plan v3 (add TTS augmentation)
```

## Data Sources for v2

| Source | Hours | Status | Quality |
|--------|-------|--------|---------|
| DigiGreen | 62h | Ready | Good (human transcripts) |
| Bible (re-aligned) | ~80h | Needs forced alignment | Will improve |
| WAXAL | 10h | Ready | Good (studio) |
| GRN pseudo-labeled | ~16h | Needs v1 inference | Medium (pseudo) |
| Radio pseudo-labeled | ~36h | Needs v1 inference | Low-medium (pseudo + noisy) |
| Swahili cross-lingual | ~10h | Needs download | Good |
| Text denoising | 12K samples | Ready | Good |

## References

1. Akera et al. (2025). "How much speech data is necessary for ASR in African languages?" arXiv:2510.07221
2. Fang et al. (2025). "Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning." arXiv:2506.05671
3. Burdisso et al. (2026). "Text-only adaptation in LLM-based ASR through text denoising." arXiv:2601.20900
4. Gupta et al. (2024). "Parameter-efficient Adaptation of Multilingual Multimodal Models for Low-resource ASR." arXiv:2410.13445
5. "Iterative LoRA Training through Focus-Feedback-Fix." arXiv:2507.08477
6. "Synthetic Voice Data for ASR in African Languages." arXiv:2507.17578
7. "Frustratingly Easy Data Augmentation for Low-Resource ASR." arXiv:2509.15373
8. "Iterative Pseudo-Label Refinement for Semi-Supervised Speech Recognition." arXiv:2602.18721
9. Microsoft Paza. https://www.microsoft.com/en-us/research/project/project-gecko/pazabench-models/

---

*Created: 2026-04-06. Execute after v1 evaluation.*
