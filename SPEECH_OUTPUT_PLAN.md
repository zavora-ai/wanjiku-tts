# Speech Output Plan: Native Audio-In/Audio-Out Kikuyu Assistant

## Goal
Build a unified model that hears Kikuyu, reasons, and speaks Kikuyu back — no pipeline, no latency.

## Architecture: VITA-Audio style MCTP on Gemma 4

```
Kikuyu Audio → [Audio Encoder] → Gemma 4 E2B (frozen)
                                       ↓
                               Hidden states + text tokens
                                       ↓
                     ┌─────────────────┴──────────────────┐
                     ↓                                     ↓
               Text output                          MCTP modules (×10)
               (reasoning)                          (lightweight transformers)
                                                          ↓
                                                    Audio tokens
                                                          ↓
                                                    [Vocoder] → Kikuyu speech
```

## Key References

- **VITA-Audio** (Tencent, 2025): First MLLM with zero audio token delay. MCTP modules generate 10 audio tokens per LLM forward pass. 3-5× speedup. Open source: github.com/VITA-MLLM/VITA-Audio
- **MGM-Omni** (2025): "Brain-mouth" dual-track. Brain reasons, mouth speaks. Available: wcy1122/MGM-Omni-TTS-2B
- **Voxtral-4B-TTS** (Mistral, 2026): 4B TTS model, 70ms latency, voice cloning
- **Phi-Omni-ST** (Microsoft): Audio transformer head on Phi-4, streaming vocoder

## Training Stages

### Stage 1: ASR + Text Understanding (CURRENT)
- Fine-tune Gemma 4 E2B with LoRA on Kikuyu audio + text tasks
- Model learns to hear and understand Kikuyu
- Data: 65K audio samples (122h) + 64K text tasks

### Stage 2: Speech Tokenizer
- Train or adapt a Kikuyu speech tokenizer (audio → discrete tokens)
- Options:
  - Use GLM-4-Voice tokenizer (pretrained, may work for Kikuyu)
  - Use SpeechTokenizer or EnCodec and fine-tune on Kikuyu audio
  - Use Whisper encoder features as pseudo-tokens
- Target: 12.5 Hz token rate (80 tokens per second of audio)
- Training data: Bible audio (80h, clean, single speaker) + Thiomi (8.6h)

### Stage 3: Single MCTP Module
- Add 1 lightweight transformer block after Gemma 4's last layer
- Freeze Gemma 4, train only the MCTP module
- Input: LLM hidden states + text token embedding
- Output: next audio token
- Initialize from Gemma 4's last layer weights
- Training: TTS data (text → audio token sequences)

### Stage 4: Scale to 10 MCTP Modules
- Add 9 more MCTP modules (cascaded)
- Each predicts the next audio token given previous MCTP's output
- Initialize all from Stage 3 weights
- Result: 10 audio tokens per LLM forward pass

### Stage 5: Speech-to-Speech Fine-tuning
- Train on speech QA pairs (Kikuyu question → Kikuyu answer)
- Interleaved text-audio output format
- Data: Course bilingual content, Mukuyu cultural Q&A, Bible Q&A

## Data Requirements

| Stage | Data | Hours | Source |
|-------|------|-------|--------|
| 1 (ASR) | Audio + transcripts | 122h | DigiGreen, Bible, radio, etc. |
| 2 (Tokenizer) | Clean audio | 80h+ | Bible, Thiomi, WAXAL |
| 3-4 (MCTP) | Text + audio tokens | 80h+ | Bible (single speaker ideal) |
| 5 (S2S) | Speech QA pairs | 10h+ | Course, Mukuyu, synthetic |

## Hardware

- MCTP modules: ~10M params each (tiny). Train on A10G.
- Gemma 4 stays frozen during Stages 3-5.
- Vocoder: Use existing (MMS-TTS-kik or gateremark/kikuyu-tts-v1) or train VITS.

## Vocoder Options

1. **GLM-4-Voice decoder** — used by VITA-Audio, converts audio tokens → waveform
2. **MMS-TTS-kik** (Meta) — pretrained Kikuyu, works today
3. **gateremark/kikuyu-tts-v1** — fine-tuned on 48.5K Kikuyu samples, better quality
4. **Train VITS on Bible audio** — best quality, single consistent voice

## Expected Outcome

A single model that:
- Hears Kikuyu speech input
- Understands, reasons, translates
- Speaks Kikuyu back in real-time (~50ms first audio token)
- Can teach Kikuyu to learners
- Can translate between Kikuyu, English, and Swahili
- Understands cultural context (proverbs, customs)

## Timeline

After ASR training completes:
- Stage 2 (tokenizer): 1-2 days
- Stage 3 (single MCTP): 1 day
- Stage 4 (10 MCTP): 1 day
- Stage 5 (S2S fine-tune): 1-2 days
- Total: ~1 week from ASR completion

## References

1. VITA-Audio: arxiv.org/abs/2505.03739
2. MGM-Omni: arxiv.org/abs/2509.25131
3. GLM-4-Voice: arxiv.org/abs/2412.02612
4. Freeze-Omni: arxiv.org/abs/2411.00774
5. Phi-Omni-ST: arxiv.org/abs/2506.04392

---
*Created: 2026-04-07*
