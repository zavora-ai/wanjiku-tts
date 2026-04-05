# Wanjiku TTS — Data Inventory

Last updated: 2026-04-05

## Summary

| Source | Type | Size | Status | License |
|--------|------|------|--------|---------|
| Google WAXAL (kik_tts) | Audio + Text | 10.3h (2,026 clips) | ✅ Downloaded, on S3 | CC-BY-SA-4.0 |
| Radio: Kameme FM | Raw audio | ~20h (10 clips) | 🔄 Recording (2 more clips) | Internal |
| Radio: Inooro FM | Raw audio | ~16h (8 clips) | 🔄 Recording (4 more clips) | Internal |
| gateremark/kikuyu_conversations | Text only | 22,924 conversations | ✅ Access granted | Gated |
| gateremark/kikuyu-tts-training | Audio + Text | 48,500 samples | ⏳ Access requested | Gated |
| gateremark/english-kikuyu-translations | Text only | 30,400 pairs | ⏳ Access requested | Gated |
| gateremark/kikuyu-data | Text | Unknown | ⏳ Access requested | Gated |
| MCAA1-MSU/anv_data_ke (Kikuyu) | Audio + Text | 754h (381h transcribed) | ⏳ Access requested | CC-BY-4.0 |

---

## 1. Google WAXAL — Kikuyu TTS Subset

- **Source**: `google/WaxalNLP`, config `kik_tts`
- **Provider**: Loud and Clear (via Google Research)
- **Content**: Studio-quality single-speaker recordings reading scripted text
- **Topics**: Image descriptions, general knowledge, geography, sports, math
- **Speakers**: 8 unique (4 female, 4 male)
- **Audio**: 24kHz mono WAV
- **Diacritics**: Uses ū, ì (macron style), not ũ, ĩ (tilde style)

| Split | Samples | Duration |
|-------|---------|----------|
| train | 1,602 | 10.3h |
| validation | 210 | 1.3h |
| test | 214 | 1.3h |

**Location on EC2**: `~/wanjiku-tts/data/waxal_kikuyu/{train,validation,test}/`
**Location on S3**: `s3://wanjiku-tts-971994957690/data/waxal_kikuyu/`
**Manifests**: `data/transcripts/waxal_{train,validation,test}.jsonl`

**Limitations**:
- Image description text (not conversational)
- Limited topic diversity (15% about colors/shapes)
- Macron diacritics differ from common Kikuyu tilde convention

---

## 2. Radio Recordings — Kameme FM

- **Source**: `https://mediamax.api.radiosphere.io/channels/kameme/stream.aac`
- **Station**: Kameme FM 101.1 (Mediamax Network, Nairobi)
- **Language**: Kikuyu with Swahili/English code-switching
- **Content**: Talk shows, news, callers, music, ads, jingles
- **Recording**: 2-hour clips, 24kHz mono WAV
- **Schedule**: 12 clips × 2 hours = 24 hours target

| Status | Clips | Duration | Size |
|--------|-------|----------|------|
| Recorded | 10 | ~20h | 2.7GB |
| Remaining | 2 | ~4h | — |

**Location on EC2**: `~/wanjiku-tts/data/radio_raw/`

---

## 3. Radio Recordings — Inooro FM

- **Source**: `https://inoorofm-atunwadigital.streamguys1.com/inoorofm`
- **Station**: Inooro FM (Royal Media Services, Nairobi)
- **Language**: Kikuyu with Swahili/English code-switching
- **Content**: Talk shows, news, callers, music, ads, jingles
- **Recording**: 2-hour clips, 24kHz mono WAV

| Status | Clips | Duration | Size |
|--------|-------|----------|------|
| Recorded | 8 | ~16h | 2.6GB |
| Remaining | 4 | ~8h | — |

**Location on EC2**: `~/wanjiku-tts/data/radio_raw_inooro/`

---

## 4. gateremark/kikuyu_conversations_unreviewed

- **Source**: HuggingFace `gateremark/kikuyu_conversations_unreviewed`
- **Type**: Text-only conversations (no audio)
- **Size**: 22,924 conversations, 3 turns each (system/user/assistant)
- **Format**: Chat format with Kikuyu text, proper tilde diacritics (ũ, ĩ)

| Category | Count | Example Topics |
|----------|-------|----------------|
| daily_life | 6,045 | Greetings, Weather, Food, Family |
| practical | 6,034 | M-Pesa, Health, Education, Sports |
| modern_life | 6,010 | Technology, current affairs |
| culture | 4,835 | Traditions, Ceremonies, Music, Storytelling |

**Use for TTS**: Text prompts for synthesis, text normalization validation
**Status**: ✅ Access granted

---

## 5. gateremark/kikuyu-tts-training (Pending)

- **Source**: HuggingFace `gateremark/kikuyu-tts-training`
- **Size**: 48,500 samples
- **Type**: Likely audio + text pairs (used to train kikuyu-tts-v1)
- **Status**: ⏳ Access requested

---

## 6. gateremark/english-kikuyu-translations (Pending)

- **Source**: HuggingFace `gateremark/english-kikuyu-translations`
- **Size**: 30,400 parallel sentence pairs
- **Type**: English ↔ Kikuyu text translations
- **Use**: Generate Kikuyu text for TTS, evaluation
- **Status**: ⏳ Access requested

---

## 7. MCAA1-MSU/anv_data_ke — African Next Voices Kenya (Pending)

- **Source**: HuggingFace `MCAA1-MSU/anv_data_ke`
- **Provider**: KenCorpus Consortium (DeKUT, LDRI, Maseno, USIU, Kabarak)
- **Funding**: Gates Foundation
- **Type**: Scripted + unscripted speech with transcriptions

| Metric | Value |
|--------|-------|
| Total Kikuyu hours | 754h |
| Scripted | 183h |
| Unscripted | 571h (381h transcribed) |
| Dialects | Gĩ-Kabete, Ki-Mathira, Ki-Muranga, Ki-Ndia, Gĩ-Gichugu |
| Domains | Agriculture, Healthcare, News, Education, Finance, Customer Care |

**This is the most important pending dataset.** 754h of Kikuyu with 5 dialects and real-world domains. Would transform both ASR and TTS quality.

- **Status**: ⏳ Access requested
- **License**: CC-BY-4.0

---

## Models Evaluated

| Model | Task | Params | Kikuyu Quality | Notes |
|-------|------|--------|----------------|-------|
| **MaryWambo/whisper-base-kikuyu4** | ASR | 73M | ★★★★ Best | Proper diacritics, trained on MCAA1-MSU data |
| Meta MMS (kik adapter) | ASR | 1B | ★★☆ | Messy but recognizable |
| Google Chirp 2 (sw-KE) | ASR | — | ★★☆ | Mixes Swahili/Kikuyu |
| Whisper large-v3 | ASR | 1.5B | ★☆☆ | No Kikuyu, hallucinates |
| Gemma 4 E2B (zero-shot) | ASR | 2.3B | ★★★ | Decent, mixed languages |
| facebook/mms-tts-kik | TTS | 36M | ★☆☆ | Terrible quality |
| BrianMwangi/African-Kikuyu-TTS | TTS | 36M | ★☆☆ | Same as above |
| gateremark/kikuyu-tts-v1 | TTS | 36M | ★☆☆ | Same architecture |
| **Qwen3-TTS-12Hz-1.7B-Base** | TTS | 1.7B | ★★★★ Target | Voice cloning works, needs fine-tuning |

---

## Pipeline Tools

| Stage | Tool | Status |
|-------|------|--------|
| Speaker diarization | SpeechBrain ECAPA-TDNN | ✅ Tested |
| ASR transcription | MaryWambo/whisper-base-kikuyu4 | ✅ Tested, best quality |
| Audio cleanup | noisereduce | ✅ Tested |
| Loudness normalization | pyloudnorm (EBU R128) | ✅ In pipeline |
| TTS base model | Qwen3-TTS-12Hz-1.7B-Base | ✅ Tested, voice cloning works |

---

## AWS Infrastructure

| Resource | Details |
|----------|---------|
| EC2 Instance | i-076aaa7423b670d2a (g5.2xlarge, A10G 24GB) |
| IP | 13.217.62.47 |
| S3 Bucket | s3://wanjiku-tts-971994957690 |
| Region | us-east-1 |
| GCS Bucket | gs://zavora-wanjiku-tts (for Google STT tests) |

---

## Next Steps

1. **Wait for dataset access**: MCAA1-MSU (754h) and gateremark TTS training data (48.5k)
2. **Complete radio recordings**: ~6h remaining across both stations
3. **Run processing pipeline**: Diarize → Filter → Clean → Transcribe → Normalize
4. **Fine-tune Qwen3-TTS**: Phase 1 on WAXAL + processed radio data
5. **Evaluate**: Compare against existing Kikuyu TTS models
