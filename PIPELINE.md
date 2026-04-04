# Radio Audio Processing Pipeline

## Overview

Process 48+ hours of raw radio recordings into clean, labeled, single-speaker Kikuyu speech segments for TTS training.

```
Raw 2hr WAV ──▶ Music/Speech Split ──▶ VAD ──▶ Speaker Diarization ──▶ Content Classifier ──▶ Quality Filter ──▶ Transcribe ──▶ Training Data
```

## Pipeline Stages

### Stage 1: Music/Speech Separation (Demucs)

Separate vocal track from music beds, jingles, background audio.

```
Input:  2hr WAV (24kHz mono)
Output: vocals.wav, background.wav
Tool:   Demucs htdemucs
```

- Splits each 2hr file into vocal-isolated and background tracks
- Process in 30-second chunks to manage memory
- Keep both tracks — background useful for jingle detection

### Stage 2: Voice Activity Detection (Silero VAD)

Find speech boundaries in the vocal track.

```
Input:  vocals.wav
Output: segments.json [{start: 0.0, end: 5.2}, ...]
Tool:   Silero VAD
```

- Min speech duration: 1.0s (skip very short utterances)
- Min silence duration: 0.5s (merge close segments)
- Output: timestamped speech regions

### Stage 3: Speaker Diarization (pyannote)

Identify who is speaking in each segment.

```
Input:  vocals.wav + segments.json
Output: diarized.json [{start, end, speaker_id}, ...]
Tool:   pyannote.audio
```

- Cluster speakers across the full recording
- Label: presenter_1, presenter_2, caller_1, etc.
- Single-speaker segments are highest priority for TTS

### Stage 4: Content Classification

Classify each speech segment by content type.

```
Input:  diarized segments
Output: classified.json [{start, end, speaker, type, language, confidence}, ...]
```

Classification categories:

| Type | Characteristics | TTS Value |
|------|----------------|-----------|
| news | Formal tone, single speaker, structured | ★★★★★ |
| story | Narrative, single speaker, expressive | ★★★★★ |
| monologue | Presenter solo, varied topics | ★★★★ |
| conversation | 2+ speakers, turn-taking | ★★★ (extract turns) |
| caller | Phone audio quality, echo | ★ (exclude) |
| ad | Different production, often English/Swahili | ✗ (exclude) |
| jingle | Short, musical, station ID | ✗ (exclude) |
| music | Songs, instrumentals | ✗ (exclude) |

Detection approach:
- **Language**: Whisper language detection per segment (keep Kikuyu, flag English/Swahili)
- **Audio quality**: SNR estimation — phone callers have lower SNR
- **Speaker count**: From diarization — single speaker = monologue/news/story
- **Duration**: Jingles < 10s, ads 15-60s, news/stories > 60s
- **Spectral features**: Music has wider frequency spread, speech is narrower
- **Production style**: Ads have compression/EQ differences detectable via spectral analysis

### Stage 5: Quality Filter

Keep only segments suitable for TTS training.

```
Criteria:
  ✓ Single speaker (from diarization)
  ✓ Kikuyu language (from Whisper lang detect)
  ✓ SNR > 20dB
  ✓ Duration 3-30 seconds (segment further if longer)
  ✓ No background music (from Demucs separation quality)
  ✓ Not a phone caller (audio quality check)
  ✓ Not an ad or jingle (content classifier)
```

### Stage 6: Segmentation

Split long clean segments into TTS-sized clips.

```
Input:  filtered segments (some > 30s)
Output: 3-15 second clips at sentence boundaries
Tool:   silence detection + punctuation from transcription
```

- Use silence gaps (>300ms) as natural break points
- Target 5-15 seconds per clip
- Preserve sentence boundaries where possible

### Stage 7: Transcription (Whisper)

Generate text transcriptions for each clip.

```
Input:  clean clips
Output: manifest.jsonl with text
Tool:   Whisper large-v3 (has Kikuyu support)
```

- Bootstrap with Whisper automatic transcription
- Flag low-confidence segments for manual review
- Preserve Kikuyu diacritics (ĩ, ũ)

### Stage 8: Loudness Normalization

Final audio normalization for training consistency.

```
Input:  transcribed clips
Output: normalized clips at -23 LUFS
Tool:   pyloudnorm
```

---

## Output Structure

```
data/
├── radio_processed/
│   ├── kameme/
│   │   ├── news/
│   │   │   ├── KAM_NEWS_00001.wav
│   │   │   └── ...
│   │   ├── story/
│   │   ├── monologue/
│   │   └── conversation/
│   └── inooro/
│       ├── news/
│       ├── story/
│       ├── monologue/
│       └── conversation/
├── radio_excluded/
│   ├── callers/
│   ├── ads/
│   ├── jingles/
│   └── music/
└── transcripts/
    ├── radio_train.jsonl
    └── radio_stats.json
```

Manifest format:
```json
{"audio": "kameme/news/KAM_NEWS_00001.wav", "text": "...", "speaker": "presenter_1", "type": "news", "duration": 8.3, "source": "kameme", "snr": 32.5, "language": "kik", "confidence": 0.94}
```

---

## Dependencies

```
pip install demucs pyannote.audio openai-whisper silero-vad resemblyzer
```

Note: pyannote.audio requires a HuggingFace token with access to pyannote models.

---

## Estimated Yield

From 48 hours of raw radio audio:

| Stage | Input | Output | Loss |
|-------|-------|--------|------|
| Music/speech split | 48h | ~30h speech | ~40% music |
| VAD | 30h | ~25h voiced | ~15% silence |
| Single speaker filter | 25h | ~15h | ~40% multi-speaker |
| Quality + language filter | 15h | ~10h | ~30% callers/ads/low quality |
| Segmentation | 10h | ~10h (3-15s clips) | minimal |

**Expected yield: ~10 hours of clean, single-speaker Kikuyu speech** from 48 hours of raw radio. Combined with 10.3h WAXAL = ~20h total training data.

---

## Processing Time Estimate (g5.2xlarge)

| Stage | Tool | Time for 48h audio |
|-------|------|-------------------|
| Demucs | GPU | ~4 hours |
| Silero VAD | CPU | ~30 min |
| Diarization | GPU | ~2 hours |
| Classification | CPU+GPU | ~1 hour |
| Whisper transcription | GPU | ~3 hours |
| Other (filter, segment, normalize) | CPU | ~30 min |
| **Total** | | **~11 hours** |
