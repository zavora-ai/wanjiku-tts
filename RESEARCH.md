# Kikuyu Speech Technology Research

Reference documentation for the Wanjiku TTS project — linguistic foundations, known blockers, prior work, and technical approach.

## 1. Kikuyu Language Overview

- **ISO 639-3:** kik
- **Speakers:** ~8.1 million (largest ethnic group in Kenya)
- **Family:** Niger-Congo > Bantu (E51)
- **Dialects:** Southern Gikuyu, Ndia, Gichugu, Mathira, Northern Gikuyu
- **Writing system:** Latin script (African Reference Alphabet, UNESCO 1978)
- **Notable literature:** Ngũgĩ wa Thiong'o — oldest and most extensive written literature of any East African language besides Swahili

## 2. Phonology

### 2.1 Vowels (7 phonemes)

Kikuyu has 7 vowel phonemes mapped to the Latin alphabet with diacritics:

| Phoneme (IPA) | Orthography | Description | Example |
|---------------|-------------|-------------|---------|
| /i/ | i | Close front | *gĩthĩ* |
| /e/ | ĩ | Close-mid front | *mũndũ* |
| /ɛ/ | e | Open-mid front | *mega* |
| /a/ | a | Open central | *baba* |
| /ɔ/ | o | Open-mid back | *moko* |
| /o/ | ũ | Close-mid back | *gũkũ* |
| /u/ | u | Close back | *guru* |

**Critical:** ĩ ≠ i and ũ ≠ u — these are distinct phonemes, not decorative diacritics. Confusing them changes word meaning.

**Vowel length** is contrastive. Long vowels are written as doubled letters (aa, ee, ii, ĩĩ, oo, ũũ, uu).

**Nasalization** exists but is not the function of the tilde in standard orthography — the tilde on ĩ/ũ marks vowel *quality* (close-mid), not nasalization.

*Source: Armstrong (1940/2017), mustgo.com/worldlanguages/kikuyu/*

### 2.2 Consonants (18 phonemes)

| Phoneme | Orthography | Notes |
|---------|-------------|-------|
| /t/ | t | Voiceless alveolar stop |
| /k/ | k | Voiceless velar stop |
| /ᵐb/ | mb | Prenasalized bilabial stop |
| /ⁿd/ | nd | Prenasalized alveolar stop |
| /ᵑg/ | ng | Prenasalized velar stop |
| /ⁿdʒ/ | nj | Prenasalized postalveolar affricate |
| /β/ | b | Bilabial fricative (not a stop) |
| /ð/ | th | Voiced interdental fricative |
| /ɣ/ | g | Voiced velar fricative (not a stop) |
| /h/ | h | Glottal fricative |
| /ʃ/ | c | Voiceless postalveolar fricative |
| /m/ | m | Bilabial nasal |
| /n/ | n | Alveolar nasal |
| /ɲ/ | ny | Palatal nasal |
| /ŋ/ | ng' | Velar nasal (distinct from prenasalized ng) |
| /r/ | r | Alveolar flap/trill |
| /w/ | w | Bilabial semivowel |
| /j/ | y | Palatal semivowel |

**Key digraphs:** mb, nd, ng, ng', nj, ny, th — these are single phonemes written as two letters.

### 2.3 Tone

Kikuyu has **two tones**: High (H) and Low (L).

- Tone is **lexically contrastive** — same segmental sequence with different tone = different word
- Tone is **NOT marked** in standard orthography
- Armstrong (1940) documented 5 tonal verb classes and 3+ noun tonal classes
- Floating tones exist at phrasal boundaries (Phonological Phrases in Kikuyu, ResearchGate 2016)

**Implication for TTS:** Tone must be learned implicitly from audio. Self-supervised speech models (wav2vec2, HuBERT) encode lexical tone even when trained on non-tonal languages (arxiv 2403.16865, NAACL 2024). Qwen3-TTS voice cloning learns prosody from reference audio, naturally capturing tone.

**Implication for ASR:** Tone distinctions are present in audio but cannot be represented in standard text output. ASR systems must rely on context for disambiguation.

### 2.4 Morphology

Kikuyu is **agglutinative** (typical Bantu):
- 10 noun classes with singular/plural prefix pairs
- Verbs built from root + multiple suffixes (subject marker, tense/aspect, mood, extensions)
- Fine-grained tense distinctions (recent past vs. remote past vs. today's past)
- High out-of-vocabulary rates for word-level models

**Implication:** Use subword tokenization (BPE/SentencePiece) rather than word-level for any text processing.

## 3. Orthographic Challenges

### 3.1 The Diacritic Problem

Three competing conventions exist in written Kikuyu:

| Convention | ĩ/ũ representation | Used by |
|------------|-------------------|---------|
| **Tilde** (standard) | ĩ, ũ | Modern publications, Bible Society of Kenya, DigiGreen dataset |
| **Macron** | ī, ū | WAXAL dataset, some academic texts |
| **Bare** (no diacritics) | i, u | Informal digital writing, SMS, social media |

The bare form is **lossy** — it merges 7 vowels into 5, creating ambiguity. The tilde and macron forms are informationally equivalent.

### 3.2 Normalization Strategy

```
Canonical form: tilde (ĩ, ũ) — the UNESCO standard

Macron → Tilde:  ī→ĩ, ū→ũ, Ī→Ĩ, Ū→Ũ
                  ì→ĩ, ù→ũ (grave accent variants)
                  í→ĩ, ú→ũ (acute accent variants, if used for vowel quality)

Bare → Ambiguous: Cannot reliably recover ĩ/ũ from bare i/u without a lexicon.
                   For ASR output, use a dictionary-based post-processor.
```

### 3.3 Additional Text Normalization

- Number words (Kikuyu numerals)
- Loanwords from Swahili and English (common in radio speech)
- Code-switching (Kikuyu-English, Kikuyu-Swahili) — prevalent in radio broadcasts
- Punctuation and sentence boundaries

## 4. Known Blockers for Kikuyu Speech Technology

| Blocker | Severity | Status in our project |
|---------|----------|----------------------|
| Data scarcity | Critical | **Addressed** — 244h+ audio collected |
| Orthographic inconsistency | High | Normalization rules defined above |
| Unmarked tone | High | Implicit learning via audio codec |
| No G2P system | Medium | Feasible as rule-based (phonemic orthography) |
| No pronunciation dictionary | Medium | Can bootstrap from spell-checker (19K words) |
| Morphological complexity | Medium | Subword tokenization |
| No research community | Low | We are the research community now |
| Dialectal variation | Low | Multi-dialect data from 3 radio stations + WAXAL |

## 5. Prior Work on Kikuyu NLP/Speech

### 5.1 Datasets involving Kikuyu

| Dataset | Type | Kikuyu content | Reference |
|---------|------|----------------|-----------|
| BibleTTS | Speech | 90.6h unaligned audio | Meyer et al. (2022), Interspeech |
| Google WAXAL | Speech | 10.3h, 2026 clips, 8 speakers | Google (2026) |
| DigiGreen/CGIAR | Speech | ~62h agriculture queries | Digital Green / Karya (2024) |
| SIB-200 | Text | Topic classification sentences | Adelani et al. (2023) |
| FLORES-200 | Text | 3001 translated sentences | Goyal et al. (2022) |
| NLLB | Text | Translation model includes Kikuyu | Costa-jussà et al. (2022) |
| WikiAnn | Text | ~1K named entities | Pan et al. (2017) |
| Bloom Library | Text | Children's stories | Leong et al. (2022) |
| African Storybook | Text | Stories in Gikuyu | Stranger-Johannessen & Norton (2017) |
| Leipzig Wortschatz | Text | 815 web-crawled sentences | University of Leipzig (2017) |
| Spell-checker corpus | Text | 19K words, POS-tagged | Chege et al. (2010), UoN |

### 5.2 Models involving Kikuyu

| Model | Task | Quality | Reference |
|-------|------|---------|-----------|
| facebook/mms-tts-kik | TTS | Very poor (36M VITS, 16kHz) | Meta MMS (2023) |
| facebook/mms-1b + kik adapter | ASR | Messy but recognizable | Meta MMS (2023) |
| MaryWambo/whisper-base-kikuyu4 | ASR | Best available for pure Kikuyu | DeKUT-DSAIL |
| NLLB-200 | Translation | Includes kik↔eng | Meta (2022) |
| SERENGETI | Language model | Includes Gikuyu | Adebara et al. (2022) |
| starnleymbote/Kikuyu_Kiswahili-translation | Translation | OpenNMT Kikuyu↔Swahili | GitHub |
| gateremark/kikuyu-tts-v1 | TTS | Poor quality | HuggingFace (2025) |
| CGIAR/KikuyuASR_agri_queries | ASR | wav2vec2, agriculture domain | CGIAR/DigiGreen |

### 5.3 Key Papers

**Directly relevant:**
- Armstrong, L.A. (1940/2017). *The Phonetic and Tonal Structure of Kikuyu*. Routledge. — The definitive reference on Kikuyu phonology and tone. ISBN 9781138098244.
- Meyer, J. et al. (2022). "BibleTTS: a large, high-fidelity, multilingual, and uniquely African speech corpus." *Interspeech*. arXiv:2207.03546.
- Amol, C.J. et al. (2024). "State of NLP in Kenya: A Survey." arXiv:2410.09948. — Comprehensive survey showing Kikuyu's underrepresentation.
- Chege, K. et al. (2010). "Developing an Open Source Spell-checker for Gĩkuyu." *AfLaT 2010*.

**African ASR/TTS:**
- Imam, S.H. et al. (2025). "Automatic Speech Recognition for African Low-Resource Languages: Challenges and Future Directions." arXiv:2505.11690.
- Ogayo, P. et al. (2022). "Building African Voices." arXiv:2207.00688.
- Ogun, S. et al. (2024). "1000 African Voices." arXiv:2406.11727.

**Tone in speech models:**
- Zuluaga-Gomez, J. et al. (2024). "Encoding of lexical tone in self-supervised models of spoken language." *NAACL 2024*. arXiv:2403.16865. — Shows SSL models encode tone even from non-tonal training data.
- Chen, Y. et al. (2025). "Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages." arXiv:2601.09050.
- Li, J. et al. (2024). "A Speech Discretization Approach for Tonal Language Speech Synthesis." arXiv:2406.08989.

**Orthography and diacritics:**
- "Dire — Critical: The intrigues of vowels and diacritics in writing Gĩkũyũ and other Eastern Bantu Languages." Medium/@fourtharchetype (2025). — Explains the 7-vowel system and diacritic conventions.

**Low-resource TTS:**
- Pratap, V. et al. (2023). "Scaling Speech Technology to 1000+ Languages." arXiv:2305.13516. — MMS project, includes Kikuyu.
- Black, A.W. (2019). "CMU Wilderness Multilingual Speech Dataset." *ICASSP*. — Bible-based multilingual speech (does NOT include Kikuyu).

## 6. G2P Rules for Kikuyu

Kikuyu orthography is largely **phonemic** — each grapheme maps predictably to a phoneme. A rule-based G2P is feasible:

### 6.1 Vowel mapping
```
a  → /a/      aa → /aː/
e  → /ɛ/      ee → /ɛː/
i  → /i/      ii → /iː/
ĩ  → /e/      ĩĩ → /eː/
o  → /ɔ/      oo → /ɔː/
u  → /u/      uu → /uː/
ũ  → /o/      ũũ → /oː/
```

### 6.2 Consonant mapping (process digraphs first)
```
mb → /ᵐb/     nd → /ⁿd/
ng'→ /ŋ/      ng → /ᵑg/    (ng' before ng to avoid ambiguity)
nj → /ⁿdʒ/   ny → /ɲ/
th → /ð/
b  → /β/      c  → /ʃ/
g  → /ɣ/      h  → /h/
k  → /k/      m  → /m/
n  → /n/      r  → /r/
t  → /t/      w  → /w/
y  → /j/
```

### 6.3 Processing order
1. Normalize diacritics (macron → tilde)
2. Handle digraphs (longest match first: ng' > ng > nj > ny > nd > mb > th)
3. Map remaining single characters
4. Handle vowel length (doubled vowels → long phoneme)

## 7. Data Inventory (as of 2026-04-05)

### 7.1 Audio datasets on EC2

| Dataset | Location | Hours | Files | Size | Transcripts | License |
|---------|----------|-------|-------|------|-------------|---------|
| Biblica Kikuyu Bible | data/bible_audio/mp3s/ | ~113h | 1,189 MP3 | 1.3GB | USFM (data/bible_audio/text/) | CC BY-SA 4.0 |
| DigiGreen Kikuyu ASR | data/digigreen/audio/KikuyuASR/dg_16/ | ~62h | 25,379 WAV | 6.4GB | CSV (data/digigreen/digital_green_recordings.csv) | Apache 2.0 |
| Radio Kameme FM | data/radio_raw/ | ~20h+ | 2hr clips | — | Needs ASR pipeline | — |
| Radio Inooro FM | data/radio_raw_inooro/ | ~16h+ | 2hr clips | — | Needs ASR pipeline | — |
| Radio Gukena FM | data/radio_raw_gukena/ | ~1h+ | Recording | — | Needs ASR pipeline | — |
| GRN 5fish programs | data/grn_kikuyu/ | 15.9h | 15 MP3 | 919MB | Needs ASR/alignment | CC (GRN terms) |
| Google WAXAL kik_tts | data/waxal_kikuyu/ | 10.3h | 2,026 WAV | — | Included in dataset | CC BY 4.0 |
| Kikuyu course videos | data/kikuyu_course/ | ~4.8h | 21 MP4 | — | Partial (data/kikuyu_course_transcripts/) | — |
| Kikuyu Gospel songs | data/archive_org/kikuyu_gospel/ | 2.0h | 20 MP3 | 109MB | None (music/singing) | — |
| UCLA Phonetics | data/ucla_phonetics/ | 8 min | 1 WAV | 42MB | Word list (HTML) | Academic |
| Lomax/Rosetta | data/archive_org/ | 5 min | 1 WAV | 59MB | None | Public domain |
| **Total** | | **~244h+** | | | | |

### 7.2 Text datasets on EC2

| Dataset | Location | Size | Notes |
|---------|----------|------|-------|
| Biblica Kikuyu Bible (USFM) | data/bible_audio/text/release/USX_1/ | 66 files | Verse-level markup, matches audio |
| DigiGreen transcripts | data/digigreen/digital_green_recordings.csv | 26,483 rows | $-separated, path+transcript |
| Leipzig Wortschatz | data/leipzig_text/ | 815 sentences, 14K words | Web-crawled 2017 |
| WAXAL transcripts | (in dataset) | 2,026 entries | Image descriptions |

### 7.3 Pending access

| Dataset | Size | Status | Importance |
|---------|------|--------|------------|
| MCAA1-MSU/anv_data_ke | 754h, 5 Kikuyu dialects | Access requested | Critical — largest Kikuyu speech dataset |
| gateremark/kikuyu-tts-training | 48.5K samples | Access requested | High — TTS training data |
| gateremark/kikuyu-data | Unknown | Access requested | Unknown |
| gateremark/english-kikuyu-translations | 30.4K pairs | Access requested | Medium — text pairs |

## 8. Technical Approach

### 8.1 TTS Pipeline
1. **Text normalization** → canonical Kikuyu orthography (tilde diacritics)
2. **G2P** → phoneme sequence (rule-based, phonemic orthography)
3. **Qwen3-TTS-12Hz-1.7B-Base** → fine-tune with voice cloning on paired audio+text
4. Tone learned implicitly from audio codec representations

### 8.2 ASR Pipeline (for data processing)
1. **MaryWambo/whisper-base-kikuyu4** → best for pure Kikuyu radio content
2. **microsoft/paza-whisper-large-v3-turbo** → bilingual content
3. **openai/whisper-large-v3-turbo** → English content (course videos)

### 8.3 Audio Processing Pipeline
1. SpeechBrain diarization → speaker segmentation
2. Noise reduction (noisereduce)
3. ASR transcription
4. Loudness normalization (pyloudnorm)
5. Forced alignment (MFA or MMS) for Bible audio → verse-level segments

---

*Last updated: 2026-04-05*
*Project: Wanjiku TTS — Kikuyu Text-to-Speech*
