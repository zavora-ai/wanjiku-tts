"""Fine-tune Gemma 4 E2B for Kikuyu ASR using Unsloth.

Based on the official Unsloth Gemma4_(E2B)-Audio.ipynb pattern.
Loads DigiGreen + WAXAL manifests, trains LoRA on audio→text.
"""
import os, json, torch
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "unsloth/gemma-4-E2B-it"
MANIFEST_DIR = os.path.expanduser("~/wanjiku-tts/data/manifests/combined")
OUTPUT_DIR = os.path.expanduser("~/wanjiku-tts/models/gemma4_kikuyu_asr_v2")
MAX_SEQ_LENGTH = 8192
TARGET_SR = 16000
INSTRUCTION = "Transcribe this Kikuyu speech accurately. Output only the transcription."

# Training hyperparams
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5
MAX_STEPS = 12000      # ~1 epoch over 44K samples (44072 / 4 = 11018)
WARMUP_RATIO = 0.03
LORA_R = 32
LORA_ALPHA = 64
LOGGING_STEPS = 100
SAVE_STEPS = 2000

# ── Load model ──────────────────────────────────────────────────
print("Loading Gemma 4 E2B...")
from unsloth import FastModel

model, processor = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    dtype=None,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
)

# ── Apply LoRA ──────────────────────────────────────────────────
print("Applying LoRA...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    random_state=42,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "post", "linear_start", "linear_end", "embedding_projection",
    ],
)

# ── Load data ───────────────────────────────────────────────────
def load_manifest(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def load_audio(path, target_sr=TARGET_SR):
    """Load audio file and resample to target_sr."""
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def convert_to_conversation(item):
    """Convert manifest item to Gemma 4 chat format."""
    audio_array = load_audio(item["audio_path"])
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["text"]}],
            },
        ]
    }

print("Loading training data...")
train_path = os.path.join(MANIFEST_DIR, "train.jsonl")
train_items = load_manifest(train_path)
print(f"  Train samples: {len(train_items)}")

# Convert to chat format (lazy — load audio on the fly during collation)
# For large datasets, we convert in batches to avoid OOM
print("Converting to conversation format...")
converted_dataset = []
errors = 0
for i, item in enumerate(train_items):
    if i % 2000 == 0 and i > 0:
        print(f"  Converted {i}/{len(train_items)}...")
    try:
        converted_dataset.append(convert_to_conversation(item))
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  Error on {item['audio_path']}: {e}")
print(f"  Converted: {len(converted_dataset)}, Errors: {errors}")

# ── Train ───────────────────────────────────────────────────────
print("Setting up trainer...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        warmup_ratio=WARMUP_RATIO,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    ),
)

print(f"Starting training: {MAX_STEPS} steps, batch={BATCH_SIZE}×{GRAD_ACCUM}...")
trainer_stats = trainer.train()
print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

# ── Save ────────────────────────────────────────────────────────
print("Saving LoRA adapters...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))

print("Saving merged model (float16)...")
model.save_pretrained_merged(
    os.path.join(OUTPUT_DIR, "merged"),
    processor,
    save_method="merged_16bit",
)

print(f"Done! Model saved to {OUTPUT_DIR}")

# ── Quick eval ──────────────────────────────────────────────────
print("\n=== Quick evaluation ===")
dev_path = os.path.join(MANIFEST_DIR, "dev.jsonl")
if os.path.exists(dev_path):
    dev_items = load_manifest(dev_path)[:5]
    for item in dev_items:
        audio_array = load_audio(item["audio_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=False, do_sample=False)
        pred = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"  REF:  {item['text'][:80]}")
        print(f"  PRED: {pred[:80]}")
        print()
