"""
Fine-tune Gemma 4 E2B for Kikuyu ASR using Unsloth.
Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Audio.ipynb

Usage:
    pip install unsloth torchcodec timm
    python scripts/finetune_gemma4_unsloth.py
"""
import json, torch
from pathlib import Path
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Audio

# Load model
model, processor = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    max_seq_length=4096,
    load_in_4bit=True,
    full_finetuning=False,
)

# Add LoRA
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "post", "linear_start", "linear_end",
        "embedding_projection",
    ],
)

# Load WAXAL Kikuyu dataset - use soundfile instead of torchcodec
print("Loading WAXAL Kikuyu data...")
import soundfile as sf
import numpy as np

def load_manifest(path, audio_dir):
    entries = [json.loads(l) for l in open(path)]
    samples = []
    for e in entries:
        audio, sr = sf.read(str(audio_dir / e["audio"]), dtype="float32")
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Cap at 30 seconds
        audio = audio[:16000 * 30]
        samples.append({"audio_array": audio, "text": e["text"]})
    return samples

train_data = load_manifest("data/transcripts/waxal_train.jsonl", Path("data/waxal_kikuyu"))
print(f"Train: {len(train_data)} samples")

# Convert to conversation format
instruction = (
    "Transcribe the following speech segment in Kikuyu into Kikuyu text. "
    "Only output the transcription, with no newlines."
)

def convert_to_conversation(sample):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "audio", "audio": sample["audio_array"]},
            {"type": "text", "text": instruction},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": sample["text"]},
        ]},
    ]}

print("Converting dataset...")
converted = [convert_to_conversation(s) for s in train_data]
print(f"Converted {len(converted)} samples")

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=converted,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="models/gemma4_kikuyu_asr",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=4096,
    ),
)

print(f"GPU = {torch.cuda.get_device_properties(0).name}")
print(f"Memory reserved: {torch.cuda.max_memory_reserved()/1e9:.1f} GB")
print("Training...")
trainer.train()

# Save
model.save_pretrained("models/gemma4_kikuyu_asr/final")
processor.save_pretrained("models/gemma4_kikuyu_asr/final")
print("Saved to models/gemma4_kikuyu_asr/final")
