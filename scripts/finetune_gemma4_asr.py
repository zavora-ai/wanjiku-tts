"""
Fine-tune Gemma 4 E2B for Kikuyu ASR using LoRA on WAXAL dataset.

Usage:
    python scripts/finetune_gemma4_asr.py
    python scripts/finetune_gemma4_asr.py --resume models/gemma4_asr/checkpoint-500
"""
import argparse
import json
from pathlib import Path

import torch
import soundfile as sf
import librosa
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "google/gemma-4-e2b-it"
MAX_AUDIO_SEC = 30
PROMPT = (
    "Transcribe the following speech segment in Kikuyu into Kikuyu text. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits."
)


def load_waxal(manifest_path, audio_dir):
    entries = [json.loads(l) for l in open(manifest_path)]
    return [{"audio_path": str(audio_dir / e["audio"]), "text": e["text"]} for e in entries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    out_dir = Path("models/gemma4_asr")

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    # LoRA config — target the inner linear layers inside ClippableLinear
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj.linear", "v_proj.linear"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # Load data
    print("Loading WAXAL data...")
    train_data = load_waxal("data/transcripts/waxal_train.jsonl", Path("data/waxal_kikuyu"))
    val_data = load_waxal("data/transcripts/waxal_validation.jsonl", Path("data/waxal_kikuyu"))
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Preprocess: build input/label tensors
    def preprocess(example):
        audio, sr = sf.read(example["audio_path"], dtype="float32")
        # Cap audio length
        max_samples = MAX_AUDIO_SEC * sr
        audio = audio[:max_samples]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write("/tmp/_tmp_audio.wav", audio, 16000)

        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": "/tmp/_tmp_audio.wav"},
                {"type": "text", "text": PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": example["text"]},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask everything except the assistant response
        # Find where assistant response starts
        assistant_tokens = processor.tokenizer.encode(example["text"], add_special_tokens=False)
        resp_len = len(assistant_tokens)
        labels[:-resp_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

    print("Preprocessing train set...")
    train_processed = []
    skipped = 0
    for ex in train_data:
        try:
            train_processed.append(preprocess(ex))
        except Exception as e:
            skipped += 1
    print(f"  {len(train_processed)} samples ready, {skipped} skipped")

    print("Preprocessing val set...")
    val_processed = []
    for ex in val_data:
        try:
            val_processed.append(preprocess(ex))
        except Exception:
            pass
    print(f"  {len(val_processed)} samples ready")

    # Custom collator for variable-length sequences
    def collator(features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - f["input_ids"].shape[0]
            batch["input_ids"].append(torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in batch.items()}

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=250,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=collator,
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved to {out_dir / 'final'}")


if __name__ == "__main__":
    main()
