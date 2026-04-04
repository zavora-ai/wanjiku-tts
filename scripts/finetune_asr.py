"""
Fine-tune Meta MMS Kikuyu ASR adapter on WAXAL dataset.

Usage:
    python scripts/finetune_asr.py --config configs/config.yaml
    python scripts/finetune_asr.py --resume models/asr_checkpoints/checkpoint-5000
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_values, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def load_manifest(manifest_path, audio_dir):
    entries = [json.loads(l) for l in open(manifest_path)]
    return Dataset.from_dict({
        "audio": [str(audio_dir / e["audio"]) for e in entries],
        "text": [e["text"] for e in entries],
    }).cast_column("audio", Audio(sampling_rate=16000))


def prepare_batch(batch, processor, max_length=160000):
    audio = batch["audio"]
    # Cap at 10 seconds (160000 samples at 16kHz)
    array = audio["array"][:max_length]
    batch["input_values"] = processor(
        array, sampling_rate=16000, return_tensors="pt"
    ).input_values[0]
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MMS Kikuyu ASR")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["waxal_dir"])
    out_dir = Path("models/asr_checkpoints")

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kik")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all", target_lang="kik", ignore_mismatched_sizes=True
    )
    model.load_adapter("kik")

    # Freeze feature extractor, only train adapter + LM head
    model.freeze_feature_encoder()
    model.freeze_base_model()
    model.lm_head.requires_grad_(True)
    # Unfreeze adapter layers
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    print("Loading datasets...")
    train_ds = load_manifest(Path("data/transcripts/waxal_train.jsonl"), data_dir)
    val_ds = load_manifest(Path("data/transcripts/waxal_validation.jsonl"), data_dir)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_ds = train_ds.map(lambda b: prepare_batch(b, processor), remove_columns=["audio", "text"])
    val_ds = val_ds.map(lambda b: prepare_batch(b, processor), remove_columns=["audio", "text"])

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        warmup_steps=200,
        num_train_epochs=20,
        fp16=False,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out_dir / "final"))
    processor.save_pretrained(str(out_dir / "final"))
    print(f"Saved to {out_dir / 'final'}")

    # Sync to S3
    s3 = cfg.get("aws", {}).get("s3_bucket")
    if s3:
        import subprocess
        subprocess.run(["aws", "s3", "sync", str(out_dir), f"s3://{s3}/models/asr_checkpoints"], check=True)
        print(f"Synced to S3")


if __name__ == "__main__":
    main()
