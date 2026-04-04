"""Fine-tune Qwen3-TTS on Kikuyu speech data."""
import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoProcessor, TrainingArguments, Trainer
from datasets import load_from_disk

from qwen_tts.core.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.processing_qwen3_tts import Qwen3TTSProcessor


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS for Kikuyu")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="1=WAXAL adaptation, 2=broadcast voice style")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    phase_cfg = cfg["training"][f"phase{args.phase}"]
    model_name = args.resume or cfg["model"]["base"]

    print(f"Phase {args.phase}: Fine-tuning {model_name}")
    print(f"Config: {phase_cfg}")

    # Load dataset
    if args.phase == 1:
        data_path = Path(cfg["data"]["waxal_dir"]) / "train"
    else:
        data_path = Path(cfg["data"]["radio_clean_dir"])

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run download_waxal.py (phase 1) or clean_audio.py (phase 2) first.")
        return

    dataset = load_from_disk(str(data_path))
    print(f"Loaded {len(dataset)} samples")

    # Register Qwen3-TTS model types
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    # Load model and processor
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    output_dir = Path(cfg["model"]["checkpoint_dir"]) / f"phase{args.phase}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=phase_cfg["batch_size"],
        gradient_accumulation_steps=phase_cfg["gradient_accumulation_steps"],
        learning_rate=phase_cfg["learning_rate"],
        warmup_steps=phase_cfg["warmup_steps"],
        max_steps=phase_cfg["max_steps"],
        fp16=phase_cfg["fp16"],
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # Sync to S3 if configured
    s3_bucket = cfg.get("aws", {}).get("s3_bucket")
    if s3_bucket:
        import subprocess
        subprocess.run([
            "aws", "s3", "sync", str(output_dir),
            f"s3://{s3_bucket}/models/checkpoints/phase{args.phase}"
        ], check=True)
        print(f"Checkpoints synced to s3://{s3_bucket}/models/checkpoints/phase{args.phase}")


if __name__ == "__main__":
    main()
