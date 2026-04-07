"""Fine-tune DeKUT Whisper-base on human-transcribed Kikuyu data."""
import json, os, torch, soundfile as sf, numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSpeechSeq2Seq, WhisperProcessor, Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL = "DeKUT-DSAIL/tunuh-whisper-base-kikuyu-v1"
OUTPUT = os.path.expanduser("~/wanjiku-tts/models/dekut_finetuned_v1")
MANIFESTS = [
    os.path.expanduser("~/wanjiku-tts/data/manifests/digigreen/train.jsonl"),
    os.path.expanduser("~/wanjiku-tts/data/manifests/bible/train.jsonl"),
    os.path.expanduser("~/wanjiku-tts/data/manifests/waxal/train.jsonl"),
    os.path.expanduser("~/wanjiku-tts/data/manifests/openbible/train.jsonl"),
]
DEV = os.path.expanduser("~/wanjiku-tts/data/manifests/combined/dev.jsonl")

# Custom dataset that loads audio on the fly
class ASRDataset(TorchDataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        audio, sr = sf.read(item["audio_path"])
        if sr != 16000:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * 16000 / sr))
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
        labels = self.processor.tokenizer(item["text"]).input_ids
        if len(labels) > 440:
            labels = labels[:440]
        return {"input_features": inputs.input_features[0], "labels": labels}

# Load manifests
print("Loading data...")
train_items = []
for path in MANIFESTS:
    with open(path) as f:
        for l in f:
            d = json.loads(l)
            if os.path.exists(d["audio_path"]):
                train_items.append(d)
print(f"Train: {len(train_items)}")

dev_items = []
with open(DEV) as f:
    for l in f:
        d = json.loads(l)
        if os.path.exists(d["audio_path"]):
            dev_items.append(d)
dev_items = dev_items[:500]
print(f"Dev: {len(dev_items)}")

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, use_safetensors=True)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

train_ds = ASRDataset(train_items, processor)
dev_ds = ASRDataset(dev_items, processor)

@dataclass
class DataCollator:
    processor: WhisperProcessor
    def __call__(self, features):
        input_features = torch.tensor(np.array([f["input_features"] for f in features]))
        labels = [f["labels"] for f in features]
        max_len = max(len(l) for l in labels)
        labels_padded = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            labels_padded[i, :len(l)] = torch.tensor(l)
        return {"input_features": input_features, "labels": labels_padded}

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # Simple WER
    errors = total = 0
    for p, r in zip(pred_str, label_str):
        pw, rw = p.split(), r.split()
        d = [[0]*(len(pw)+1) for _ in range(len(rw)+1)]
        for i in range(len(rw)+1): d[i][0] = i
        for j in range(len(pw)+1): d[0][j] = j
        for i in range(1, len(rw)+1):
            for j in range(1, len(pw)+1):
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(0 if rw[i-1]==pw[j-1] else 1))
        errors += d[len(rw)][len(pw)]
        total += max(len(rw), 1)
    return {"wer": errors / max(total, 1)}

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    eval_strategy="no",
    save_steps=1000,
    save_total_limit=3,
    logging_steps=100,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    report_to="none",
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=DataCollator(processor),
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Training...")
trainer.train()
trainer.save_model(OUTPUT)
processor.save_pretrained(OUTPUT)
print(f"Saved to {OUTPUT}")
