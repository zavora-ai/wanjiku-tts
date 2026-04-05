"""
Fine-tune Gemma 4 E2B for Kikuyu ASR using the official gemma/kauldron library.

Install:
    pip install gemma kauldron

Usage:
    python -m kauldron.main --cfg=scripts/finetune_gemma4_asr_official.py --cfg.workdir=models/gemma4_asr
"""
from kauldron import konfig

with konfig.imports():
    from gemma import gm
    from kauldron import kd
    import optax


def get_config():
    batch_size = 2
    max_length = 512

    tokenizer = gm.text.Gemma3Tokenizer()

    return kd.train.Trainer(
        seed=42,
        # Dataset
        train_ds=_make_dataset(
            training=True,
            batch_size=batch_size,
            max_length=max_length,
        ),
        # Model — Gemma 4 E2B with audio support
        model=gm.nn.Gemma4_E2B(
            tokens="batch.input",
            audio="batch.audio",
        ),
        # Load pretrained weights
        init_transform=gm.ckpts.LoadCheckpoint(
            path=gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,
        ),
        # Training
        num_train_steps=3000,
        train_losses={
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        },
        optimizer=optax.adafactor(learning_rate=1e-4),
        checkpointer=kd.ckpts.Checkpointer(
            save_interval_steps=500,
        ),
        # Evaluation
        evals={
            "test": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(500),
                ds=_make_dataset(
                    training=False,
                    batch_size=batch_size,
                    max_length=max_length,
                ),
            ),
            "sampling": gm.evals.SamplerEvaluator(
                run=kd.evals.EveryNSteps(500),
                max_new_tokens=200,
                num_batches=1,
                ds=_make_dataset(training=False, sampling=True),
            ),
        },
    )


def _make_dataset(
    *,
    training: bool,
    sampling: bool = False,
    batch_size: int | None = None,
    max_length: int | None = None,
):
    tokenizer = gm.text.Gemma3Tokenizer()

    # Load WAXAL Kikuyu data from JSONL
    # Each entry: {"audio_path": "...", "text": "..."}
    return kd.data.py.JsonL(
        path="data/transcripts/waxal_train.jsonl" if training else "data/transcripts/waxal_validation.jsonl",
        shuffle=True if training else False,
        num_epochs=None if training else 1,
        batch_size=None if sampling else batch_size,
        num_workers=2,
        transforms=[
            # Load audio from path
            gm.data.LoadAudio(
                key="audio",
                path_key="audio",
                base_dir="data/waxal_kikuyu",
                sample_rate=16000,
                max_duration=30.0,
            ),
            # Create ASR task: audio prompt -> text response
            gm.data.Seq2SeqTask(
                in_prompt="<start_of_audio>",
                in_response="text",
                out_input="input",
                out_target="target",
                out_target_mask="loss_mask",
                tokenizer=tokenizer,
                max_length=None if sampling else max_length,
                truncate=True,
                sampling=sampling,
            ),
        ],
    )
