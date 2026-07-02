"""Fine-tune a token-classification model on weak-labeled judgment data.

CLI:
    python -m training.train --data data/labeled/weak_labels.jsonl --epochs 5
    python -m training.train --max-steps 1            # smoke run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ID2LABEL, LABEL2ID, LABELED_DIR, MODEL_NAME, MODELS_DIR  # noqa: E402
from training.dataset import build_dataset_dict, make_tokenize_fn  # noqa: E402


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(ID2LABEL[int(l)])
            pred_seq.append(ID2LABEL[int(p)])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)
    print(classification_report(true_seqs, pred_seqs, zero_division=0))
    return {"f1": f1_score(true_seqs, pred_seqs, zero_division=0)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train legal NER token classifier")
    parser.add_argument("--data", default=str(LABELED_DIR / "weak_labels.jsonl"),
                        help="weak-labeled BIO JSONL (default: data/labeled/weak_labels.jsonl)")
    parser.add_argument("--model", default=MODEL_NAME, help=f"base model (default: {MODEL_NAME})")
    parser.add_argument("--out", default=str(MODELS_DIR / "legal-ner"),
                        help="checkpoint output dir")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="override epochs; use 1 for a smoke run")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    # --- memory-fit flags (GPU is shared; only ~3.6GB VRAM free) ---
    parser.add_argument("--fp16", type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="mixed-precision fp16 (default True)")
    parser.add_argument("--grad-checkpointing",
                        type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="gradient checkpointing + use_cache=False (default True)")
    parser.add_argument("--freeze-bottom-layers", type=int, default=8,
                        help="freeze embeddings + bottom N of 12 encoder layers (default 8)")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="gradient_accumulation_steps (default 2)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="cap tokenization length (default: config MAX_SEQ_LENGTH)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    splits = build_dataset_dict(Path(args.data), dev_ratio=args.dev_ratio)
    print(f"  train={len(splits['train'])} validation={len(splits['validation'])} chunks")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- memory-fit: gradient checkpointing ---
    if args.grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # frozen embeddings produce no grad; this re-enables grad flow through
        # checkpointed blocks so backprop reaches the trainable top layers.
        model.enable_input_require_grads()

    # --- memory-fit: freeze embeddings + bottom N encoder layers ---
    n_freeze = max(0, min(args.freeze_bottom_layers, len(model.roberta.encoder.layer)))
    if n_freeze > 0:
        for p in model.roberta.embeddings.parameters():
            p.requires_grad = False
        for layer in model.roberta.encoder.layer[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen embeddings + bottom {n_freeze}/{len(model.roberta.encoder.layer)} layers | "
          f"trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.1f}%)")

    tok_max_len = args.max_length if args.max_length else None
    tok_fn = make_tokenize_fn(tokenizer, max_length=tok_max_len) if tok_max_len \
        else make_tokenize_fn(tokenizer)
    tokenized = splits.map(
        tok_fn, batched=True,
        remove_columns=splits["train"].column_names,
    )

    full_run = args.max_steps < 0
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_checkpointing else None,
        eval_strategy="epoch" if full_run else "no",
        save_strategy="epoch" if full_run else "no",
        load_best_model_at_end=full_run,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()

    final_dir = Path(args.out) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to {final_dir}")
    if args.max_steps < 0:
        print("Final evaluation:")
        trainer.evaluate()


if __name__ == "__main__":
    main()
