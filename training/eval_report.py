"""Per-entity evaluation of a trained legal-NER checkpoint.

Reproduces the EXACT validation split train.py uses
(build_dataset_dict(path, dev_ratio=0.1, seed=42)), runs prediction with a
HuggingFace Trainer, and prints the seqeval classification_report
(per-entity precision/recall/F1/support) plus overall micro/macro F1.

CLI:
    python -m training.eval_report \
        --model data/models/legal-ner-v3r-full/final \
        --data  data/labeled/weak_labels_combined.jsonl \
        --out   data/eval_combined.txt
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
from config import ID2LABEL  # noqa: E402
from training.dataset import build_dataset_dict, make_tokenize_fn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-entity eval of a legal-NER checkpoint")
    parser.add_argument("--model", required=True, help="fine-tuned checkpoint dir")
    parser.add_argument("--data", required=True, help="weak-labeled BIO JSONL")
    parser.add_argument("--out", default=None, help="write the report here (default: stdout only)")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # SAME split as train.py: dev_ratio=0.1, seed=42 (baked into build_dataset_dict)
    splits = build_dataset_dict(Path(args.data), dev_ratio=args.dev_ratio)
    val = splits["validation"]
    print(f"validation chunks: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)

    tokenized = splits.map(
        make_tokenize_fn(tokenizer),
        batched=True,
        remove_columns=splits["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir="/tmp/legal_ner_eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=False,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )

    pred_out = trainer.predict(tokenized["validation"])
    logits = pred_out.predictions
    labels = pred_out.label_ids
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

    report = classification_report(true_seqs, pred_seqs, zero_division=0, digits=4)
    micro = f1_score(true_seqs, pred_seqs, average="micro", zero_division=0)
    macro = f1_score(true_seqs, pred_seqs, average="macro", zero_division=0)

    header = (
        f"Model: {args.model}\n"
        f"Data:  {args.data}\n"
        f"Validation split: dev_ratio={args.dev_ratio}, seed=42 (same as train.py)\n"
        f"Validation chunks: {len(val)}\n"
        f"{'=' * 70}\n"
    )
    footer = f"\nOverall micro-F1: {micro:.4f}\nOverall macro-F1: {macro:.4f}\n"
    out_text = header + report + footer
    print(out_text)

    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
