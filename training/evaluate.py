"""Per-entity evaluation of a trained legal-NER checkpoint (reusable).

Reproduces the EXACT validation split train.py uses
(``build_dataset_dict(path, dev_ratio=0.1, seed=42)`` — seed is baked in), runs
fp16/no-grad batched inference over the validation split, and prints the seqeval
``classification_report`` (per-entity precision/recall/F1/support) plus the
overall micro-F1 (and macro-F1).

Inference uses a plain manual loop (no Trainer): batch size 8, fp16 + no_grad on
GPU when the checkpoint fits, else CPU (fp32). The GPU here is shared and
near-full (~4GB free); the ~1GB checkpoint usually fits, but we degrade
gracefully to CPU on CUDA OOM rather than crash.

CLI:
    python -m training.evaluate \
        --model data/models/legal-ner-v3r-full/final \
        --data  data/labeled/weak_labels_combined.jsonl \
        --dev-ratio 0.1
"""

import argparse
import sys
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ID2LABEL  # noqa: E402
from training.dataset import build_dataset_dict, make_tokenize_fn  # noqa: E402


def _pick_device(model) -> tuple[str, bool]:
    """Try to place the model on CUDA; return (device, use_fp16).

    Falls back to CPU (fp32) on CUDA OOM / runtime error so a busy shared GPU
    never crashes the eval.
    """
    if torch.cuda.is_available():
        try:
            model.to("cuda")
            return "cuda", True
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # type: ignore[attr-defined]
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"[evaluate] CUDA placement failed ({exc}); using CPU", flush=True)
    model.to("cpu")
    return "cpu", False


@torch.no_grad()
def _run_inference(model, loader, device: str, use_fp16: bool):
    """Yield (pred_ids, label_ids) numpy rows for each example, fp16/no-grad."""
    model.eval()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast:
            logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu()
        for pred_row, label_row in zip(preds.tolist(), labels.tolist()):
            yield pred_row, label_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-entity seqeval eval of a legal-NER checkpoint"
    )
    parser.add_argument("--model", required=True, help="fine-tuned checkpoint dir")
    parser.add_argument("--data", required=True, help="weak-labeled BIO JSONL")
    parser.add_argument("--dev-ratio", type=float, default=0.1,
                        help="validation fraction (default 0.1, matches train.py)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=None,
                        help="also write the report here (default: stdout only)")
    args = parser.parse_args()

    # SAME split as train.py: dev_ratio=0.1, seed=42 (baked into build_dataset_dict)
    splits = build_dataset_dict(Path(args.data), dev_ratio=args.dev_ratio)
    val = splits["validation"]
    print(f"validation chunks: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)

    tokenized_val = val.map(
        make_tokenize_fn(tokenizer),
        batched=True,
        remove_columns=val.column_names,
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(
        tokenized_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device, use_fp16 = _pick_device(model)
    print(f"device={device} fp16={use_fp16} batch_size={args.batch_size}")

    true_seqs, pred_seqs = [], []
    for pred_row, label_row in _run_inference(model, loader, device, use_fp16):
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
        f"Validation chunks: {len(val)} | device={device} fp16={use_fp16}\n"
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
