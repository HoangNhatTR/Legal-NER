"""Span-level evaluation of a prediction source against the human GOLD set.

Breaks the self-referential loop: gold was annotated by reading the judgments
(INDEPENDENT of the regex), so this measures CORRECTNESS, not label-learnability.

Prediction sources (--pred):
  * regex      : run the CURRENT labeling.patterns.find_entities (= v1) live.
  * regex_v2   : reuse the preserved v2 regex output (data/labeled/weak_spans_v2.jsonl);
                 same char offsets into the same flattened stream as gold.
  * regex_file : score an arbitrary preserved weak_spans*.jsonl (--spans PATH).
  * model      : run a trained checkpoint via training.infer (--model PATH).

Both EXACT-match and OVERLAP-match (any char overlap, greedy 1-1) F1 are
reported per entity, focused on the 3 weak entities but covering all.

CLI:
    python -m training.eval_gold --gold gold/gold.jsonl --pred regex
    python -m training.eval_gold --gold gold/gold.jsonl --pred regex_v2
    python -m training.eval_gold --gold gold/gold.jsonl --pred model \
        --model data/models/legal-ner/final
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.normalize import flatten_for_matching  # noqa: E402

WEAK = ["DECISION", "LEGAL_BASIS", "VIOLATION_ACT"]


def load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_text(rec: dict, gold_dir: Path) -> str:
    return (gold_dir / rec["text_file"]).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Prediction sources
# --------------------------------------------------------------------------
def pred_regex_live(rec, text):
    from labeling.patterns import find_entities
    return [{"start": s.start, "end": s.end, "label": s.label, "text": s.text}
            for s in find_entities(text)]


def _load_spans_file(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["doc"]] = row["spans"]
    return out


def pred_from_spans_file(spans_by_doc: dict, rec):
    # gold id "civil_100043" maps to preserved-spans doc "civil_100043";
    # criminal "102375" maps to "102375".
    return spans_by_doc.get(rec["id"], [])


def pred_model(rec, text, model, tokenizer, device):
    from training.infer import infer_entities
    return infer_entities(text, model, tokenizer, device)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["start"] + (a["end"] - a["start"]) \
        and a["start"] < b["end"] and a["end"] > b["start"]


def _overlap(a, b):
    return a["start"] < b["end"] and a["end"] > b["start"]


def score(gold_spans, pred_spans, mode):
    """Greedy 1-1 matching per label. mode in {'exact','overlap'}.
    Returns dict label -> (tp, n_gold, n_pred)."""
    stats = defaultdict(lambda: [0, 0, 0])  # tp, gold, pred
    by_label_g = defaultdict(list)
    by_label_p = defaultdict(list)
    for g in gold_spans:
        by_label_g[g["label"]].append(g)
    for p in pred_spans:
        by_label_p[p["label"]].append(p)
    labels = set(by_label_g) | set(by_label_p)
    for lab in labels:
        gs = by_label_g[lab]
        ps = list(by_label_p[lab])
        stats[lab][1] += len(gs)
        stats[lab][2] += len(ps)
        used = [False] * len(ps)
        for g in gs:
            for i, p in enumerate(ps):
                if used[i]:
                    continue
                ok = (g["start"] == p["start"] and g["end"] == p["end"]) if mode == "exact" \
                    else _overlap(g, p)
                if ok:
                    used[i] = True
                    stats[lab][0] += 1
                    break
    return stats


def prf(tp, ng, npd):
    p = tp / npd if npd else 0.0
    r = tp / ng if ng else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate(all_stats):
    agg = defaultdict(lambda: [0, 0, 0])
    for st in all_stats:
        for lab, (tp, ng, npd) in st.items():
            agg[lab][0] += tp
            agg[lab][1] += ng
            agg[lab][2] += npd
    return agg


def print_report(agg_exact, agg_overlap, title):
    labels = sorted(set(agg_exact) | set(agg_overlap))
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    print(f"{'ENTITY':16s} | {'gold':>4s} {'pred':>4s} | "
          f"{'P_ex':>6s} {'R_ex':>6s} {'F_ex':>6s} | {'P_ov':>6s} {'R_ov':>6s} {'F_ov':>6s}")
    print("-" * 92)

    def row(lab):
        tp_e, ng, npd = agg_exact.get(lab, [0, 0, 0])
        tp_o, _, _ = agg_overlap.get(lab, [0, 0, 0])
        pe, re_, fe = prf(tp_e, ng, npd)
        po, ro, fo = prf(tp_o, ng, npd)
        print(f"{lab:16s} | {ng:4d} {npd:4d} | {pe:6.3f} {re_:6.3f} {fe:6.3f} | "
              f"{po:6.3f} {ro:6.3f} {fo:6.3f}")

    print("-- WEAK ENTITIES --")
    for lab in WEAK:
        row(lab)
    print("-- OTHER ANNOTATED --")
    for lab in labels:
        if lab not in WEAK:
            row(lab)
    # weak-only micro
    for name, agg in [("EXACT", agg_exact), ("OVERLAP", agg_overlap)]:
        tp = sum(agg.get(l, [0, 0, 0])[0] for l in WEAK)
        ng = sum(agg.get(l, [0, 0, 0])[1] for l in WEAK)
        npd = sum(agg.get(l, [0, 0, 0])[2] for l in WEAK)
        p, r, f = prf(tp, ng, npd)
        print(f"WEAK micro-{name:8s}: P={p:.3f} R={r:.3f} F1={f:.3f}  (tp={tp} gold={ng} pred={npd})")


def main():
    ap = argparse.ArgumentParser(description="Span-level eval vs human gold")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True,
                    choices=["regex", "regex_v2", "regex_file", "model"])
    ap.add_argument("--spans", default=None, help="weak_spans*.jsonl for regex_file")
    ap.add_argument("--model", default=None, help="checkpoint dir for --pred model")
    ap.add_argument("--criminal-only", action="store_true",
                    help="score only criminal docs (v2 regex output exists only here)")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    gold_dir = gold_path.resolve().parent
    gold = load_gold(gold_path)
    if args.criminal_only:
        gold = [r for r in gold if r.get("source") == "criminal"]

    spans_by_doc = None
    if args.pred == "regex_v2":
        spans_by_doc = _load_spans_file(
            Path(__file__).resolve().parents[1] / "data/labeled/weak_spans_v2.jsonl")
    elif args.pred == "regex_file":
        spans_by_doc = _load_spans_file(Path(args.spans))

    model = tokenizer = device = None
    if args.pred == "model":
        import torch
        from training.infer import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_model(args.model, device)

    stats_exact, stats_overlap = [], []
    for rec in gold:
        text = gold_text(rec, gold_dir)
        if args.pred == "regex":
            pred = pred_regex_live(rec, text)
        elif args.pred in ("regex_v2", "regex_file"):
            pred = pred_from_spans_file(spans_by_doc, rec)
        else:
            pred = pred_model(rec, text, model, tokenizer, device)
        # restrict scoring to the label set present in gold (anchors + weak),
        # so we don't penalize predictors for entities we did not annotate.
        gold_labels = {s["label"] for r in gold for s in r["spans"]}
        pred = [p for p in pred if p["label"] in gold_labels]
        stats_exact.append(score(rec["spans"], pred, "exact"))
        stats_overlap.append(score(rec["spans"], pred, "overlap"))

    title = f"PRED={args.pred}" + (f"  MODEL={args.model}" if args.model else "")
    print_report(aggregate(stats_exact), aggregate(stats_overlap), title)


if __name__ == "__main__":
    main()
