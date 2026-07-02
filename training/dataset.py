"""Load weak-labeled BIO JSONL into HF datasets with subword label alignment.

Syllable tokens are passed pre-split to the tokenizer; only the first
subword of each syllable carries the label, the rest get -100.
"""

import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, MAX_SEQ_LENGTH  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset_dict(jsonl_path: Path, dev_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    records = load_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"No records in {jsonl_path}")
    data = {
        "id": [r["id"] for r in records],
        "tokens": [r["tokens"] for r in records],
        "ner_tags": [[LABEL2ID[t] for t in r["tags"]] for r in records],
    }
    full = Dataset.from_dict(data)
    if len(records) < 10 or dev_ratio <= 0:
        # tiny smoke data: same split for train/dev
        return DatasetDict(train=full, validation=full)
    split = full.train_test_split(test_size=dev_ratio, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def make_tokenize_fn(tokenizer, max_length: int = MAX_SEQ_LENGTH):
    def tokenize_and_align(batch):
        encoded = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = encoded.word_ids(batch_index=i)
            labels = []
            previous = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != previous:
                    labels.append(tags[wid])      # first subword: real label
                else:
                    labels.append(-100)           # continuation subword
                previous = wid
            all_labels.append(labels)
        encoded["labels"] = all_labels
        return encoded

    return tokenize_and_align
