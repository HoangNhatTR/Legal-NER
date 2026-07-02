"""BIO decoding: robust span recovery + confidence aggregation.

Pure-Python (no model weights): we hand-craft tag sequences over real tokens
and assert tags_to_entities decodes them correctly.
"""

from labeling.weak_label import tokenize_with_offsets
from training.infer import tags_to_entities


def _decode(text, tags, scores=None):
    tokens = tokenize_with_offsets(text)
    assert len(tokens) == len(tags), (len(tokens), len(tags), [t for t, _, _ in tokens])
    return tags_to_entities(tokens, tags, text, scores)


def test_standard_bio():
    text = "Điều 250"  # 2 tokens: "Điều", "250"
    ents = _decode(text, ["B-ARTICLE", "I-ARTICLE"])
    assert len(ents) == 1
    assert ents[0]["label"] == "ARTICLE"
    assert ents[0]["text"] == "Điều 250"


def test_orphan_i_tag_is_recovered():
    """An I-X with no preceding B-X used to be silently dropped."""
    text = "Điều 250"
    ents = _decode(text, ["I-ARTICLE", "I-ARTICLE"])
    assert len(ents) == 1, "leading I- span must be recovered, not dropped"
    assert ents[0]["text"] == "Điều 250"


def test_label_switch_starts_new_span():
    """B-Y immediately followed by I-X opens an X span (not dropped)."""
    text = "khoản 1"  # 2 tokens
    ents = _decode(text, ["B-CLAUSE", "I-ARTICLE"])
    labels = sorted(e["label"] for e in ents)
    assert labels == ["ARTICLE", "CLAUSE"]


def test_two_adjacent_same_label_spans():
    text = "Điều 250 Điều 9"  # 4 tokens
    ents = _decode(text, ["B-ARTICLE", "I-ARTICLE", "B-ARTICLE", "I-ARTICLE"])
    assert [e["text"] for e in ents] == ["Điều 250", "Điều 9"]


def test_o_tags_close_spans():
    text = "bị cáo Nguyễn Văn A"  # tokens: bị, cáo, Nguyễn, Văn, A
    tags = ["O", "O", "B-DEFENDANT", "I-DEFENDANT", "I-DEFENDANT"]
    ents = _decode(text, tags)
    assert len(ents) == 1
    assert ents[0]["text"] == "Nguyễn Văn A"


def test_score_is_mean_over_span():
    text = "Điều 250"
    ents = _decode(text, ["B-ARTICLE", "I-ARTICLE"], scores=[0.9, 0.7])
    assert ents[0]["score"] == 0.8  # mean(0.9, 0.7)


def test_no_scores_means_no_score_key():
    text = "Điều 250"
    ents = _decode(text, ["B-ARTICLE", "I-ARTICLE"])
    assert "score" not in ents[0]
