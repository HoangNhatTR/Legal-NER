"""Inference windowing: subword budget is large enough that a window is never
silently truncated, and the guard fires if it ever is."""

import config
import training.infer as infer


def test_subword_budget_covers_hard_cap_window():
    # A window can grow to INFER_WINDOW + (INFER_WINDOW - INFER_STRIDE) syllables
    # (the sentence-extension hard cap). Even at a pessimistic ~2 subwords per
    # syllable that must fit under the budget, else the tail is dropped.
    hard_cap = config.INFER_WINDOW + (config.INFER_WINDOW - config.INFER_STRIDE)
    assert config.INFER_MAX_SUBWORDS >= 2 * hard_cap, (
        f"budget {config.INFER_MAX_SUBWORDS} too small for {hard_cap}-syllable "
        "windows; dense judgments will truncate"
    )


def test_guard_fires_on_truncation(capsys):
    infer._TRUNCATION_WARNED = False
    # 5 syllables fed, but word_ids only cover indices 0..2 -> tail dropped.
    infer._warn_if_truncated([None, 0, 1, 2, None], n_words=5)
    assert infer._TRUNCATION_WARNED is True
    assert "truncated" in capsys.readouterr().err.lower()


def test_guard_silent_when_fully_covered(capsys):
    infer._TRUNCATION_WARNED = False
    infer._warn_if_truncated([None, 0, 1, 2, 3, 4, None], n_words=5)
    assert infer._TRUNCATION_WARNED is False
    assert capsys.readouterr().err == ""


def test_guard_warns_only_once(capsys):
    infer._TRUNCATION_WARNED = False
    infer._warn_if_truncated([0, 1], n_words=10)
    infer._warn_if_truncated([0, 1], n_words=10)
    # second call must be a no-op (no second warning line)
    assert capsys.readouterr().err.lower().count("truncated") == 1
