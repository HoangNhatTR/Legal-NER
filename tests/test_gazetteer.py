"""Gazetteer + factor matching: the bundle ships a working gazetteer and the
folded match keeps char offsets aligned to the original text."""

from labeling.gazetteer import load_gazetteer
from labeling.patterns import find_entities


def test_gazetteer_lists_non_empty():
    gz = load_gazetteer()
    assert gz["mitigating"], "mitigating list must not be empty"
    assert gz["aggravating"], "aggravating list must not be empty"
    # contract: each entry is {"phrase": str}
    for bucket in gz.values():
        for row in bucket:
            assert isinstance(row["phrase"], str) and row["phrase"]


def test_find_entities_does_not_crash():
    # regression: labeling.gazetteer was missing -> ModuleNotFoundError here.
    find_entities("Bị cáo thành khẩn khai báo và ăn năn hối cải.")


def test_factor_offsets_are_aligned():
    text = ("Bị cáo thành khẩn khai báo, ăn năn hối cải; phạm tội lần đầu. "
            "Phạm tội có tổ chức và tái phạm nguy hiểm.")
    factors = [s for s in find_entities(text)
               if s.label in ("MITIGATING_FACTOR", "AGGRAVATING_FACTOR")]
    assert factors, "expected sentencing factors to be matched"
    for s in factors:
        # the length-preserving fold must keep span.text == the raw slice
        assert text[s.start:s.end] == s.text


def test_both_factor_classes_detected():
    text = "thành khẩn khai báo nhưng phạm tội có tổ chức"
    labels = {s.label for s in find_entities(text)}
    assert "MITIGATING_FACTOR" in labels
    assert "AGGRAVATING_FACTOR" in labels
