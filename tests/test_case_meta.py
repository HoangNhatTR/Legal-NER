"""Document metadata derived from the CASE_NUMBER suffix."""

from labeling.patterns import Span, derive_doc_meta


def _meta(case_number):
    span = Span(0, len(case_number), "CASE_NUMBER", case_number)
    return derive_doc_meta([span])


def test_criminal_first_instance():
    m = _meta("17/2018/HS-ST")
    assert m["case_type"] == "hình sự"
    assert m["procedure_stage"] == "sơ thẩm"


def test_criminal_appeal():
    m = _meta("111/2017/HSPT")
    assert m["case_type"] == "hình sự"
    assert m["procedure_stage"] == "phúc thẩm"


def test_family_not_misread_as_civil_or_admin():
    # "HNGĐ" must win over the "HC"/"DS" substrings (longest-code-first).
    m = _meta("08/2018/QĐST-HNGĐ")
    assert m["case_type"] == "hôn nhân và gia đình"
    assert m["procedure_stage"] == "sơ thẩm"


def test_civil():
    m = _meta("31/2018/DS-PT")
    assert m["case_type"] == "dân sự"
    assert m["procedure_stage"] == "phúc thẩm"


def test_no_case_number_is_empty_meta():
    assert derive_doc_meta([]) == {
        "case_number": None, "case_type": None, "procedure_stage": None
    }
