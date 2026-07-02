"""lawdb — legal database for Vietnamese judgment verification.

Pipeline: fetch -> parse -> load SQLite -> lookup.

    from lawdb.lookup import verify_citation, get_article, penalty_frame, check_validity

Holds MULTIPLE law documents in one SQLite DB (``data/lawdb/lawvn.db``), each
keyed by a stable ``law_key``:

    BLHS2015    Bộ luật Hình sự 2015        (100/2015/QH13)
    BLTTHS2015  Bộ luật Tố tụng hình sự 2015 (101/2015/QH13)
    BLDS2015    Bộ luật Dân sự 2015          (91/2015/QH13)
    NQ326_2016  Nghị quyết 326/2016/UBTVQH14 (án phí, lệ phí Tòa án)

Powers verification Layer 1 (cited-law existence + validity-at-date) and
Layer 3 (charge <-> article <-> sentencing-frame) for criminal judgments.

Backward compatibility: ``DOC_META`` (the original single-document BLHS dict)
is preserved and equals ``LAW_REGISTRY["BLHS2015"]`` projected onto the old key
names. The default ``law_key`` for every lookup is ``BLHS2015``.
"""

from __future__ import annotations

# Default law for all lookups when none specified (backward-compat = BLHS).
DEFAULT_LAW_KEY = "BLHS2015"

# ---------------------------------------------------------------------------
# Law registry — one entry per document. ``effective_from`` / ``effective_to``
# drive the validity-at-judgment-date checks. ``total_dieu`` is the coverage
# denominator (official article count of the document).
# ---------------------------------------------------------------------------
LAW_REGISTRY: dict[str, dict] = {
    "BLHS2015": {
        "law_key": "BLHS2015",
        "ten": "Bộ luật Hình sự",
        "so_hieu": "100/2015/QH13",
        "loai": "bo_luat",
        "total_dieu": 426,
        "ngay_ban_hanh": "2015-11-27",
        "effective_from": "2018-01-01",  # hợp nhất với Luật 12/2017/QH14
        "effective_to": None,
        "status": "in_force",
        "version_note": (
            "BLHS 2015 (Luật số 100/2015/QH13) đã được sửa đổi, bổ sung bởi Luật "
            "số 12/2017/QH14; bản hợp nhất, hiệu lực từ 01/01/2018."
        ),
        "amendments": [
            {
                "amending_key": None,
                "amending_ten": "Luật sửa đổi, bổ sung một số điều của Bộ luật Hình sự số 100/2015/QH13",
                "so_hieu": "12/2017/QH14",
                "effective_from": "2018-01-01",
                "mo_ta": "Sửa đổi, bổ sung BLHS 2015; cùng hiệu lực thi hành 01/01/2018.",
            }
        ],
    },
    "BLTTHS2015": {
        "law_key": "BLTTHS2015",
        "ten": "Bộ luật Tố tụng hình sự",
        "so_hieu": "101/2015/QH13",
        "loai": "bo_luat",
        "total_dieu": 510,
        "ngay_ban_hanh": "2015-11-27",
        "effective_from": "2018-01-01",
        "effective_to": None,
        "status": "in_force",
        "version_note": (
            "BLTTHS 2015 (Luật số 101/2015/QH13), hiệu lực thi hành 01/01/2018 "
            "(theo Nghị quyết 41/2017/QH14)."
        ),
        "amendments": [],
    },
    "BLDS2015": {
        "law_key": "BLDS2015",
        "ten": "Bộ luật Dân sự",
        "so_hieu": "91/2015/QH13",
        "loai": "bo_luat",
        "total_dieu": 689,
        "ngay_ban_hanh": "2015-11-24",
        "effective_from": "2017-01-01",
        "effective_to": None,
        "status": "in_force",
        "version_note": "BLDS 2015 (Luật số 91/2015/QH13), hiệu lực thi hành 01/01/2017.",
        "amendments": [],
    },
    "NQ326_2016": {
        "law_key": "NQ326_2016",
        "ten": "Nghị quyết về án phí, lệ phí Tòa án",
        "so_hieu": "326/2016/UBTVQH14",
        "loai": "nghi_quyet",
        "total_dieu": 48,
        "ngay_ban_hanh": "2016-12-30",
        "effective_from": "2017-01-01",
        "effective_to": None,
        "status": "in_force",
        "version_note": (
            "Nghị quyết 326/2016/UBTVQH14 của UBTVQH quy định mức thu, miễn, giảm, "
            "thu, nộp, quản lý và sử dụng án phí, lệ phí Tòa án; hiệu lực 01/01/2017."
        ),
        "amendments": [],
    },
}

# Backward-compatible single-document metadata (BLHS), kept for any importer of
# ``lawdb.DOC_META``.
DOC_META = {
    "ten": LAW_REGISTRY["BLHS2015"]["ten"],
    "so_hieu": LAW_REGISTRY["BLHS2015"]["so_hieu"],
    "ngay_ban_hanh": LAW_REGISTRY["BLHS2015"]["ngay_ban_hanh"],
    "ngay_hieu_luc": LAW_REGISTRY["BLHS2015"]["effective_from"],
    "version_note": LAW_REGISTRY["BLHS2015"]["version_note"],
}


def normalize_law_key(law_key: str | None) -> str:
    """Map various spellings / aliases to a canonical registry key.

    Accepts the coarse tags produced by ``verify.citation.classify_law``
    ('BLHS', 'BLTTHS', 'OTHER', 'UNKNOWN') as well as the canonical keys.
    Unknown / None falls back to the default law (BLHS2015).
    """
    if not law_key:
        return DEFAULT_LAW_KEY
    k = law_key.strip().upper().replace(" ", "").replace("-", "_")
    if k in LAW_REGISTRY:
        return k
    alias = {
        "BLHS": "BLHS2015",
        "BLTTHS": "BLTTHS2015",
        "BLDS": "BLDS2015",
        "NQ326": "NQ326_2016",
        "NQ326_2016_UBTVQH14": "NQ326_2016",
    }
    return alias.get(k, DEFAULT_LAW_KEY)
