# lawdb — Vietnamese legal database (multi-document)

A self-contained SQLite database (`data/lawdb/lawvn.db`) of multiple Vietnamese
law documents, keyed by a stable `law_key`, with a query API:

| `law_key`    | document | số hiệu | hiệu lực |
| ------------ | -------- | ------- | -------- |
| `BLHS2015`   | Bộ luật Hình sự 2015 | 100/2015/QH13 (sđbs 12/2017/QH14) | 01/01/2018 |
| `BLTTHS2015` | Bộ luật Tố tụng hình sự 2015 | 101/2015/QH13 | 01/01/2018 |
| `BLDS2015`   | Bộ luật Dân sự 2015 | 91/2015/QH13 | 01/01/2017 |
| `NQ326_2016` | Nghị quyết 326/2016/UBTVQH14 (án phí, lệ phí) | 326/2016/UBTVQH14 | 01/01/2017 |

It unlocks judgment-verification **Layer 1** (cited-law existence + validity at
the judgment date) and provides structured data for **Layer 3** (charge ↔
article ↔ sentencing-frame logic, BLHS only) on criminal judgments.

## Pipeline

```
fetch.py  -> data/lawdb/<law_key>_raw.txt  (+ .meta.json)   acquire full text per doc
parser.py -> điều/khoản/điểm tree + penalty frames (BLHS)   structure it
schema.sql + build.py -> data/lawdb/lawvn.db                load SQLite (all docs)
lookup.py -> query API (Layer 1 / Layer 3 / validity)       use it, scoped by law_key
```

## Build

```bash
python -m lawdb.build                  # fetch (if needed) -> parse -> load ALL docs
python -m lawdb.build --law BLTTHS2015  # one document
python -m lawdb.build --refetch        # force re-download of raw text
```

Build output (verified):

| law_key      | articles | clauses | points | penalty_frames | offense | coverage |
| ------------ | -------- | ------- | ------ | -------------- | ------- | -------- |
| `BLHS2015`   | 426/426  | 1433    | 2902   | 1565           | 318     | 100.0%   |
| `BLTTHS2015` | 501/510  | 1419    | 988    | 0              | 0       | 98.2%    |
| `BLDS2015`   | 689/689  | 1635    | 312    | 0              | 0       | 100.0%   |
| `NQ326_2016` | 47/48    | 169     | 65     | 0              | 0       | 97.9%    |

## Source & coverage (honest)

All documents are acquired from the luatvietnam.vn "nội dung hợp nhất" full-text
pages. BLHS additionally has a HuggingFace backfill for the few articles
luatvietnam omits.

| law_key | luatvietnam URL id | obtained | missing |
| ------- | ------------------ | -------- | ------- |
| BLHS2015 | `...-101324-d1.html` (+ HF `xuanhungttm/bo-luat-hinh-su-2015`) | 426/426 | none |
| BLTTHS2015 | `...-101322-d1.html` | 501/510 | 9 (Điều 413, 416, 418–420, 422–425 — điều khoản thi hành cuối) |
| BLDS2015 | `...-101333-d1.html` | 689/689 | none |
| NQ326_2016 | `...-111767-d1.html` | 47/48 | 1 (Điều 23 — bỏ qua do định dạng nguồn) |

Raw text is cached to `data/lawdb/<law_key>_raw.txt` so parsing is fully
reproducible offline, with `<law_key>_raw.meta.json` recording the source and
Phần/Chương/Mục context of every article. The luatvietnam consolidated view
occasionally duplicates a clause block (a comparison artifact); `build.py`
de-duplicates by keeping the first occurrence of each khoản — affecting only
BLTTHS Điều 417/421 and NQ326 Điều 22.

**Phần boundaries** are assigned deterministically by article range
(1–107 Phần I, 108–425 Phần II "Các tội phạm", 426 Phần III) because the
standalone "Phần thứ hai/ba" header lines are collapsed in the source HTML.
Chương and Mục are parsed from the live text.

## Penalty frames (Layer 3)

For each khoản of an offense article, sentencing phrases are parsed into
structured `(penalty_type, min_value, max_value, unit)`:

| penalty_type            | meaning              | unit | example raw_text |
| ----------------------- | -------------------- | ---- | ---------------- |
| `tu_co_thoi_han`        | tù có thời hạn       | nam  | "phạt tù từ 02 năm đến 07 năm" |
| `tu_chung_than`         | tù chung thân        | —    | "tù chung thân" |
| `tu_hinh`               | tử hình              | —    | "tử hình" |
| `phat_tien`             | phạt tiền            | dong | "phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng" |
| `cai_tao_khong_giam_giu`| cải tạo không giam giữ | nam | "cải tạo không giam giữ đến 03 năm" |
| `canh_cao`              | cảnh cáo             | —    | "phạt cảnh cáo" |

Month values ("06 tháng") are normalized to fractional years (0.5) with
`unit="nam"`; money keeps đồng as integers. A single khoản may carry several
frames (e.g. tù 20 năm + tù chung thân + tử hình).

## Query API

All lookups are scoped by `law_key` (default `BLHS2015` — old positional calls
keep working unchanged).

```python
from lawdb.lookup import get_article, get_clause, verify_citation, penalty_frame, check_validity

verify_citation(250)                              # BLHS by default -> exists, content, message_vi
verify_citation(250, 2)                           # + penalty_frame of khoản 2
verify_citation(106, law_key="BLTTHS2015")        # scope to another code
verify_citation(250, 2, on_date="2017-06-01")     # + validity block (in_force=False: chưa hiệu lực)
get_article(688, law_key="BLDS2015")              # full khoản/điểm tree
penalty_frame(250, 2)                             # {type:'tu_co_thoi_han', min:7.0, max:15.0, unit:'nam'}
check_validity("BLHS2015", 250, on_date="2019-01-01")  # {in_force:True, effective_from, ...}
```

Backward compatibility: `law_key` defaults to `BLHS2015`; `on_date=None`
(default) omits the validity block entirely (identical to prior behaviour). If
`lawvn.db` is absent, lookups fall back to the legacy `blhs2015.db`.

CLI:

```bash
python -m lawdb.lookup --dieu 250 --khoan 2
python -m lawdb.lookup --law BLTTHS2015 --dieu 106
python -m lawdb.lookup --law BLHS2015 --dieu 250 --on-date 2017-06-01   # in_force=False
python -m lawdb.lookup --dieu 999                                       # exists=False
```

## Schema

`documents` → `articles` (điều) → `clauses` (khoản) → `points` (điểm), with
`penalty_frames` attached to clauses and an `amendments` table. `documents`
carries `law_key`, `total_dieu`, `effective_from`, `effective_to`, `status`.
See `schema.sql`.

## Validity at judgment date

`check_validity(law_key, so_dieu, on_date)` answers whether a document was in
force on a date. It is **document-level** (per-Điều amendment granularity is not
captured — stated honestly in `message_vi` and the `granularity` field). This
catches anomalies such as a 2016 judgment citing BLHS 2015, which only took
effect 01/01/2018 → `in_force=False`, flagged. `verify_citation(..., on_date=)`
attaches the same block; `verify.citation.check_citations(..., on_date=)` threads
the judgment date through and counts `summary.validity_flags`.

## Known limitations

- 4 of 318 offense articles carry no penalty frame — all are correct
  (definitional articles: Điều 122, 352, 367, 392 — no sentencing range in text).
- One source typo ("phạt tù từ 06 tháng năm đến 03 năm") is parsed leniently
  (min taken as 06 tháng); value is still correct.
- Phần III (Điều 426) inherits the last Chương label from Phần II; harmless
  because `is_offense=False` for it.
- BLHS is the **consolidated 2015+2017** text as published by luatvietnam; it is
  not a diffable amendment history. `documents.version_note` records this.
- **Validity is checked at document level, not per-article.** The `amendments`
  table records document-level relationships (e.g. BLHS 2015 ← Luật 12/2017/QH14,
  hiệu lực 01/01/2018) but cannot say whether a *specific Điều* was later amended.
- **BLTTHS / BLDS / NQ326 have no penalty frames** (only BLHS defines offenses);
  Layer 3 sentencing checks apply to BLHS citations only.
- **NQ326 is partial (47/48) and out of `check_citations` scope** — Nghị quyết
  citations are still reported `out_of_scope`. Only BLHS/BLTTHS/BLDS are routed
  to the DB. Luật THADS and Bộ luật Tố tụng dân sự are also out of scope.
