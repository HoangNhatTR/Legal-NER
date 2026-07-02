# Layer 2 — Judgment Existence Lookup (`verify/existence.py`)

Part of **Model 2** judgment verification. Given a judgment's extracted fields
(`case_number`, `court`, `judgment_date` — produced by the Model 1 NER), this
module searches the official portal **congbobanan.toaan.gov.vn** and reports
whether a matching judgment is published there.

## ⚖️ Core principle: ABSENCE IS NOT PROOF OF FORGERY

A **"not found" result is INCONCLUSIVE, never "fake"**. A real judgment can be
missing from the portal because:

- **Publication lag** — recent judgments are not published instantly.
- **Excluded categories** — many judgments are *never* published at all
  (state-secret, juvenile, morality/decency, family-privacy, etc.), per
  **Nghị quyết 03/2017/NQ-HĐTP, Điều 4**.
- **Imperfect search** — OCR noise in the source fields, suffix variants,
  typos, and portal index gaps cause false negatives.

Therefore the module never outputs a forgery verdict. Every `not_found` /
`error` report carries `caveat_vi` and an explicit "KHÔNG kết luận" summary.

## Statuses

| status          | meaning                                                            | inconclusive? |
|-----------------|--------------------------------------------------------------------|---------------|
| `found`         | exact case-number match **and** court+date corroborate (score ≥ 0.85) | no            |
| `found_partial` | case number matches but court/date differ or weren't supplied; OR only a same-number/different-suffix near-match was found | no (but verify) |
| `not_found`     | no candidate passed the case-number gate                           | **yes**       |
| `error`         | portal unreachable / search transport failure                     | **yes**       |

## Search mechanism (verified live 2026-06-11)

The portal homepage hosts a single ASP.NET WebForms `aspnetForm`. No captcha;
**no `__EVENTVALIDATION`** field on this page (only `__VIEWSTATE` /
`__VIEWSTATEGENERATOR`).

1. `GET /` → harvest all hidden inputs.
2. `POST /` with the hidden state plus:
   - `ctl00$Content_home_Public$ctl00$txtKeyword` = `<case number>`
   - `ctl00$Content_home_Public$ctl00$cmd_search_home` = `Tìm kiếm`
3. Server 200-redirects to `/0t15at1cvn1/Tra-cu-ban-an` and returns 20 hits/page.
   Each hit:
   ```html
   <a class="echo_id_pub" href="/2ta{id}t1cvn/chi-tiet-ban-an">
     <span>số {case_no} ngày {dd/mm/yyyy} của {court} (publish_date)</span>
   </a>
   ```

The keyword search is substring/relevance based — querying `17/2018/HS-ST` also
returns `217/2018/HS-ST`. The match logic therefore re-validates the
**canonical** case number strictly (core `num/year` must agree) instead of
trusting result order.

The shared portal session (`verify=False` for the incomplete TLS chain, browser
UA) is reused from `crawler/portal_client.make_session`; the search field names
live there too (`SEARCH_KEYWORD_FIELD` / `SEARCH_SUBMIT_FIELD`).

## Case-number normalization

`normalize_case_number` collapses real-world variants to a canonical key:

- suffix punctuation: `DS-ST` = `DSST` = `DS_ST`
- spacing: `17 / 2018 / HSST` = `17/2018/HS-ST`
- accent on `Đ/đ`: `QĐDS` = `QDDS`
- OCR lowercase: `hs_st` = `HS-ST`

A `_core_number` guard (leading `num/year`) rejects substring-relevance noise
(`172018` ≠ `2172018`).

## Scoring

`score = 0.6·case + 0.25·court_overlap + 0.15·date_match`, where:
- `case` = 1.0 (exact canonical) or 0.6 (same num/year, different suffix).
- `court_overlap` = token Jaccard on accent-stripped court strings (0.5 if no
  court supplied in the query).
- `date_match` = 1.0 on exact ISO date, 0.5 if no date supplied, 0.0 on
  mismatch. `found` requires an exact case-number match and score ≥ 0.85.

## Usage

### Python
```python
from verify.existence import check_existence
report = check_existence(
    "17/2018/HS-ST",
    court="TAND Quận 11, TP. Hồ Chí Minh",
    judgment_date="20/03/2019",   # dd/mm/yyyy or ISO
)
print(report.status, report.confidence)
print(report.model_dump_json(indent=2))
```

### CLI
```bash
python -m verify.existence --case-number "17/2018/HS-ST" \
    --court "TAND Quận 11, TP. Hồ Chí Minh" --date 20/03/2019 [--json]
```
Polite by default: 1 s delay, 30 s timeout, one retry on transient transport
error.

## Output schema (`ExistenceReport`, pydantic v2)
```json
{
  "status": "found|found_partial|not_found|error",
  "confidence": 0.0,
  "queried": {"case_number": "...", "court": "...", "judgment_date": "..."},
  "matches": [{"case_number": "...", "court": "...", "date": "YYYY-MM-DD",
               "detail_url": "https://...", "score": 0.0}],
  "caveat_vi": "Không tìm thấy KHÔNG đồng nghĩa với giả mạo ...",
  "summary_vi": "..."
}
```

## Reliability (observed live, 2026-06-11)

- **12/12 distinct known-real judgments were located** (`found` or
  `found_partial`) — zero false negatives. Search-by-number reliably finds a
  published judgment.
- The obviously-fake `9999/2099/HS-ST` correctly returned `not_found` with the
  caveat.
- **Common case numbers are ambiguous.** Numbers like `16/2018/HS-ST` exist for
  70+ different courts; the portal returns only the first 20 (its own
  relevance order). If the exact court/date isn't on page 1, the module
  honestly reports `found_partial` (the number exists) rather than over-claiming
  `found`. Supplying the full court name + exact date upgrades the
  unambiguous cases to `found` (conf 1.0).
- No rate limiting or captcha observed at a 1–2 s cadence; the portal was
  stable across the test runs. TLS chain remains incomplete → `verify=False`.

### Known limitations
- **Page-1 only** — for very common numbers the exact judgment may be on a
  later results page; the module degrades safely to `found_partial`. A future
  improvement is to drive the result pager (additional `__VIEWSTATE` postbacks)
  when the first page is exhausted without an exact court/date hit.
- Court matching is token-overlap; abbreviation differences (`Q.` vs `Quận`)
  lower the score but rarely flip an exact-number match below the gate when the
  province is present.
