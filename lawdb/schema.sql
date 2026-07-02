-- Vietnamese legal database — SQLite schema (multi-document).
-- Holds several law documents (BLHS / BLTTHS / BLDS / Nghị quyết ...) keyed by
-- a stable `law_key`. Hierarchy per document:
--   documents -> articles (điều) -> clauses (khoản) -> points (điểm)
-- plus penalty_frames attached to clauses for Layer 3 sentencing checks, and an
-- amendments table recording document-level amendment relationships and the
-- validity-at-date metadata (effective_from / effective_to / status).

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS amendments;
DROP TABLE IF EXISTS penalty_frames;
DROP TABLE IF EXISTS points;
DROP TABLE IF EXISTS clauses;
DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id             INTEGER PRIMARY KEY,
    law_key        TEXT NOT NULL UNIQUE,      -- stable key: BLHS2015 | BLTTHS2015 | BLDS2015 | NQ326_2016
    ten            TEXT NOT NULL,             -- "Bộ luật Hình sự"
    so_hieu        TEXT NOT NULL,             -- "100/2015/QH13"
    loai           TEXT,                      -- bo_luat | luat | nghi_quyet
    total_dieu     INTEGER,                   -- expected article count (coverage denominator)
    ngay_ban_hanh  TEXT,                      -- ISO date
    effective_from TEXT,                      -- ISO date the document took effect
    effective_to   TEXT,                      -- ISO date it ceased (NULL = still in force)
    status         TEXT DEFAULT 'in_force',   -- in_force | repealed | amended | not_yet_effective
    nguon_url      TEXT,                      -- acquisition source
    version_note   TEXT
);

CREATE TABLE articles (
    id        INTEGER PRIMARY KEY,
    doc_id    INTEGER NOT NULL REFERENCES documents(id),
    so_dieu   INTEGER NOT NULL,             -- article number
    tieu_de   TEXT,                         -- title, e.g. "Tội trộm cắp tài sản"
    phan      TEXT,                         -- Phần context
    chuong    TEXT,                         -- Chương context
    muc       TEXT,                         -- Mục context (nullable)
    noi_dung  TEXT NOT NULL,                -- full article body
    is_offense INTEGER NOT NULL DEFAULT 0,  -- 1 if offense-defining (BLHS Phần II)
    UNIQUE(doc_id, so_dieu)
);

CREATE TABLE clauses (
    id         INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    so_khoan   INTEGER NOT NULL,            -- clause number within article
    noi_dung   TEXT NOT NULL,
    UNIQUE(article_id, so_khoan)
);

CREATE TABLE points (
    id        INTEGER PRIMARY KEY,
    clause_id INTEGER NOT NULL REFERENCES clauses(id),
    ky_hieu   TEXT NOT NULL,                -- point letter: a, b, đ, ...
    noi_dung  TEXT NOT NULL
);

CREATE TABLE penalty_frames (
    id           INTEGER PRIMARY KEY,
    clause_id    INTEGER NOT NULL REFERENCES clauses(id),
    penalty_type TEXT NOT NULL,             -- tu_co_thoi_han | tu_chung_than |
                                            -- tu_hinh | phat_tien |
                                            -- cai_tao_khong_giam_giu | canh_cao
    min_value    REAL,                      -- nullable (life/death/warning)
    max_value    REAL,
    unit         TEXT,                       -- nam | thang | dong | NULL
    raw_text     TEXT NOT NULL
);

-- Document-level amendment relationships. Each row: amending_law modifies
-- target_doc, taking effect on `effective_from`. Article-level granularity is
-- not captured (honest limitation) — validity is checked at document level.
CREATE TABLE amendments (
    id             INTEGER PRIMARY KEY,
    target_doc_id  INTEGER NOT NULL REFERENCES documents(id),
    amending_key   TEXT,                     -- law_key of the amending instrument (may be external)
    amending_ten   TEXT NOT NULL,            -- "Luật số 12/2017/QH14"
    so_hieu        TEXT,                      -- "12/2017/QH14"
    effective_from TEXT,                      -- ISO date the amendment took effect
    mo_ta          TEXT                       -- short Vietnamese description
);

CREATE INDEX idx_articles_doc_so_dieu ON articles(doc_id, so_dieu);
CREATE INDEX idx_articles_so_dieu ON articles(so_dieu);
CREATE INDEX idx_clauses_article ON clauses(article_id);
CREATE INDEX idx_points_clause ON points(clause_id);
CREATE INDEX idx_penalty_clause ON penalty_frames(clause_id);
CREATE INDEX idx_amend_target ON amendments(target_doc_id);
