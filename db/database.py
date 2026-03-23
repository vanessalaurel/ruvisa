"""SQLite database setup and connection management."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "skincare.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    name          TEXT,
    email         TEXT,
    password_hash TEXT,
    skin_type     TEXT,
    concerns      TEXT,   -- JSON array of concern strings
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    image_path      TEXT,
    concern_vector  TEXT,   -- JSON array of 7 floats
    acne_summary    TEXT,   -- JSON object
    wrinkle_summary TEXT,   -- JSON object
    full_report     TEXT,   -- JSON object (entire skin_analysis output)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id   INTEGER REFERENCES analyses(id),
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    product_url   TEXT,
    product_title TEXT,
    brand         TEXT,
    category      TEXT,
    similarity    REAL,
    price         REAL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS purchases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    product_url   TEXT,
    product_title TEXT,
    price         REAL,
    purchased_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bag (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    product_url   TEXT NOT NULL,
    product_title TEXT,
    brand         TEXT,
    price         REAL,
    image_url     TEXT,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_url)
);

CREATE TABLE IF NOT EXISTS liked (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    product_url   TEXT NOT NULL,
    product_title TEXT,
    brand         TEXT,
    price         REAL,
    image_url     TEXT,
    liked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_url)
);

CREATE TABLE IF NOT EXISTS product_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    product_url     TEXT NOT NULL,
    analysis_before INTEGER REFERENCES analyses(id),
    analysis_after  INTEGER REFERENCES analyses(id),
    concern_deltas  TEXT,   -- JSON: per-concern change (positive = worsened)
    outcome         TEXT,   -- 'improved', 'worsened', 'mixed', 'no_change'
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_url     TEXT PRIMARY KEY,
    category        TEXT,
    price_value     REAL,
    rating          REAL,
    brand           TEXT,
    title           TEXT,
    data            TEXT    -- JSON: full product dict
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating DESC);

CREATE TABLE IF NOT EXISTS product_review_scores (
    product_url     TEXT PRIMARY KEY,
    data            TEXT    -- JSON: concern_scores, skin_type_scores, etc.
);
"""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
