"""CRUD operations for the skincare database."""

import hashlib
import json
import secrets
from datetime import datetime

from .database import get_db


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, h = stored.split(":", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == h


def register_user(
    email: str, password: str, name: str,
    skin_type: str | None = None, concerns: list[str] | None = None,
) -> dict | None:
    """Register a new user. Returns user dict or None if email exists."""
    conn = get_db()
    existing = conn.execute(
        "SELECT user_id FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    if existing:
        conn.close()
        return None

    user_id = secrets.token_hex(8)
    pw_hash = _hash_password(password)
    conn.execute(
        """INSERT INTO users (user_id, name, email, password_hash, skin_type, concerns)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, name, email.lower(), pw_hash, skin_type,
         json.dumps(concerns) if concerns else None),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_dict(row)


def login_user(email: str, password: str) -> dict | None:
    """Authenticate user. Returns user dict or None if invalid."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return _user_dict(row)


def _user_dict(row) -> dict:
    d = dict(row)
    d.pop("password_hash", None)
    if d.get("concerns") and isinstance(d["concerns"], str):
        try:
            d["concerns"] = json.loads(d["concerns"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def create_user(user_id: str, name: str | None = None,
                skin_type: str | None = None) -> dict:
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, name, skin_type) VALUES (?, ?, ?)",
        (user_id, name, skin_type),
    )
    if skin_type:
        conn.execute(
            "UPDATE users SET skin_type = ? WHERE user_id = ?",
            (skin_type, user_id),
        )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def update_user_profile(user_id: str, skin_type: str | None = None,
                        concerns: list[str] | None = None) -> dict | None:
    conn = get_db()
    if skin_type:
        conn.execute("UPDATE users SET skin_type = ? WHERE user_id = ?", (skin_type, user_id))
    if concerns is not None:
        conn.execute("UPDATE users SET concerns = ? WHERE user_id = ?",
                     (json.dumps(concerns), user_id))
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_dict(row) if row else None


def update_user_settings(user_id: str, name: str | None = None,
                         email: str | None = None,
                         current_password: str | None = None,
                         new_password: str | None = None) -> dict | str:
    """Update user name, email, password. Returns updated user dict or error string."""
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return "User not found"

    if new_password:
        if not current_password or not _verify_password(current_password, row["password_hash"]):
            conn.close()
            return "Current password is incorrect"
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                     (_hash_password(new_password), user_id))

    if name and name.strip():
        conn.execute("UPDATE users SET name = ? WHERE user_id = ?", (name.strip(), user_id))

    if email and email.strip():
        existing = conn.execute("SELECT user_id FROM users WHERE email = ? AND user_id != ?",
                                (email.lower().strip(), user_id)).fetchone()
        if existing:
            conn.close()
            return "Email already in use"
        conn.execute("UPDATE users SET email = ? WHERE user_id = ?", (email.lower().strip(), user_id))

    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_dict(updated)


def get_user(user_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return _user_dict(row)


def save_analysis(
    user_id: str,
    image_path: str | None = None,
    concern_vector: list[float] | None = None,
    acne_summary: dict | None = None,
    wrinkle_summary: dict | None = None,
    full_report: dict | None = None,
) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO analyses
           (user_id, image_path, concern_vector, acne_summary,
            wrinkle_summary, full_report)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            image_path,
            json.dumps(concern_vector) if concern_vector else None,
            json.dumps(acne_summary) if acne_summary else None,
            json.dumps(wrinkle_summary) if wrinkle_summary else None,
            json.dumps(full_report) if full_report else None,
        ),
    )
    conn.commit()
    analysis_id = cur.lastrowid
    conn.close()
    return analysis_id


def get_analysis_history(user_id: str, limit: int = 10) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM analyses WHERE user_id = ?
           ORDER BY created_at DESC, id DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        for key in ("concern_vector", "acne_summary", "wrinkle_summary", "full_report"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    return results


def save_recommendations(
    analysis_id: int, user_id: str, products: list[dict]
) -> int:
    conn = get_db()
    count = 0
    for p in products:
        conn.execute(
            """INSERT INTO recommendations
               (analysis_id, user_id, product_url, product_title, brand,
                category, similarity, price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                user_id,
                p.get("product_url"),
                p.get("title", p.get("product_title")),
                p.get("brand"),
                p.get("category"),
                p.get("cosine_similarity", p.get("similarity")),
                p.get("price_value", p.get("price")),
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def get_recommendations(user_id: str, limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT r.*, a.concern_vector, a.created_at as analysis_date
           FROM recommendations r
           JOIN analyses a ON r.analysis_id = a.id
           WHERE r.user_id = ?
           ORDER BY r.created_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_purchase(
    user_id: str, product_url: str,
    product_title: str | None = None, price: float | None = None,
) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO purchases (user_id, product_url, product_title, price)
           VALUES (?, ?, ?, ?)""",
        (user_id, product_url, product_title, price),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def get_purchase_history(user_id: str, limit: int = 20) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM purchases WHERE user_id = ?
           ORDER BY purchased_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_purchased_product_urls(user_id: str) -> set[str]:
    """Return set of product URLs the user has purchased (for filtering recommendations)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT product_url FROM purchases WHERE user_id = ? AND product_url IS NOT NULL AND product_url != ''",
        (user_id,),
    ).fetchall()
    conn.close()
    return {r["product_url"] for r in rows}


def add_to_bag(user_id: str, product_url: str, product_title: str | None = None,
               brand: str | None = None, price: float | None = None,
               image_url: str | None = None) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT OR IGNORE INTO bag (user_id, product_url, product_title, brand, price, image_url)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, product_url, product_title, brand, price, image_url),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def remove_from_bag(user_id: str, product_url: str) -> bool:
    conn = get_db()
    cur = conn.execute(
        "DELETE FROM bag WHERE user_id = ? AND product_url = ?",
        (user_id, product_url),
    )
    conn.commit()
    removed = cur.rowcount > 0
    conn.close()
    return removed


def get_bag(user_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM bag WHERE user_id = ? ORDER BY added_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def toggle_like(user_id: str, product_url: str, product_title: str | None = None,
                brand: str | None = None, price: float | None = None,
                image_url: str | None = None) -> bool:
    """Toggle like. Returns True if liked, False if unliked."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM liked WHERE user_id = ? AND product_url = ?",
        (user_id, product_url),
    ).fetchone()
    if existing:
        conn.execute("DELETE FROM liked WHERE id = ?", (existing["id"],))
        conn.commit()
        conn.close()
        return False
    else:
        conn.execute(
            """INSERT INTO liked (user_id, product_url, product_title, brand, price, image_url)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, product_url, product_title, brand, price, image_url),
        )
        conn.commit()
        conn.close()
        return True


def get_liked(user_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM liked WHERE user_id = ? ORDER BY liked_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_liked_urls(user_id: str) -> set[str]:
    conn = get_db()
    rows = conn.execute(
        "SELECT product_url FROM liked WHERE user_id = ?", (user_id,),
    ).fetchall()
    conn.close()
    return {r["product_url"] for r in rows}


def save_product_outcome(
    user_id: str,
    product_url: str,
    analysis_before: int,
    analysis_after: int,
    concern_deltas: dict[str, float],
    outcome: str,
) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO product_outcomes
           (user_id, product_url, analysis_before, analysis_after,
            concern_deltas, outcome)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, product_url, analysis_before, analysis_after,
         json.dumps(concern_deltas), outcome),
    )
    conn.commit()
    oid = cur.lastrowid
    conn.close()
    return oid


def get_product_outcomes(user_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM product_outcomes WHERE user_id = ?
           ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        if d.get("concern_deltas"):
            try:
                d["concern_deltas"] = json.loads(d["concern_deltas"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def get_failed_product_urls(user_id: str) -> dict[str, dict]:
    """Return {product_url: {concern_deltas, outcome}} for products that worsened/didn't help."""
    outcomes = get_product_outcomes(user_id)
    failed = {}
    for o in outcomes:
        if o["outcome"] in ("worsened", "mixed", "no_change"):
            failed[o["product_url"]] = {
                "concern_deltas": o.get("concern_deltas", {}),
                "outcome": o["outcome"],
            }
    return failed


def evaluate_product_outcomes(user_id: str) -> dict:
    """Compare last 2 scans and attribute outcomes to purchased/recommended products.
    Returns {evaluated_count, outcomes: [{product_url, deltas, outcome}]}."""
    concerns = ["acne", "comedonal_acne", "pigmentation",
                 "acne_scars_texture", "pores", "redness", "wrinkles"]

    analyses = get_analysis_history(user_id, limit=2)
    if len(analyses) < 2:
        return {"evaluated_count": 0, "outcomes": [],
                "reason": "Need at least 2 scans"}

    latest = analyses[0]
    previous = analyses[1]

    cv_new = latest.get("concern_vector", [])
    cv_old = previous.get("concern_vector", [])
    if isinstance(cv_new, str):
        cv_new = json.loads(cv_new) if cv_new else []
    if isinstance(cv_old, str):
        cv_old = json.loads(cv_old) if cv_old else []

    if len(cv_new) != 7 or len(cv_old) != 7:
        return {"evaluated_count": 0, "outcomes": [],
                "reason": "Invalid concern vectors"}

    deltas = {c: round(cv_new[i] - cv_old[i], 4) for i, c in enumerate(concerns)}

    conn = get_db()
    prev_date = previous.get("created_at", "1970-01-01")
    recs = conn.execute(
        """SELECT DISTINCT product_url FROM recommendations
           WHERE user_id = ? AND created_at >= ?""",
        (user_id, prev_date),
    ).fetchall()
    purchases = conn.execute(
        """SELECT DISTINCT product_url FROM purchases
           WHERE user_id = ? AND purchased_at >= ?""",
        (user_id, prev_date),
    ).fetchall()
    conn.close()

    used_urls = {r["product_url"] for r in recs} | {p["product_url"] for p in purchases}
    used_urls.discard(None)
    used_urls.discard("")

    evaluated = []
    for url in used_urls:
        worsened = sum(1 for c, d in deltas.items() if d > 0.03)
        improved = sum(1 for c, d in deltas.items() if d < -0.03)

        if worsened > improved:
            outcome = "worsened"
        elif improved > worsened:
            outcome = "improved"
        elif worsened == 0 and improved == 0:
            outcome = "no_change"
        else:
            outcome = "mixed"

        save_product_outcome(
            user_id, url, previous["id"], latest["id"], deltas, outcome,
        )
        evaluated.append({
            "product_url": url,
            "concern_deltas": deltas,
            "outcome": outcome,
        })

    return {"evaluated_count": len(evaluated), "outcomes": evaluated,
            "deltas": deltas}


def count_catalog_rows() -> tuple[int, int]:
    """Return (n_products, n_review_score_rows) without loading JSON blobs."""
    conn = get_db()
    np = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    nr = conn.execute("SELECT COUNT(*) FROM product_review_scores").fetchone()[0]
    conn.close()
    return int(np), int(nr)


def get_all_products() -> dict:
    """Load all products from DB as {product_url: product_dict}. Fast indexed read."""
    conn = get_db()
    rows = conn.execute("SELECT product_url, data FROM products").fetchall()
    conn.close()
    products = {}
    for row in rows:
        try:
            d = json.loads(row["data"])
            products[row["product_url"]] = d
        except (json.JSONDecodeError, TypeError):
            continue
    return products


def get_all_product_review_scores() -> dict:
    """Load all product review scores from DB as {product_url: scores_dict}."""
    conn = get_db()
    rows = conn.execute("SELECT product_url, data FROM product_review_scores").fetchall()
    conn.close()
    reviews = {}
    for row in rows:
        try:
            reviews[row["product_url"]] = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue
    return reviews


def get_products_paginated(
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int, list[str]]:
    """Get products with optional category filter. Returns (items, total, categories)."""
    conn = get_db()
    if category:
        total = conn.execute(
            "SELECT COUNT(*) FROM products WHERE LOWER(category) = LOWER(?)",
            (category,),
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT data FROM products WHERE LOWER(category) = LOWER(?)
               ORDER BY rating DESC LIMIT ? OFFSET ?""",
            (category, limit, offset),
        ).fetchall()
    else:
        total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        rows = conn.execute(
            """SELECT data FROM products ORDER BY rating DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    categories = [r[0] for r in conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()]
    conn.close()
    items = []
    for row in rows:
        try:
            items.append(json.loads(row["data"]))
        except (json.JSONDecodeError, TypeError):
            continue
    return items, total, categories


def compute_skin_improvement(user_id: str) -> dict | None:
    """Compare latest vs previous analysis. Returns improvement_pct (positive = improved) and details."""
    analyses = get_analysis_history(user_id, limit=2)
    if len(analyses) < 2:
        return None

    latest = analyses[0]
    previous = analyses[1]

    cv_latest = latest.get("concern_vector", [])
    cv_prev = previous.get("concern_vector", [])
    if isinstance(cv_latest, str):
        cv_latest = json.loads(cv_latest) if cv_latest else []
    if isinstance(cv_prev, str):
        cv_prev = json.loads(cv_prev) if cv_prev else []

    if not cv_latest or not cv_prev or len(cv_latest) != 7 or len(cv_prev) != 7:
        return None

    # Lower concern = better. Improvement = (old_avg - new_avg) / old_avg * 100
    avg_old = sum(cv_prev) / 7
    avg_new = sum(cv_latest) / 7
    if avg_old < 0.01:
        improvement_pct = 0.0
    else:
        improvement_pct = ((avg_old - avg_new) / avg_old) * 100  # positive = improved

    return {
        "improvement_pct": round(improvement_pct, 1),
        "latest_date": latest.get("created_at"),
        "previous_date": previous.get("created_at"),
        "latest_score": round(100 - avg_new * 80, 0),
        "previous_score": round(100 - avg_old * 80, 0),
        "concern_changes": {
            c: round(cv_latest[i] - cv_prev[i], 3)
            for i, c in enumerate(
                ["acne", "comedonal_acne", "pigmentation", "acne_scars_texture", "pores", "redness", "wrinkles"]
            )
        },
    }
