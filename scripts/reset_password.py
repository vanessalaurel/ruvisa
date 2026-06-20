#!/usr/bin/env python3
"""Reset a user's password by email (local dev / server admin use).

Usage:
  PYTHONPATH=. python3 scripts/reset_password.py nessa04@gmail.com newpassword123
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.database import get_db, init_db
from db.crud import _hash_password


def main():
    if len(sys.argv) != 3:
        print("Usage: PYTHONPATH=. python3 scripts/reset_password.py <email> <new_password>")
        sys.exit(1)

    email = sys.argv[1].strip().lower()
    new_password = sys.argv[2]
    if len(new_password) < 6:
        print("Password must be at least 6 characters.")
        sys.exit(1)

    init_db()
    conn = get_db()
    row = conn.execute("SELECT user_id, name, email FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        print(f"No user found for email: {email}")
        print("\nRegistered emails:")
        for r in conn.execute("SELECT email, name FROM users ORDER BY created_at DESC"):
            print(f"  - {r['email']} ({r['name']})")
        conn.close()
        sys.exit(1)

    conn.execute(
        "UPDATE users SET password_hash = ? WHERE user_id = ?",
        (_hash_password(new_password), row["user_id"]),
    )
    conn.commit()
    conn.close()
    print(f"Password updated for {row['name']} <{row['email']}>")


if __name__ == "__main__":
    main()
