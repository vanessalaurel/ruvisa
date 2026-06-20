#!/usr/bin/env python3
"""
Re-split INCI from ingredients_raw (fixes 1,2-hexanediol style breaks),
sync ingredients_scraped, then re-run evidence labeling.

Updates:
  labeling/products_labeled.jsonl
  labeling/products_evidence_labeled.jsonl

Optional:
  labeling/products_evidence_labeled.json  (array) if --write-json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scrapper.inci_split import split_inci_list  # noqa: E402

LABELING = PROJECT / "labeling"
LABELED = LABELING / "products_labeled.jsonl"
EVIDENCE = LABELING / "products_evidence_labeled.jsonl"
SCRIPT_LABEL = PROJECT / "scripts" / "label_products_evidence.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-json", action="store_true", help="Also write products_evidence_labeled.json array")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not LABELED.exists():
        raise SystemExit(f"Missing {LABELED}")

    if not args.no_backup:
        shutil.copy2(LABELED, LABELED.with_suffix(".jsonl.bak"))
        if EVIDENCE.exists():
            shutil.copy2(EVIDENCE, EVIDENCE.with_suffix(".jsonl.bak"))

    products = []
    changed = 0
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            products.append(json.loads(line))

    for p in products:
        raw = p.get("ingredients_raw")
        if not raw or not isinstance(raw, str):
            continue
        if raw.count(",") < 1:
            continue
        new_list = split_inci_list(raw)
        if not new_list:
            continue
        old = p.get("ingredients")
        if old != new_list:
            changed += 1
        p["ingredients"] = new_list
        p["ingredients_scraped"] = list(new_list)
        p["ingredients_count"] = len(new_list)

    with open(LABELED, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Updated {changed} products with re-parsed INCI lists -> {LABELED}")

    subprocess.run([sys.executable, str(SCRIPT_LABEL)], check=True, cwd=str(PROJECT))
    print(f"Evidence labels written -> {EVIDENCE}")

    if args.write_json:
        out_json = LABELING / "products_evidence_labeled.json"
        rows = []
        with open(EVIDENCE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(rows)} products -> {out_json}")


if __name__ == "__main__":
    main()
