import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeling.sephora_hk_labels import label_product, LESION_LABELS

IN_PATH = Path("/home/vanessa/project/data/raw/sephora/products/products.jsonl")
OUT_PATH = Path("/home/vanessa/project/labeling/products_labeled.jsonl")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Ingredients fixers (ADDED)
# ----------------------------

def split_ingredients_robust(ingredients_raw: str) -> List[str]:
    """
    Split on commas EXCEPT:
      - commas between digits (e.g., "1,2-Hexanediol")
      - commas inside parentheses
    """
    if not ingredients_raw:
        return []

    s = str(ingredients_raw)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(";", ",")  # optional normalize

    out: List[str] = []
    buf: List[str] = []
    paren_depth = 0

    for i, ch in enumerate(s):
        if ch == "(":
            paren_depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            buf.append(ch)
            continue

        if ch == "," and paren_depth == 0:
            prev_ch = s[i - 1] if i > 0 else ""
            next_ch = s[i + 1] if i + 1 < len(s) else ""

            # Don't split 1,2-... etc
            if prev_ch.isdigit() and next_ch.isdigit():
                buf.append(ch)
            else:
                token = "".join(buf).strip().strip(" .")
                if token:
                    out.append(token)
                buf = []
        else:
            buf.append(ch)

    token = "".join(buf).strip().strip(" .")
    if token:
        out.append(token)

    return out

def repair_numeric_comma_splits(ingredients_list: List[Any]) -> List[str]:
    """
    Fix scraper artifacts like ["1", "2-Hexanediol"] -> ["1,2-Hexanediol"].
    """
    if not ingredients_list:
        return []

    items = [str(x).strip() for x in ingredients_list if x is not None and str(x).strip()]
    out: List[str] = []
    i = 0

    while i < len(items):
        cur = items[i]

        if re.fullmatch(r"\d+", cur) and i + 1 < len(items):
            nxt = items[i + 1]
            # allow spaces around hyphen: "2 - Hexanediol"
            if re.match(r"^\d+\s*-\s*\w", nxt):
                nxt_clean = re.sub(r"\s*", "", nxt)  # "2-Hexanediol"
                out.append(f"{cur},{nxt_clean}")     # "1,2-Hexanediol"
                i += 2
                continue

        out.append(cur)
        i += 1

    return out

def get_ingredients_list(row: Dict[str, Any]) -> List[str]:
    """
    Unified access:
      - If ingredients_raw exists: robust split from raw
      - Else if ingredients is list: repair it
      - Else if ingredients is string: robust split
    """
    raw = row.get("ingredients_raw")
    if raw:
        return split_ingredients_robust(str(raw))

    ing = row.get("ingredients")
    if isinstance(ing, list):
        return repair_numeric_comma_splits(ing)

    if isinstance(ing, str) and ing.strip():
        return split_ingredients_robust(ing)

    return []

# ----------------------------
# Run labeling + patch ingredients
# ----------------------------

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad JSON on line {line_no}: {e}") from e

        # Fix ingredients BEFORE labeling / saving
        fixed_ingredients = get_ingredients_list(row)
        if fixed_ingredients:
            # keep original if you want to audit what the scraper produced
            if "ingredients" in row and "ingredients_scraped" not in row:
                row["ingredients_scraped"] = row["ingredients"]
            row["ingredients"] = fixed_ingredients

        labels = label_product(row)
        row.update(labels)                 # <- flatten labels into top-level keys
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")