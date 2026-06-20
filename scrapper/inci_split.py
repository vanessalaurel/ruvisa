"""
Split INCI ingredient strings on commas while respecting:
  - Parentheses (commas inside INCI names)
  - Numbered diols / glycols: "1, 2-Hexanediol", "1,2-Hexanediol" (do not split)
"""

from __future__ import annotations

import re
from typing import List


def normalize_inci_raw(text: str) -> str:
    """Normalize unicode dashes and whitespace for consistent splitting."""
    if not text:
        return ""
    t = text.replace("\xa0", " ")
    t = re.sub(r"[\u2013\u2014\u2212]", "-", t)  # en dash, em dash, minus → hyphen
    t = re.sub(r"\s+", " ", t).strip()
    # Common Sephora/INCI typo: comma should be hyphen in this INCI name
    t = re.sub(r"\bAlpha,Glucan\b", "Alpha-Glucan", t, flags=re.IGNORECASE)
    return t


def _is_single_digit_token(s: str) -> bool:
    """True if current segment is only a single digit (e.g. '1' before ', 2-...')."""
    s = s.strip()
    return len(s) == 1 and s.isdigit()


def _comma_after_letter_then_digit(text: str, i: int, cur: str) -> bool:
    """INCI typos: 'Biosaccharide Gum,1' (Gum-1); merge comma.

    Do NOT merge 'Coco-Glucoside, 1' when followed by ',2-...' — that starts 1,2-Hexanediol.
    """
    cst = cur.rstrip()
    if not cst or not cst[-1].isalpha():
        return False
    j = i + 1
    n = len(text)
    while j < n and text[j] in " \t":
        j += 1
    if j >= n or not text[j].isdigit():
        return False
    k = j
    while k < n and text[k].isdigit():
        k += 1
    while k < n and text[k] in " \t":
        k += 1
    if k < n and text[k] == ",":
        k2 = k + 1
        while k2 < n and text[k2] in " \t":
            k2 += 1
        if k2 < n and text[k2].isdigit():
            return False
    return True


def split_inci_list(ingredients_raw: str) -> List[str]:
    """
    Split a comma-separated INCI blob into ingredient names.

    Commas inside (...) are ignored. Commas between a lone digit and the next
    digit (e.g. 1,2- or 1, 2-) are kept (numbered organic names).
    """
    text = normalize_inci_raw(ingredients_raw)
    if not text:
        return []

    out: List[str] = []
    cur = ""
    depth = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth = max(0, depth - 1)
            cur += ch
        elif ch == "," and depth == 0:
            if _is_single_digit_token(cur):
                j = i + 1
                while j < n and text[j] in " \t":
                    j += 1
                if j < n and text[j].isdigit():
                    cur += ch
                    i += 1
                    continue
            if _comma_after_letter_then_digit(text, i, cur):
                cur += ch
                i += 1
                continue
            part = cur.strip()
            if part:
                out.append(part)
            cur = ""
        else:
            cur += ch
        i += 1

    if cur.strip():
        out.append(cur.strip())

    cleaned: List[str] = []
    for p in out:
        p = re.sub(r"\s+", " ", p).strip()
        p = p.rstrip(".;")
        if p:
            cleaned.append(p)
    return cleaned
