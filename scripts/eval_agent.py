"""Quantitative evaluation harness for the Ruvisa agentic LLM.

Runs a curated query set against the real ReAct agent (Ollama) over the real
SQLite DB and reports:
  - Tool-selection accuracy (required-tool recall, full-set rate, forbidden violations)
  - Tool-chain length and per-turn latency
  - Grounding ablation: hallucinated-product rate with tools ON vs OFF
    (hallucination = product-like mention not found in the catalogue)

Usage:
  PYTHONPATH=. python3 scripts/eval_agent.py [--repeats N] [--out FILE]

NOTE: some queries call track_purchase / evaluate_outcomes which write to the DB.
Back up data/skincare.db before running and restore afterwards.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

import db.crud as crud
from agent.graph import _build_user_context
from agent.llm import get_llm
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# Curated query set. required_tools = tools that SHOULD be called for the intent.
# Users chosen to have the history each query needs.
# ---------------------------------------------------------------------------
U_HIST = "928d6b37a001ac29"   # 3 analyses, 5 purchases, 4 outcomes
U_HIST2 = "4d105fdd9f9f7cae"  # 2 analyses, 5 purchases, 6 outcomes
U_SCANS = "a17ad82ed25544fe"  # 7 analyses

QUERIES = [
    # --- single-intent (tool-selection) ---
    {"id": "Q1", "user": U_HIST, "cat": "recommend",
     "msg": "Recommend me some products for my acne.",
     "required": ["recommend_products"]},
    {"id": "Q2", "user": U_HIST, "cat": "routine",
     "msg": "Build me a full skincare routine under $80.",
     "required": ["recommend_routine"]},
    {"id": "Q3", "user": U_HIST, "cat": "search",
     "msg": "Search for cheap products under $30 for pores.",
     "required": ["search_products"]},
    {"id": "Q4", "user": U_HIST, "cat": "product_info",
     "msg": "Tell me the ingredients and details of the Blemish Clearing Cleanser.",
     "required": ["get_product_info"]},
    {"id": "Q5", "user": U_HIST, "cat": "profile",
     "msg": "What is in my profile and what have I purchased so far?",
     "required": ["get_user_profile"]},
    {"id": "Q6", "user": U_HIST2, "cat": "compare",
     "msg": "How has my skin changed since my last scan?",
     "required": ["compare_analyses"]},
    {"id": "Q7", "user": U_HIST2, "cat": "outcomes",
     "msg": "Did the products I bought actually work for my skin?",
     "required": ["evaluate_outcomes"]},
    {"id": "Q8", "user": U_HIST, "cat": "purchase",
     "msg": "I just bought the Daily Milkfoliant Exfoliator, please record it.",
     "required": ["track_purchase"]},
    {"id": "Q9", "user": U_HIST, "cat": "explain",
     "msg": "Why are you recommending the Blemish Clearing Cleanser for my acne?",
     "required": ["explain_recommendation"]},
    # --- compound (chaining) ---
    {"id": "P1", "user": U_HIST2, "cat": "compound",
     "msg": "Evaluate my previous recommendations, compare my latest scan to the last one, "
            "and then recommend updated products.",
     "required": ["evaluate_outcomes", "compare_analyses", "recommend_products"]},
    {"id": "P2", "user": U_HIST, "cat": "compound",
     "msg": "Build a full routine under $80, suggest 3 extra products for acne and redness, "
            "and look up details on the top serum.",
     "required": ["recommend_routine", "search_products", "get_product_info"]},
    {"id": "P4", "user": U_HIST2, "cat": "compound",
     "msg": "Record that I purchased the Lala Retro Whipped Cream, re-evaluate my outcomes, "
            "and recommend my next products.",
     "required": ["track_purchase", "evaluate_outcomes", "recommend_products"]},
]

# ---------------------------------------------------------------------------
# Catalogue for hallucination checking
# ---------------------------------------------------------------------------
_GENERIC = {
    "the", "and", "for", "with", "a", "of", "to", "in", "skin", "face", "facial",
    "cream", "serum", "oil", "gel", "cleanser", "toner", "lotion", "mask", "moisturizer",
    "moisturiser", "acid", "treatment", "exfoliant", "exfoliator", "spf", "sunscreen",
    "daily", "night", "day", "repair", "clear", "clearing", "hydrating", "brightening",
    "2%", "1%", "5%", "10%", "%", "bha", "aha", "vitamin", "c", "&", "-",
}


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9%& ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_catalog_norm() -> list[set]:
    cat = crud.get_all_products()
    out = []
    for v in cat.values():
        for field in ("full_name", "title"):
            t = v.get(field)
            if t:
                toks = set(_norm(f"{v.get('brand','')} {t}").split())
                if toks:
                    out.append(toks)
    return out


_REJECT_PREFIX = (
    "recommend", "here", "since", "note", "based", "i ", "your", "this", "these",
    "for ", "to ", "if ", "you ", "step", "use ", "apply", "morning", "evening",
    "first", "next", "then", "finally", "lastly", "exfoliate", "cleanse",
    "moisturize", "moisturise", "avoid", "track", "tracking", "consider", "try",
    "remember", "start", "continue", "look", "check", "build", "search",
    "evaluate", "compare", "results", "result", "tips", "tip", "important",
    "summary", "overall", "additional", "other", "more", "optional", "in ",
    "with ", "after", "before", "do ", "we ", "let ", "feel", "keep", "make",
    "find", "get ", "see ", "ask", "would", "could", "should", "also", "by ",
)

# Tokens that mark a markdown *section header* rather than a product name.
_HEADER_WORDS = {
    "information", "recommendation", "recommendations", "comparison", "progress",
    "takeaway", "takeaways", "routine", "outcome", "outcomes", "profile",
    "history", "cost", "scans", "scan", "summary", "recorded", "evaluating",
    "evaluation", "overview", "breakdown", "notes", "ingredients", "active",
    "name", "id", "type", "total", "key", "steps", "step", "purchase",
    "purchases", "reducing", "targeted", "extra", "options", "option",
}


def extract_product_candidates(text: str) -> list[str]:
    """Heuristically pull product-name-like spans (bold spans + list heads)."""
    cands = []
    cands += re.findall(r"\*\*(.+?)\*\*", text)
    for line in text.splitlines():
        m = re.match(r"\s*(?:\d+\.|[-*•])\s+(.+)", line)
        if m:
            seg = re.split(r"[:–—]| - | \(| by ", m.group(1))[0]
            cands.append(seg)
    out = []
    for c in cands:
        c = re.sub(r"\*+", "", c).strip(" .*:-")
        low = c.lower()
        nwords = len(c.split())
        if not (3 <= len(c) <= 70) or nwords > 7 or nwords < 2:
            continue
        if c.endswith((".", "!", "?")) or "," in c:
            continue
        if low.startswith(_REJECT_PREFIX):
            continue
        if "$" in c or " under " in low or low.startswith("under "):
            continue
        toks = set(re.sub(r"[^a-z0-9 ]", " ", low).split())
        if toks & _HEADER_WORDS:
            continue
        # product names are mostly Title-Case / contain a brand or digit
        caps = sum(1 for w in c.split() if w[:1].isupper())
        if caps < 2 and not re.search(r"\d", c):
            continue
        out.append(c)
    return list(dict.fromkeys(out))


def candidate_in_catalog(cand: str, catalog: list[set], thresh: float = 0.6) -> bool:
    ctoks = set(_norm(cand).split()) - _GENERIC
    if not ctoks:
        return True  # too generic to judge; don't count as hallucination
    best = 0.0
    for ttoks in catalog:
        dist = ttoks - _GENERIC
        if not dist:
            continue
        cover = len(ctoks & dist) / len(ctoks)
        if cover > best:
            best = cover
            if best >= thresh:
                return True
    return best >= thresh


# ---------------------------------------------------------------------------
# Agent runners (grounded = tools on; ungrounded = no tools)
# ---------------------------------------------------------------------------
_grounded_graph = None
_ungrounded_graph = None


def get_graph(grounded: bool):
    global _grounded_graph, _ungrounded_graph
    if grounded:
        if _grounded_graph is None:
            _grounded_graph = create_react_agent(
                get_llm(), tools=ALL_TOOLS, prompt=SystemMessage(content=SYSTEM_PROMPT))
        return _grounded_graph
    if _ungrounded_graph is None:
        _ungrounded_graph = create_react_agent(
            get_llm(), tools=[], prompt=SystemMessage(content=SYSTEM_PROMPT))
    return _ungrounded_graph


async def run_once(user_id: str, message: str, grounded: bool) -> dict:
    graph = get_graph(grounded)
    ctx = _build_user_context(user_id)
    full = ctx + "User message: " + message
    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=full)]},
        config={"configurable": {"thread_id": user_id}},
    )
    latency = time.perf_counter() - t0
    tools_used = []
    for m in result["messages"]:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tools_used.append(tc.get("name", "unknown"))
    ai = [m for m in result["messages"] if m.type == "ai" and m.content]
    response = ai[-1].content if ai else ""
    return {"latency": latency, "tools_used": tools_used, "response": response}


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="docs/agent_eval_results.json")
    args = ap.parse_args()

    catalog = load_catalog_norm()
    records = []

    for cond in ("grounded", "ungrounded"):
        grounded = cond == "grounded"
        reps = args.repeats if grounded else 1
        for q in QUERIES:
            for r in range(reps):
                out = await run_once(q["user"], q["msg"], grounded)
                used = out["tools_used"]
                req = set(q["required"])
                req_hit = len(req & set(used)) / len(req) if req else 1.0
                full_set = req.issubset(set(used))
                cands = extract_product_candidates(out["response"])
                hall = [c for c in cands if not candidate_in_catalog(c, catalog)]
                rec = {
                    "id": q["id"], "cat": q["cat"], "cond": cond, "rep": r,
                    "required": q["required"], "tools_used": used,
                    "n_tools": len(used), "req_recall": req_hit, "full_set": full_set,
                    "latency": round(out["latency"], 2),
                    "n_cands": len(cands), "n_hall": len(hall),
                    "hall_rate": (len(hall) / len(cands)) if cands else None,
                    "hall_examples": hall[:4],
                    "resp_len": len(out["response"]),
                    "response": out["response"][:1500],
                }
                records.append(rec)
                print(f"[{cond:10}] {q['id']:3} rep{r} "
                      f"tools={used} req_recall={req_hit:.2f} "
                      f"lat={out['latency']:.1f}s cands={len(cands)} hall={len(hall)}",
                      flush=True)

    # ---- aggregate ----
    g = [x for x in records if x["cond"] == "grounded"]
    u = [x for x in records if x["cond"] == "ungrounded"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else float("nan")

    def std(xs):
        xs = [x for x in xs if x is not None]
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    g_hall = [x["hall_rate"] for x in g if x["hall_rate"] is not None]
    u_hall = [x["hall_rate"] for x in u if x["hall_rate"] is not None]

    summary = {
        "n_queries": len(QUERIES),
        "grounded_repeats": args.repeats,
        "model": "llama3.2:latest",
        "tool_selection": {
            "mean_required_recall": round(mean([x["req_recall"] for x in g]), 4),
            "full_required_set_rate": round(mean([1.0 if x["full_set"] else 0.0 for x in g]), 4),
            "mean_tools_per_query": round(mean([x["n_tools"] for x in g]), 2),
        },
        "latency_s": {
            "mean": round(mean([x["latency"] for x in g]), 2),
            "std": round(std([x["latency"] for x in g]), 2),
            "min": round(min(x["latency"] for x in g), 2),
            "max": round(max(x["latency"] for x in g), 2),
        },
        "hallucination_rate": {
            "grounded_mean": round(mean(g_hall), 4),
            "ungrounded_mean": round(mean(u_hall), 4),
            "grounded_n_with_products": len(g_hall),
            "ungrounded_n_with_products": len(u_hall),
        },
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
