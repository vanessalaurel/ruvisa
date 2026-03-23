"""LangGraph agent graph: ReAct-style tool-calling loop."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .llm import get_llm
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS

_graph_cache: dict = {}


def build_graph(model: str | None = None):
    key = model or "default"
    if key in _graph_cache:
        return _graph_cache[key]

    llm = get_llm(model=model)
    graph = create_react_agent(
        llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    _graph_cache[key] = graph
    return graph


def _build_user_context(user_id: str) -> str:
    """Build rich context string from user profile, latest scan results, purchases, and improvement."""
    import db.crud as crud
    import json

    concern_names = [
        "acne", "comedonal_acne", "pigmentation",
        "acne_scars_texture", "pores", "redness", "wrinkles",
    ]

    user = crud.get_user(user_id)
    if not user:
        return ""

    analyses = crud.get_analysis_history(user_id, limit=3)
    purchases = crud.get_purchase_history(user_id, limit=10)
    improvement = crud.compute_skin_improvement(user_id)

    parts = [
        f"[User: {user.get('name') or user_id} | Skin type: {user.get('skin_type') or 'not set'} | Total scans: {len(analyses)} | Total purchases: {len(purchases)}]",
    ]

    if analyses:
        latest = analyses[0]
        cv = latest.get("concern_vector", [])
        if isinstance(cv, str):
            try:
                cv = json.loads(cv)
            except (json.JSONDecodeError, TypeError):
                cv = []

        if cv and len(cv) == 7:
            overall = max(20, min(98, int(100 - (sum(cv) / 7) * 80)))
            concern_strs = []
            for c, v in zip(concern_names, cv):
                if v > 0.05:
                    concern_strs.append(f"{c}={v:.2f}")
            parts.append(
                f"[Latest scan ({latest.get('created_at', 'unknown')}): "
                f"skin score={overall}/100, concerns: {', '.join(concern_strs) if concern_strs else 'none detected'}]"
            )

        acne_s = latest.get("acne_summary", {})
        if isinstance(acne_s, str):
            try:
                acne_s = json.loads(acne_s)
            except (json.JSONDecodeError, TypeError):
                acne_s = {}
        if acne_s.get("total_detections"):
            region_info = ""
            regions = acne_s.get("regions", {})
            if regions:
                region_info = ", ".join(f"{r}: {d.get('count', 0)} lesions" for r, d in regions.items() if d.get("count", 0) > 0)
            parts.append(
                f"[Acne: {acne_s['total_detections']} total lesions. "
                f"Severity: {acne_s.get('severity_distribution', {})}. "
                f"Regions: {region_info or 'n/a'}]"
            )

        wrinkle_s = latest.get("wrinkle_summary", {})
        if isinstance(wrinkle_s, str):
            try:
                wrinkle_s = json.loads(wrinkle_s)
            except (json.JSONDecodeError, TypeError):
                wrinkle_s = {}
        if wrinkle_s.get("wrinkle_pct", 0) > 0 or wrinkle_s.get("severity", "none") != "none":
            wr = wrinkle_s.get("wrinkle_regions", {})
            wr_details = ", ".join(
                f"{r}: {d.get('wrinkle_pct', 0):.2f}% ({d.get('severity', 'none')})"
                for r, d in wr.items()
            ) if wr else "n/a"
            parts.append(
                f"[Wrinkles: overall {wrinkle_s.get('wrinkle_pct', 0):.2f}% coverage, "
                f"severity={wrinkle_s.get('severity', 'none')} ({wrinkle_s.get('severity_score', 0)}/3). "
                f"Regions: {wr_details}]"
            )

    if improvement:
        imp = improvement["improvement_pct"]
        direction = "improved" if imp > 0 else "worsened" if imp < 0 else "unchanged"
        parts.append(
            f"[Skin progress: {direction} {abs(imp):.1f}% since last scan "
            f"(score {improvement['previous_score']} → {improvement['latest_score']})]"
        )

    if purchases:
        titles = [p.get("product_title") or p.get("product_url", "")[:40] for p in purchases[:5]]
        parts.append(f"[Recent purchases: {', '.join(titles)}]")

    return "\n".join(parts) + "\n\n"


async def invoke_agent(
    user_id: str,
    message: str,
    model: str | None = None,
    inject_user_context: bool = True,
) -> dict:
    """Returns {"response": str, "tools_used": list[dict]}"""
    graph = build_graph(model=model)

    if inject_user_context:
        context = _build_user_context(user_id)
        if context:
            message = context + "User message: " + message

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": user_id}},
    )

    tools_used = []
    for m in result["messages"]:
        if m.type == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tools_used.append({
                    "name": tc.get("name", "unknown"),
                    "args": {k: str(v)[:100] for k, v in (tc.get("args") or {}).items()},
                })

    ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
    response = ai_messages[-1].content if ai_messages else "I wasn't able to generate a response. Could you rephrase your question?"

    return {"response": response, "tools_used": tools_used}
