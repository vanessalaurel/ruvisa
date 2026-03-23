"""API routes for the skincare advisory system."""

import asyncio
import json
import logging
import os
import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse

import db.crud as crud
from agent.graph import invoke_agent

from .schemas import (
    AnalyzeResponse,
    AuthResponse,
    BagRequest,
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    LikeRequest,
    LoginRequest,
    ProfileResponse,
    PurchaseRequest,
    RecommendRequest,
    RegisterRequest,
    UpdateSettingsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]


# ── Auth endpoints ────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    user = crud.register_user(
        email=req.email,
        password=req.password,
        name=req.name,
        skin_type=req.skin_type,
        concerns=req.concerns,
    )
    if not user:
        raise HTTPException(409, "An account with this email already exists")
    return AuthResponse(
        user_id=user["user_id"],
        name=user.get("name"),
        email=user.get("email"),
        skin_type=user.get("skin_type"),
        concerns=user.get("concerns"),
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    user = crud.login_user(email=req.email, password=req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    return AuthResponse(
        user_id=user["user_id"],
        name=user.get("name"),
        email=user.get("email"),
        skin_type=user.get("skin_type"),
        concerns=user.get("concerns"),
    )


@router.put("/profile/{user_id}")
async def update_profile(user_id: str, skin_type: str | None = None,
                         concerns: list[str] | None = None):
    user = crud.update_user_profile(user_id, skin_type=skin_type, concerns=concerns)
    if not user:
        raise HTTPException(404, "User not found")
    return user


@router.post("/settings")
async def update_settings(req: UpdateSettingsRequest):
    result = crud.update_user_settings(
        req.user_id, name=req.name, email=req.email,
        current_password=req.current_password, new_password=req.new_password,
    )
    if isinstance(result, str):
        raise HTTPException(400, result)
    return result


# ── Chat ──────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    """Send a message to the skincare agent and get a response."""
    crud.create_user(req.user_id)

    context = req.message
    if req.image_path:
        context += f"\n[User uploaded image: {req.image_path}]"

    result = await invoke_agent(
        user_id=req.user_id,
        message=context,
    )

    return {
        "user_id": req.user_id,
        "response": result["response"],
        "tools_used": result.get("tools_used", []),
    }


def _run_cv_pipeline(save_path: Path, skin_type: str, out_dir: Path):
    """Run the full CV pipeline synchronously (called in a thread)."""
    from test_all import run_inference

    class InferenceArgs:
        pass

    args = InferenceArgs()
    args.image = str(save_path)
    args.skin_type = skin_type
    args.detection_model = str(PROJECT_ROOT / "acne_yolo_runs/roboflow_6classes/weights/best.pt")
    args.severity_model = str(Path.home() / "acne_severity_runs/20251110_172153/best_model.pt")
    args.bisenet_model = str(PROJECT_ROOT / "79999_iter.pth")
    args.wrinkle_checkpoint = str(
        PROJECT_ROOT / "ffhq_wrinkle_data/pretrained_ckpt/stage2_wrinkle_finetune_unet/stage2_unet.pth"
    )
    args.wrinkle_network = "UNet"
    args.conf_thres = 0.1
    args.iou_thres = 0.3
    args.img_size = 640
    args.output_dir = str(out_dir)
    args.output = str(out_dir / "report.json")
    args.save_acne_vis = str(out_dir / "result_acne.png")
    args.save_wrinkle_vis = str(out_dir / "result_wrinkle.png")
    args.save_rec_vis = str(out_dir / "result_recommendations.png")

    run_inference(args)
    return args


@router.post("/analyze")
async def analyze(
    user_id: str = Form(...),
    skin_type: str = Form("oily"),
    image: UploadFile = File(...),
):
    """Upload a face image, run the full CV pipeline, and return analysis."""
    ext = Path(image.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    crud.create_user(user_id, skin_type=skin_type)

    out_dir = UPLOAD_DIR / f"analysis_{uuid.uuid4().hex}"
    out_dir.mkdir(exist_ok=True)
    error_msg = None

    try:
        args = await asyncio.to_thread(_run_cv_pipeline, save_path, skin_type, out_dir)

        with open(args.output) as f:
            report = json.load(f)

        concern_vector = [report["user_concern_vector"].get(c, 0.0) for c in CONCERNS]

        acne = report.get("acne", {})
        detections = acne.get("detections", [])
        severity_counts = {}
        class_counts = {}
        for det in detections:
            sev = det.get("severity_name", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            cls = det.get("class_name", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        acne_summary = {
            "total_detections": acne.get("total_detections", 0),
            "severity_distribution": severity_counts,
            "class_distribution": class_counts,
            "regions": acne.get("regions", {}),
            "detections": [
                {
                    "class_name": d.get("class_name", "unknown"),
                    "severity_name": d.get("severity_name", "unknown"),
                    "confidence": round(d.get("confidence", 0), 2),
                    "face_region": d.get("face_region", "unknown"),
                }
                for d in detections
            ],
        }

        wrinkle = report.get("wrinkle", {})
        wrinkle_regions = report.get("wrinkle_regions", {}) or wrinkle.get("wrinkle_regions", {})
        wrinkle_summary = {
            "severity": wrinkle.get("severity", "none"),
            "severity_score": wrinkle.get("severity_score", 0),
            "wrinkle_pct": wrinkle.get("wrinkle_pct", 0),
            "wrinkle_regions": {
                k: {
                    "wrinkle_pct": v.get("wrinkle_pct", 0),
                    "severity": v.get("severity", "none"),
                    "severity_score": v.get("severity_score", 0),
                }
                for k, v in wrinkle_regions.items()
            },
        }

        recommendations = report.get("recommendations", {})

    except Exception as e:
        error_msg = str(e)
        logger.error("CV pipeline failed for user %s: %s\n%s", user_id, e, traceback.format_exc())
        concern_vector = [0.0] * 7
        acne_summary = {"total_detections": 0, "error": error_msg}
        wrinkle_summary = {"severity": "unknown", "error": error_msg}
        recommendations = {}
        report = {}

    analysis_id = crud.save_analysis(
        user_id=user_id,
        image_path=str(save_path),
        concern_vector=concern_vector,
        acne_summary=acne_summary,
        wrinkle_summary=wrinkle_summary,
        full_report=report,
    )

    zone_scores = _build_zone_scores(report)

    acne_vis = out_dir / "result_acne.png"
    wrinkle_vis = out_dir / "result_wrinkle.png"

    return {
        "user_id": user_id,
        "analysis_id": analysis_id,
        "concern_vector": concern_vector,
        "concerns": {c: round(v, 3) for c, v in zip(CONCERNS, concern_vector)},
        "acne_summary": acne_summary,
        "wrinkle_summary": wrinkle_summary,
        "wrinkle_regions": wrinkle_summary.get("wrinkle_regions") or {},
        "zone_scores": zone_scores,
        "overall_score": _compute_overall_score(concern_vector),
        "recommendations": recommendations,
        "images": {
            "acne": f"/api/uploads/{acne_vis.relative_to(UPLOAD_DIR)}" if acne_vis.exists() else None,
            "wrinkle": f"/api/uploads/{wrinkle_vis.relative_to(UPLOAD_DIR)}" if wrinkle_vis.exists() else None,
        },
        "error": error_msg,
    }


def _compute_overall_score(concern_vector):
    """Convert concern vector (higher = worse) to health score (higher = better)."""
    if not concern_vector or all(v == 0 for v in concern_vector):
        return 75
    avg_concern = sum(concern_vector) / len(concern_vector)
    return max(20, min(98, int(100 - avg_concern * 80)))


def _build_zone_scores(report):
    """Build face zone scores from the CV report for the frontend."""
    zones = []
    acne = report.get("acne", {})
    wrinkle = report.get("wrinkle", {})
    wrinkle_regions = report.get("wrinkle_regions", {})

    total_det = acne.get("total_detections", 0)
    wrinkle_sev = wrinkle.get("severity_score", 0)

    zones.append({
        "zone": "Forehead",
        "score": max(30, 90 - wrinkle_regions.get("forehead", {}).get("severity_score", 0) * 15),
        "issues": _zone_issues(wrinkle_regions.get("forehead", {}), "wrinkle"),
    })
    zones.append({
        "zone": "T-Zone",
        "score": max(30, 85 - min(total_det, 10) * 4),
        "issues": [f"{total_det} acne detections"] if total_det > 0 else ["Clear"],
    })
    zones.append({
        "zone": "Cheeks",
        "score": max(30, 88 - wrinkle_regions.get("nasolabial", {}).get("severity_score", 0) * 12),
        "issues": _zone_issues(wrinkle_regions.get("nasolabial", {}), "nasolabial"),
    })
    zones.append({
        "zone": "Under Eyes",
        "score": max(30, 85 - wrinkle_regions.get("under_eye", {}).get("severity_score", 0) * 15),
        "issues": _zone_issues(wrinkle_regions.get("under_eye", {}), "wrinkle"),
    })
    zones.append({
        "zone": "Crow's Feet",
        "score": max(30, 87 - wrinkle_regions.get("crow_feet", {}).get("severity_score", 0) * 15),
        "issues": _zone_issues(wrinkle_regions.get("crow_feet", {}), "wrinkle"),
    })

    return zones


def _zone_issues(region_data, label):
    if not region_data:
        return ["No data"]
    sev = region_data.get("severity", "none")
    pct = region_data.get("wrinkle_pct", 0)
    if sev == "none":
        return ["Clear"]
    return [f"{sev.capitalize()} {label} ({pct:.1f}%)"]


@router.post("/purchase")
async def record_purchase(req: PurchaseRequest):
    """Record that a user purchased a product (from recommendations)."""
    crud.create_user(req.user_id)
    pid = crud.save_purchase(
        req.user_id,
        req.product_url,
        req.product_title,
        req.price,
    )
    return {"purchase_id": pid, "product_url": req.product_url}


def _full_product_item(url, p, sim, skin_match):
    """Build full product dict for API response."""
    cat = p["category"][-1] if p.get("category") else "Unknown"
    ev = p.get("evidence_scores", {})
    return {
        "product_url": url,
        "brand": p.get("brand", ""),
        "title": p.get("title", ""),
        "full_name": p.get("full_name", ""),
        "category": cat,
        "price": p.get("price", ""),
        "price_value": p.get("price_value"),
        "rating": p.get("rating"),
        "review_count": p.get("review_count", 0),
        "similarity": round(sim, 4),
        "skin_match": skin_match,
        "evidence_scores": {c: round(ev.get(c, 0), 3) for c in CONCERNS},
        "top_ingredients": p.get("evidence_matched_ingredients", [])[:8],
        "skin_type": p.get("skin_type", ""),
        "skin_concerns": p.get("skin_concerns", ""),
        "formulation": p.get("formulation", ""),
        "what_it_is": p.get("what_it_is", ""),
        "what_it_does": p.get("what_it_does", ""),
        "ingredients": p.get("ingredients") or [],
        "product_claims": p.get("product_claims", []),
        "image_url": p.get("image_url", ""),
        "variant": p.get("variant", ""),
        "age_range": p.get("age_range", ""),
        "description_raw": p.get("description_raw", ""),
    }


@router.post("/recommend")
async def recommend(req: RecommendRequest):
    """Get product recommendations and optimized routine based on concern vector or latest analysis."""
    from agent.tools import (
        _load_product_data, _build_product_vector,
        _build_user_vec, _compute_outcome_penalties, _adaptive_score,
    )
    from labeling.routine_optimizer import optimize_routine

    products, reviews = _load_product_data()

    user_vec = req.concern_vector
    if not user_vec:
        analyses = crud.get_analysis_history(req.user_id, limit=1)
        if analyses:
            cv = analyses[0].get("concern_vector", [])
            if isinstance(cv, str):
                cv = json.loads(cv)
            user_vec = cv
    if not user_vec:
        user_vec = [0.5] * 7
    if len(user_vec) < 7:
        user_vec = (user_vec + [0.0] * 7)[:7]

    direct_pen, failed_ings, worsened = _compute_outcome_penalties(req.user_id)
    base_vec = _build_user_vec(
        user_vec[0], user_vec[6], user_vec[2], user_vec[4], user_vec[5]
    ) if len(user_vec) >= 7 else user_vec

    scored = []
    for url, p in products.items():
        rs = reviews.get(url, {})
        r = _adaptive_score(
            url, p, rs, base_vec, req.skin_type,
            direct_pen, failed_ings, worsened, req.budget,
        )
        if r:
            scored.append(_full_product_item(url, p, r["adaptive_score"], r["skin_match"]))

    scored.sort(key=lambda x: (-int(x["skin_match"]), -x["similarity"],
                                x["price_value"] or 9999))

    by_category = {}
    for item in scored:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        if len(by_category[cat]) < req.top_n:
            by_category[cat].append(item)

    # Ensure all catalog categories have at least one product
    all_cats = {p["category"][-1] if p.get("category") else "Unknown" for p in products.values()}
    scored_urls = {item["product_url"] for item in scored}
    for cat in all_cats:
        if cat not in by_category or len(by_category[cat]) == 0:
            best = None
            best_sim = -1
            for url, p in products.items():
                if url in scored_urls:
                    continue
                pcat = p["category"][-1] if p.get("category") else "Unknown"
                if pcat != cat:
                    continue
                ev = p.get("evidence_scores", {})
                sim = sum(ev.get(c, 0) for c in CONCERNS) / max(1, len(CONCERNS))
                if sim > best_sim:
                    best_sim = sim
                    best = _full_product_item(url, p, sim, bool(p.get(f"skin_{req.skin_type}", 0)))
            if best:
                by_category[cat] = [best]

    exclude_urls = set(direct_pen.keys()) if direct_pen else None
    routine_budget = (req.budget * 5) if req.budget else None
    routine_result = optimize_routine(
        products, reviews, user_vec, req.skin_type,
        budget=routine_budget,
        build_product_vector=lambda url, p: _build_product_vector(p, reviews.get(url, {})),
        exclude_urls=exclude_urls,
        lambda_conflict=5.0,  # Heavily penalize ingredient conflicts
    )

    routine_for_frontend = []
    step_display = {"SPF": "Sunscreen"}
    for r in routine_result.get("routine", []):
        step = r["step"]
        url = r["product_url"]
        p = products.get(url, r.get("product", {}))
        if not p:
            continue
        ev = p.get("evidence_scores", {})
        sim = sum(ev.get(c, 0) for c in CONCERNS) / max(1, len(CONCERNS))
        routine_for_frontend.append({
            "step": step_display.get(step, step),
            "product": _full_product_item(url, p, sim, bool(p.get(f"skin_{req.skin_type}", 0))),
        })

    return {
        "user_id": req.user_id,
        "skin_type": req.skin_type,
        "concern_vector": user_vec,
        "recommendations": by_category,
        "routine": routine_for_frontend,
        "routine_meta": {
            "coverage": routine_result.get("coverage"),
            "conflict_penalty": routine_result.get("conflict_penalty"),
            "total_cost": routine_result.get("total_cost"),
        },
        "total_products": len(scored),
    }


def _product_to_api_item(p: dict) -> dict:
    """Convert product dict to API response shape."""
    url = p.get("product_url", "")
    cat = p["category"][-1] if p.get("category") else "Unknown"
    ev = p.get("evidence_scores", {})
    return {
        "product_url": url,
        "brand": p.get("brand", ""),
        "title": p.get("title", ""),
        "full_name": p.get("full_name", ""),
        "category": cat,
        "price": p.get("price", ""),
        "price_value": p.get("price_value"),
        "rating": p.get("rating"),
        "review_count": p.get("review_count", 0),
        "evidence_scores": {c: round(ev.get(c, 0), 3) for c in CONCERNS},
        "top_ingredients": p.get("evidence_matched_ingredients", [])[:8],
        "skin_type": p.get("skin_type", ""),
        "skin_concerns": p.get("skin_concerns", ""),
        "formulation": p.get("formulation", ""),
        "what_it_is": p.get("what_it_is", ""),
        "what_it_does": p.get("what_it_does", ""),
        "ingredients": p.get("ingredients") or [],
        "product_claims": p.get("product_claims", []),
        "image_url": p.get("image_url", ""),
        "variant": p.get("variant", ""),
        "age_range": p.get("age_range", ""),
        "description_raw": p.get("description_raw", ""),
    }


@router.get("/products")
async def list_products(
    category: str | None = Query(None),
    limit: int = Query(100, le=2000),
    offset: int = Query(0),
):
    """List products from SQLite (fast paginated query). Falls back to JSONL if DB empty."""
    items, total, categories = crud.get_products_paginated(category=category, limit=limit, offset=offset)
    if total == 0:
        from agent.tools import _load_product_data
        products, _ = _load_product_data()
        all_items = []
        for url, p in products.items():
            p = dict(p)
            p["product_url"] = url
            cat = p["category"][-1] if p.get("category") else "Unknown"
            if category and cat.lower() != category.lower():
                continue
            all_items.append(p)
        all_items.sort(key=lambda x: -(x.get("rating") or 0))
        total = len(all_items)
        page = [_product_to_api_item(p) for p in all_items[offset:offset + limit]]
        categories = sorted(set(_product_to_api_item(p)["category"] for p in all_items))
        return {"products": page, "total": total, "categories": categories}
    page = [_product_to_api_item(p) for p in items]
    return {"products": page, "total": total, "categories": categories}


def _product_path_key(url: str) -> str:
    """Normalize URL to path for matching (e.g. /products/foo/v/50ml)."""
    from urllib.parse import urlparse, unquote
    if not url:
        return ""
    p = urlparse(unquote(url).strip())
    return (p.path or "").rstrip("/").lower()


@router.get("/products/reviews")
async def get_product_reviews(product_url: str = Query(..., description="Product URL to fetch reviews for")):
    """Get all scraped user reviews for a product."""
    import urllib.parse

    def _url_matches(a: str, b: str) -> bool:
        if not a or not b:
            return False
        da, db = urllib.parse.unquote(a).strip(), urllib.parse.unquote(b).strip()
        if da == db or da.rstrip("/") == db.rstrip("/"):
            return True
        # Fallback: match by path (handles domain/protocol differences)
        pa, pb = _product_path_key(a), _product_path_key(b)
        return bool(pa and pa == pb)

    def _load_from(path) -> list:
        out = []
        if not path.exists():
            return out
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if _url_matches(d.get("product_url", ""), product_url):
                        out.append({
                            "reviewer_name": d.get("reviewer_name", ""),
                            "rating": d.get("rating"),
                            "headline": d.get("headline", ""),
                            "review_text": d.get("review_text", ""),
                            "date_published": d.get("date_published", ""),
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    review_paths = [
        PROJECT_ROOT / "labeling" / "reviews_labeled.jsonl",
        PROJECT_ROOT / "data" / "raw" / "sephora" / "reviews" / "reviews.jsonl",
        PROJECT_ROOT / "data" / "processed" / "sephora" / "reviews_clean.jsonl",
    ]
    reviews = []
    for path in review_paths:
        reviews = _load_from(path)
        if reviews:
            break
    return {"reviews": reviews, "count": len(reviews)}


@router.get("/products/search")
async def search_products_api(
    concern: str = Query(..., description="Skin concern to search for"),
    skin_type: str | None = Query(None),
    max_price: float | None = Query(None),
    min_rating: float | None = Query(None),
    sort_by: str = Query("evidence"),
    limit: int = Query(10, le=50),
):
    """Search products by concern, skin type, budget, and rating."""
    from agent.tools import search_products

    result = search_products.invoke({
        "concern": concern,
        "skin_type": skin_type,
        "max_price": max_price,
        "min_rating": min_rating,
        "sort_by": sort_by,
        "limit": limit,
    })
    return {"results": result}


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get user profile with analysis and purchase history."""
    user = crud.get_user(user_id)
    if not user:
        raise HTTPException(404, f"User '{user_id}' not found")

    analyses = crud.get_analysis_history(user_id, limit=5)
    purchases = crud.get_purchase_history(user_id, limit=5)

    return ProfileResponse(
        user_id=user["user_id"],
        name=user.get("name"),
        skin_type=user.get("skin_type"),
        created_at=user.get("created_at"),
        analyses_count=len(analyses),
        purchases_count=len(purchases),
        latest_analysis=analyses[0] if analyses else None,
        recent_purchases=[dict(p) for p in purchases],
    )


@router.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str):
    """Get full analysis + purchase + recommendation history for a user."""
    user = crud.get_user(user_id)
    if not user:
        raise HTTPException(404, f"User '{user_id}' not found")

    analyses = crud.get_analysis_history(user_id, limit=20)
    purchases = crud.get_purchase_history(user_id, limit=20)
    recs = crud.get_recommendations(user_id, limit=20)

    return HistoryResponse(
        user_id=user_id,
        analyses=analyses,
        purchases=purchases,
        recommendations=recs,
    )


@router.post("/bag/add")
async def add_to_bag(req: BagRequest):
    crud.create_user(req.user_id)
    rid = crud.add_to_bag(
        req.user_id, req.product_url, req.product_title,
        req.brand, req.price, req.image_url,
    )
    return {"id": rid, "product_url": req.product_url}


@router.post("/bag/remove")
async def remove_from_bag(req: BagRequest):
    removed = crud.remove_from_bag(req.user_id, req.product_url)
    return {"removed": removed, "product_url": req.product_url}


@router.get("/bag/{user_id}")
async def get_bag(user_id: str):
    return {"items": crud.get_bag(user_id)}


@router.post("/like")
async def toggle_like(req: LikeRequest):
    crud.create_user(req.user_id)
    liked = crud.toggle_like(
        req.user_id, req.product_url, req.product_title,
        req.brand, req.price, req.image_url,
    )
    return {"liked": liked, "product_url": req.product_url}


@router.get("/liked/{user_id}")
async def get_liked(user_id: str):
    return {"items": crud.get_liked(user_id)}


@router.get("/trending")
async def get_trending(limit: int = Query(10, le=30)):
    """Return top-rated products as 'trending' (SQL LIMIT — does not load full catalog)."""
    items, total, _ = crud.get_products_paginated(category=None, limit=limit, offset=0)
    if total > 0:
        out = []
        for p in items:
            url = p.get("product_url", "")
            cat = p["category"][-1] if p.get("category") else "Unknown"
            rc = p.get("review_count", 0) or 0
            rating = p.get("rating") or 0
            out.append({
                "product_url": url,
                "brand": p.get("brand", ""),
                "title": p.get("title", ""),
                "category": cat,
                "price": p.get("price", ""),
                "price_value": p.get("price_value"),
                "rating": rating,
                "review_count": rc,
                "image_url": p.get("image_url", ""),
                "what_it_does": p.get("what_it_does", ""),
            })
        return {"products": out}

    from agent.tools import _load_product_data

    products, _reviews = _load_product_data()
    items = []
    for url, p in products.items():
        cat = p["category"][-1] if p.get("category") else "Unknown"
        rc = p.get("review_count", 0) or 0
        rating = p.get("rating") or 0
        items.append({
            "product_url": url,
            "brand": p.get("brand", ""),
            "title": p.get("title", ""),
            "category": cat,
            "price": p.get("price", ""),
            "price_value": p.get("price_value"),
            "rating": rating,
            "review_count": rc,
            "image_url": p.get("image_url", ""),
            "what_it_does": p.get("what_it_does", ""),
        })
    items.sort(key=lambda x: (-(x["rating"] or 0), -(x["review_count"] or 0)))
    return {"products": items[:limit]}


@router.get("/journey/{user_id}")
async def get_journey(user_id: str):
    """Get skin journey: past scans, purchases, and improvement % since last scan."""
    user = crud.get_user(user_id)
    if not user:
        raise HTTPException(404, f"User '{user_id}' not found")

    analyses = crud.get_analysis_history(user_id, limit=10)
    purchases = crud.get_purchase_history(user_id, limit=20)
    improvement = crud.compute_skin_improvement(user_id)

    return {
        "user_id": user_id,
        "name": user.get("name"),
        "skin_type": user.get("skin_type"),
        "analyses": analyses,
        "purchases": purchases,
        "improvement": improvement,
    }
