#!/usr/bin/env python3
"""
Combined Skin Analyzer + Product Recommender
- Acne: YOLO detection -> severity classification -> face region assignment
- Wrinkle: BiSeNet face masking -> UNet segmentation -> overall severity grading
- Recommender: cosine similarity between user concern vector and product vectors
Outputs annotated images, product recommendations, and a combined JSON report.
"""

import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

from trialfaceparsing import parse_face

sys.path.insert(0, "face-parsing.PyTorch")

# ── Acne constants ──────────────────────────────────────────────────────────
SEVERITY_NAMES = ["level0", "level1", "level2", "level3"]
SEVERITY_TO_SCORE = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}

# ── Wrinkle severity thresholds (per region) ───────────────────────────────
# Used for severity label only; concern score uses direct percentage
WRINKLE_THRESHOLDS = {"mild": 0.8, "moderate": 2.0, "severe": 4.0}

# Wrinkle concern: linear scale 0–100% coverage → 0–1.0 (5% coverage = max concern)
WRINKLE_PCT_TO_CONCERN = 5.0  # wrinkle_concern = min(1.0, wrinkle_pct / 5.0)

# Wrinkle region definitions: (y_lo, y_hi, x_lo, x_hi) as fraction of face bbox
# Value can be single tuple or list of tuples for composite regions (e.g. crow_feet L+R)
WRINKLE_REGIONS = {
    "forehead": (0.0, 0.32, 0.0, 1.0),           # top 32%
    "under_eye": (0.32, 0.48, 0.25, 0.75),       # eyebag: middle 50%
    "crow_feet": [(0.32, 0.48, 0.0, 0.25), (0.32, 0.48, 0.75, 1.0)],  # lateral L+R
    "nasolabial": (0.48, 0.70, 0.2, 0.8),        # beside nose
    "perioral": (0.70, 1.0, 0.0, 1.0),           # beside mouth
}


# ============================================================================
# Acne helpers
# ============================================================================

def load_severity_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(SEVERITY_NAMES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model


def get_detection_region(bbox, region_masks, image_shape):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = image_shape[:2]
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None
    for name, mask in region_masks.items():
        if mask.shape != (h, w):
            m = cv2.resize(mask.astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST) > 0
        else:
            m = mask
        if m[cy, cx]:
            return name
    return None


# ============================================================================
# Wrinkle helpers
# ============================================================================

def create_masked_face(image_np, bisenet_seg, device):
    """Create a face-only image (face + nose regions kept, rest blacked out)."""
    h, w = image_np.shape[:2]
    seg = bisenet_seg
    if seg.shape != (h, w):
        seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    face_mask = (seg == 1) | (seg == 10)
    masked = image_np.copy()
    masked[~face_mask] = 0
    return masked, face_mask


def generate_texture_map(image_np, face_mask):
    """
    Approximate a texture/wrinkle-hint map via high-pass filtering.
    Mimics the weak wrinkle masks used during model training.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    hp = np.abs(gray - blur)
    hp = np.clip(hp, 0, 255).astype(np.uint8)
    hp[~face_mask] = 0
    return hp


def load_wrinkle_model(checkpoint_path, network, device):
    if network.lower() == "unet":
        from unet.unet_model import UNet
        model = UNet(n_channels=4, n_classes=2, bilinear=True)
    else:
        from unet.swin_unetr import SwinUNETR
        model = SwinUNETR(in_channels=4, out_channels=2)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    from collections import OrderedDict
    cleaned = OrderedDict()
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned)
    model.to(device).eval()
    return model


def run_wrinkle_inference(model, masked_face, texture_map, device, img_size=1024):
    """Run wrinkle segmentation on a single image, return binary mask."""
    face_pil = Image.fromarray(masked_face).convert("RGB").resize(
        (img_size, img_size), Image.BILINEAR)
    tex_pil = Image.fromarray(texture_map).convert("L").resize(
        (img_size, img_size), Image.BILINEAR)

    np_img = np.array(face_pil, dtype=np.float32).transpose((2, 0, 1))
    np_tex = np.array(tex_pil, dtype=np.float32)[np.newaxis, ...]

    if np_img.max() > 1.0:
        np_img = np_img / 255.0
    np_img = np_img * 2.0 - 1.0
    if np_tex.max() > 1.0:
        np_tex = np_tex / 255.0
    np_tex = np_tex * 2.0 - 1.0

    combined = np.concatenate([np_img, np_tex], axis=0)
    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():
        pred = model(tensor).argmax(dim=1)[0].cpu().numpy()

    return (pred > 0).astype(np.uint8) * 255


def classify_wrinkle_severity(wrinkle_pct):
    """Return (severity_name, severity_score 0-3) for display."""
    if wrinkle_pct < WRINKLE_THRESHOLDS["mild"]:
        return "none", 0
    elif wrinkle_pct < WRINKLE_THRESHOLDS["moderate"]:
        return "mild", 1
    elif wrinkle_pct < WRINKLE_THRESHOLDS["severe"]:
        return "moderate", 2
    else:
        return "severe", 3


def compute_wrinkle_regions(face_mask_1024, wrinkle_mask_1024):
    """
    Compute wrinkle coverage per region (forehead, under_eye, nasolabial, crow_feet, perioral).
    Returns dict of {region: {wrinkle_pct, severity, severity_score, wrinkle_pixels, face_pixels}}.
    """
    face_bin = (face_mask_1024 > 0).astype(np.uint8)
    wrinkle_bin = (wrinkle_mask_1024 > 0) & (face_bin.astype(bool))
    h, w = face_mask_1024.shape

    pts = np.column_stack(np.where(face_bin > 0))
    if len(pts) == 0:
        return {}
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    face_h = y_max - y_min + 1
    face_w = x_max - x_min + 1

    regions = {}
    for name, spec in WRINKLE_REGIONS.items():
        rects = spec if isinstance(spec, list) else [spec]
        region_face = np.zeros_like(face_bin)
        for r in rects:
            y_lo, y_hi, x_lo, x_hi = r
            y1 = int(y_min + y_lo * face_h)
            y2 = int(y_min + y_hi * face_h)
            x1 = int(x_min + x_lo * face_w)
            x2 = int(x_min + x_hi * face_w)
            region_face[y1:y2, x1:x2] = np.maximum(region_face[y1:y2, x1:x2], face_bin[y1:y2, x1:x2])
        region_face = cv2.bitwise_and(region_face, face_bin)

        region_px = np.sum(region_face > 0)
        wrinkle_px = np.sum(wrinkle_bin & region_face.astype(bool))
        wrinkle_pct = 100.0 * wrinkle_px / region_px if region_px > 0 else 0.0
        sev_name, sev_score = classify_wrinkle_severity(wrinkle_pct)

        regions[name] = {
            "wrinkle_pct": round(wrinkle_pct, 3),
            "severity": sev_name,
            "severity_score": sev_score,
            "wrinkle_pixels": int(wrinkle_px),
            "face_pixels": int(region_px),
        }
    return regions


# ============================================================================
# Product Recommender (Cosine Similarity)
# ============================================================================

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]

# Map YOLO detection class names to concern dimensions
CLASS_TO_CONCERN = {
    "acne": "acne", "papule": "acne", "pustule": "acne",
    "nodule": "acne", "cyst": "acne",
    "blackhead": "comedonal_acne", "whitehead": "comedonal_acne",
    "comedone": "comedonal_acne",
    "acne_scars": "acne_scars_texture", "scar": "acne_scars_texture",
    "dark_spot": "pigmentation", "pigmentation": "pigmentation",
    "redness": "redness",
}


def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a, b):
    na, nb = vec_norm(a), vec_norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return vec_dot(a, b) / (na * nb)


def build_user_concern_vector(detections, wrinkle_info):
    """
    Convert skin analysis results into a 7-dim concern vector [0-1].
    Higher value = stronger need for that concern.
    Wrinkle: uses actual coverage % (0-5% maps to 0-1.0), not severity tiers.
    """
    concern_counts = {c: 0 for c in CONCERNS}
    concern_severity = {c: [] for c in CONCERNS}

    for det in detections:
        cls_name = det["class_name"].lower()
        concern = CLASS_TO_CONCERN.get(cls_name, "acne")
        concern_counts[concern] += 1
        concern_severity[concern].append(det["severity_score"])

    vec = []
    for c in CONCERNS:
        if c == "wrinkles":
            vec.append(wrinkle_info.get("wrinkle_concern", 0))
        elif concern_counts[c] > 0:
            count_score = min(1.0, concern_counts[c] / 10.0)
            avg_sev = np.mean(concern_severity[c]) if concern_severity[c] else 0
            score = count_score * (0.5 + 0.5 * avg_sev)
            vec.append(round(score, 4))
        else:
            vec.append(0.0)

    return vec


def load_product_data():
    """Load product labels and review scores."""
    base = os.path.join(os.path.dirname(__file__), "labeling")
    products = {}
    with open(os.path.join(base, "products_evidence_labeled.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            products[d["product_url"]] = d

    with open(os.path.join(base, "product_review_scores.json")) as f:
        review_scores = json.load(f)

    return products, review_scores


def build_product_vector(product, review_entry):
    """
    Build a combined 7-dim concern vector per product.
    Uses INCI position-weighted evidence scores instead of binary labels.
    """
    vec = []
    cs = review_entry.get("concern_scores", {}) if review_entry else {}
    for c in CONCERNS:
        ing = float(product.get(c, 0))
        eff = cs.get(c, {}).get("effectiveness")
        if eff is not None:
            rev = (eff + 1.0) / 2.0
            val = 0.5 * ing + 0.5 * rev
        else:
            val = ing
        vec.append(round(val, 4))
    return vec


def recommend_products(user_vec, products, review_scores,
                       skin_type=None, budget=None, top_n=3):
    """
    Compute cosine similarity between user concern vector and each product's
    combined vector. Returns ranked recommendations grouped by category.
    """
    scored = []
    for url, p in products.items():
        if budget and (p.get("price_value") or float("inf")) > budget:
            continue

        rs_entry = review_scores.get(url, {})
        prod_vec = build_product_vector(p, rs_entry)

        if all(v == 0 for v in prod_vec):
            continue

        sim = cosine_sim(user_vec, prod_vec)
        if sim <= 0:
            continue

        skin_match = None
        if skin_type:
            skin_match = bool(p.get(f"skin_{skin_type}", 0))

        cat = p["category"][-1] if p.get("category") else "Unknown"
        scored.append({
            "product_url": url,
            "brand": p.get("brand", ""),
            "title": p.get("title", ""),
            "full_name": p.get("full_name", ""),
            "category": cat,
            "price": p.get("price", ""),
            "price_value": p.get("price_value"),
            "rating": p.get("rating"),
            "cosine_similarity": round(sim, 4),
            "product_vector": prod_vec,
            "skin_match": skin_match,
        })

    # Sort: skin match first, then cosine similarity desc, then price asc
    def sort_key(x):
        skin_pri = 0 if x["skin_match"] is None else (0 if x["skin_match"] else 1)
        return (skin_pri, -x["cosine_similarity"], x["price_value"] or 0)

    scored.sort(key=sort_key)

    by_cat = {}
    for item in scored:
        cat = item["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < top_n:
            by_cat[cat].append(item)

    return by_cat


def draw_recommendations(user_vec, recommendations, skin_type, save_path):
    """Draw a recommendation summary image."""
    line_h = 28
    pad = 20
    cat_header_h = 35

    categories = sorted(recommendations.keys())
    total_items = sum(len(v) for v in recommendations.values())
    img_h = pad * 2 + 120 + len(categories) * cat_header_h + total_items * line_h + 40
    img_w = 900
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    y = pad
    cv2.putText(canvas, "PRODUCT RECOMMENDATIONS", (pad, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += 35
    cv2.putText(canvas, "Based on: cosine_sim(your_skin_analysis, product_vector)",
                (pad, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y += 25
    if skin_type:
        cv2.putText(canvas, f"Skin type: {skin_type.upper()} (soft filter)",
                    (pad, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 25

    concern_names = ["acne", "comedonal", "pigment", "scars", "pores", "redness", "wrinkles"]
    vec_str = "Your concerns: " + "  ".join(
        f"{n}={v:.2f}" for n, v in zip(concern_names, user_vec) if v > 0)
    if not any(v > 0 for v in user_vec):
        vec_str = "Your concerns: (none detected)"
    cv2.putText(canvas, vec_str, (pad, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)
    y += 35

    for cat in categories:
        items = recommendations[cat]
        cv2.rectangle(canvas, (pad, y), (img_w - pad, y + cat_header_h - 5),
                      (50, 50, 50), -1)
        cv2.putText(canvas, cat.upper(), (pad + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        y += cat_header_h

        for i, item in enumerate(items):
            skin_tag = ""
            if item["skin_match"] is not None:
                skin_tag = " MATCH" if item["skin_match"] else ""

            line = (f"  #{i+1}  sim={item['cosine_similarity']:.3f}"
                    f"  {item['price']:>10s}"
                    f"  {item['full_name'][:50]}{skin_tag}")

            color = (200, 255, 200) if item.get("skin_match") else (200, 200, 200)
            cv2.putText(canvas, line, (pad + 5, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += line_h

    canvas = canvas[:y + pad]
    cv2.imwrite(save_path, canvas)


# ============================================================================
# Visualization
# ============================================================================

def draw_acne_result(image_np, detections, region_masks, roi_by_region, save_path):
    canvas = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    region_colors_bgr = {
        "forehead": (0, 0, 255), "left_cheek": (0, 255, 0),
        "right_cheek": (255, 0, 0), "chin": (255, 0, 255),
    }
    overlay = canvas.copy()
    for name, mask in region_masks.items():
        overlay[mask] = region_colors_bgr.get(name, (128, 128, 128))
    canvas = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        sev = det["severity_score"]
        color = (0, 0, 255) if sev >= 0.66 else (0, 165, 255) if sev >= 0.33 else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} | {det['severity_name']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    y = 25
    for name in ["forehead", "left_cheek", "right_cheek", "chin"]:
        stats = roi_by_region.get(name, {"roi": 0, "count": 0})
        txt = f"{name}: ROI={stats['roi']:.2f}  Count={stats['count']}"
        cv2.putText(canvas, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        y += 24

    cv2.imwrite(save_path, canvas)


def draw_wrinkle_result(image_np, wrinkle_mask_1024, face_mask, wrinkle_info, save_path):
    h, w = image_np.shape[:2]
    mask_resized = cv2.resize(wrinkle_mask_1024, (w, h), interpolation=cv2.INTER_NEAREST)

    panel_face = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    panel_overlay = panel_face.copy()
    wrinkle_px = mask_resized > 0
    panel_overlay[wrinkle_px, 2] = np.minimum(
        panel_overlay[wrinkle_px, 2].astype(int) + 150, 255).astype(np.uint8)
    panel_overlay[wrinkle_px, 1] = (panel_overlay[wrinkle_px, 1] * 0.4).astype(np.uint8)
    panel_overlay[wrinkle_px, 0] = (panel_overlay[wrinkle_px, 0] * 0.4).astype(np.uint8)

    panel_mask = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    target_h = 512
    panels = []
    for p in [panel_face, panel_overlay, panel_mask]:
        scale = target_h / p.shape[0]
        panels.append(cv2.resize(p, None, fx=scale, fy=scale))

    strip = np.concatenate(panels, axis=1)
    bar_h = 50
    bar = np.zeros((bar_h, strip.shape[1], 3), dtype=np.uint8)

    sev = wrinkle_info["severity"]
    pct = wrinkle_info["wrinkle_pct"]
    sev_colors = {"none": (180, 180, 180), "mild": (0, 255, 255),
                  "moderate": (0, 165, 255), "severe": (0, 0, 255)}
    color = sev_colors.get(sev, (255, 255, 255))
    text = f"Wrinkle: {pct:.2f}% coverage  |  Severity: {sev.upper()} ({wrinkle_info['severity_score']}/3)"
    cv2.putText(bar, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    headers = ["Original Face", "Wrinkle Detection", "Wrinkle Mask"]
    pw = panels[0].shape[1]
    for i, hdr in enumerate(headers):
        cx = i * pw + pw // 2
        (tw2, _), _ = cv2.getTextSize(hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, hdr, (cx - tw2 // 2, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1)

    result = np.concatenate([bar, strip], axis=0)
    cv2.imwrite(save_path, result)


# ============================================================================
# Main pipeline
# ============================================================================

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Step 1: Face Parsing ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Face Parsing (BiSeNet)")
    print("=" * 60)
    face_result = parse_face(args.image, args.bisenet_model, device=device)
    if face_result is None:
        print("Face parsing failed!")
        return
    image_np = face_result["image"]
    seg_mask = face_result["segmentation"]
    region_masks = face_result["regions"]
    h, w = image_np.shape[:2]
    print(f"Parsed {len(region_masks)} regions: {list(region_masks.keys())}")

    # ── Step 2: Acne Detection ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Acne Lesion Detection (YOLO)")
    print("=" * 60)
    det_model = YOLO(args.detection_model)
    results = det_model.predict(
        source=args.image,
        conf=getattr(args, "conf_thres", 0.1),
        iou=getattr(args, "iou_thres", 0.5),
        device=0 if device == "cuda" else None,
    )
    result = results[0]
    print(f"Raw detections: {len(result.boxes)}")

    # ── Step 3: Severity Classification ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Acne Severity Classification")
    print("=" * 60)
    sev_model = load_severity_model(args.severity_model, device)
    sev_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    detections = []
    severity_by_region = {}
    for box, cls, score in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), \
                          min(w, int(box[2])), min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_t = sev_transform(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(sev_model(crop_t), dim=1)
            prob_val, pred_idx = torch.max(probs, dim=1)

        sev_level = int(pred_idx.item())
        region = get_detection_region([x1, y1, x2, y2], region_masks, image_np.shape)
        det = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(score),
            "class_id": int(cls),
            "class_name": result.names[int(cls)],
            "severity_level": sev_level,
            "severity_name": SEVERITY_NAMES[sev_level],
            "severity_score": float(SEVERITY_TO_SCORE[sev_level]),
            "severity_confidence": float(prob_val.item()),
            "face_region": region or "unknown",
        }
        detections.append(det)
        if region:
            severity_by_region.setdefault(region, []).append(det["severity_score"])

    roi_by_region = {}
    for name in ["forehead", "left_cheek", "right_cheek", "chin"]:
        scores = severity_by_region.get(name, [])
        roi_by_region[name] = {
            "count": len(scores),
            "roi": float(np.mean(scores)) if scores else 0.0,
        }

    print(f"Classified {len(detections)} lesions")
    for name in ["forehead", "left_cheek", "right_cheek", "chin"]:
        s = roi_by_region[name]
        print(f"  {name:15s}: {s['count']} lesions, ROI={s['roi']:.2f}")

    # ── Step 4: Wrinkle Segmentation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Wrinkle Segmentation (UNet)")
    print("=" * 60)
    wrinkle_ckpt = getattr(args, "wrinkle_checkpoint",
        "./ffhq_wrinkle_data/pretrained_ckpt/stage2_wrinkle_finetune_unet/stage2_unet.pth")
    wrinkle_net = getattr(args, "wrinkle_network", "UNet")

    masked_face, face_mask = create_masked_face(image_np, seg_mask, device)
    texture_map = generate_texture_map(image_np, face_mask)
    print(f"Face mask pixels: {np.sum(face_mask)}")
    print(f"Texture map non-zero: {np.sum(texture_map > 0)}")

    wrinkle_model = load_wrinkle_model(wrinkle_ckpt, wrinkle_net, device)
    wrinkle_mask = run_wrinkle_inference(wrinkle_model, masked_face, texture_map, device)

    face_pixels_1024 = cv2.resize(face_mask.astype(np.uint8) * 255,
                                  (1024, 1024), interpolation=cv2.INTER_NEAREST)
    face_px = np.sum(face_pixels_1024 > 0)
    wrinkle_px = np.sum((wrinkle_mask > 0) & (face_pixels_1024 > 0))
    wrinkle_pct = 100.0 * wrinkle_px / face_px if face_px > 0 else 0.0
    sev_name, sev_score = classify_wrinkle_severity(wrinkle_pct)

    wrinkle_regions = compute_wrinkle_regions(face_pixels_1024, wrinkle_mask)
    wrinkle_concern = min(1.0, wrinkle_pct / WRINKLE_PCT_TO_CONCERN)

    wrinkle_info = {
        "wrinkle_pct": round(wrinkle_pct, 3),
        "severity": sev_name,
        "severity_score": sev_score,
        "wrinkle_pixels": int(wrinkle_px),
        "face_pixels": int(face_px),
        "wrinkle_regions": wrinkle_regions,
        "wrinkle_concern": round(wrinkle_concern, 4),
    }
    print(f"Wrinkle coverage: {wrinkle_pct:.2f}%")
    print(f"Wrinkle severity: {sev_name.upper()} ({sev_score}/3)")
    print(f"Wrinkle concern (0-1): {wrinkle_concern:.3f}")
    for rname, rdata in wrinkle_regions.items():
        print(f"  {rname:12s}: {rdata['wrinkle_pct']:.2f}% ({rdata['severity']})")

    # ── Step 5: Product Recommendation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Product Recommendation (Cosine Similarity)")
    print("=" * 60)
    user_skin_type = getattr(args, "skin_type", "oily")
    user_budget = getattr(args, "budget", None)
    rec_top_n = getattr(args, "rec_top_n", 3)

    user_vec = build_user_concern_vector(detections, wrinkle_info)
    concern_names = ["acne", "comedonal", "pigment", "scars", "pores", "redness", "wrinkles"]
    print(f"User skin type: {user_skin_type}")
    print(f"User concern vector:")
    for name, val in zip(concern_names, user_vec):
        bar = "#" * int(val * 20)
        print(f"    {name:12s}: {val:.3f}  {bar}")

    products, review_scores = load_product_data()
    print(f"\nProducts loaded: {len(products)}")

    recommendations = recommend_products(
        user_vec, products, review_scores,
        skin_type=user_skin_type, budget=user_budget, top_n=rec_top_n)

    total_recs = sum(len(v) for v in recommendations.values())
    print(f"Recommendations: {total_recs} products across {len(recommendations)} categories")

    print(f"\n{'─' * 60}")
    for cat in sorted(recommendations.keys()):
        items = recommendations[cat]
        print(f"\n  [{cat}]")
        for i, item in enumerate(items):
            skin_tag = "MATCH" if item.get("skin_match") else "     "
            print(f"    #{i+1}  sim={item['cosine_similarity']:.3f}  {skin_tag}  "
                  f"★{item['rating'] or 0:.1f}  {item['price']:>10s}  "
                  f"{item['full_name'][:45]}")

    # ── Step 6: Save outputs ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Saving Results")
    print("=" * 60)
    acne_vis = getattr(args, "save_acne_vis", "result_acne.png")
    wrinkle_vis = getattr(args, "save_wrinkle_vis", "result_wrinkle.png")
    rec_vis = getattr(args, "save_rec_vis", "result_recommendations.png")
    out_json = getattr(args, "output", "skin_analysis.json")

    draw_acne_result(image_np, detections, region_masks, roi_by_region, acne_vis)
    print(f"Acne visualization     -> {acne_vis}")

    draw_wrinkle_result(image_np, wrinkle_mask, face_mask, wrinkle_info, wrinkle_vis)
    print(f"Wrinkle visualization  -> {wrinkle_vis}")

    draw_recommendations(user_vec, recommendations, user_skin_type, rec_vis)
    print(f"Recommendation image   -> {rec_vis}")

    rec_json = {}
    for cat, items in recommendations.items():
        rec_json[cat] = items

    report = {
        "image": os.path.abspath(args.image),
        "skin_type": user_skin_type,
        "user_concern_vector": {c: v for c, v in zip(CONCERNS, user_vec)},
        "acne": {
            "total_detections": len(detections),
            "regions": roi_by_region,
            "detections": detections,
        },
        "wrinkle": wrinkle_info,
        "wrinkle_regions": wrinkle_info.get("wrinkle_regions", {}),
        "recommendations": rec_json,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report            -> {out_json}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SKIN ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n  ACNE ({len(detections)} lesions detected)")
    for det in detections:
        print(f"    - {det['class_name']:12s} | {det['severity_name']} "
              f"| conf={det['confidence']:.2f} | region={det['face_region']}")
    print(f"\n  Region ROI scores:")
    for name in ["forehead", "left_cheek", "right_cheek", "chin"]:
        s = roi_by_region[name]
        print(f"    {name:15s}: {s['count']} lesions, ROI = {s['roi']:.2f}")

    print(f"\n  WRINKLE")
    print(f"    Coverage : {wrinkle_pct:.2f}%")
    print(f"    Severity : {sev_name.upper()} ({sev_score}/3)")

    print(f"\n  RECOMMENDED PRODUCTS (top {rec_top_n} per category, skin={user_skin_type})")
    for cat in sorted(recommendations.keys()):
        items = recommendations[cat]
        print(f"    [{cat}]")
        for i, item in enumerate(items):
            print(f"      #{i+1} sim={item['cosine_similarity']:.3f}  "
                  f"{item['price']:>8s}  {item['full_name'][:42]}")
    print("=" * 60)


# ============================================================================
# Entry point
# ============================================================================

class Args:
    image = "/home/vanessa/project/levle3_113 copy.jpg"
    detection_model = "/home/vanessa/project/acne_yolo_runs/roboflow_6classes/weights/best.pt"
    severity_model = "/home/vanessa/acne_severity_runs/20251110_172153/best_model.pt"
    bisenet_model = "/home/vanessa/project/79999_iter.pth"
    wrinkle_checkpoint = "/home/vanessa/project/ffhq_wrinkle_data/pretrained_ckpt/stage2_wrinkle_finetune_unet/stage2_unet.pth"
    wrinkle_network = "UNet"
    conf_thres = 0.1
    iou_thres = 0.5
    output = "skin_analysis.json"
    save_acne_vis = "result_acne.png"
    save_wrinkle_vis = "result_wrinkle.png"
    save_rec_vis = "result_recommendations.png"
    skin_type = "oily"
    budget = None
    rec_top_n = 3


if __name__ == "__main__":
    print("=" * 60)
    print("COMBINED SKIN ANALYZER + PRODUCT RECOMMENDER")
    print("  Acne Detection + Classification + Severity")
    print("  Wrinkle Segmentation + Severity Grading")
    print("  Product Recommendation (Cosine Similarity)")
    print("=" * 60)
    print(f"Image:             {Args.image}")
    print(f"YOLO model:        {Args.detection_model}")
    print(f"Severity model:    {Args.severity_model}")
    print(f"BiSeNet model:     {Args.bisenet_model}")
    print(f"Wrinkle checkpoint:{Args.wrinkle_checkpoint}")
    print(f"Skin type:         {Args.skin_type}")
    print()

    run_inference(Args())
