"""
Wrinkle Severity Scorer

Calculates per-region wrinkle density from binary segmentation masks
and face-parsed labels, then maps to severity tiers.

Face parsing labels (CelebAMask-HQ / face-parsing.PyTorch):
  0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
  6=eyeglasses, 7=l_ear, 8=r_ear, 9=mouth, 10=nose,
  11=u_lip, 12=l_lip, 13=hair, 14=hat, 15=earring, 16=necklace, 17=neck
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom, binary_dilation
from tqdm import tqdm


SEVERITY_THRESHOLDS = {
    "forehead":    {"mild": 0.3, "moderate": 1.5, "severe": 3.5},
    "glabella":    {"mild": 0.2, "moderate": 1.0, "severe": 2.5},
    "crow_feet":   {"mild": 0.3, "moderate": 1.5, "severe": 3.0},
    "under_eye":   {"mild": 0.2, "moderate": 1.0, "severe": 2.5},
    "nasolabial":  {"mild": 0.4, "moderate": 2.0, "severe": 4.0},
    "perioral":    {"mild": 0.2, "moderate": 1.0, "severe": 2.5},
    "overall":     {"mild": 0.3, "moderate": 1.2, "severe": 3.0},
}


def classify_severity(wrinkle_pct, region):
    thresholds = SEVERITY_THRESHOLDS.get(region, SEVERITY_THRESHOLDS["overall"])
    if wrinkle_pct < thresholds["mild"]:
        return "none", 0
    elif wrinkle_pct < thresholds["moderate"]:
        return "mild", 1
    elif wrinkle_pct < thresholds["severe"]:
        return "moderate", 2
    else:
        return "severe", 3


def get_landmark_centers(parsed_labels):
    """Extract approximate center positions of face landmarks from parsing map."""
    centers = {}
    for label_id, name in [(2, "l_brow"), (3, "r_brow"), (4, "l_eye"), (5, "r_eye"),
                            (10, "nose"), (9, "mouth"), (11, "u_lip"), (12, "l_lip")]:
        ys, xs = np.where(parsed_labels == label_id)
        if len(ys) > 0:
            centers[name] = (int(np.mean(ys)), int(np.mean(xs)))
    return centers


def define_face_regions(parsed_labels, scale=2):
    """
    Define wrinkle-relevant face zones from parsing labels.
    Returns a dict of region_name -> boolean mask at target resolution.
    """
    h, w = parsed_labels.shape
    th, tw = h * scale, w * scale

    parsed_up = zoom(parsed_labels, (scale, scale), order=0).astype(np.int32)
    face_skin = (parsed_up == 1) | (parsed_up == 10)

    centers = get_landmark_centers(parsed_labels)
    for k in centers:
        y, x = centers[k]
        centers[k] = (y * scale, x * scale)

    regions = {}

    brow_y = None
    if "l_brow" in centers and "r_brow" in centers:
        brow_y = min(centers["l_brow"][0], centers["r_brow"][0])
    elif "l_brow" in centers:
        brow_y = centers["l_brow"][0]
    elif "r_brow" in centers:
        brow_y = centers["r_brow"][0]

    eye_y = None
    if "l_eye" in centers and "r_eye" in centers:
        eye_y = (centers["l_eye"][0] + centers["r_eye"][0]) // 2
    elif "l_eye" in centers:
        eye_y = centers["l_eye"][0]
    elif "r_eye" in centers:
        eye_y = centers["r_eye"][0]

    nose_y = centers.get("nose", (None,))[0]

    mouth_y = None
    for key in ["mouth", "u_lip", "l_lip"]:
        if key in centers:
            mouth_y = centers[key][0]
            break

    face_cx = tw // 2
    if "l_eye" in centers and "r_eye" in centers:
        face_cx = (centers["l_eye"][1] + centers["r_eye"][1]) // 2

    # Forehead: skin above eyebrows
    if brow_y is not None:
        forehead_mask = np.zeros((th, tw), dtype=bool)
        forehead_mask[:brow_y, :] = True
        regions["forehead"] = forehead_mask & face_skin

    # Glabella: between the eyebrows
    if "l_brow" in centers and "r_brow" in centers:
        glabella_mask = np.zeros((th, tw), dtype=bool)
        left_x = min(centers["l_brow"][1], centers["r_brow"][1])
        right_x = max(centers["l_brow"][1], centers["r_brow"][1])
        top = brow_y - int(0.03 * th) if brow_y else 0
        bot = brow_y + int(0.05 * th) if brow_y else th // 3
        glabella_mask[max(0, top):bot, left_x:right_x] = True
        regions["glabella"] = glabella_mask & face_skin

    # Crow's feet: lateral eye regions
    if "l_eye" in centers and "r_eye" in centers:
        eye_spread = abs(centers["r_eye"][1] - centers["l_eye"][1])
        margin = int(eye_spread * 0.4)
        eye_height = int(0.06 * th)

        left_eye_x = min(centers["l_eye"][1], centers["r_eye"][1])
        right_eye_x = max(centers["l_eye"][1], centers["r_eye"][1])
        ey = eye_y if eye_y else th // 3

        crow_mask = np.zeros((th, tw), dtype=bool)
        crow_mask[ey - eye_height:ey + eye_height, :max(0, left_eye_x - margin)] = True
        crow_mask[ey - eye_height:ey + eye_height, right_eye_x + margin:] = True
        regions["crow_feet"] = crow_mask & face_skin

    # Under-eye: below eyes, above nose
    if eye_y is not None and nose_y is not None:
        under_eye_mask = np.zeros((th, tw), dtype=bool)
        under_eye_mask[eye_y:nose_y, :] = True
        if "glabella" in regions:
            under_eye_mask &= ~regions.get("glabella", np.zeros_like(under_eye_mask))
        regions["under_eye"] = under_eye_mask & face_skin

    # Nasolabial: from nose sides down to mouth
    if nose_y is not None and mouth_y is not None:
        naso_mask = np.zeros((th, tw), dtype=bool)
        nose_x = centers["nose"][1] if "nose" in centers else face_cx
        spread = int(0.15 * tw)
        naso_mask[nose_y:mouth_y + int(0.03 * th),
                  nose_x - spread:nose_x + spread] = True
        nose_region = (parsed_up == 10)
        naso_mask &= ~nose_region
        regions["nasolabial"] = naso_mask & face_skin

    # Perioral: around the mouth
    if mouth_y is not None:
        perioral_mask = np.zeros((th, tw), dtype=bool)
        mouth_h = int(0.08 * th)
        mouth_w = int(0.12 * tw)
        perioral_mask[mouth_y - mouth_h:mouth_y + mouth_h,
                      face_cx - mouth_w:face_cx + mouth_w] = True
        mouth_region = (parsed_up == 9) | (parsed_up == 11) | (parsed_up == 12)
        perioral_mask &= ~mouth_region
        regions["perioral"] = perioral_mask & face_skin

    return regions


def score_single_image(wrinkle_mask_path, parsed_label_path):
    """
    Score wrinkle severity for a single image.

    Returns dict with per-region and overall scores.
    """
    wrinkle_mask = np.array(Image.open(wrinkle_mask_path).convert("L"))
    wrinkle_binary = wrinkle_mask > 0

    parsed_labels = np.load(parsed_label_path)
    regions = define_face_regions(parsed_labels, scale=2)

    parsed_up = zoom(parsed_labels, (2, 2), order=0).astype(np.int32)
    face_skin = (parsed_up == 1) | (parsed_up == 10)

    results = {}
    total_wrinkle = 0
    total_skin = 0

    for region_name, region_mask in regions.items():
        region_pixels = np.sum(region_mask)
        if region_pixels == 0:
            results[region_name] = {
                "wrinkle_pct": 0.0,
                "severity": "none",
                "severity_score": 0,
                "region_pixels": 0,
                "wrinkle_pixels": 0,
            }
            continue

        wrinkle_in_region = np.sum(wrinkle_binary & region_mask)
        wrinkle_pct = 100.0 * wrinkle_in_region / region_pixels
        severity, score = classify_severity(wrinkle_pct, region_name)

        results[region_name] = {
            "wrinkle_pct": round(wrinkle_pct, 3),
            "severity": severity,
            "severity_score": score,
            "region_pixels": int(region_pixels),
            "wrinkle_pixels": int(wrinkle_in_region),
        }

    total_skin = np.sum(face_skin)
    total_wrinkle = np.sum(wrinkle_binary & face_skin)
    overall_pct = 100.0 * total_wrinkle / total_skin if total_skin > 0 else 0.0
    overall_sev, overall_score = classify_severity(overall_pct, "overall")

    region_scores = [r["severity_score"] for r in results.values() if r["region_pixels"] > 0]
    weighted_score = np.mean(region_scores) if region_scores else 0.0

    results["overall"] = {
        "wrinkle_pct": round(overall_pct, 3),
        "severity": overall_sev,
        "severity_score": overall_score,
        "weighted_avg_score": round(weighted_score, 2),
        "face_skin_pixels": int(total_skin),
        "total_wrinkle_pixels": int(total_wrinkle),
    }

    return results


def visualize_severity(image_id, base_folder, results, output_path, model="unet"):
    """Create a visual showing the face with color-coded wrinkle severity per region."""
    results_dir = "test_outputs" if model == "unet" else "test_outputs_swinunetr"

    face_img = Image.open(os.path.join(base_folder, "face_images", f"{image_id}.png")).convert("RGB")
    face_img = face_img.resize((1024, 1024), Image.LANCZOS)
    wrinkle_mask = np.array(
        Image.open(os.path.join(base_folder, results_dir, f"{image_id}_mask.png")).convert("L")
    )

    parsed_labels = np.load(os.path.join(base_folder, "etcs", "face_parsed_labels", f"{image_id}.npy"))
    regions = define_face_regions(parsed_labels, scale=2)

    severity_colors = {
        "none": (80, 80, 80, 60),
        "mild": (255, 255, 0, 100),
        "moderate": (255, 140, 0, 120),
        "severe": (255, 0, 0, 140),
    }

    panel_w = 1024
    total_w = panel_w * 3 + 40
    total_h = 1024 + 200
    canvas = Image.new("RGBA", (total_w, total_h), (30, 30, 30, 255))

    # Panel 1: Original face
    canvas.paste(face_img, (10, 10))

    # Panel 2: Wrinkle overlay
    overlay_arr = np.array(face_img).copy()
    wrinkle_px = wrinkle_mask > 0
    overlay_arr[wrinkle_px, 0] = np.minimum(overlay_arr[wrinkle_px, 0].astype(int) + 150, 255).astype(np.uint8)
    overlay_arr[wrinkle_px, 1] = (overlay_arr[wrinkle_px, 1] * 0.4).astype(np.uint8)
    overlay_arr[wrinkle_px, 2] = (overlay_arr[wrinkle_px, 2] * 0.4).astype(np.uint8)
    canvas.paste(Image.fromarray(overlay_arr), (panel_w + 20, 10))

    # Panel 3: Region severity map
    region_overlay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    face_base = face_img.copy().convert("RGBA")
    darkened = np.array(face_base)
    darkened[:, :, :3] = (darkened[:, :, :3] * 0.4).astype(np.uint8)
    face_base = Image.fromarray(darkened)

    for region_name, region_mask in regions.items():
        if region_name in results and results[region_name]["region_pixels"] > 0:
            sev = results[region_name]["severity"]
            color = severity_colors.get(sev, (80, 80, 80, 60))
            colored = np.zeros((1024, 1024, 4), dtype=np.uint8)
            colored[region_mask, :] = color
            region_layer = Image.fromarray(colored, "RGBA")
            region_overlay = Image.alpha_composite(region_overlay, region_layer)

    wrinkle_layer = np.zeros((1024, 1024, 4), dtype=np.uint8)
    wrinkle_layer[wrinkle_px, :] = (255, 255, 255, 200)
    region_overlay = Image.alpha_composite(region_overlay, Image.fromarray(wrinkle_layer, "RGBA"))

    panel3 = Image.alpha_composite(face_base, region_overlay)
    canvas.paste(panel3, (panel_w * 2 + 30, 10))

    # Labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except:
        font = font_sm = font_title = ImageFont.load_default()

    headers = ["Original Face", "Wrinkle Detection", "Severity Map"]
    for i, h in enumerate(headers):
        x = 10 + i * (panel_w + 10) + panel_w // 2
        draw.text((x, 1040), h, fill=(220, 220, 220), font=font_title, anchor="mt")

    # Severity report
    y_text = 1080
    sev_colors_rgb = {"none": (120, 120, 120), "mild": (255, 255, 0),
                      "moderate": (255, 140, 0), "severe": (255, 50, 50)}
    region_labels = {"forehead": "Forehead", "glabella": "Glabella (11 lines)",
                     "crow_feet": "Crow's Feet", "under_eye": "Under Eye",
                     "nasolabial": "Nasolabial", "perioral": "Perioral (Mouth)"}

    x_start = 10
    for i, (key, label) in enumerate(region_labels.items()):
        if key in results:
            r = results[key]
            sev = r["severity"]
            col = sev_colors_rgb.get(sev, (180, 180, 180))
            text = f"{label}: {r['wrinkle_pct']:.2f}% - {sev.upper()}"
            col_x = x_start + (i % 3) * (panel_w + 10)
            col_y = y_text + (i // 3) * 28
            draw.text((col_x, col_y), text, fill=col, font=font_sm)

    overall = results.get("overall", {})
    ov_sev = overall.get("severity", "none")
    ov_col = sev_colors_rgb.get(ov_sev, (180, 180, 180))
    draw.text((x_start, y_text + 70),
              f"OVERALL: {overall.get('wrinkle_pct', 0):.2f}% wrinkle coverage - "
              f"{ov_sev.upper()} (avg region score: {overall.get('weighted_avg_score', 0):.1f}/3.0)",
              fill=ov_col, font=font_title)

    canvas = canvas.convert("RGB")
    canvas.save(output_path, quality=95)
    return canvas


def batch_score(base_folder, model="unet", output_json=None, visualize_ids=None):
    """Score all images and optionally produce visualizations."""
    results_dir = "test_outputs" if model == "unet" else "test_outputs_swinunetr"
    mask_dir = os.path.join(base_folder, results_dir)
    label_dir = os.path.join(base_folder, "etcs", "face_parsed_labels")

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith("_mask.png")])
    all_scores = {}

    for mf in tqdm(mask_files, desc=f"Scoring ({model})"):
        img_id = mf.replace("_mask.png", "")
        label_path = os.path.join(label_dir, f"{img_id}.npy")
        mask_path = os.path.join(mask_dir, mf)

        if not os.path.exists(label_path):
            continue

        scores = score_single_image(mask_path, label_path)
        all_scores[img_id] = scores

        if visualize_ids and img_id in visualize_ids:
            vis_path = os.path.join(base_folder, f"severity_{model}_{img_id}.png")
            visualize_severity(img_id, base_folder, scores, vis_path, model)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"Saved scores to {output_json}")

    return all_scores


def print_summary(all_scores):
    """Print a summary table of severity distribution."""
    from collections import Counter

    region_names = ["forehead", "glabella", "crow_feet", "under_eye", "nasolabial", "perioral", "overall"]
    print(f"\n{'Region':<15} {'None':>6} {'Mild':>6} {'Moderate':>8} {'Severe':>8} {'Avg%':>8}")
    print("-" * 55)

    for region in region_names:
        counts = Counter()
        pcts = []
        for img_id, scores in all_scores.items():
            if region in scores:
                counts[scores[region]["severity"]] += 1
                pcts.append(scores[region]["wrinkle_pct"])
        avg_pct = np.mean(pcts) if pcts else 0
        print(f"{region:<15} {counts.get('none', 0):>6} {counts.get('mild', 0):>6} "
              f"{counts.get('moderate', 0):>8} {counts.get('severe', 0):>8} {avg_pct:>7.2f}%")


if __name__ == "__main__":
    base = "./ffhq_wrinkle_data"
    vis_ids = ["00001", "00048", "00125", "01044", "03140"]

    print("=== UNet Wrinkle Severity Analysis ===")
    unet_scores = batch_score(base, model="unet",
                              output_json=os.path.join(base, "wrinkle_severity_unet.json"),
                              visualize_ids=vis_ids)
    print_summary(unet_scores)

    swinunetr_dir = os.path.join(base, "test_outputs_swinunetr")
    if os.path.exists(swinunetr_dir):
        print("\n=== SwinUNETR Wrinkle Severity Analysis ===")
        swin_scores = batch_score(base, model="swinunetr",
                                  output_json=os.path.join(base, "wrinkle_severity_swinunetr.json"),
                                  visualize_ids=vis_ids)
        print_summary(swin_scores)
