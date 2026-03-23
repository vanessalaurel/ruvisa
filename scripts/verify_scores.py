#!/usr/bin/env python3
"""
Verify that the app's analysis scores match test_all.py output.

Usage:
  python scripts/verify_scores.py <image_path> [--skin-type oily]
  python scripts/verify_scores.py <image_path> --api   # Compare with live API

Runs test_all.run_inference on the image, then applies the same post-processing
as the API (/api/analyze) and prints both outputs side-by-side for comparison.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONCERNS = [
    "acne", "comedonal_acne", "pigmentation",
    "acne_scars_texture", "pores", "redness", "wrinkles",
]


def _compute_overall_score(concern_vector):
    if not concern_vector or all(v == 0 for v in concern_vector):
        return 75
    avg_concern = sum(concern_vector) / len(concern_vector)
    return max(20, min(98, int(100 - avg_concern * 80)))


def _zone_issues(region_data, label):
    if not region_data:
        return ["No data"]
    sev = region_data.get("severity", "none")
    pct = region_data.get("wrinkle_pct", 0)
    if sev == "none":
        return ["Clear"]
    return [f"{sev.capitalize()} {label} ({pct:.1f}%)"]


def _build_zone_scores(report):
    zones = []
    acne = report.get("acne", {})
    wrinkle = report.get("wrinkle", {})
    wrinkle_regions = report.get("wrinkle_regions", {})

    total_det = acne.get("total_detections", 0)

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


def run_test_all(image_path: Path, skin_type: str, out_dir: Path):
    """Run test_all.run_inference with same args as the API."""
    from test_all import run_inference

    class InferenceArgs:
        pass

    args = InferenceArgs()
    args.image = str(image_path)
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


def build_app_response(report: dict) -> dict:
    """Build the same response structure the app receives from /api/analyze."""
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
    }

    wrinkle = report.get("wrinkle", {})
    wrinkle_summary = {
        "severity": wrinkle.get("severity", "none"),
        "severity_score": wrinkle.get("severity_score", 0),
        "wrinkle_pct": wrinkle.get("wrinkle_pct", 0),
    }

    zone_scores = _build_zone_scores(report)
    overall = _compute_overall_score(concern_vector)

    return {
        "concern_vector": concern_vector,
        "concerns": {c: round(v, 3) for c, v in zip(CONCERNS, concern_vector)},
        "acne_summary": acne_summary,
        "wrinkle_summary": wrinkle_summary,
        "zone_scores": zone_scores,
        "overall_score": overall,
    }


def fetch_api_response(image_path: Path, skin_type: str, api_url: str = "http://localhost:8001") -> dict | None:
    """Call /api/analyze and return the response (requires server running)."""
    try:
        import httpx
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/jpeg")}
            data = {"user_id": "verify-test", "skin_type": skin_type}
            r = httpx.post(f"{api_url}/api/analyze", files=files, data=data, timeout=120)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Verify app scores match test_all output")
    parser.add_argument("image", type=Path, help="Path to face image (or existing report.json)")
    parser.add_argument("--skin-type", default="oily", help="Skin type (default: oily)")
    parser.add_argument("--api", action="store_true", help="Also call /api/analyze and compare")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--report-only", action="store_true", help="Use existing report.json instead of running inference")
    args = parser.parse_args()

    report = None
    run_args = None

    if args.report_only and args.image.suffix == ".json":
        # Use existing report
        if not args.image.exists():
            print(f"Error: Report not found: {args.image}")
            sys.exit(1)
        with open(args.image) as f:
            report = json.load(f)
        print("=" * 60)
        print("VERIFY SCORES (from existing report.json)")
        print("=" * 60)
        print(f"Report:     {args.image}")
    else:
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        out_dir = PROJECT_ROOT / "uploads" / "verify_scores"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("VERIFY SCORES: test_all vs App")
        print("=" * 60)
        print(f"Image:      {args.image}")
        print(f"Skin type:  {args.skin_type}")
        print()

        # 1. Run test_all
        print("Running test_all.run_inference...")
        run_args = run_test_all(args.image, args.skin_type, out_dir)
        with open(run_args.output) as f:
            report = json.load(f)

    # 2. Build app-style response from report
    app_from_testall = build_app_response(report)

    print("\n" + "-" * 60)
    print("APP-STYLE OUTPUT (from test_all report.json)")
    print("-" * 60)
    print(f"Overall Score:     {app_from_testall['overall_score']}")
    print(f"Concern Vector:   {[round(v, 3) for v in app_from_testall['concern_vector']]}")
    print(f"Concerns:         {app_from_testall['concerns']}")
    print(f"Acne Summary:     {app_from_testall['acne_summary']}")
    print(f"Wrinkle Summary:  {app_from_testall['wrinkle_summary']}")
    print(f"Zone Scores:      {[(z['zone'], z['score']) for z in app_from_testall['zone_scores']]}")

    if args.api:
        print("\n" + "-" * 60)
        print("Calling /api/analyze...")
        api_resp = fetch_api_response(args.image, args.skin_type, args.api_url)
        if api_resp:
            print("API Response:")
            print(f"  Overall Score:     {api_resp.get('overall_score')}")
            print(f"  Concern Vector:   {api_resp.get('concern_vector')}")
            print(f"  Concerns:         {api_resp.get('concerns')}")
            print(f"  Acne Summary:     {api_resp.get('acne_summary')}")
            print(f"  Wrinkle Summary:  {api_resp.get('wrinkle_summary')}")

            # Compare
            print("\n" + "-" * 60)
            print("COMPARISON")
            print("-" * 60)
            match = (
                app_from_testall["overall_score"] == api_resp.get("overall_score")
                and app_from_testall["concern_vector"] == api_resp.get("concern_vector")
            )
            if match:
                print("✓ Scores MATCH between test_all and API")
            else:
                print("✗ Scores DIFFER:")
                if app_from_testall["overall_score"] != api_resp.get("overall_score"):
                    print(f"  Overall: test_all={app_from_testall['overall_score']} vs API={api_resp.get('overall_score')}")
                if app_from_testall["concern_vector"] != api_resp.get("concern_vector"):
                    print(f"  Concern vector differs")
        else:
            print("(Start the backend with: test-venv/bin/python -m uvicorn api.main:app --port 8001)")

    print("\n" + "=" * 60)
    if run_args:
        print(f"Report saved to: {run_args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
