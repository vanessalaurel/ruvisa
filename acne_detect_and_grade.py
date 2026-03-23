import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from ultralytics import YOLO

SEVERITY_NAMES = ["level0", "level1", "level2", "level3"]
SEVERITY_TO_SCORE = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}
DEFAULT_OUTPUT_JSON = "prediction.json"
DEFAULT_VIS_PATH = "annotated.png"


def load_severity_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(SEVERITY_NAMES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def build_severity_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def draw_annotations(image_np: np.ndarray, detections: List[Dict], save_path: str):
    canvas = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["class_name"]
        score = det["confidence"]
        severity = det["severity_name"]
        sev_score = det["severity_score"]

        color = (0, 255, 0)
        if sev_score >= 0.66:
            color = (0, 0, 255)
        elif sev_score >= 0.33:
            color = (0, 165, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {score:.2f} | {severity} ({sev_score:.2f})"
        cv2.putText(canvas, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return save_path


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    det_model = YOLO(args.detection_model)
    print(f"Loaded YOLO model from {args.detection_model}")

    severity_model = load_severity_model(args.severity_model, device=device)
    print(f"Loaded severity classifier from {args.severity_model}")

    severity_transform = build_severity_transform()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")
    pil_image = Image.open(args.image).convert("RGB")
    image_np = np.array(pil_image)

    results = det_model.predict(
        source=args.image,
        conf=args.conf_thres,
        iou=args.iou_thres,
        device=0 if device == "cuda" else None,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO model returned no results")
    result = results[0]

    detections = []

    for box, cls, score in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(int, box)
        crop = image_np[max(0, y1): max(1, y2), max(0, x1): max(1, x2)]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)
        crop_tensor = severity_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = severity_model(crop_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            prob_vals, pred_idx = torch.max(probs, dim=1)

        severity_level = int(pred_idx.item())
        severity_score = float(SEVERITY_TO_SCORE[severity_level])
        severity_conf = float(prob_vals.item())

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(score),
                "class_id": int(cls),
                "class_name": result.names[int(cls)],
                "severity_level": severity_level,
                "severity_name": SEVERITY_NAMES[severity_level],
                "severity_score": severity_score,
                "severity_confidence": severity_conf,
            }
        )

    aggregates = {}
    overall = {}
    if detections:
        severity_by_class = defaultdict(list)
        severity_all = []
        for det in detections:
            severity_by_class[det["class_name"]].append(det["severity_score"])
            severity_all.append(det["severity_score"])

        for cls_name, scores in severity_by_class.items():
            aggregates[cls_name] = {
                "count": len(scores),
                "mean_severity": float(np.mean(scores)),
                "max_severity": float(np.max(scores)),
            }
        overall = {
            "count": len(severity_all),
            "mean_severity": float(np.mean(severity_all)),
            "max_severity": float(np.max(severity_all)),
        }

    output = {
        "image": os.path.abspath(args.image),
        "detections": detections,
        "aggregates": aggregates,
        "overall_roi": overall,
        "metadata": {
            "detection_model": os.path.abspath(args.detection_model),
            "severity_model": os.path.abspath(args.severity_model),
            "conf_threshold": args.conf_thres,
            "iou_threshold": args.iou_thres,
        },
    }

    out_path = args.output or DEFAULT_OUTPUT_JSON
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetections with severity written to {out_path}")

    if detections and args.save_vis:
        vis_path = args.save_vis if args.save_vis.lower() != "auto" else DEFAULT_VIS_PATH
        vis_path = os.path.abspath(vis_path)
        draw_annotations(image_np, detections, vis_path)
        print(f"Annotated image saved to {vis_path}")

    if aggregates:
        print("\nAggregate severity scores:")
        for cls_name, stats in aggregates.items():
            print(
                f" - {cls_name}: count={stats['count']}, "
                f"mean={stats['mean_severity']:.2f}, max={stats['max_severity']:.2f}"
            )
        if overall:
            print(
                f"\nOverall ROI severity → count={overall['count']}, "
                f"mean={overall['mean_severity']:.2f}, max={overall['max_severity']:.2f}"
            )
    else:
        print("No detections found above threshold.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO detection and ACNE severity grading on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--detection-model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--severity-model", required=True, help="Path to trained severity classifier checkpoint (.pt)")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--output", default=None, help="Optional path to save JSON results")
    parser.add_argument(
        "--save-vis",
        nargs="?",
        const="auto",
        default=None,
        help="Optional path to save annotated image (use without value to auto-name)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
