"""
Acne Detection with Face Region Segmentation and ROI Scoring
- Uses EXACT face parsing code from test_face_parsing_only.py (imported directly)
- YOLO detects lesions in each segmented region
- Severity classifier grades each lesion
- Calculates ROI scores per face region (forehead, left_cheek, right_cheek, chin)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from ultralytics import YOLO

# Import parse_face directly from test_face_parsing_only.py to use exact same code
from trialfaceparsing import parse_face

SEVERITY_NAMES = ["level0", "level1", "level2", "level3"]
SEVERITY_TO_SCORE = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}

# Colors for visualization (RGB format)
REGION_COLORS = {
    "forehead": (255, 0, 0),      # Red
    "left_cheek": (0, 255, 0),    # Green
    "right_cheek": (0, 0, 255),   # Blue
    "chin": (255, 0, 255),        # Magenta
}


# ============================================================================
# Detection and Classification Functions
# ============================================================================

def load_severity_model(checkpoint_path: str, device: str = "cuda"):
    """Load ACNE04 severity classifier."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(SEVERITY_NAMES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def build_severity_transform():
    """Transform for severity classifier input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_detection_region(bbox: List[float], region_masks: Dict[str, np.ndarray], 
                        image_shape: Tuple[int, int]) -> Optional[str]:
    """Determine which face region a detection belongs to based on bbox center."""
    if not region_masks:
        return None
    
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    h, w = image_shape[:2]

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    for region_name, mask in region_masks.items():
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_uint8 = mask.astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 0).astype(bool)
        else:
            mask_resized = mask
        
        if cy < mask_resized.shape[0] and cx < mask_resized.shape[1]:
            if mask_resized[cy, cx]:
                return region_name

    return None


def draw_annotations_with_regions(image_np: np.ndarray, detections: List[Dict],
    seg_mask: np.ndarray, region_masks: Dict[str, np.ndarray],
    roi_by_region: Dict[str, Dict], save_path: str, alpha: float = 0.4):
    """Draw detections, face regions, and ROI scores on image."""
    canvas = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Draw region masks
    for region_name, mask in region_masks.items():
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask
        
        color = REGION_COLORS.get(region_name, (128, 128, 128))
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        
        overlay = canvas.copy()
        overlay[mask] = color_bgr
        canvas = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, color_bgr, 2)
    
    # Draw detection boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["class_name"]
        score = det["confidence"]
        severity = det["severity_name"]
        sev_score = det["severity_score"]
        region = det.get("face_region", "unknown")

        if sev_score >= 0.66:
            color = (0, 0, 255)  # Red
        elif sev_score >= 0.33:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {score:.2f} | {severity} | {region}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(canvas, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw ROI scores per region
    y_offset = 30
    region_names = ["forehead", "left_cheek", "right_cheek", "chin"]
    for region_name in region_names:
        if region_name in roi_by_region:
            stats = roi_by_region[region_name]
            text = f"{region_name}: ROI={stats['roi']:.3f}, Count={stats['count']}"
            cv2.putText(canvas, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    cv2.imwrite(save_path, canvas)


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Validate required arguments
    required_attrs = ['image', 'detection_model', 'severity_model', 'bisenet_model']
    for attr in required_attrs:
        if not hasattr(args, attr):
            raise AttributeError(f"Args object missing required attribute: {attr}")
        if getattr(args, attr) is None:
            raise ValueError(f"Required argument {attr} is None")

    # Load models
    det_model = YOLO(args.detection_model)
    print(f"✅ Loaded YOLO model from {args.detection_model}")

    severity_model = load_severity_model(args.severity_model, device=device)
    print(f"✅ Loaded severity classifier from {args.severity_model}")

    severity_transform = build_severity_transform()

    # Step 1: Parse face into regions (using EXACT code from test_face_parsing_only.py)
    print("\n" + "="*60)
    print("Step 1: Face Parsing (using test_face_parsing_only.py code)")
    print("="*60)
    face_result = parse_face(args.image, args.bisenet_model, device=device)
    
    if face_result is None:
        print("❌ Face parsing failed!")
        return
    
    image_np = face_result["image"]
    seg_mask = face_result["segmentation"]
    region_masks = face_result["regions"]
    h, w = image_np.shape[:2]
    
    print(f"✅ Face parsed into {len(region_masks)} regions: {list(region_masks.keys())}")

    # Step 2: Run YOLO detections on full image
    print("\n" + "="*60)
    print("Step 2: Lesion Detection")
    print("="*60)
    # Get optional arguments with defaults
    conf_thres = getattr(args, 'conf_thres', 0.3)
    iou_thres = getattr(args, 'iou_thres', 0.5)
    
    results = det_model.predict(
        source=args.image,
        conf=conf_thres,
        iou=iou_thres,
        device=0 if device == "cuda" else None
    )
    if not results:
        raise RuntimeError("YOLO model returned no results")
    result = results[0]

    # Step 3: Classify severity and assign to regions
    print("\n" + "="*60)
    print("Step 3: Severity Classification & Region Assignment")
    print("="*60)
    detections = []
    severity_by_region = defaultdict(list)

    for box, cls, score in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Get severity
        crop_pil = Image.fromarray(crop)
        crop_tensor = severity_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = severity_model(crop_tensor)
            probs = F.softmax(logits, dim=1)
            prob_vals, pred_idx = torch.max(probs, dim=1)

        severity_level = int(pred_idx.item())
        severity_score = float(SEVERITY_TO_SCORE[severity_level])
        severity_conf = float(prob_vals.item())

        # Assign to region
        region = get_detection_region([x1, y1, x2, y2], region_masks, image_np.shape)

        detection = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(score),
            "class_id": int(cls),
            "class_name": result.names[int(cls)],
            "severity_level": severity_level,
            "severity_name": SEVERITY_NAMES[severity_level],
            "severity_score": severity_score,
            "severity_confidence": severity_conf,
            "face_region": region if region else "unknown",
        }
        detections.append(detection)

        if region:
            severity_by_region[region].append(severity_score)

    print(f"✅ Processed {len(detections)} detections")
    for region_name, scores in severity_by_region.items():
        print(f"   {region_name}: {len(scores)} detections")

    # Step 4: Calculate ROI per region
    print("\n" + "="*60)
    print("Step 4: ROI Calculation")
    print("="*60)
    roi_by_region = {}
    region_names = ["forehead", "left_cheek", "right_cheek", "chin"]
    
    for region_name in region_names:
        if region_name in severity_by_region and severity_by_region[region_name]:
            scores = severity_by_region[region_name]
            roi_by_region[region_name] = {
                "count": len(scores),
                "roi": float(np.mean(scores)),
            }
        else:
            roi_by_region[region_name] = {
                "count": 0,
                "roi": 0.0,
            }

    # Save JSON output
    output = {
        "image": os.path.abspath(args.image),
        "regions": roi_by_region,
        "total_detections": len(detections),
        "detections": detections,
    }

    out_path = getattr(args, 'output', None) or "face_severity_regions.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Results written to {out_path}")

    # Visualization
    save_vis = getattr(args, 'save_vis', "auto")
    if save_vis and save_vis.lower() != "none":
        vis_path = save_vis if save_vis != "auto" else "annotated_regions.png"
        draw_annotations_with_regions(
            image_np, detections, seg_mask, region_masks, roi_by_region, vis_path
        )
        print(f"✅ Visualization saved to {vis_path}")

    # Print summary
    print("\n" + "="*60)
    print("📊 Results by Face Region (ROI & Counts)")
    print("="*60)
    for region_name in region_names:
        stats = roi_by_region[region_name]
        print(f"   {region_name:15s}: ROI={stats['roi']:.3f}, Count={stats['count']}")
    
    total_count = sum(roi_by_region[r]['count'] for r in region_names)
    print(f"\n   Total lesions detected: {total_count}")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection + severity grading with face region segmentation"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--detection-model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--severity-model", required=True,
                       help="Path to trained severity classifier checkpoint (.pt)")
    parser.add_argument("--bisenet-model", required=True,
                       help="Path to BiSeNet face parsing checkpoint (.pth)")
    parser.add_argument("--conf-thres", type=float, default=0.3,
                       help="YOLO confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--save-vis", default="auto",
                       help="Path to save visualization (or 'auto' for default, or 'none' to skip)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save_vis.lower() == "none":
        args.save_vis = None
    run_inference(args)
