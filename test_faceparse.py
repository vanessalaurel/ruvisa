"""
Quick test script for face regions pipeline with pre-configured model paths
Update the paths below to point to your trained models
"""

from acne_detect_with_face_region import run_inference

class Args:
    # Update these paths to your actual model locations
    image = "/home/vanessa/project/levle3_113 copy.jpg"  # Change this to your test image
    
    # YOLO detection model (from acne_yolo_roboflow_only.py training)
    detection_model = "/home/vanessa/project/acne_yolo_runs/roboflow_6classes/weights/best.pt"
    
    # Severity classifier (from acne_severity_classifier.py training)
    severity_model = "/home/vanessa/acne_severity_runs/20251110_172153/best_model.pt"  # Update with actual run folder
    
    # BiSeNet face parsing model
    bisenet_model = "/home/vanessa/project/models/bisenet.py"  # Update with your BiSeNet checkpoint path
    bisenet_num_classes = 19  # Adjust if your BiSeNet uses different number of classes
    bisenet_backbone = "resnet34"  # or "resnet18"
    
    # Optional: Custom ResNet weights
    resnet_weights = "/home/vanessa/project/resnet34.onnx"  # Optional, set to None if not using
    
    # Detection thresholds
    conf_thres = 0.3
    iou_thres = 0.5
    
    # Output paths
    output = "face_regions_roi.json"
    save_vis = "annotated_regions.png"  # Set to None to skip visualization

if __name__ == "__main__":
    # Set resnet_weights to None if not provided
    if not Args.resnet_weights or Args.resnet_weights == "/home/vanessa/project/resnet34.onnx":
        Args.resnet_weights = None
    
    run_inference(Args())

