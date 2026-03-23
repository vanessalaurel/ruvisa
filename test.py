from acne_detect_and_grade import run_inference

class Args:
    image = "/home/vanessa/project/levle0_2.jpg"
    detection_model = "/home/vanessa/project/acne_yolo_runs/roboflow_6classes/weights/best.pt"
    severity_model = "/home/vanessa/acne_severity_runs/20251110_172153/best_model.pt"
    conf_thres = 0.1
    iou_thres = 0.5
    output = "face_severity.json"
    save_vis = "auto"  # or "annotated.png", or None if you don’t want an image

run_inference(Args())