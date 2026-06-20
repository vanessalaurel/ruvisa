"""Evaluate wrinkle segmentation: predicted masks vs. manual ground-truth masks.

Compares ffhq_wrinkle_data/test_outputs/{id}_mask.png (U-Net predictions) against
ffhq_wrinkle_data/manual_wrinkle_masks/{id}.png (ground truth).

Reports pixel-pooled (micro) precision/recall/F1(=Dice)/IoU and pixel accuracy,
plus mean per-image Dice/IoU over images with non-empty ground truth.
"""
import os
import numpy as np
from PIL import Image

BASE = "ffhq_wrinkle_data"
PRED_DIR = os.path.join(BASE, "test_outputs")
GT_DIR = os.path.join(BASE, "manual_wrinkle_masks")


def load_bin(path, size=None):
    im = Image.open(path).convert("L")
    if size is not None and im.size != size:
        im = im.resize(size, Image.NEAREST)
    return (np.array(im) > 127)


def main():
    gt_files = {f.replace(".png", ""): f for f in os.listdir(GT_DIR) if f.endswith(".png")}
    pred_files = {f.replace("_mask.png", ""): f for f in os.listdir(PRED_DIR) if f.endswith("_mask.png")}
    ids = sorted(set(gt_files) & set(pred_files))
    print(f"GT={len(gt_files)} pred={len(pred_files)} matched={len(ids)}")

    TP = FP = FN = TN = 0
    dices, ious = [], []
    n_gt_pos = 0
    for i in ids:
        gt = load_bin(os.path.join(GT_DIR, gt_files[i]))
        pred = load_bin(os.path.join(PRED_DIR, pred_files[i]), size=(gt.shape[1], gt.shape[0]))
        tp = int(np.sum(gt & pred)); fp = int(np.sum(~gt & pred))
        fn = int(np.sum(gt & ~pred)); tn = int(np.sum(~gt & ~pred))
        TP += tp; FP += fp; FN += fn; TN += tn
        if gt.sum() > 0:
            n_gt_pos += 1
            denom_d = 2 * tp + fp + fn
            denom_i = tp + fp + fn
            dices.append((2 * tp / denom_d) if denom_d else 1.0)
            ious.append((tp / denom_i) if denom_i else 1.0)

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
    pix_acc = (TP + TN) / (TP + TN + FP + FN)

    print("\n=== Wrinkle segmentation (pixel-pooled / micro) ===")
    print(f"Pixel accuracy : {pix_acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 / Dice      : {f1:.4f}")
    print(f"IoU (Jaccard)  : {iou:.4f}")
    print(f"\n=== Mean per-image (over {n_gt_pos} images with non-empty GT) ===")
    print(f"Mean Dice      : {np.mean(dices):.4f}")
    print(f"Mean IoU       : {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
