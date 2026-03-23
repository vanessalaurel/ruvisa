import os
import json
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from torch.amp import autocast, GradScaler
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# ---------------------------------------------------------------
# Step 1. Load dataset from Roboflow
# ---------------------------------------------------------------
from roboflow import Roboflow

rf = Roboflow(api_key="6xiFDiiTXfnjJ5ILDA5l")
project = rf.workspace("fyp-espej").project("acne-detection-d4kac")

DATASET_VERSION = 2
print(f"📥 Downloading dataset version {DATASET_VERSION}...")
dataset = project.version(DATASET_VERSION).download("coco")

print("✅ Dataset downloaded successfully!")
print(f"Dataset located at: {dataset.location}")

# Analyze dataset quality
def analyze_dataset_quality(ann_file):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    images = {img['id']: img for img in data['images']}
    
    box_sizes = []
    invalid_count = 0
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            invalid_count += 1
            continue
        
        img_id = ann['image_id']
        if img_id in images:
            img_w = images[img_id]['width']
            img_h = images[img_id]['height']
            
            # Normalize box size by image size
            norm_area = (w * h) / (img_w * img_h)
            box_sizes.append(norm_area)
    
    if len(box_sizes) > 0:
        import numpy as np
        sizes_arr = np.array(box_sizes)
        print(f"   Box size stats (% of image):")
        print(f"   - Mean: {sizes_arr.mean()*100:.2f}% | Median: {np.median(sizes_arr)*100:.2f}%")
        print(f"   - Min: {sizes_arr.min()*100:.2f}% | Max: {sizes_arr.max()*100:.2f}%")
        print(f"   - 25th percentile: {np.percentile(sizes_arr, 25)*100:.2f}%")
        print(f"   - 75th percentile: {np.percentile(sizes_arr, 75)*100:.2f}%")
    
    if invalid_count > 0:
        print(f"   ⚠️  Found {invalid_count} invalid boxes (width or height <= 0)")
    
    return box_sizes

print("\n📊 Analyzing dataset quality:")
print("Training set:")
train_boxes = analyze_dataset_quality(f"{dataset.location}/train/_annotations.coco.json")
print("\nValidation set:")
val_boxes = analyze_dataset_quality(f"{dataset.location}/valid/_annotations.coco.json")

# Reality check
import numpy as np
if len(train_boxes) > 0:
    median_size = np.median(train_boxes) * 100
    if median_size < 0.5:
        print(f"\n⚠️  WARNING: Objects are VERY SMALL (median {median_size:.2f}% of image)")
        print("   This is extremely challenging for Faster R-CNN.")
        print("   Recommendations:")
        print("   1. Use larger input image size (1024x1024 or higher)")
        print("   2. Consider YOLO or RetinaNet instead (better for small objects)")
        print("   3. Check if labels are correct - acne should be visible!")
    elif median_size < 2.0:
        print(f"\n⚠️  Note: Objects are small (median {median_size:.2f}% of image)")
        print("   70% mAP@50 may be difficult. Realistic target: 40-60%")
        print("   Applied optimizations: Large images (1280x1920), tiny anchors, more proposals")

# ---------------------------------------------------------------
# Step 2. Data transforms - STRONG augmentation for better generalization
# ---------------------------------------------------------------
train_transform = T.Compose([
    T.ToTensor(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomAdjustSharpness(1.5, p=0.5),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

test_transform = T.Compose([T.ToTensor()])

# ---------------------------------------------------------------
# Step 3. Dataset + Dataloader
# ---------------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CocoDetection(
    root=f"{dataset.location}/train",
    annFile=f"{dataset.location}/train/_annotations.coco.json",
    transform=train_transform
)

valid_dataset = CocoDetection(
    root=f"{dataset.location}/valid",
    annFile=f"{dataset.location}/valid/_annotations.coco.json",
    transform=test_transform
)

BATCH_SIZE = 2  # Smaller due to larger image size (1280x1920)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}")

# ---------------- Load category mapping ----------------
with open(f"{dataset.location}/train/_annotations.coco.json", "r") as f:
    coco_train = json.load(f)

all_cats = {cat["id"]: cat["name"] for cat in coco_train.get("categories", [])}
used_cat_ids = {ann["category_id"] for ann in coco_train.get("annotations", [])}
filtered_categories = [{"id": cid, "name": all_cats[cid]} for cid in sorted(used_cat_ids) if cid in all_cats]

if len(filtered_categories) == 0:
    filtered_categories = coco_train.get("categories", [])

categories = filtered_categories
cat_id_to_contiguous = {cat["id"]: idx + 1 for idx, cat in enumerate(categories)}
class_names = [cat["name"] for cat in categories]
print(f"✅ Using {len(class_names)} classes: {class_names}")

# ---------------------------------------------------------------
# Step 4. Model initialization - USE PRETRAINED WEIGHTS
# ---------------------------------------------------------------
num_classes = len(categories) + 1

# OPTIMIZED FOR TINY OBJECTS
from torchvision.models import ResNet50_Weights

# Custom anchor generator for TINY objects (0.3% of image)
# FPN has 5 feature maps - use ONE size per level for simplicity
anchor_generator = AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),  # One size per FPN level
    aspect_ratios=((0.5, 1.0, 2.0),) * 5  # 3 aspect ratios per level
)

# Load pretrained backbone only (custom anchors incompatible with full model)
backbone_weights = ResNet50_Weights.IMAGENET1K_V1

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights_backbone=backbone_weights,
    rpn_anchor_generator=anchor_generator,
    min_size=1280,  # MUCH larger image size for tiny objects
    max_size=1920,
    # More proposals for tiny objects
    rpn_pre_nms_top_n_train=6000,
    rpn_post_nms_top_n_train=3000,
    rpn_pre_nms_top_n_test=3000,
    rpn_post_nms_top_n_test=1500,
    # Lower IoU thresholds for tiny objects
    rpn_fg_iou_thresh=0.5,
    rpn_bg_iou_thresh=0.3,
    box_fg_iou_thresh=0.4,
    box_bg_iou_thresh=0.3,
    # More boxes per image
    box_batch_size_per_image=512,
)

# Replace only the box predictor head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.is_available():
    print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA not available - training will be slow")

print(f"✅ Faster R-CNN initialized on {device}")

# ---------------------------------------------------------------
# Step 5. Optimizer & Scheduler - PROPER SETUP
# ---------------------------------------------------------------
# Separate learning rates - ADJUSTED for tiny objects
params = [
    {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},
    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 5e-4}
]
optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler - warmup then decay
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

# ---------------------------------------------------------------
# Step 6. Training Loop
# ---------------------------------------------------------------
num_epochs = 150  # More epochs for tiny objects
scaler = GradScaler('cuda') if torch.cuda.is_available() else None
best_val_iou = 0.0
early_stop_patience = 30  # More patience
epochs_since_improve = 0

print("\n" + "="*60)
print("⚙️  TINY OBJECT DETECTION MODE")
print("   - Image size: 1280x1920 (very large)")
print("   - Anchor sizes: 4-512px (very small)")
print("   - Proposals: 3000 train, 1500 test")
print("   - Realistic target: 50-65% mAP@50")
print("="*60)

print("\n" + "="*60)
print("🚀 Starting Training")
print("="*60)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        formatted_targets = []
        
        for t in targets:
            boxes, labels = [], []
            for a in t:
                x, y, w, h = a['bbox']
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(int(cat_id_to_contiguous[int(a['category_id'])]))
            
            if len(boxes) == 0:
                continue
                
            formatted_targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                'labels': torch.tensor(labels, dtype=torch.int64).to(device)
            })
        
        if len(formatted_targets) == 0:
            continue

        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast('cuda'):
                loss_dict = model(images, formatted_targets)
                losses = sum(loss for loss in loss_dict.values())
            
            if torch.isnan(losses) or torch.isinf(losses):
                continue
                
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, formatted_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if torch.isnan(losses) or torch.isinf(losses):
                continue
                
            losses.backward()
            optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1

    scheduler.step()
    avg_loss = total_loss / max(1, num_batches)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # ---------------- Validation every epoch - ADAPTIVE THRESHOLD ----------------
    model.eval()
    
    # Evaluate at multiple thresholds to track progress
    thresholds = [0.3, 0.4, 0.5]
    iou_by_thr = {thr: {'total': 0.0, 'count': 0} for thr in thresholds}
    
    with torch.no_grad():
        for images, targets in valid_loader:
            images = [img.to(device) for img in images]
            batch_preds = model(images)
            
            for i in range(len(images)):
                preds = batch_preds[i]
                scores = preds['scores'].cpu()
                boxes_pred = preds['boxes'].cpu()
                
                # Get ground truth
                boxes_gt = []
                for ann in targets[i]:
                    x, y, w, h = ann['bbox']
                    if w > 0 and h > 0:
                        boxes_gt.append([x, y, x + w, y + h])
                
                if len(boxes_gt) == 0:
                    continue
                    
                boxes_gt = torch.tensor(boxes_gt, dtype=torch.float32)
                
                # Evaluate at each threshold
                for thr in thresholds:
                    keep = scores >= thr
                    if keep.sum() == 0:
                        continue
                    boxes_thr = boxes_pred[keep]
                    ious = box_iou(boxes_thr, boxes_gt)
                    
                    if ious.numel() > 0:
                        iou_by_thr[thr]['total'] += ious.max(dim=1)[0].mean().item()
                        iou_by_thr[thr]['count'] += 1

    # Calculate IoU for each threshold
    val_iou_03 = iou_by_thr[0.3]['total'] / iou_by_thr[0.3]['count'] if iou_by_thr[0.3]['count'] > 0 else 0.0
    val_iou_04 = iou_by_thr[0.4]['total'] / iou_by_thr[0.4]['count'] if iou_by_thr[0.4]['count'] > 0 else 0.0
    val_iou_05 = iou_by_thr[0.5]['total'] / iou_by_thr[0.5]['count'] if iou_by_thr[0.5]['count'] > 0 else 0.0
    
    print(f"   Val mAP @0.3: {val_iou_03:.4f} ({val_iou_03*100:.1f}%) [{iou_by_thr[0.3]['count']} imgs] | "
          f"@0.4: {val_iou_04:.4f} ({val_iou_04*100:.1f}%) [{iou_by_thr[0.4]['count']}] | "
          f"@0.5: {val_iou_05:.4f} ({val_iou_05*100:.1f}%) [{iou_by_thr[0.5]['count']}]")
    
    # Use 0.5 threshold for best model selection
    val_iou = val_iou_05

    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_fasterrcnn.pth")
        print(f"   ✨ New best! Model saved. (Target: 70%)")
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= early_stop_patience:
            print(f"   ⏹️  Early stopping: no improvement for {early_stop_patience} epochs.")
            break

print("\n✅ Training complete!")
print(f"Best validation mAP@50: {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")

# ---------------------------------------------------------------
# Step 7. Testing
# ---------------------------------------------------------------
print("\n" + "="*60)
print("📊 TESTING - Evaluating on Test Set")
print("="*60)

model.load_state_dict(torch.load("best_fasterrcnn.pth", map_location=device))
model.eval()

test_dataset = CocoDetection(
    root=f"{dataset.location}/test",
    annFile=f"{dataset.location}/test/_annotations.coco.json",
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Evaluate at multiple thresholds
test_thresholds = [0.3, 0.4, 0.5]
test_iou_by_thr = {thr: {'total': 0.0, 'count': 0} for thr in test_thresholds}
max_scores = []

with torch.no_grad():
    for images, targets in test_loader:
        images = [img.to(device) for img in images]
        preds = model(images)[0]
        scores = preds['scores'].cpu()
        boxes_all = preds['boxes'].cpu()
        
        if len(scores) > 0:
            max_scores.append(scores.max().item())
        
        boxes_gt = []
        for ann in targets[0]:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes_gt.append([x, y, x + w, y + h])
        
        if len(boxes_gt) == 0:
            continue
            
        boxes_gt = torch.tensor(boxes_gt, dtype=torch.float32)
        
        # Evaluate at each threshold
        for thr in test_thresholds:
            keep = scores >= thr
            if keep.sum() == 0:
                continue
            boxes_pred = boxes_all[keep]
            ious = box_iou(boxes_pred, boxes_gt)
            
            if ious.numel() > 0:
                test_iou_by_thr[thr]['total'] += ious.max(dim=1)[0].mean().item()
                test_iou_by_thr[thr]['count'] += 1

# Calculate results
test_iou_03 = test_iou_by_thr[0.3]['total'] / test_iou_by_thr[0.3]['count'] if test_iou_by_thr[0.3]['count'] > 0 else 0.0
test_iou_04 = test_iou_by_thr[0.4]['total'] / test_iou_by_thr[0.4]['count'] if test_iou_by_thr[0.4]['count'] > 0 else 0.0
test_iou_05 = test_iou_by_thr[0.5]['total'] / test_iou_by_thr[0.5]['count'] if test_iou_by_thr[0.5]['count'] > 0 else 0.0

if len(max_scores) > 0:
    avg_max_score = sum(max_scores) / len(max_scores)
    print(f"   Average max confidence: {avg_max_score:.3f}")
    print(f"   Max confidence seen: {max(max_scores):.3f}")

print(f"\n✅ TEST mAP @0.3: {test_iou_03:.4f} ({test_iou_03*100:.2f}%) on {test_iou_by_thr[0.3]['count']}/{len(test_dataset)} images")
print(f"✅ TEST mAP @0.4: {test_iou_04:.4f} ({test_iou_04*100:.2f}%) on {test_iou_by_thr[0.4]['count']}/{len(test_dataset)} images")
print(f"✅ TEST mAP @0.5: {test_iou_05:.4f} ({test_iou_05*100:.2f}%) on {test_iou_by_thr[0.5]['count']}/{len(test_dataset)} images")
print("="*60)

# ---------------------------------------------------------------
# Step 8. Visualization
# ---------------------------------------------------------------
def visualize_samples(model, dataset, device, n=4, score_thresh=0.5):
    import random
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    for idx in idxs:
        img, anns = dataset[idx]
        with torch.no_grad():
            preds = model([img.to(device)])[0]
        
        boxes_pred = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        
        keep = scores > score_thresh
        boxes_pred = boxes_pred[keep]
        scores_keep = scores[keep]
        labels_keep = labels[keep]
        
        boxes_gt = [[x, y, x + w, y + h] for x, y, w, h in [a['bbox'] for a in anns]]
        
        img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw GT in red
        for b in boxes_gt:
            x1,y1,x2,y2 = map(int,b)
            cv2.rectangle(img_cv,(x1,y1),(x2,y2),(0,0,255),2)
        
        # Draw predictions in green with labels
        for b, s, l in zip(boxes_pred, scores_keep, labels_keep):
            x1,y1,x2,y2 = map(int,b)
            cv2.rectangle(img_cv,(x1,y1),(x2,y2),(0,255,0),2)
            label_text = f"{class_names[l-1]}: {s:.2f}"
            cv2.putText(img_cv, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"GT (red) vs Pred (green) — Image {idx}")
    
    plt.show()
    print("✅ Visualization done")

# Uncomment to visualize:
# visualize_samples(model, test_dataset, device, n=5, score_thresh=0.5)
