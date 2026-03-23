"""
YOLOv8 Acne Detection - Roboflow Dataset Only (6 Classes)
Simple training script using only Roboflow dataset
"""

import os
import shutil
import torch
from roboflow import Roboflow
from ultralytics import YOLO
import yaml

print("="*60)
print("🚀 YOLOv8 Acne Detection - Roboflow Only (6 Classes)")
print("="*60)

# Configuration
ROBOFLOW_API_KEY = "6xiFDiiTXfnjJ5ILDA5l"
ROBOFLOW_WORKSPACE = "fyp-espej"
ROBOFLOW_PROJECT = "acne-detection-d4kac"
ROBOFLOW_VERSION = 2

# Expected 6 classes (final class order)
TARGET_CLASS_NAMES = ['Acne', 'nodule', 'blackhead', 'whitehead', 'acne_scars', 'flat_wart']
# Any existing Roboflow classes that should merge into 'acne'
REMAP_TO_ACNE = {'Pustule', 'pustule', 'papular', 'Papular', 'Pimples-acne', 'Pimples-Acne'}

# Will hold the class names actually used for training (after remap)
CLASS_NAMES = TARGET_CLASS_NAMES.copy()

print(f"\n📊 Configuration:")
print(f"   Classes: {len(CLASS_NAMES)}")
print(f"   Class names: {CLASS_NAMES}")

# Step 1: Download Roboflow dataset
print(f"\n" + "="*60)
print(f"Step 1: Downloading Roboflow dataset...")
print(f"="*60)

DOWNLOAD_DIR = "/home/vanessa/project/roboflow_v2_latest"

if os.path.exists(DOWNLOAD_DIR):
    print(f"🧹 Removing previous download at {DOWNLOAD_DIR} to force a fresh copy...")
    shutil.rmtree(DOWNLOAD_DIR)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
roboflow_dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=DOWNLOAD_DIR, overwrite=True)

dataset_root = os.path.abspath(roboflow_dataset.location)
print(f"✅ Dataset downloaded to: {dataset_root}")
try:
    top_entries = os.listdir(dataset_root)
    print(f"   Top-level entries ({len(top_entries)}): {top_entries[:10]}")
except Exception as e:
    print(f"   ⚠️  Could not list dataset contents: {e}")

# Step 2: Verify dataset structure and classes
print(f"\n" + "="*60)
print(f"Step 2: Verifying dataset...")
print(f"="*60)

# Read data.yaml from Roboflow
data_yaml_path = os.path.join(dataset_root, "data.yaml")
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    roboflow_classes = data_config.get('names', [])
    if not roboflow_classes:
        print("❌ data.yaml has no class names. Please verify the Roboflow export.")
        raise SystemExit(1)

    print(f"📋 Classes in Roboflow dataset:")
    print(f"   Found {len(roboflow_classes)} classes: {roboflow_classes}")

    missing_targets = [name for name in TARGET_CLASS_NAMES if name not in roboflow_classes]
    if missing_targets:
        print(f"\n⚠️  Target classes missing from Roboflow export: {missing_targets}")
        print("   We'll proceed, but labels for these classes cannot be generated.")

    # Build mapping from original class id -> new class id (or None to drop)
    orig_id_to_new = {}
    dropped_source_classes = set()
    remapped_source_classes = set()

    acne_new_id = TARGET_CLASS_NAMES.index('Acne')

    for idx, name in enumerate(roboflow_classes):
        if name in TARGET_CLASS_NAMES:
            orig_id_to_new[idx] = TARGET_CLASS_NAMES.index(name)
        elif name in REMAP_TO_ACNE:
            orig_id_to_new[idx] = acne_new_id
            remapped_source_classes.add(name)
        else:
            orig_id_to_new[idx] = None
            dropped_source_classes.add(name)

    print(f"\n🧩 Class remapping summary:")
    print(f"   → Keeping classes: {TARGET_CLASS_NAMES}")
    if remapped_source_classes:
        print(f"   → Remapping {sorted(remapped_source_classes)} → 'acne'")
    if dropped_source_classes:
        print(f"   → Dropping {sorted(dropped_source_classes)} (no target mapping)")
else:
    print(f"⚠️  data.yaml not found, will create one with 6 classes")
    # Fallback: assume direct mapping order
    roboflow_classes = TARGET_CLASS_NAMES
    orig_id_to_new = {i: i for i in range(len(TARGET_CLASS_NAMES))}

# Locate split directories (handles Train/Valid/Test variations)
def find_split_dir(split_name: str):
    """Locate the directory inside dataset_root that contains the images for a given split."""
    candidates = []
    for entry in os.listdir(dataset_root):
        path = os.path.join(dataset_root, entry)
        if os.path.isdir(path) and entry.lower().startswith(split_name):
            candidates.append(path)

    # Prefer exact match (e.g., 'train' before 'Train200')
    candidates.sort(key=lambda p: (os.path.basename(p).lower() != split_name, os.path.basename(p)))

    for candidate in candidates:
        images_dir = os.path.join(candidate, "images")
        if os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0:
            labels_dir = os.path.join(candidate, "labels")
            if not os.path.isdir(labels_dir):
                labels_dir = None
            return {
                "root": candidate,
                "images": images_dir,
                "labels": labels_dir,
                "name": os.path.basename(candidate),
            }

    # Fallback to standard structure (root/train/images, etc.)
    fallback_root = os.path.join(dataset_root, split_name)
    fallback_images = os.path.join(fallback_root, "images")
    if os.path.isdir(fallback_images) and len(os.listdir(fallback_images)) > 0:
        fallback_labels = os.path.join(fallback_root, "labels")
        if not os.path.isdir(fallback_labels):
            fallback_labels = None
        return {
            "root": fallback_root,
            "images": fallback_images,
            "labels": fallback_labels,
            "name": os.path.basename(fallback_root),
        }

    return None

split_info = {}
for split in ["train", "valid", "test"]:
    info = find_split_dir(split)
    if info:
        split_info[split] = info
        print(f"   🔎 {split}: using folder '{info['name']}'")
    else:
        print(f"   ⚠️  Could not locate images for split '{split}'.")

# Count images
print(f"\n📊 Dataset counts:")
for split in ['train', 'valid', 'test']:
    info = split_info.get(split)
    if info:
        if os.path.isdir(info["images"]):
            img_count = len([f for f in os.listdir(info["images"])
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            img_count = 0
        if info["labels"]:
            label_count = len([f for f in os.listdir(info["labels"])
                               if f.lower().endswith('.txt')])
        else:
            label_count = 0
        print(f"   {split}: {img_count} images, {label_count} label files")
    else:
        print(f"   {split}: 0 images (split not found)")

required_missing = [s for s in ["train", "valid"] if s not in split_info]
if required_missing:
    print(f"\n❌ Required splits missing: {required_missing}")
    print("   Please verify the Roboflow download contains these splits.")
    raise SystemExit(1)

# Step 2.5: Sanitize labels (remove any classes outside expected range)
print(f"\n" + "="*60)
print(f"Step 2.5: Remapping / sanitizing label files...")
print(f"="*60)

def remap_label_file(path: str) -> tuple[int, int]:
    """Rewrite YOLO label file with new class ids. Returns (kept, dropped)."""
    kept = 0
    dropped = 0
    changed = False

    try:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception:
        return (0, 0)

    new_lines = []
    for line in lines:
        parts = line.split()
        try:
            orig_cid = int(float(parts[0]))
        except Exception:
            dropped += 1
            changed = True
            continue

        new_cid = orig_id_to_new.get(orig_cid, None)
        if new_cid is None:
            dropped += 1
            changed = True
            continue

        if orig_cid != new_cid:
            changed = True
        new_lines.append(" ".join([str(new_cid)] + parts[1:]))
        kept += 1

    if changed:
        if new_lines:
            with open(path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        else:
            os.remove(path)

    return kept, dropped

sanitization_stats = {}
for split, info in split_info.items():
    labels_dir = info.get("labels")
    if not labels_dir:
        continue
    total_removed = 0
    total_files = 0
    for label_file in os.listdir(labels_dir):
        if not label_file.lower().endswith(".txt"):
            continue
        total_files += 1
        _, removed = remap_label_file(os.path.join(labels_dir, label_file))
        total_removed += removed
    sanitization_stats[split] = (total_files, total_removed)
    if total_removed > 0:
        print(f"   ⚠️  {split}: removed {total_removed} invalid label entries across {total_files} files")
    else:
        print(f"   ✅ {split}: all labels within expected class set")

# Step 3: Update data.yaml to ensure 6 classes
print(f"\n" + "="*60)
print(f"Step 3: Configuring dataset...")
print(f"="*60)

# Update data.yaml with correct class count and names
data_yaml_content = {
    'path': dataset_root,
    'train': split_info['train']['images'],
    'val': split_info['valid']['images'],
    'nc': len(CLASS_NAMES),
    'names': CLASS_NAMES
}

if 'test' in split_info:
    data_yaml_content['test'] = split_info['test']['images']

with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"✅ data.yaml updated:")
print(f"   Classes: {len(CLASS_NAMES)}")
print(f"   Names: {CLASS_NAMES}")
print(f"   Train images dir: {data_yaml_content['train']}")
print(f"   Val images dir:   {data_yaml_content['val']}")
print(f"   Test images dir:  {data_yaml_content.get('test', '(not provided)')}")

# Step 4: Train YOLOv8 model
print(f"\n" + "="*60)
print(f"Step 4: Training YOLOv8 model...")
print(f"="*60)

model = YOLO('yolov8m.pt')  # Medium model for better accuracy

training_config = {
    'data': data_yaml_path,
    'epochs': 100,
    'patience': 20,
    'imgsz': 1280,  # Large image size for tiny objects
    'batch': 8,
    'lr0': 0.01,
    'optimizer': 'AdamW',
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'project': 'acne_yolo_runs',
    'name': 'roboflow_6classes',
    'exist_ok': True,
}

print(f"🚀 Starting training...")
print(f"   Model: YOLOv8-Medium")
print(f"   Image size: 1280x1280")
print(f"   Epochs: 100")
print(f"   Batch size: 8")
print(f"   Classes: {len(CLASS_NAMES)}")
print(f"   GPU: {'✅ Available' if torch.cuda.is_available() else '❌ CPU only'}")

results = model.train(**training_config)

print(f"\n✅ Training complete!")
print(f"   Best model: acne_yolo_runs/roboflow_6classes/weights/best.pt")

# Step 5: Evaluate model
print(f"\n" + "="*60)
print(f"Step 5: Evaluating Model")
print(f"="*60)

print(f"\n📊 Validation Set Results:")
val_metrics = model.val(data=data_yaml_path, split='val')
print(f"   mAP@50: {val_metrics.box.map50:.1%}")
print(f"   mAP@50-95: {val_metrics.box.map:.1%}")
print(f"   Precision: {val_metrics.box.mp:.1%}")
print(f"   Recall: {val_metrics.box.mr:.1%}")

print(f"\n📊 Per-Class Performance (mAP@50):")
if hasattr(val_metrics, 'names') and hasattr(val_metrics.box, 'maps50'):
    for i, class_name in enumerate(val_metrics.names):
        if i < len(val_metrics.box.maps50):
            map50 = val_metrics.box.maps50[i]
            print(f"   {class_name}: {map50:.1%}")

print(f"\n📊 Test Set Results:")
test_metrics = model.val(data=data_yaml_path, split='test')
print(f"   mAP@50: {test_metrics.box.map50:.1%}")
print(f"   mAP@50-95: {test_metrics.box.map:.1%}")
print(f"   Precision: {test_metrics.box.mp:.1%}")
print(f"   Recall: {test_metrics.box.mr:.1%}")

print(f"\n📊 Per-Class Performance on Test Set (mAP@50):")
if hasattr(test_metrics, 'names') and hasattr(test_metrics.box, 'maps50'):
    for i, class_name in enumerate(test_metrics.names):
        if i < len(test_metrics.box.maps50):
            map50 = test_metrics.box.maps50[i]
            print(f"   {class_name}: {map50:.1%}")

print(f"\n" + "="*60)
print(f"✅ Training Complete!")
print(f"="*60)
print(f"\n📁 Model saved at:")
print(f"   acne_yolo_runs/roboflow_6classes/weights/best.pt")
print(f"\n💡 Next steps:")
print(f"   1. Check training results in: acne_yolo_runs/roboflow_6classes/")
print(f"   2. Use the model for inference:")
print(f"      model = YOLO('acne_yolo_runs/roboflow_6classes/weights/best.pt')")
print(f"      results = model('path/to/image.jpg')")

