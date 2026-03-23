"""
Debug script to see what class IDs your BiSeNet model outputs
This helps you update FACE_REGION_MAPPING correctly
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your BiSeNet loading function
from acne_detect_with_face_region import load_bisenet_model, build_bisenet_transform

# Configuration - update these paths
BISENET_MODEL_PATH = "/home/vanessa/project/79999_iter.pth"
TEST_IMAGE = "/home/vanessa/project/levle3_113 copy.jpg"
LOAD_FULL_MODEL = True

print("="*60)
print("🔍 BiSeNet Class ID Inspector")
print("="*60)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    # Try with load_full_model parameter (updated version)
    bisenet_model = load_bisenet_model(
        BISENET_MODEL_PATH,
        num_classes=19,
        backbone_name="resnet34",
        load_full_model=LOAD_FULL_MODEL,
        device=device
    )
except TypeError:
    # Fallback for older version without load_full_model parameter
    print("⚠️  Using older load_bisenet_model (without load_full_model parameter)")
    # Try loading the checkpoint
    print(f"📥 Loading checkpoint from {BISENET_MODEL_PATH}")
    try:
        checkpoint = torch.load(BISENET_MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(BISENET_MODEL_PATH, map_location=device)
    
    # Check if it's a full model or state_dict
    if isinstance(checkpoint, torch.nn.Module):
        bisenet_model = checkpoint
        bisenet_model.to(device)
        bisenet_model.eval()
        print("✅ Loaded pre-trained BiSeNet model directly (full model)")
    elif isinstance(checkpoint, dict):
        # It's a state_dict - we need to build the model first
        print("   Checkpoint is a state_dict, building model architecture...")
        # Try to infer num_classes from checkpoint keys
        if 'conv_out.conv.weight' in checkpoint:
            # Count output channels to infer num_classes
            num_classes = checkpoint['conv_out.conv.weight'].shape[0]
            print(f"   Inferred num_classes: {num_classes}")
        else:
            num_classes = 19  # Default
            print(f"   Using default num_classes: {num_classes}")
        
        # Build model and load state_dict
        from models.bisenet import BiSeNet
        bisenet_model = BiSeNet(num_classes=num_classes, backbone_name="resnet34")
        bisenet_model.load_state_dict(checkpoint, strict=False)
        bisenet_model.to(device)
        bisenet_model.eval()
        print("✅ Loaded BiSeNet from state_dict")
    else:
        raise ValueError(f"Unknown checkpoint format. Type: {type(checkpoint)}")

# Load and process image
image_pil = Image.open(TEST_IMAGE).convert("RGB")
image_np = np.array(image_pil)  # For visualization
bisenet_transform = build_bisenet_transform()
image_tensor = bisenet_transform(image_pil)

# Run inference
print("\n🔍 Running BiSeNet on test image...")
with torch.no_grad():
    outputs = bisenet_model(image_tensor.unsqueeze(0).to(device))
    seg_logits = outputs[0]
    seg_pred = seg_logits.argmax(dim=1).squeeze().cpu().numpy()

# Analyze output
unique_classes = np.unique(seg_pred)
print(f"\n📊 BiSeNet Output Analysis:")
print(f"   Image shape: {seg_pred.shape}")
print(f"   Unique class IDs: {unique_classes.tolist()}")
print(f"   Total classes found: {len(unique_classes)}")
print(f"\n   Class distribution:")
for cls_id in sorted(unique_classes):
    count = np.sum(seg_pred == cls_id)
    percentage = (count / seg_pred.size) * 100
    print(f"      Class {cls_id}: {count:8d} pixels ({percentage:5.1f}%)")

# Visualize segmentation - create clearer overlays
print(f"\n📸 Creating visualizations...")

# Create individual class overlays
num_classes = len(unique_classes)
cols = 3
rows = (num_classes + 2) // cols + 1  # +2 for original and combined view

fig = plt.figure(figsize=(18, 6 * rows))
gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)

# Original image
ax = fig.add_subplot(gs[0, 0])
ax.imshow(image_pil)
ax.set_title("Original Image", fontsize=12, fontweight='bold')
ax.axis('off')

# Combined segmentation overlay
ax = fig.add_subplot(gs[0, 1])
overlay = image_np.copy().astype(float) / 255.0
# Create distinct colors for each class
class_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
for i, cls_id in enumerate(unique_classes):
    mask = seg_pred == cls_id
    overlay[mask] = overlay[mask] * 0.5 + np.array(class_colors[i][:3]) * 0.5
ax.imshow(overlay)
ax.set_title(f"All Classes Overlay", fontsize=12, fontweight='bold')
ax.axis('off')

# Legend for combined view
ax = fig.add_subplot(gs[0, 2])
ax.axis('off')
legend_text = "Class IDs:\n"
for i, cls_id in enumerate(unique_classes):
    color = class_colors[i]
    count = np.sum(seg_pred == cls_id)
    pct = (count / seg_pred.size) * 100
    ax.scatter([], [], c=[color], s=100, label=f"Class {cls_id}: {pct:.1f}%")
ax.legend(loc='center', fontsize=10)
ax.set_title("Class Legend", fontsize=12, fontweight='bold')

# Individual class overlays
for idx, cls_id in enumerate(unique_classes):
    row = (idx // cols) + 1
    col = idx % cols
    
    ax = fig.add_subplot(gs[row, col])
    
    # Create overlay: original image with class highlighted
    overlay = image_np.copy().astype(float) / 255.0
    mask = seg_pred == cls_id
    color = class_colors[idx]
    
    # Highlight this class in bright color, dim everything else
    overlay[~mask] = overlay[~mask] * 0.3  # Dim non-class pixels
    overlay[mask] = np.array(color[:3])  # Bright color for this class
    
    ax.imshow(overlay)
    count = np.sum(mask)
    pct = (count / seg_pred.size) * 100
    ax.set_title(f"Class {cls_id}\n({count:,} px, {pct:.1f}%)", 
                 fontsize=11, fontweight='bold',
                 color=tuple(color[:3]))
    ax.axis('off')

plt.suptitle(f"BiSeNet Segmentation - Individual Class Analysis\n(Classes found: {unique_classes.tolist()})", 
             fontsize=14, fontweight='bold', y=0.995)

output_path = "bisenet_segmentation_debug.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved detailed visualization to: {output_path}")

# Suggest mapping based on common face parsing conventions
print(f"\n💡 Suggested FACE_REGION_MAPPING (update in acne_detect_with_face_regions.py):")
print(f"   Based on common BiSeNet face parsing conventions:")
print(f"   ")
print(f"   FACE_REGION_MAPPING = {{")
print(f"       'forehead': [<class_id>],  # Usually upper face region")
print(f"       'left_cheek': [<class_id>],  # Left side of face")
print(f"       'right_cheek': [<class_id>],  # Right side of face")
print(f"       'nose': [<class_id>],  # Nose region")
print(f"       'chin': [<class_id>],  # Lower face region")
print(f"   }}")
print(f"\n   To find the correct class IDs:")
print(f"   1. Look at the visualization image: {output_path}")
print(f"   2. Identify which class ID corresponds to each face region")
print(f"   3. Update FACE_REGION_MAPPING in acne_detect_with_face_regions.py")
print(f"\n   Common mappings (may vary by your model):")
print(f"   - Class 0: background")
print(f"   - Class 1: skin (might need to split into forehead/cheeks/chin)")
print(f"   - Class 6: nose")
print(f"   - Class 7: mouth")
print(f"   - etc.")

plt.show()

