"""
Standalone Face Parsing Test Script
- Clones official BiSeNet repo
- Downloads pretrained model
- Parses face into skin regions
- Divides into 4 regions: forehead, left_cheek, right_cheek, chin
- Visualizes results
"""

import os
import sys
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms

# Try to import gdown for model download
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False
    print("⚠️  gdown not installed. Model download will be skipped if model doesn't exist.")
    print("   Install with: pip install gdown")


def setup_bisenet_repo():
    """Clone the official BiSeNet repository if it doesn't exist."""
    repo_dir = "face-parsing.PyTorch"
    model_file = os.path.join(repo_dir, "model.py")
    
    # Check if repo exists and has the model file
    if os.path.exists(repo_dir) and os.path.exists(model_file):
        print(f"✅ Repository already exists at {repo_dir}")
        print(f"   (Skipping clone - using existing repo)")
    else:
        if os.path.exists(repo_dir):
            print(f"⚠️  Repository directory exists but model.py not found")
            print(f"   Re-cloning repository...")
            # Remove incomplete repo
            import shutil
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
        
        print(f"📥 Cloning BiSeNet repository...")
        try:
            result = subprocess.run(["git", "clone", "https://github.com/zllrunning/face-parsing.PyTorch"], 
                         check=True, capture_output=True, text=True)
            print(f"✅ Cloned repository to {repo_dir}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository")
            if e.stderr:
                print(f"   Error: {e.stderr}")
            if e.stdout:
                print(f"   Output: {e.stdout}")
            return False
        except FileNotFoundError:
            print(f"❌ git not found. Please install git or clone manually:")
            print(f"   git clone https://github.com/zllrunning/face-parsing.PyTorch")
            return False
    
    # Verify model.py exists
    if not os.path.exists(model_file):
        print(f"❌ model.py not found in {repo_dir}")
        return False
    
    # Add repo to path (only if not already there)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        print(f"   Added {repo_dir} to Python path")
    
    return True


def download_bisenet_model(model_file: str = "79999_iter.pth", model_id: str = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"):
    """Download pretrained BiSeNet model if it doesn't exist."""
    if os.path.exists(model_file):
        print(f"✅ Model file already exists: {model_file}")
        print(f"   Size: {os.path.getsize(model_file)} bytes")
        return model_file
    
    if not HAS_GDOWN:
        print(f"⚠️  Model file not found: {model_file}")
        print(f"   gdown not available. Please download manually:")
        print(f"   https://drive.google.com/uc?id={model_id}")
        return None
    
    print(f"📥 Downloading pretrained model...")
    try:
        gdown.download(id=model_id, output=model_file, quiet=False)
        if os.path.exists(model_file):
            print(f"✅ Downloaded model: {model_file}")
            print(f"   Size: {os.path.getsize(model_file)} bytes")
            return model_file
        else:
            print(f"❌ Download failed")
            return None
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


def load_bisenet_model(checkpoint_path: str, num_classes: int = 19, device: str = "cuda"):
    """Load BiSeNet face parsing model from official repo."""
    print(f"📥 Loading BiSeNet model from {checkpoint_path}")
    
    # Import BiSeNet from cloned repo
    try:
        from model import BiSeNet
    except ImportError:
        print(f"❌ Could not import BiSeNet from face-parsing.PyTorch")
        print(f"   Make sure the repository is cloned and in the path")
        return None
    
    # Build model (matching your working code)
    print(f"   Building BiSeNet model (n_classes={num_classes})...")
    net = BiSeNet(n_classes=num_classes)
    
    # Load state dict (matching your working code)
    try:
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded BiSeNet checkpoint")
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        return None
    
    net.to(device)
    net.eval()
    print(f"✅ BiSeNet loaded successfully!")
    return net


def split_skin_regions(skin_mask: np.ndarray) -> dict:
    """
    Split face skin mask into 4 regions: forehead, left_cheek, right_cheek, chin.
    Matches your working code exactly, with chin added.
    """
    h, w = skin_mask.shape
    total_pixels = np.sum(skin_mask > 0)
    
    print(f"\n   Splitting skin mask into regions...")
    print(f"   Skin mask shape: {skin_mask.shape}, total skin pixels: {total_pixels}")
    
    if total_pixels == 0:
        print("   ❌ Skin mask is empty!")
        return {}
    
    # Match your working code: cv2.boundingRect(skin_mask) directly on mask
    x, y, ww, hh = cv2.boundingRect(skin_mask)
    
    print(f"   Face bounding box: x={x}, y={y}, w={ww}, h={hh}")
    print(f"   Face covers {ww*hh} pixels ({ww*hh/(h*w)*100:.1f}% of image)")
    
    # Validate bounding box
    if ww < 10 or hh < 10:
        print(f"   ❌ Face bounding box too small ({ww}x{hh})")
        return {}
    
    # Match your working code exactly: 32% for forehead, 78% for cheek bottom
    # Chin: 78% to bottom
    forehead_bottom = int(y + 0.32 * hh)
    cheek_top = forehead_bottom
    cheek_bottom = int(y + 0.78 * hh)
    chin_top = cheek_bottom
    chin_bottom = y + hh
    mid_x = x + ww // 2
    
    print(f"   Region boundaries (matching your working code):")
    print(f"      Forehead: y={y} to {forehead_bottom} (top 32%)")
    print(f"      Cheeks: y={cheek_top} to {cheek_bottom} (32% to 78%)")
    print(f"      Chin: y={chin_top} to {chin_bottom} (78% to bottom)")
    print(f"      Left/Right split at x={mid_x}")
    
    # Create region masks (matching your working code exactly)
    forehead = np.zeros_like(skin_mask)
    forehead[y:forehead_bottom, x:x+ww] = skin_mask[y:forehead_bottom, x:x+ww]
    forehead = cv2.bitwise_and(forehead, skin_mask)

    left_cheek = np.zeros_like(skin_mask)
    left_cheek[cheek_top:cheek_bottom, x:mid_x] = skin_mask[cheek_top:cheek_bottom, x:mid_x]
    left_cheek = cv2.bitwise_and(left_cheek, skin_mask)

    right_cheek = np.zeros_like(skin_mask)
    right_cheek[cheek_top:cheek_bottom, mid_x:x+ww] = skin_mask[cheek_top:cheek_bottom, mid_x:x+ww]
    right_cheek = cv2.bitwise_and(right_cheek, skin_mask)

    chin = np.zeros_like(skin_mask)
    chin[chin_top:chin_bottom, x:x+ww] = skin_mask[chin_top:chin_bottom, x:x+ww]
    chin = cv2.bitwise_and(chin, skin_mask)

    # Check pixel counts
    forehead_pixels = np.sum(forehead > 0)
    left_cheek_pixels = np.sum(left_cheek > 0)
    right_cheek_pixels = np.sum(right_cheek > 0)
    chin_pixels = np.sum(chin > 0)
    
    print(f"   Region pixel counts:")
    print(f"      Forehead: {forehead_pixels} pixels")
    print(f"      Left cheek: {left_cheek_pixels} pixels")
    print(f"      Right cheek: {right_cheek_pixels} pixels")
    print(f"      Chin: {chin_pixels} pixels")
    
    if forehead_pixels == 0 and left_cheek_pixels == 0 and right_cheek_pixels == 0 and chin_pixels == 0:
        print(f"   ❌ All regions are empty!")
        return {}

    return {
        "forehead": forehead.astype(bool),
        "left_cheek": left_cheek.astype(bool),
        "right_cheek": right_cheek.astype(bool),
        "chin": chin.astype(bool),
    }


def parse_face(image_path: str, bisenet_model_path: str = None, device: str = "cuda"):
    """Parse face into regions."""
    print("="*60)
    print("🔍 Face Parsing Test")
    print("="*60)
    
    # Step 1: Setup BiSeNet repository
    if not setup_bisenet_repo():
        return None
    
    # Step 2: Handle model path
    if bisenet_model_path is None or not os.path.exists(bisenet_model_path):
        # Check common locations
        common_paths = [
            os.path.join("face-parsing.PyTorch", "79999_iter.pth"),  # In repo
            "/home/vanessa/project/79999_iter.pth",  # Common server location
            "79999_iter.pth",  # Current directory
        ]
        
        found_path = None
        for path in common_paths:
            if os.path.exists(path):
                found_path = path
                print(f"✅ Found model at: {path}")
                break
        
        if found_path:
            bisenet_model_path = found_path
        else:
            # Try to download to repo directory
            repo_model_path = os.path.join("face-parsing.PyTorch", "79999_iter.pth")
            print(f"📥 Model not found in common locations, attempting to download...")
            downloaded_path = download_bisenet_model(repo_model_path)
            if downloaded_path:
                bisenet_model_path = downloaded_path
            else:
                print(f"❌ Could not find or download model file")
                print(f"   Please provide model path with --bisenet-model or ensure gdown is installed")
                return None
    else:
        print(f"✅ Using provided model: {bisenet_model_path}")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    print(f"\n📸 Image: {image_path}")
    print(f"   Size: {w}x{h}")
    
    # Load BiSeNet model
    bisenet_model = load_bisenet_model(bisenet_model_path, num_classes=19, device=device)
    if bisenet_model is None:
        return None
    
    # Prepare image for BiSeNet (matching your working code exactly)
    # Use ImageOps.exif_transpose to handle EXIF orientation
    try:
        from PIL import ImageOps
        image_pil = ImageOps.exif_transpose(image_pil)
    except:
        pass  # ImageOps might not be available, continue without it
    
    # Match your working code: Resize to 512x512, then normalize
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image_tensor = transform(image_pil)
    
    # Run BiSeNet (matching your working code exactly)
    print(f"\n🔍 Running BiSeNet segmentation...")
    with torch.no_grad():
        # Match your working code exactly: net(inp)[0]
        inp = image_tensor.unsqueeze(0).to(device)
        out = bisenet_model(inp)[0]
        
        # Match your working code: out.squeeze(0).cpu().numpy().argmax(0)
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    # Resize to original image size
    if parsing.shape != (h, w):
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    seg_pred = parsing
    
    # Show what classes were detected
    unique_classes = np.unique(seg_pred)
    print(f"\n📊 BiSeNet Segmentation Results:")
    print(f"   Unique class IDs: {unique_classes.tolist()}")
    print(f"   Segmentation shape: {seg_pred.shape}")
    
    for cls_id in unique_classes:
        count = np.sum(seg_pred == cls_id)
        percentage = (count / seg_pred.size) * 100
        print(f"      Class {cls_id}: {count} pixels ({percentage:.1f}%)")
    
    # Extract skin mask (class 1) - matching your working code exactly
    # Your code: skin_mask = (parsing == 1).astype(np.uint8)*255
    skin_mask = (parsing == 1).astype(np.uint8) * 255
    skin_pixels = np.sum(skin_mask > 0)
    
    print(f"\n🎭 Face Skin Detection:")
    print(f"   Skin (class 1): {skin_pixels} pixels ({skin_pixels/parsing.size*100:.1f}% of image)")
    
    if skin_pixels == 0:
        print(f"   ⚠️  WARNING: Class 1 (skin) not found!")
        print(f"   Available classes: {unique_classes.tolist()}")
        print(f"   Your model might use a different class ID for skin.")
        print(f"   Please check which class corresponds to face skin in your model.")
        return None
    
    # Filter to face region (remove wall/background if detected as class 1)
    print(f"\n🔍 Filtering to face region (removing background/wall)...")
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        print(f"   Found {len(contours)} contour(s)")
        
        # Sort contours by area (largest first)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check each contour to find the face (not the wall)
        # Face should be reasonable size (5-50% of image) and not too large
        image_area = h * w
        face_contour = None
        
        for idx, contour in enumerate(contours_sorted):
            area = cv2.contourArea(contour)
            percentage = (area / image_area) * 100
            x, y, ww, hh = cv2.boundingRect(contour)
            
            print(f"   Contour {idx+1}: {area:.0f} pixels ({percentage:.1f}% of image), bbox: {ww}x{hh}")
            
            # Face criteria: reasonable size (5-50% of image) and reasonable aspect ratio
            if 5.0 <= percentage <= 50.0:
                aspect_ratio = ww / hh if hh > 0 else 0
                # Face aspect ratio is usually between 0.6 and 1.5
                if 0.6 <= aspect_ratio <= 1.5:
                    face_contour = contour
                    print(f"      ✅ This looks like a face (size {percentage:.1f}%, aspect {aspect_ratio:.2f})")
                    break
        
        # If no contour matches criteria, use largest that's not too big
        if face_contour is None:
            largest_contour = contours_sorted[0]
            largest_area = cv2.contourArea(largest_contour)
            largest_percentage = (largest_area / image_area) * 100
            
            if largest_percentage > 50.0:
                # Largest is too big (probably wall), try second largest
                if len(contours_sorted) > 1:
                    second_contour = contours_sorted[1]
                    second_area = cv2.contourArea(second_contour)
                    second_percentage = (second_area / image_area) * 100
                    print(f"   Largest contour too big ({largest_percentage:.1f}%), trying second largest ({second_percentage:.1f}%)")
                    if second_percentage <= 50.0:
                        face_contour = second_contour
                    else:
                        face_contour = largest_contour  # Use largest anyway
                else:
                    face_contour = largest_contour
            else:
                face_contour = largest_contour
        
        # Create mask with the selected contour (the face)
        skin_mask_filtered = np.zeros_like(skin_mask)
        cv2.fillPoly(skin_mask_filtered, [face_contour], 255)
        
        filtered_pixels = np.sum(skin_mask_filtered > 0)
        print(f"   ✅ Using face contour: {filtered_pixels} pixels")
        skin_mask = skin_mask_filtered
    else:
        print(f"   ⚠️  No contours found, using full skin mask")
    
    # Split into regions
    region_masks = split_skin_regions(skin_mask)
    
    if not region_masks:
        print(f"\n❌ Failed to create regions!")
        return None
    
    print(f"\n✅ Successfully created {len(region_masks)} regions: {list(region_masks.keys())}")
    
    # Visualize results
    print(f"\n📊 Creating visualization...")
    visualize_results(image_np, seg_pred, skin_mask, region_masks)
    
    return {
        "image": image_np,
        "segmentation": seg_pred,
        "skin_mask": skin_mask,
        "regions": region_masks,
    }


def visualize_results(image_np, seg_pred, skin_mask, region_masks):
    """Create visualization of face parsing results."""
    # Layout: 2 rows, 3 columns (top row: original, segmentation, skin mask)
    # Bottom row: combined overlay + 2 individual regions
    # Then add 2 more individual regions below
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_np)
    ax.set_title("Original Image", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Full segmentation (colored)
    seg_colored = plt.cm.tab20(seg_pred / seg_pred.max())
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(seg_colored)
    ax.set_title("BiSeNet Segmentation (All Classes)", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Skin mask
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(skin_mask, cmap='gray')
    ax.set_title(f"Face Skin Mask\n({np.sum(skin_mask > 0)} pixels)", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Region overlays
    colors = {
        "forehead": (255, 0, 0),      # Red
        "left_cheek": (0, 255, 0),    # Green
        "right_cheek": (0, 0, 255),   # Blue
        "chin": (255, 0, 255),        # Magenta
    }
    
    # Combined regions
    combined = image_np.copy()
    for region_name, mask in region_masks.items():
        color = colors.get(region_name, (128, 128, 128))
        combined[mask] = combined[mask] * 0.6 + np.array(color) * 0.4
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(combined)
    ax.set_title("All Regions Overlay", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Individual regions (all 4)
    region_list = list(region_masks.items())
    positions = [(1, 1), (1, 2), (2, 0), (2, 1)]  # Positions for 4 regions
    
    for idx, (region_name, mask) in enumerate(region_list):
        if idx < len(positions):
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            overlay = image_np.copy()
            color = colors.get(region_name, (128, 128, 128))
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
            ax.imshow(overlay)
            pixel_count = np.sum(mask)
            ax.set_title(f"{region_name}\n({pixel_count} pixels)", fontsize=14, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle("Face Parsing Results - 4 Regions", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = "face_parsing_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved visualization to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    import sys
    
    # Default paths - UPDATE THESE
    DEFAULT_IMAGE = "/home/vanessa/project/levle3_113 copy.jpg"
    DEFAULT_MODEL = None  # Will download to face-parsing.PyTorch/ if not provided
    
    # Check if running in IPython/Jupyter
    try:
        get_ipython()
        in_jupyter = True
    except NameError:
        in_jupyter = False
    
    # Check if running with command line args or using defaults
    if not in_jupyter and len(sys.argv) > 1 and not any('--f=' in arg or 'kernel' in arg for arg in sys.argv):
        # Command line mode (not in Jupyter)
        parser = argparse.ArgumentParser(description="Test face parsing only")
        parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input image")
        parser.add_argument("--bisenet-model", default=None,
                           help="Path to BiSeNet model checkpoint (optional, will download if not provided)")
        parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
        args = parser.parse_args()
        image_path = args.image
        model_path = args.bisenet_model  # None means will download
        device = args.device
    else:
        # Simple mode - use defaults (works in Jupyter or when run without args)
        print("🚀 Starting Face Parsing Test (using default paths)")
        print(f"   Image: {DEFAULT_IMAGE}")
        print(f"   Model: Will download to face-parsing.PyTorch/ if not found")
        if not in_jupyter:
            print("   (To use different paths, run with --image and --bisenet-model arguments)")
        print()
        image_path = DEFAULT_IMAGE
        model_path = DEFAULT_MODEL  # None means will download
        device = "cuda"
    
    # Run face parsing
    result = parse_face(image_path, model_path, device)
    
    if result:
        print("\n" + "="*60)
        print("✅ Face parsing completed successfully!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Regions created: {list(result['regions'].keys())}")
        for region_name, mask in result['regions'].items():
            print(f"      {region_name}: {np.sum(mask)} pixels")
        print(f"\n   ✅ Check 'face_parsing_result.png' to see the visualization.")
    else:
        print("\n" + "="*60)
        print("❌ Face parsing failed!")
        print("="*60)

