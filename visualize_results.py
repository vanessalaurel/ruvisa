import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_comparison(image_ids, base_folder, output_path, model_name="UNet"):
    """Create a grid comparing: original face | masked face | wrinkle mask (UNet) | overlay"""
    results_dir = "test_outputs" if model_name == "UNet" else "test_outputs_swinunetr"

    cell_size = 512
    cols = 4
    rows = len(image_ids)
    padding = 10
    header_h = 40

    headers = ["Original Face", "Masked Face", f"Wrinkle Mask ({model_name})", "Overlay"]
    total_w = cols * cell_size + (cols + 1) * padding
    total_h = rows * cell_size + (rows + 1) * padding + header_h

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        font = ImageFont.load_default()

    for col_idx, header in enumerate(headers):
        x = padding + col_idx * (cell_size + padding) + cell_size // 2
        draw.text((x, 8), header, fill=(220, 220, 220), font=font, anchor="mt")

    for row_idx, img_id in enumerate(image_ids):
        y = header_h + padding + row_idx * (cell_size + padding)

        face_path = os.path.join(base_folder, "face_images", f"{img_id}.png")
        masked_path = os.path.join(base_folder, "masked_face_images", f"{img_id}.png")
        mask_path = os.path.join(base_folder, results_dir, f"{img_id}_mask.png")

        for col_idx, path in enumerate([face_path, masked_path, mask_path]):
            x = padding + col_idx * (cell_size + padding)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.text((x + cell_size // 2, y + cell_size // 2), "N/A",
                          fill=(128, 128, 128), font=font, anchor="mm")

        if os.path.exists(face_path) and os.path.exists(mask_path):
            x = padding + 3 * (cell_size + padding)
            face = Image.open(face_path).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
            mask = Image.open(mask_path).convert("L").resize((cell_size, cell_size), Image.NEAREST)
            face_arr = np.array(face)
            mask_arr = np.array(mask)

            overlay = face_arr.copy()
            wrinkle_pixels = mask_arr > 0
            overlay[wrinkle_pixels, 0] = np.minimum(overlay[wrinkle_pixels, 0].astype(int) + 150, 255).astype(np.uint8)
            overlay[wrinkle_pixels, 1] = (overlay[wrinkle_pixels, 1] * 0.4).astype(np.uint8)
            overlay[wrinkle_pixels, 2] = (overlay[wrinkle_pixels, 2] * 0.4).astype(np.uint8)

            canvas.paste(Image.fromarray(overlay), (x, y))

    canvas.save(output_path, quality=95)
    print(f"Saved comparison to {output_path}")
    return canvas


if __name__ == "__main__":
    base = "./ffhq_wrinkle_data"
    sample_ids = ["00001", "00048", "00125", "01044", "03140"]

    create_comparison(sample_ids, base, "./ffhq_wrinkle_data/comparison_unet.png", "UNet")

    swinunetr_dir = os.path.join(base, "test_outputs_swinunetr")
    if os.path.exists(swinunetr_dir) and len(os.listdir(swinunetr_dir)) > 0:
        create_comparison(sample_ids, base, "./ffhq_wrinkle_data/comparison_swinunetr.png", "SwinUNETR")

        create_comparison(sample_ids, base, "./ffhq_wrinkle_data/comparison_both.png", "UNet")
