"""
Run inference using the trained model.

The trained model weights must be at ./mambairv2_Colouriser_Final.pth

Usage: python predict.py [image_path/image_name.jpg]

If an image path is provided, the script will run inference on that image and
save it to images/output/image_name.jpg

If no argument is provided, the script will run inference on all images in
images/test/ and save all results to images/output/
"""

import sys
import os
import cv2
import torch
import numpy as np
from basicsr.utils import imwrite
from modules import MambaIRv2
from options import FINAL_MODEL_PATH
from dataset import read_image_lab, lab_split_tensor, AB_SCALE

TEST_DIR = "images/test/"
OUTPUT_DIR = "images/output/"

MAX_PIXELS = 1600 * 1200  # Rangpur runs out of memory on larger images


def predict_image(img_path: str, model: MambaIRv2):
    """Run inference on a single image and save the output."""
    # Read image in LAB colour space
    img = read_image_lab(img_path)

    # Resize if too large
    h, w, _ = img.shape
    if h * w > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (h * w)) ** 0.5
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized {img_path} from ({w},{h}) to ({new_w},{new_h})")

    # l_img is a tensor
    l_img, _, L = lab_split_tensor(img)
    l_tensor = l_img.unsqueeze(0).cuda()  # batch size 1 basically

    # Inference
    with torch.no_grad():
        ab_tensor = model(l_tensor)

    # Convert tensor to numpy
    ab_np = ab_tensor.squeeze(0).cpu().numpy()  # [2,H,W]

    # Stack L + ab: combine grayscale and colour
    lab = np.stack([L, ab_np[0] * AB_SCALE, ab_np[1] * AB_SCALE], axis=2)  # [H,W,3]

    # Convert Lab to BGR
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    bgr_img = (bgr * 255.0).clip(0, 255).astype(np.uint8)

    # Save output image
    img_name = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, img_name)
    imwrite(bgr_img, save_path)


def predict_test_folder(model: MambaIRv2):
    """Run inference on all images in the test directory"""
    for img_name in os.listdir(TEST_DIR):
        print(f"Processing {img_name}...")
        img_path = os.path.join(TEST_DIR, img_name)
        predict_image(img_path, model)
        torch.cuda.empty_cache()  # free up memory


if __name__ == "__main__":
    model = MambaIRv2().cuda()
    model.load_state_dict(torch.load(FINAL_MODEL_PATH), strict=True)
    model.eval()

    if len(sys.argv) > 1:
        # Predict single image
        img_path = sys.argv[1]
        predict_image(img_path, model)
    else:
        predict_test_folder(model)
