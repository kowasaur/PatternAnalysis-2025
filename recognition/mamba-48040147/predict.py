import sys
import os
import cv2
import torch
import numpy as np
from basicsr.utils import img2tensor, imwrite
from modules import MambaIRv2
from options import FINAL_MODEL_PATH

TEST_DIR = "images/test/"
OUTPUT_DIR = "images/output/"

MAX_PIXELS = 1600 * 1200  # Rangpur runs out of memory on larger images


def predict_image(img_path: str, model: MambaIRv2):
    """Run inference on a single image and save the output."""
    # Make image grayscale
    y_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize if too large
    h, w = y_img.shape
    if h * w > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (h * w)) ** 0.5
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        y_img = cv2.resize(y_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized {img_path} from ({w},{h}) to ({new_w},{new_h})")

    # Convert into tensor
    y_img = y_img / 255.0
    y_tensor = np.expand_dims(y_img, axis=2)  # img2tensor needs 3 dimensions
    y_tensor = img2tensor(y_tensor, float32=True)
    y_tensor = y_tensor.unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        crcb_tensor = model(y_tensor)

    # Convert tensor to numpy
    crcb_np = crcb_tensor.squeeze(0).cpu().numpy()  # [2,H,W]

    # Stack Y + CbCr: combine grayscale and colour
    ycrcb = np.stack([y_img, crcb_np[0], crcb_np[1]], axis=2)  # [H,W,3]

    # Convert YCbCr -> BGR for OpenCV
    ycrcb = (ycrcb * 255.0).clip(0, 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Make output dir if doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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


if __name__ == "__main__":
    model = MambaIRv2().cuda()
    model.load_state_dict(torch.load(FINAL_MODEL_PATH)["params"], strict=True)
    model.eval()

    if len(sys.argv) > 1:
        # Predict single image
        img_path = sys.argv[1]
        predict_image(img_path, model)
    else:
        predict_test_folder(model)
