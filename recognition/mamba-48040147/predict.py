import sys
import cv2
import torch
import numpy as np
from basicsr.utils import img2tensor, imwrite
from modules import MambaIRv2
from options import FINAL_MODEL_PATH

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    model = MambaIRv2().cuda()
    # TODO: change to strict=True because should be using trained model
    model.load_state_dict(torch.load(FINAL_MODEL_PATH)["params"], strict=False)
    model.eval()

    img_path = sys.argv[1]

    # Make image grayscale and right format
    y_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
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

    save_path = "output.png"
    imwrite(bgr_img, save_path)
