import sys
import cv2
import torch
from basicsr.utils import img2tensor, tensor2img, imwrite
from modules import MambaIRv2
from options import FINAL_MODEL_PATH

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    model = MambaIRv2()
    model.load_state_dict(torch.load(FINAL_MODEL_PATH)["params"], strict=True)

    img_path = sys.argv[1]

    img = cv2.imread(img_path) / 255.0
    img = img2tensor(img, bgr2rgb=True, float32=True)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    output = tensor2img([output.detach().cpu()])
    save_path = "output.png"
    imwrite(output, save_path)
