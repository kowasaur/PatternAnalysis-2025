"""
This file was modified from https://github.com/csguoh/MambaIR/blob/main/basicsr/data/paired_image_dataset.py
"""

from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
import cv2
import torch

# https://docs.opencv.org/4.9.0/de/d25/imgproc_color_conversions.html
# Docs say L: 0~100, a: -127~127, b: -127~127
L_SCALE = 100
AB_SCALE = 127


def read_image_lab(imgpath: str):
    """Read an image aand convert it to Lab color space."""
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)


def lab_split_tensor(img_lab):
    """Split a Lab image and convert to tensors.

    Returns img_l (1, H, W), img_ab (2, H, W), L channel numpy array (H, W)
    """
    L, a, b = cv2.split(img_lab)

    # normalize to [0, 1]
    a /= AB_SCALE
    b /= AB_SCALE

    img_l = torch.from_numpy(L / L_SCALE).unsqueeze(0)  # shape (1, H, W)
    img_ab = torch.from_numpy(np.stack([a, b], axis=0))  # shape (2, H, W)
    return img_l, img_ab, L


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.task = opt["task"] if "task" in opt else None

        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder],
            ["lq", "gt"],
            self.filename_tmpl,
            self.task,
        )

    def __getitem__(self, index):
        scale = self.opt["scale"]
        path = self.paths[index]["gt_path"]

        img = read_image_lab(path)

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # random crop. Since paired, need to pass image twice
            img, _ = paired_random_crop(img, img, gt_size, scale, path)
            # flip, rotation
            img = augment(img, self.opt["use_hflip"], self.opt["use_rot"])

        # Split channels and convert to tensor
        img_l, img_ab, _ = lab_split_tensor(img)

        return {"lq": img_l, "gt": img_ab, "lq_path": path, "gt_path": path}

    def __len__(self):
        return len(self.paths)
