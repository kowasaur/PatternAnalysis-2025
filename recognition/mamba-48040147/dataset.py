"""
This file was modified from https://github.com/csguoh/MambaIR/blob/main/basicsr/data/paired_image_dataset.py
"""

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
import cv2
import torch


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
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.task = opt["task"] if "task" in opt else None
        self.noise = opt["noise"] if "noise" in opt else 0

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

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        gt_path = self.paths[index]["gt_path"]
        lq_path = gt_path
        img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        img_gt = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
        img_lq = img_gt

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )

        # Split channels
        Y, Cr, Cb = cv2.split(img_gt)
        img_lq = torch.from_numpy(Y).unsqueeze(0)  # shape (1, H, W)
        img_gt = torch.from_numpy(np.stack([Cr, Cb], axis=0))  # shape (2, H, W)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)
