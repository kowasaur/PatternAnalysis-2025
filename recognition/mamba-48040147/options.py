import random
import torch
from os import path as osp
from basicsr.utils import set_random_seed

PRETRAINED_MODEL_PATH = "mambairv2_classicSR_Small_x2.pth"
INITIAL_MODEL_PATH = "mambairv2_Colouriser.pth"
FINAL_MODEL_PATH = INITIAL_MODEL_PATH
DATASET_PATH = "./images/train"

# Model
UPSCALE = 1
IN_CHANS = 1
OUT_CHANS = 2
IMG_SIZE = 128
IMG_RANGE = 1
EMBED_DIM = 132
D_STATE = 16
DEPTHS = [4, 4, 4, 4, 4, 4]
NUM_HEADS = [4, 4, 4, 4, 4, 4]
WINDOW_SIZE = 16
INNER_RANK = 64
NUM_TOKENS = 128
CONVFFN_KERNEL_SIZE = 5
MLP_RATIO = 2

# Training
NUM_GPU = 1
BATCH_SIZE = 3
UNFREEZE_ITER = 60000
CROP_SIZE = 128
DATASET_ENLARGE_RATIO = 100

# Modified from https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_ColorDN_level25.yml and https://github.com/csguoh/MambaIR/blob/main/options/train/mambairv2/train_MambaIRv2_SRSmall_x2.yml
OPTIONS = {
    "name": "MambaIRv2_Colouriser",
    "model_type": "MambaIRv2Model",
    "scale": 1,
    "num_gpu": NUM_GPU,
    "manual_seed": 10,
    "datasets": {
        "train": {
            "task": "denoising_color",
            "name": "DIV2K",
            "type": "PairedImageDataset",
            "dataroot_gt": DATASET_PATH,
            "dataroot_lq": DATASET_PATH,
            "filename_tmpl": "{}",
            "io_backend": {"type": "disk"},
            "gt_size": CROP_SIZE,
            "use_hflip": True,
            "use_rot": True,
            "use_shuffle": True,
            "num_worker_per_gpu": 1,
            "batch_size_per_gpu": BATCH_SIZE,
            "dataset_enlarge_ratio": DATASET_ENLARGE_RATIO,
            "prefetch_mode": None,
        },
    },
    "network_g": {
        "type": "MambaIRv2",
        "upscale": UPSCALE,
        "in_chans": IN_CHANS,
        "out_chans": OUT_CHANS,
        "img_size": IMG_SIZE,
        "img_range": IMG_RANGE,
        "embed_dim": EMBED_DIM,
        "d_state": D_STATE,
        "depths": DEPTHS,
        "num_heads": NUM_HEADS,
        "window_size": WINDOW_SIZE,
        "inner_rank": INNER_RANK,
        "num_tokens": NUM_TOKENS,
        "convffn_kernel_size": CONVFFN_KERNEL_SIZE,
        "mlp_ratio": MLP_RATIO,
    },
    "path": {
        "pretrain_network_g": INITIAL_MODEL_PATH,
        "strict_load_g": False,
        "resume_state": None,
    },
    "train": {
        "optim_g": {
            "type": "Adam",
            "lr": 2e-4,
            "weight_decay": 0,
            "betas": [0.9, 0.99],
        },
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": [50000, 100000, 160000, 180000, 190000],
            "gamma": 0.5,
        },
        "total_iter": 200000,
        "warmup_iter": -1,
        "pixel_opt": {
            "type": "CharbonnierLoss",
            "loss_weight": 1.0,
            "reduction": "mean",
            "eps": 1e-3,
        },
    },
    "logger": {
        "print_freq": 500,
        "save_checkpoint_freq": 1e4,
        "use_tb_logger": True,
        "wandb": {"project": None, "resume_id": None},
    },
    "dist_params": {"backend": "nccl", "port": 29500},
    "dist": False,
    "rank": 0,
    "world_size": 1,
    "is_train": True,
    "auto_resume": False,
    "debug": False,
}


# Modified from https://github.com/csguoh/MambaIR/blob/main/basicsr/utils/options.py
def parse_options(root_path) -> dict:
    opt = OPTIONS

    # random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    if opt["num_gpu"] == "auto":
        opt["num_gpu"] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt["datasets"].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        if "scale" in opt:
            dataset["scale"] = opt["scale"]

    # paths
    for key, val in opt["path"].items():
        if (val is not None) and ("resume_state" in key or "pretrain_network" in key):
            opt["path"][key] = osp.expanduser(val)

    if opt["is_train"]:
        experiments_root = osp.join(root_path, "experiments", opt["name"])
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_states"] = osp.join(experiments_root, "training_states")
        opt["path"]["log"] = experiments_root
        opt["path"]["visualization"] = osp.join(experiments_root, "visualization")

        # change some options for debug mode
        if "debug" in opt["name"]:
            if "val" in opt:
                opt["val"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(root_path, "results", opt["name"])
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root
        opt["path"]["visualization"] = osp.join(results_root, "visualization")

    return opt
