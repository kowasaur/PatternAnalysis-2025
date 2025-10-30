"""
This file was modified from https://github.com/csguoh/MambaIR/blob/main/basicsr/train.py
"""

import datetime
import logging
import time
import torch
from os import path as osp
from torch.nn import functional as F
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.utils import (
    AvgTimer,
    MessageLogger,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    make_exp_dirs,
    mkdir_and_rename,
)
from basicsr.utils.options import dict2str
from basicsr.models import lr_scheduler as lr_scheduler
from options import (
    INITIAL_MODEL_PATH,
    UNFREEZE_ITER,
    parse_options,
    PRETRAINED_MODEL_PATH,
)
from predict import predict_test_folder
from dataset import create_train_val_dataloader
from modules import MambaIRv2


def transform_pretrained_model_state():
    """Transform the pretrained model state dictionary to match the current
    model architecture."""
    state_dict = torch.load(PRETRAINED_MODEL_PATH)
    params = state_dict["params"]

    # Average the rgb weights to create a single-channel weight
    # Since lightness is sort of mean of rgb
    params["conv_first.weight"] = params["conv_first.weight"].mean(dim=1, keepdim=True)

    # Remove the last convolutional layer weights
    # These will be randomly initialized in the new model
    del params["conv_last.weight"]
    del params["conv_last.bias"]

    # Remove unused upsampling layers from super resolution model
    UNUSED = [
        "conv_before_upsample.0.weight",
        "conv_before_upsample.0.bias",
        "upsample.0.weight",
        "upsample.0.bias",
    ]
    for key in UNUSED:
        del params[key]

    torch.save(state_dict, INITIAL_MODEL_PATH)


def set_all_requires_grad(model: MambaIRv2, requires_grad: bool):
    """Set requires_grad for all the parameters in the model."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_except_last_and_first(model: MambaIRv2):
    """Freeze all layers except the first and last convolutional layers."""
    set_all_requires_grad(model, False)
    for param in model.conv_first.parameters():
        param.requires_grad = True
    for param in model.conv_last.parameters():
        param.requires_grad = True


def wrong_colour_loss(pred, target, delta=0.08, k=30):
    """

    pred and target are shaped [B, 2, H, W], in ab space scaled to [-1, 1]
    """
    # euclidean distance in ab space is perceived colour difference
    diff = torch.norm(pred - target, dim=1)  # [B, H, W]

    # differentiable approximation of step function
    soft = torch.sigmoid(k * (diff - delta))  # [B, H, W]

    return soft.mean()


def loss(pred, target, wrong_colour_weight=0.2):
    """Returns the overall loss and breakdown of the loss components."""
    total = 0
    l1 = F.l1_loss(pred, target)
    total += l1
    wc = wrong_colour_loss(pred, target)
    total += wrong_colour_weight * wc
    return total, {"l1": l1.item(), "wc": wc.item(), "total": total.item()}


def save_and_validate(model, current_iter, opt, val_loader, epoch, optim, msg_logger):
    """Save the model, calculate validation loss and log the results."""
    save_path = osp.join(opt["path"]["models"], f"mamba_colouriser_{current_iter}.pth")
    torch.save(model.state_dict(), save_path)

    losses = {}
    model.eval()
    with torch.no_grad():
        for l, ab in val_loader:
            l, ab = l.cuda(), ab.cuda()
            output = model(l)
            _, loss_dict = loss(output, ab)
            for key, value in loss_dict.items():
                if key not in losses:
                    losses[key] = 0
                losses[key] += value
    model.train()

    num_batches = len(val_loader)
    val_losses = {"val_" + k: v / num_batches for k, v in losses.items()}

    log_vars = {"epoch": epoch, "iter": current_iter}
    log_vars.update({"lrs": [param_group["lr"] for param_group in optim.param_groups]})
    log_vars.update(val_losses)
    msg_logger(log_vars)


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path)
    opt["root_path"] = root_path

    # Should improve performance
    torch.backends.cudnn.benchmark = True

    # mkdir for experiments and logger
    make_exp_dirs(opt)
    mkdir_and_rename(osp.join(opt["root_path"], "tb_logger", opt["name"]))

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize tb logger
    tb_logger = init_tb_logger(
        log_dir=osp.join(opt["root_path"], "tb_logger", opt["name"])
    )

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # must be called before building the model
    transform_pretrained_model_state()

    # create model
    model = MambaIRv2().cuda()
    model.load_state_dict(torch.load(INITIAL_MODEL_PATH)["params"], strict=False)
    model.train()

    # freeze all layers except the first and last conv layers
    freeze_except_last_and_first(model)

    # Learning
    optim = torch.optim.Adam(model.parameters(), **opt["train"]["optim_g"])
    scheduler = lr_scheduler.MultiStepRestartLR(optim, **opt["train"]["scheduler"])

    start_epoch = 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetcher = CPUPrefetcher(train_loader)

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            # training
            l, ab = train_data[0].cuda(), train_data[1].cuda()
            optim.zero_grad()
            output = model(l)
            total_loss, loss_dict = loss(output, ab)
            total_loss.backward()
            optim.step()

            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update(
                    {"lrs": [param_group["lr"] for param_group in optim.param_groups]}
                )
                log_vars.update(
                    {
                        "time": iter_timer.get_avg_time(),
                        "data_time": data_timer.get_avg_time(),
                    }
                )
                log_vars.update(loss_dict)
                msg_logger(log_vars)

            # save model and validate
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving model.")
                save_and_validate(
                    model, current_iter, opt, val_loader, epoch, optim, msg_logger
                )

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

            # Unfreeze everything
            if current_iter == UNFREEZE_ITER:
                set_all_requires_grad(model, True)

            # update learning rate
            scheduler.step()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Saving the latest model.")
    save_and_validate(model, current_iter, opt, val_loader, epoch, optim, msg_logger)

    if tb_logger:
        tb_logger.close()

    return model


if __name__ == "__main__":
    current_path = osp.dirname(osp.abspath(__file__))
    model = train_pipeline(current_path)
    predict_test_folder(model)
