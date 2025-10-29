"""
This file was modified from https://github.com/csguoh/MambaIR/blob/main/basicsr/train.py
"""

import datetime
import logging
import time
import torch
from os import path as osp
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.models import build_model
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
from options import (
    INITIAL_MODEL_PATH,
    UNFREEZE_ITER,
    parse_options,
    PRETRAINED_MODEL_PATH,
)
from predict import predict_test_folder
from dataset import create_train_val_dataloader
import modules


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


def set_all_requires_grad(model: modules.MambaIRv2, requires_grad: bool):
    """Set requires_grad for all the parameters in the model."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_except_last_and_first(model: modules.MambaIRv2):
    """Freeze all layers except the first and last convolutional layers."""
    set_all_requires_grad(model, False)
    for param in model.conv_first.parameters():
        param.requires_grad = True
    for param in model.conv_last.parameters():
        param.requires_grad = True


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
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # must be called before build_model()
    transform_pretrained_model_state()

    # create model
    model = build_model(opt)

    # freeze all layers except the first and last conv layers
    freeze_except_last_and_first(model.net_g)

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

            # Unfreeze everything
            if current_iter == UNFREEZE_ITER:
                set_all_requires_grad(model.net_g, True)

            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update(
                    {
                        "time": iter_timer.get_avg_time(),
                        "data_time": data_timer.get_avg_time(),
                    }
                )
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving model.")
                model.save_network(model.net_g, "net_g", current_iter)

            # validation
            if opt.get("val") is not None and (
                current_iter % opt["val"]["val_freq"] == 0
            ):
                if len(val_loaders) > 1:
                    logger.warning(
                        "Multiple validation datasets are *only* supported by SRModel."
                    )
                for val_loader in val_loaders:
                    model.validation(
                        val_loader, current_iter, tb_logger, opt["val"]["save_img"]
                    )

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get("val") is not None:
        for val_loader in val_loaders:
            model.validation(
                val_loader, current_iter, tb_logger, opt["val"]["save_img"]
            )
    if tb_logger:
        tb_logger.close()

    return model.net_g


if __name__ == "__main__":
    current_path = osp.dirname(osp.abspath(__file__))
    model = train_pipeline(current_path)
    predict_test_folder(model)
