from utils.logger import setup_logger
import logging
from datasets import make_dataloader
from models import make_model
from solver import build_optimizer, build_lr_scheduler
from losses import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from configs import cfg
import os.path as osp

# Monkey patch for yacs library to fix Windows encoding issue
import yacs.config
_original_merge_from_file = yacs.config.CfgNode.merge_from_file

def _patched_merge_from_file(self, cfg_filename):
    with open(cfg_filename, 'r', encoding='utf-8') as f:
        cfg = self.load_cfg(f)
    self.merge_from_other_cfg(cfg)

yacs.config.CfgNode.merge_from_file = _patched_merge_from_file

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reid baseline training")
    parser.add_argument(
        "--config_file",default="",help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    log_dir = osp.join(cfg.LOG_DIR, cfg.EXP_NAME)
    output_dir = osp.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid", log_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r', encoding='utf-8') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == 'cuda' and not torch.cuda.is_available():
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        logging.warning("No CUDA available, falling back to CPU")
        cfg.MODEL.DEVICE = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE
    loaders, bundle = make_dataloader(cfg)
    logger.info("Loaded splits: {}".format(bundle))

    # F' is a clothing classifier, so it needs the training clothing vocabulary
    num_cloth_classes = bundle.num_train_clothes if cfg.MODEL.DUAL_BRANCH else 0
    model = make_model(cfg, num_classes=bundle.num_train_pids,
                       num_cloth_classes=num_cloth_classes)
    logger.info("Model heads: {} id classes, {} cloth classes (dual_branch={})".format(
        bundle.num_train_pids, num_cloth_classes, cfg.MODEL.DUAL_BRANCH))

    device = torch.device(cfg.MODEL.DEVICE)
    if cfg.MODEL.DEVICE == 'cuda':
        model = model.to(device)

    loss_func = make_loss(cfg, num_classes=bundle.num_train_pids,
                          num_cloth_classes=num_cloth_classes)

    optimizer = build_optimizer(
        model,
        optim=cfg.SOLVER.OPTIMIZER_NAME.lower() if cfg.SOLVER.OPTIMIZER_NAME.lower() in ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam'] else 'adam',
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        momentum=cfg.SOLVER.MOMENTUM
    )

    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler=cfg.SOLVER.LR_SCHEDULER,
        stepsize=cfg.SOLVER.STEPSIZE,
        gamma=cfg.SOLVER.GAMMA,
        max_epoch=cfg.SOLVER.MAX_EPOCHS,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_type=cfg.SOLVER.WARMUP_TYPE
    )

    do_train(
        cfg,
        model,
        loaders,
        bundle,
        optimizer,
        scheduler,
        loss_func,
        device=device
    )    