import os
import torch
from configs import cfg
import argparse
from datasets import make_dataloader
from models import make_model
from processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 使用cuda或cpu
    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")

    # loaders: dict of {'train', 'query_diff', 'query_same', ...} -> (loader, num_query, data)
    loaders, bundle = make_dataloader(cfg)

    # only the person-id classifier depends on the training split size;
    # F' (cloth branch) is unused at test time, but the head must exist for the
    # state dict to match, so build it with the same width as during training
    num_cloth_classes = bundle.num_train_clothes if cfg.MODEL.DUAL_BRANCH else 0
    model = make_model(cfg, num_classes=bundle.num_train_pids,
                       num_cloth_classes=num_cloth_classes)
    model.load_parameter(cfg.TEST.WEIGHT)  # 修复: load_parameter 而不是 load_param
    model.to(device)

    do_inference(cfg,
                 model,
                 loaders,
                 bundle,
                 device  # 添加device参数
            )
