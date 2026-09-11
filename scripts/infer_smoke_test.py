"""Inference-path smoke test: load a checkpoint and run do_inference.

Reuses the synthetic tree produced by ``train_smoke_test.py`` (kept on disk
under ``_smoke_data``) and the checkpoint under ``_smoke_run``.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import yacs.config  # noqa: E402


def _patched_merge_from_file(self, cfg_filename):
    with open(cfg_filename, 'r', encoding='utf-8') as f:
        cfg = self.load_cfg(f)
    self.merge_from_other_cfg(cfg)


yacs.config.CfgNode.merge_from_file = _patched_merge_from_file

MOCK_ROOT = os.path.join(ROOT, '_smoke_data')
OUT_DIR = os.path.join(ROOT, '_smoke_run', 'outputs')
LOG_DIR = os.path.join(ROOT, '_smoke_run', 'logs')


def main():
    from configs import cfg
    from datasets import make_dataloader
    from models import make_model
    from processor import do_inference
    from utils.logger import setup_logger

    weight = os.path.join(OUT_DIR, 'smoke', 'resnet_best.pth')
    if not os.path.isfile(weight):
        print('missing checkpoint {}, run train_smoke_test.py first'.format(weight))
        return 1

    c = cfg.clone()
    c.merge_from_list([
        'MODEL.DEVICE', 'cpu',
        'MODEL.DUAL_BRANCH', True,
        'DATASETS.NAMES', 'prcc',
        'DATASETS.ROOT_DIR', MOCK_ROOT,
        'DATALOADER.NUM_WORKERS', 0,
        'TEST.IMS_PER_BATCH', 8,
        'TEST.MODE', 'both',
        'TEST.WEIGHT', weight,
        'LOG_DIR', LOG_DIR,
        'OUTPUT_DIR', OUT_DIR,
        'EXP_NAME', 'smoke_infer',
    ])
    c.freeze()

    logger = setup_logger("reid", os.path.join(LOG_DIR, 'smoke_infer'), if_train=False)
    loaders, bundle = make_dataloader(c)

    num_cloth_classes = bundle.num_train_clothes if c.MODEL.DUAL_BRANCH else 0
    model = make_model(c, num_classes=bundle.num_train_pids,
                       num_cloth_classes=num_cloth_classes)
    # exactly what test.py does
    model.load_parameter(c.TEST.WEIGHT)
    model.to(torch.device('cpu'))

    rank1, rank5 = do_inference(c, model, loaders, bundle, torch.device('cpu'))
    print('\n=== INFERENCE RESULT ===')
    print('rank1={:.4f} rank5={:.4f}'.format(float(rank1), float(rank5)))
    ok = rank1 == rank1 and rank5 == rank5  # not NaN
    print('INFERENCE SMOKE TEST ' + ('PASSED' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
