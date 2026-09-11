"""End-to-end training smoke test for CC-DeepReID (CPU-friendly).

Builds a small synthetic PRCC tree, then drives the *real* code paths:

    make_dataloader -> make_model -> make_loss -> build_optimizer
        -> build_lr_scheduler -> do_train  (train + eval + checkpointing)

Run from the CC-DeepReID directory::

    python scripts/train_smoke_test.py

Exit code 0 means the training pipeline ran end to end. Any exception in the
train/eval loop surfaces as a traceback (that is the point of the test).
"""

import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# yacs on Windows needs utf-8 when reading config files
import yacs.config  # noqa: E402

_orig_merge = yacs.config.CfgNode.merge_from_file


def _patched_merge_from_file(self, cfg_filename):
    with open(cfg_filename, 'r', encoding='utf-8') as f:
        cfg = self.load_cfg(f)
    self.merge_from_other_cfg(cfg)


yacs.config.CfgNode.merge_from_file = _patched_merge_from_file

MOCK_ROOT = os.path.join(ROOT, '_smoke_data')
LOG_DIR = os.path.join(ROOT, '_smoke_run', 'logs')
OUT_DIR = os.path.join(ROOT, '_smoke_run', 'outputs')

N_TRAIN_IDS = 2
IMGS_PER_ID = 8           # 3 cams x 8 = 24 images per identity
N_TEST_IDS = 2


def write_img(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (np.random.rand(16, 8, 3) * 255).astype('uint8')
    Image.fromarray(arr).save(path, quality=20)


def build_prcc(root):
    d = os.path.join(root, 'prcc', 'rgb')
    for pid in range(1, N_TRAIN_IDS + 1):
        for cam in ['A', 'B', 'C']:
            for i in range(IMGS_PER_ID // 3 + 1):
                write_img(os.path.join(d, 'train', '{:03d}'.format(pid),
                                       '{}_cropped_rgb{:03d}.jpg'.format(cam, i)))
        for cam in ['A', 'B', 'C']:
            for i in range(2):
                write_img(os.path.join(d, 'val', '{:03d}'.format(pid),
                                       '{}_cropped_rgb{:03d}.jpg'.format(cam, i)))
    for pid in range(1, N_TEST_IDS + 1):
        for cam in ['A', 'B', 'C']:
            for i in range(2):
                write_img(os.path.join(d, 'test', cam, '{:03d}'.format(pid),
                                       'cropped_rgb{:03d}.jpg'.format(i)))


def timed_first_batch(loaders):
    """Measure the cost of a real forward+backward step to size the smoke run."""
    from models import make_model
    from losses import make_loss
    from configs import cfg

    print('\n--- timing one train step (CPU) ---')
    t0 = time.time()
    _ = next(iter(loaders['train'][0]))
    print('  dataloader first batch: {:.2f}s'.format(time.time() - t0))

    t0 = time.time()
    model = make_model(cfg, num_classes=N_TRAIN_IDS)
    model.train()
    out = model(torch.randn(4, 3, 256, 128))
    loss = sum(o.float().sum() for o in out[:2])
    loss.backward()
    dt = time.time() - t0
    print('  fwd+bwd (batch 4, 256x128): {:.2f}s'.format(dt))
    return dt


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    shutil.rmtree(MOCK_ROOT, ignore_errors=True)
    shutil.rmtree(os.path.join(ROOT, '_smoke_run'), ignore_errors=True)
    build_prcc(MOCK_ROOT)

    from configs import cfg
    from datasets import make_dataloader
    from models import make_model
    from losses import make_loss
    from solver import build_optimizer, build_lr_scheduler
    from processor import do_train
    from utils.logger import setup_logger

    c = cfg.clone()
    c.merge_from_list([
        'MODEL.DEVICE', 'cpu',
        'MODEL.DUAL_BRANCH', True,
        'DATASETS.NAMES', 'prcc',
        'DATASETS.ROOT_DIR', MOCK_ROOT,
        'DATALOADER.NUM_WORKERS', 0,
        'DATALOADER.SAMPLER', 'triplet',
        'DATALOADER.NUM_INSTANCE', 4,
        'SOLVER.IMS_PER_BATCH', 8,
        'SOLVER.MAX_EPOCHS', 2,
        'SOLVER.LOG_PERIOD', 1,
        'SOLVER.EVAL_PERIOD', 1,
        'SOLVER.CHECKPOINT_PERIOD', 1,
        'SOLVER.WARMUP_EPOCHS', 0,
        'SOLVER.BASE_LR', 0.001,
        'TEST.IMS_PER_BATCH', 8,
        'TEST.MODE', 'both',
        'LOG_DIR', LOG_DIR,
        'OUTPUT_DIR', OUT_DIR,
        'EXP_NAME', 'smoke',
    ])
    c.freeze()

    logger = setup_logger("reid", os.path.join(LOG_DIR, 'smoke'), if_train=True)
    logger.info('=== CC-DeepReID training smoke test ===')

    # train.py creates these before do_train; do_train does not create them itself
    exp_out = os.path.join(OUT_DIR, 'smoke')
    os.makedirs(exp_out, exist_ok=True)

    loaders, bundle = make_dataloader(c)
    print('\nbundle: {}'.format(bundle))
    for key in ['train', 'query_diff', 'query_same', 'gallery']:
        if key in loaders:
            loader, nq, holder = loaders[key]
            # third element: the ImageDataset for val loaders, the raw prcc object for train
            inner = getattr(holder, 'dataset', holder)
            n_items = len(inner) if hasattr(inner, '__len__') else -1
            print('  loader {:12s} batches={:3d} items={:5d} num_query={}'.format(
                key, len(loader), n_items, nq))

    timed_first_batch(loaders)

    num_cloth_classes = bundle.num_train_clothes
    model = make_model(c, num_classes=bundle.num_train_pids,
                       num_cloth_classes=num_cloth_classes)
    print('\nheads: {} id classes, {} cloth classes'.format(
        bundle.num_train_pids, num_cloth_classes))
    loss_func = make_loss(c, num_classes=bundle.num_train_pids,
                          num_cloth_classes=num_cloth_classes)
    optimizer = build_optimizer(model, optim='adam', lr=c.SOLVER.BASE_LR,
                                weight_decay=c.SOLVER.WEIGHT_DECAY,
                                momentum=c.SOLVER.MOMENTUM)
    scheduler = build_lr_scheduler(optimizer, lr_scheduler=c.SOLVER.LR_SCHEDULER,
                                   stepsize=c.SOLVER.STEPSIZE, gamma=c.SOLVER.GAMMA,
                                   max_epoch=c.SOLVER.MAX_EPOCHS,
                                   warmup_epochs=c.SOLVER.WARMUP_EPOCHS,
                                   warmup_type=c.SOLVER.WARMUP_TYPE)

    t0 = time.time()
    rank1, mAP = do_train(c, model, loaders, bundle, optimizer, scheduler,
                          loss_func, device=torch.device('cpu'))
    elapsed = time.time() - t0

    print('\n=== RESULT ===')
    print('do_train finished in {:.1f}s'.format(elapsed))
    print('best rank1={:.4f} mAP={:.4f}'.format(float(rank1), float(mAP)))

    exp_dir = os.path.join(OUT_DIR, 'smoke')
    saved = sorted(os.listdir(exp_dir)) if os.path.isdir(exp_dir) else []
    print('checkpoints: {}'.format(saved))

    ok = True
    if not saved:
        print('FAIL: no checkpoint written')
        ok = False
    if not np.isfinite(rank1):
        print('FAIL: rank1 is not finite')
        ok = False

    # the model must be usable for inference after training, through the same
    # path test.py uses
    model.eval()
    with torch.no_grad():
        feat = model(torch.randn(2, 3, 256, 128))
    if feat.shape != (2, 2048):
        print('FAIL: eval forward shape {}'.format(tuple(feat.shape)))
        ok = False
    else:
        print('eval forward after training: {}'.format(tuple(feat.shape)))

    print('SMOKE TEST ' + ('PASSED' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
