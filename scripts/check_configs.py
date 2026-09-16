"""Validate every shipped config: load it, build the model, forward/backward it.

This exists because a yaml that parses is not the same as a yaml that runs. It
checks, for each config file:

  * it loads through the same path train.py uses (utf-8 merge_from_file)
  * MODEL.S2A_MODE round-trips, and the built model actually uses that block type
  * the batch recipe is self-consistent (IMS_PER_BATCH divisible by NUM_INSTANCE)
  * the batch recipe fits the dataset's identity count, if the data is present
  * TEST.MODE / EVAL_PROTOCOL are legal values
  * a real forward + backward runs and returns the 8-slot training tuple

Run: python scripts/check_configs.py
Exit code is non-zero if any config fails.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yacs.config                                                  # noqa: E402
from configs import cfg                                             # noqa: E402
from models.make_model import ResNet                                # noqa: E402


# train.py's monkey patch: plain merge_from_file is not utf-8 safe on Windows
def _patched_merge_from_file(self, cfg_filename):
    with open(cfg_filename, 'r', encoding='utf-8') as f:
        cfg = self.load_cfg(f)
    self.merge_from_other_cfg(cfg)


yacs.config.CfgNode.merge_from_file = _patched_merge_from_file

CONFIGS = [
    'configs/datasets/prcc/resnet.yaml',
    'configs/datasets/prcc/s2a.yaml',
    'configs/datasets/prcc/sealed.yaml',
    'configs/datasets/ltcc/resnet.yaml',
    'configs/datasets/ltcc/s2a.yaml',
    'configs/datasets/ltcc/sealed.yaml',
]

FAILS = []


def check(name, ok, detail=''):
    print('  {} {}{}'.format('[OK  ]' if ok else '[FAIL]', name,
                             (' -> ' + str(detail)) if detail != '' else ''))
    if not ok:
        FAILS.append(name)


def real_root():
    """The real dataset root, if this machine has it."""
    for cand in (os.path.join(os.path.dirname(ROOT), 'data'),
                 os.path.join(ROOT, '..', 'data')):
        if os.path.isdir(os.path.join(cand, 'prcc')):
            return os.path.abspath(cand)
    return None


def train_id_count(root, name):
    """Number of training identity folders, for the batch-recipe sanity check."""
    if root is None:
        return None
    if name == 'prcc':
        d = os.path.join(root, 'prcc', 'rgb', 'train')
        return len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]) \
            if os.path.isdir(d) else None
    if name == 'ltcc':
        d = os.path.join(root, 'LTCC_ReID', 'train')
        if not os.path.isdir(d):
            return None
        return len([x for x in os.listdir(d)
                    if os.path.isdir(os.path.join(d, x))
                    and x.lower() not in ('info',)])
    return None


root = real_root()
print('real dataset root:', root)
print()

for rel in CONFIGS:
    path = os.path.join(ROOT, rel)
    print('=== %s ===' % rel)
    if not os.path.isfile(path):
        check('exists', False, path)
        continue

    try:
        cfg.merge_from_file(path)
    except Exception as exc:
        check('loads', False, repr(exc))
        continue
    check('loads', True)

    mode = cfg.MODEL.S2A_MODE
    check("S2A_MODE is 's2a' or 'sealed'", mode in ('s2a', 'sealed'), mode)
    # the filename should agree with the setting, so nobody runs the wrong thing
    if 'sealed' in rel:
        check('filename matches S2A_MODE=sealed', mode == 'sealed', mode)
    elif 's2a' in os.path.basename(rel):
        check('filename matches S2A_MODE=s2a', mode == 's2a', mode)

    check('TEST.MODE legal', cfg.TEST.MODE in ('cc', 'both'), cfg.TEST.MODE)
    check('EVAL_PROTOCOL legal', cfg.TEST.EVAL_PROTOCOL in ('CC', 'General'),
          cfg.TEST.EVAL_PROTOCOL)

    bs = cfg.SOLVER.IMS_PER_BATCH
    ni = cfg.DATALOADER.NUM_INSTANCE
    check('IMS_PER_BATCH divisible by NUM_INSTANCE', bs % ni == 0, '%d / %d' % (bs, ni))
    ids_per_batch = bs // ni
    check('ids per batch is sane (2..16)', 2 <= ids_per_batch <= 16, ids_per_batch)

    n_ids = train_id_count(root, cfg.DATASETS.NAMES)
    if n_ids:
        check('batch recipe fits the dataset (%d train ids)' % n_ids,
              ids_per_batch <= n_ids, ids_per_batch)

    # ---- build the real model and run it ----
    try:
        model = ResNet(num_classes=12, cfg=cfg, num_cloth_classes=24)
        model.train()
        blk = model.base.layer4[0]
        if mode == 's2a':
            used = hasattr(blk, 'conv_q') and hasattr(blk, 'conv_kv')
        else:
            used = hasattr(blk, 'conv_shared') and not hasattr(blk, 'conv_kv')
        check('built trunk is the %s block' % mode, used,
              type(blk).__name__)

        n_par = sum(p.numel() for p in model.parameters())
        out = model(torch.randn(8, 3, *cfg.INPUT.SIZE_TRAIN))
        check('8-slot training forward', len(out) == 8, len(out))
        check("F and F' are %d wide" % cfg.MODEL.S2A_BRANCH_WIDTH,
              out[1].shape[1] == cfg.MODEL.S2A_BRANCH_WIDTH
              and out[3].shape[1] == cfg.MODEL.S2A_BRANCH_WIDTH,
              (tuple(out[1].shape), tuple(out[3].shape)))
        check('params %.2fM reported' % (n_par / 1e6), n_par > 0, '%.2fM' % (n_par / 1e6))

        out[0].pow(2).sum().backward()
        had_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                       for p in model.parameters())
        check('backward reaches the parameters', had_grad)
    except Exception as exc:
        check('build + forward + backward', False, repr(exc))
    print()

print('=' * 66)
if FAILS:
    print('%d CHECK(S) FAILED:' % len(FAILS))
    for f in FAILS:
        print('  -', f)
    sys.exit(1)
print('ALL CONFIGS OK (%d files)' % len(CONFIGS))
