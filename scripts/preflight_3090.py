"""Preflight check for the 3090 training host (py310 + CUDA).

Answers, in order, the questions that decide whether a run can start:

1. Is this torch a CUDA build, and does it see the GPU?   (a `+cpu` wheel fails here)
2. Can the dual-branch model build, forward and backward *on the GPU* with AMP?
3. How much GPU memory does one training step need for the configured batch size?
4. Is the dataset on disk with the layout the loaders expect?
5. Can one real train+eval step run through the actual processor code?

Run it on the 3090 host from the CC-DeepReID directory::

    python scripts/preflight_3090.py
    python scripts/preflight_3090.py --config configs/datasets/prcc/resnet.yaml \
        --opts DATASETS.ROOT_DIR /path/to/data SOLVER.IMS_PER_BATCH 64

It never touches the datasets (except a read-only listing) and never writes
checkpoints.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []
WARNINGS = []


def ok(label, extra=''):
    print('  [OK]   {}{}'.format(label, ' -> ' + str(extra) if extra else ''))


def fail(label, extra=''):
    print('  [FAIL] {}{}'.format(label, ' -> ' + str(extra) if extra else ''))
    FAILURES.append(label)


def warn(label, extra=''):
    print('  [WARN] {}{}'.format(label, ' -> ' + str(extra) if extra else ''))
    WARNINGS.append(label)


def section(title):
    print('\n=== {} ==='.format(title))


def check_environment():
    section('1. environment')
    import torch

    print('  python      : {}'.format(sys.version.split()[0]))
    print('  torch       : {}'.format(torch.__version__))
    print('  cuda build  : {}'.format(torch.version.cuda))
    print('  cudnn       : {}'.format(torch.backends.cudnn.version()))

    if torch.version.cuda is None:
        fail('torch is a CPU-only build (+cpu wheel)',
             'install a CUDA wheel before training, e.g.\n'
             '         pip install torch==2.14.0 torchvision --index-url '
             'https://download.pytorch.org/whl/cu124')
        return None

    if not torch.cuda.is_available():
        fail('torch.cuda.is_available() is False (driver / GPU not visible)')
        return None

    ok('CUDA is available', '{} device(s)'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print('    cuda:{} {} | sm_{}{} | {:.1f} GB'.format(
            i, p.name, p.major, p.minor, p.total_memory / 1024 ** 3))
    if torch.cuda.get_device_properties(0).major == 8 and \
            torch.cuda.get_device_properties(0).minor == 6:
        ok('device 0 is SM 8.6 (RTX 3090 supported by this torch build)')
    return torch.device('cuda:0')


def check_model_and_amp(cfg, device):
    section('2. dual-branch model on GPU + AMP')
    import torch
    from models import make_model
    from losses import make_loss

    n_id, n_cloth = 150, 300
    try:
        model = make_model(cfg, num_classes=n_id, num_cloth_classes=n_cloth).to(device)
    except Exception as e:
        fail('model build on GPU raised', '{}: {}'.format(type(e).__name__, e))
        return None, None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok('model built on GPU', '{:.2f}M trainable params'.format(n_params / 1e6))

    batch = int(cfg.SOLVER.IMS_PER_BATCH)
    x = torch.randn(batch, 3, 256, 128, device=device)
    target = torch.randint(0, n_id, (batch,), device=device)
    target_cloth = torch.randint(0, n_cloth, (batch,), device=device)

    loss_func = make_loss(cfg, num_classes=n_id, num_cloth_classes=n_cloth)
    scaler = torch.GradScaler('cuda')
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type='cuda', enabled=True):
            cls_score, F, cloth_score, Fp, _, _, _ = model(x, target_cloth=target_cloth)
            cls_score, F = cls_score.float(), F.float()
            if cloth_score is not None:
                cloth_score = cloth_score.float()
        loss, parts = loss_func(cls_score, F, target, target_cloth,
                                cloth_score=cloth_score, cloth_feat=Fp,
                                return_parts=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            fail('AMP train step ran out of GPU memory at batch {}'.format(batch),
                 'lower SOLVER.IMS_PER_BATCH')
        else:
            fail('AMP train step raised', '{}: {}'.format(type(e).__name__, e))
        return None, None

    ok('AMP fwd+bwd+step ran', 'loss={:.4f} (id {:.3f} | cloth {} | dis {})'.format(
        float(loss), parts['id'],
        'n/a' if parts['cloth'] is None else '{:.3f}'.format(parts['cloth']),
        'n/a' if parts['disentangle'] is None else '{:.3f}'.format(parts['disentangle'])))

    if not torch.isfinite(loss):
        fail('loss is not finite on GPU (AMP overflow?)', float(loss))
    if parts['cloth'] is None:
        fail('cloth branch returned None on GPU')
    if parts['disentangle'] is None:
        fail('disentangle term was skipped on GPU')

    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    ok('peak memory for one step (batch {})'.format(batch),
       '{:.1f} GB / {:.1f} GB ({:.0f}%)'.format(peak, total, 100 * peak / total))
    if peak / total > 0.85:
        warn('peak memory is close to the 3090 limit',
             'reduce SOLVER.IMS_PER_BATCH (currently {})'.format(batch))

    # fp16 stability of the triplet distance under autocast
    with torch.autocast(device_type='cuda', enabled=True):
        Fh = F.detach()
        d = torch.pow(Fh, 2).sum(1, keepdim=True).expand(batch, batch)
        d = d + d.t()
        d = d.addmm_(Fh, Fh.t(), beta=1, alpha=-2)
    if torch.isinf(d).any() or torch.isnan(d).any():
        fail('fp16 squared-distance overflow under autocast', float(d.max()))
    else:
        ok('triplet squared-distance is finite under autocast', float(d.max()))

    model.eval()
    with torch.no_grad():
        feat = model(torch.randn(2, 3, 256, 128, device=device))
    if feat.shape != (2, 2048):
        fail('eval forward shape', tuple(feat.shape))
    else:
        ok('eval forward returns F', tuple(feat.shape))

    return model, loss_func


def check_dataset(cfg):
    section('4. dataset on disk')
    import os.path as osp

    name = cfg.DATASETS.NAMES
    root = cfg.DATASETS.ROOT_DIR
    print('  DATASETS.NAMES={} ROOT_DIR={}'.format(name, root))
    if not osp.isdir(root):
        fail('dataset root does not exist', osp.abspath(root))
        return False

    if name == 'prcc':
        expected = {
            'rgb/train': ('dir', None),
            'rgb/val': ('dir', None),
            'rgb/test/A': ('dir', None),
            'rgb/test/B': ('dir', None),
            'rgb/test/C': ('dir', None),
        }
        base = osp.join(root, 'prcc')
    else:
        expected = {'train': ('dir', None), 'query': ('dir', None), 'test': ('dir', None)}
        base = osp.join(root, 'LTCC_ReID')

    if not osp.isdir(base):
        fail('dataset dir missing', osp.abspath(base))
        return False

    all_ok = True
    for rel in expected:
        p = osp.join(base, rel)
        if not osp.isdir(p):
            fail('missing {}'.format(osp.join(name, rel)), osp.abspath(p))
            all_ok = False
    if all_ok:
        ok('expected split directories exist', osp.abspath(base))

        # sanity: count a few files and check the naming convention
        if name == 'prcc':
            train_dir = osp.join(base, 'rgb/train')
            pdirs = sorted(os.listdir(train_dir))[:3]
            pdirs = [d for d in pdirs if osp.isdir(osp.join(train_dir, d))]
            if pdirs:
                sample_dir = osp.join(train_dir, pdirs[0])
                files = sorted(os.listdir(sample_dir))[:3]
                print('    sample: {}/{}'.format(pdirs[0], files))
                bad = [f for f in files if '_cropped_rgb' not in f]
                if bad:
                    warn('train file names do not look like <cam>_cropped_rgb###.jpg', bad)
                else:
                    ok('train file naming matches the loader expectation')
            test_dir = osp.join(base, 'rgb/test/A')
            if osp.isdir(test_dir):
                cams = sorted(os.listdir(test_dir))[:3]
                cams = [d for d in cams if osp.isdir(osp.join(test_dir, d))]
                if cams:
                    files = sorted(os.listdir(osp.join(test_dir, cams[0])))[:3]
                    print('    sample: A/{}/{}'.format(cams[0], files))
                    if any('_cropped_rgb' in f for f in files):
                        fail('test/ file names use the TRAIN convention '
                             '(<cam>_cropped_rgb), the test loader expects cropped_rgb###.jpg')
                    else:
                        ok('test file naming matches the loader expectation')
        else:
            train_dir = osp.join(base, 'train')
            files = sorted(os.listdir(train_dir))[:3]
            print('    sample: {}'.format(files))
            import re
            pat = re.compile(r'(\d+)_(\d+)_c(\d+)')
            if files and not all(pat.search(f) for f in files):
                fail('LTCC file names do not match <pid>_<cam>_c<cloth>_<frame>.png', files)
            else:
                ok('LTCC file naming matches the loader expectation')
    return all_ok


def check_real_step(cfg, device):
    section('5. one real train + eval step through the processor')
    import torch
    from datasets import make_dataloader
    from models import make_model
    from losses import make_loss
    from processor import evaluate

    try:
        loaders, bundle = make_dataloader(cfg)
    except Exception as e:
        fail('make_dataloader raised', '{}: {}'.format(type(e).__name__, e))
        return
    print('  bundle: {}'.format(bundle))
    ok('dataloaders built', '{} train images'.format(len(bundle.train)))

    # only sample a couple of batches instead of an epoch
    saved = {}
    for key, (loader, nq, holder) in list(loaders.items()):
        if key == 'train':
            saved[key] = (loader, nq, holder)
            continue
    if not saved:
        fail('no train loader in the bundle')
        return

    model = make_model(cfg, num_classes=bundle.num_train_pids,
                       num_cloth_classes=bundle.num_train_clothes).to(device)
    loss_func = make_loss(cfg, num_classes=bundle.num_train_pids,
                          num_cloth_classes=bundle.num_train_clothes)
    scaler = torch.GradScaler('cuda')
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    model.train()
    loader = saved['train'][0]
    t0 = time.time()
    n = 0
    try:
        for img, pid, _, cloth_id in loader:
            opt.zero_grad()
            img, pid, cloth_id = img.to(device), pid.to(device), cloth_id.to(device)
            with torch.autocast(device_type='cuda', enabled=True):
                cls_score, F, cloth_score, Fp, _, _, _ = model(img, target_cloth=cloth_id)
                cls_score, F = cls_score.float(), F.float()
                if cloth_score is not None:
                    cloth_score = cloth_score.float()
            loss, _ = loss_func(cls_score, F, pid, cloth_id,
                                cloth_score=cloth_score, cloth_feat=Fp, return_parts=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            n += 1
            if n >= 3:
                break
    except RuntimeError as e:
        fail('real train step raised', '{}: {}'.format(type(e).__name__, e))
        return
    ok('{} real train batches on GPU'.format(n),
       '{:.2f}s total, {:.2f}s/batch'.format(time.time() - t0, (time.time() - t0) / max(n, 1)))

    # evaluation on a subset: just make sure the protocol code runs on GPU
    model.eval()
    try:
        import logging
        logger = logging.getLogger('preflight')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)
        evaluate(cfg, model, loaders, bundle, device, logger, cfg.TEST.MODE)
        ok('evaluation protocol ran on GPU')
    except Exception as e:
        fail('evaluate raised', '{}: {}'.format(type(e).__name__, e))


def check_pretrain(cfg):
    section('3. pretrained backbone weights')
    import os.path as osp

    path = cfg.MODEL.PRETRAIN
    print('  MODEL.PRETRAIN = {}'.format(path))
    if not path:
        warn('MODEL.PRETRAIN is empty')
        return
    if not osp.isfile(path):
        warn('MODEL.PRETRAIN does not exist at that path',
             osp.abspath(path))
    else:
        ok('pretrained file exists', osp.abspath(path))
    warn('MODEL.PRETRAIN is never used by the training code',
         'train.py/make_model never call load_parameter, so the ResNet50 '
         'backbone trains FROM SCRATCH. Expected ImageNet-pretrained numbers '
         'will not be reproduced until backbone init is wired up.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='', help='dataset config yaml to merge')
    ap.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                    help='extra yacs overrides, e.g. DATASETS.ROOT_DIR /data')
    args = ap.parse_args()

    import yacs.config
    _orig = yacs.config.CfgNode.merge_from_file

    def _patched(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            self.merge_from_other_cfg(self.load_cfg(f))

    yacs.config.CfgNode.merge_from_file = _patched

    from configs import cfg
    c = cfg.clone()
    if args.config:
        c.merge_from_file(args.config)
    if args.opts:
        c.merge_from_list(args.opts)
    c.freeze()
    print('running preflight with: DATASET={} ROOT={} BATCH={} DEVICE={}'.format(
        c.DATASETS.NAMES, c.DATASETS.ROOT_DIR, c.SOLVER.IMS_PER_BATCH, c.MODEL.DEVICE))

    device = check_environment()
    if device is None:
        print('\n' + '=' * 66)
        print('PREFLIGHT FAILED: fix the environment first (see above)')
        return 1

    check_pretrain(c)
    check_model_and_amp(c, device)
    has_data = check_dataset(c)
    if has_data:
        check_real_step(c, device)
    else:
        print('\n  (skipping the real train/eval step: dataset not usable yet)')

    print('\n' + '=' * 66)
    print('{} failure(s), {} warning(s)'.format(len(FAILURES), len(WARNINGS)))
    for f in FAILURES:
        print('  FAIL: {}'.format(f))
    for w in WARNINGS:
        print('  WARN: {}'.format(w))
    if FAILURES:
        print('PREFLIGHT FAILED')
        return 1
    print('PREFLIGHT PASSED' + (' (with warnings)' if WARNINGS else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
