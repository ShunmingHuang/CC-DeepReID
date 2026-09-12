"""Run a few real training batches + a full evaluation on the REAL datasets (CPU).

This is the step between `audit_real_datasets.py` (data semantics) and a full GPU
run: it drives the real dataloaders, the dual-branch model and the real
evaluation protocol over the actual PRCC/LTCC query and gallery sets, but only
samples a couple of training batches so it stays CPU-affordable.

    python scripts/real_data_cpu_check.py --dataset prcc --root ../data
    python scripts/real_data_cpu_check.py --dataset ltcc --root ../data
"""

import argparse
import logging
import os
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='prcc', choices=['prcc', 'ltcc'])
    ap.add_argument('--root', default=os.path.join(ROOT, '..', 'data'))
    ap.add_argument('--batches', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--instances', type=int, default=4)
    args = ap.parse_args()

    from configs import cfg
    from datasets import make_dataloader
    from models import make_model
    from losses import make_loss
    from processor import evaluate

    c = cfg.clone()
    c.merge_from_list([
        'MODEL.DEVICE', 'cpu',
        'MODEL.DUAL_BRANCH', True,
        'DATASETS.NAMES', args.dataset,
        'DATASETS.ROOT_DIR', os.path.abspath(args.root),
        'DATALOADER.NUM_WORKERS', 0,
        'DATALOADER.NUM_INSTANCE', args.instances,
        'SOLVER.IMS_PER_BATCH', args.batch_size,
        'TEST.IMS_PER_BATCH', 32,
        'TEST.MODE', 'both',
    ])
    c.freeze()

    logger = logging.getLogger('reid')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    print('=== real-data CPU check: {} ==='.format(args.dataset))
    loaders, bundle = make_dataloader(c)
    print('bundle: {}'.format(bundle))
    print('cloth head will have {} classes'.format(bundle.num_train_clothes))

    model = make_model(c, num_classes=bundle.num_train_pids,
                       num_cloth_classes=bundle.num_train_clothes)
    loss_func = make_loss(c, num_classes=bundle.num_train_pids,
                          num_cloth_classes=bundle.num_train_clothes)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # ---- a few real training batches ----
    model.train()
    loader = loaders['train'][0]
    print('\n-- {} real train batches (batch_size={}, instances={}) --'.format(
        args.batches, args.batch_size, args.instances))
    t0 = time.time()
    n = 0
    first_loss = None
    for img, pid, cam, cloth_id in loader:
        opt.zero_grad()
        target, target_cloth = pid, cloth_id
        cls_score, F, cloth_score, Fp, _, _, _ = model(img, target_cloth=target_cloth)
        cls_score, F = cls_score.float(), F.float()
        if cloth_score is not None:
            cloth_score = cloth_score.float()
        loss, parts = loss_func(cls_score, F, target, target_cloth,
                                cloth_score=cloth_score, cloth_feat=Fp,
                                return_parts=True)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss)
            print('  batch 0: img{} pid{} cloth{}  ids/batch={} uniq_cloth={}'.format(
                tuple(img.shape), tuple(target[:4].tolist()),
                tuple(target_cloth[:4].tolist()),
                len(set(target.tolist())), len(set(target_cloth.tolist()))))
            print('  batch 0: loss={:.3f} (id {:.3f} | cloth {} | dis {})'.format(
                float(loss), parts['id'],
                'n/a' if parts['cloth'] is None else '{:.3f}'.format(parts['cloth']),
                'n/a' if parts['disentangle'] is None else '{:.3f}'.format(parts['disentangle'])))
        n += 1
        if n >= args.batches:
            break
    dt = time.time() - t0
    print('  {} batches ok in {:.1f}s ({:.2f}s/batch on CPU)'.format(n, dt, dt / max(n, 1)))

    # ---- full evaluation on the real query/gallery ----
    print('\n-- full evaluation on the real test split --')
    t0 = time.time()
    results = evaluate(c, model, loaders, bundle, torch.device('cpu'), logger, c.TEST.MODE)
    print('  evaluation finished in {:.1f}s'.format(time.time() - t0))
    for name, (cmc, mAP) in results.items():
        print('  {:<9s} Rank-1={:.1%} Rank-5={:.1%} mAP={:.1%}'.format(
            name, float(cmc[0]), float(cmc[min(5, len(cmc)) - 1]), float(mAP)))

    ok = all(len(cmc) > 0 for cmc, _ in results.values())
    print('\nREAL-DATA CPU CHECK ' + ('PASSED' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
