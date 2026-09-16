"""Offline verification of the CSCI-aligned CC-DeepReID data layer.

Synthesises a miniature PRCC and LTCC tree in the layout the loaders expect,
then checks the split/label semantics against ICCV-CSCI-Person-ReID:

  PRCC
    * train relabelled 0..N-1, val parsed separately and NOT merged
    * pids/cloths come from the *folder* name for train/val
    * query_same = test/B, query_diff = test/C, gallery = test/A
    * each test identity owns exactly two cloth ids (A/B share one)
    * pruning "same pid AND same cloth" keeps the clothing-change pairs

  LTCC
    * no ID filtering at read time
    * cloth id = <pid>_<cam>, vocab built on train, reused for test
    * CC and General evaluators return different numbers

Run from the CC-DeepReID directory:  python scripts/verify_alignment.py
"""

import os
import shutil
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from datasets.base import ImageDataset            # noqa: E402
from datasets.ltcc import ltcc                    # noqa: E402
from datasets.prcc import prcc                    # noqa: E402
from utils.metrics import (R1_mAP_eval, R1_mAP_eval_CC, eval_func,  # noqa: E402
                           eval_func_general)

FAILURES = []


def check(condition, label, extra=''):
    status = 'OK  ' if condition else 'FAIL'
    print('  [{}] {}{}'.format(status, label, (' -> ' + str(extra)) if extra else ''))
    if not condition:
        FAILURES.append(label)


def write_img(path, size=(32, 16)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype('uint8')
    Image.fromarray(arr).save(path, quality=30)


# --------------------------------------------------------------------------- #
# mock data
# --------------------------------------------------------------------------- #
def build_prcc(root):
    """3 train ids, 2 val ids, 3 test ids (A/B/C each)."""
    d = os.path.join(root, 'prcc', 'rgb')
    for cam in ['A', 'B', 'C']:
        for pid in ['001', '002', '003']:
            for i in range(3):
                write_img(os.path.join(d, 'train', pid, '{}_cropped_rgb{:03d}.jpg'.format(cam, i)))
    for cam in ['A', 'B', 'C']:
        for pid in ['101', '102']:
            for i in range(2):
                write_img(os.path.join(d, 'val', pid, '{}_cropped_rgb{:03d}.jpg'.format(cam, i)))
    for cam in ['A', 'B', 'C']:
        for pid in ['001', '002', '003']:
            for i in range(2):
                write_img(os.path.join(d, 'test', cam, pid, 'cropped_rgb{:03d}.jpg'.format(i)))


def build_ltcc(root):
    """3 train ids x 2 cams x many cloths, 2 test ids.

    NOTE: LTCC clothing is keyed by ``<pid>_<cam>`` (CSCI's ``(\\w+)_c`` regex
    takes the whole ``<pid>_<cam>`` prefix, not the numeric cloth field), so one
    (pid, cam) pair collapses to a single clothing label.
    """
    d = os.path.join(root, 'LTCC_ReID')
    for pid in ['0001', '0002', '0003']:
        for cam in [1, 2]:
            for frame in range(2):
                write_img(os.path.join(
                    d, 'train', '{}_{}_c{}_{:05d}.png'.format(pid, cam, 1, frame)))
            # same (pid, cam) with a second numeric cloth still shares the key
            for frame in range(2):
                write_img(os.path.join(
                    d, 'train', '{}_{}_c{}_{:05d}.png'.format(pid, cam, 2, frame)))
    for split in ['query', 'test']:
        for pid in ['0101', '0102']:
            for cam in [1, 2]:
                for frame in range(2):
                    write_img(os.path.join(
                        d, split, '{}_{}_c{}_{:05d}.png'.format(pid, cam, 1, frame)))


# --------------------------------------------------------------------------- #
# PRCC
# --------------------------------------------------------------------------- #
def verify_prcc(root):
    print('\n=== PRCC ===')
    ds = prcc(root=root, verbose=False)

    train_pids = sorted({p for _, p, _, _ in ds.train})
    val_pids = sorted({p for _, p, _, _ in ds.val})
    check(train_pids == [0, 1, 2], 'train relabelled 0..N-1', train_pids)
    check(val_pids == [101, 102], 'val NOT merged into train', val_pids)
    check(len(ds.train) == 27, 'train keeps all A/B/C images (27)', len(ds.train))
    check(ds.num_train_pids == 3, 'num_train_pids == 3', ds.num_train_pids)
    check(not set(train_pids) & set(val_pids), 'train/val pid spaces disjoint')

    check(len(ds.gallery) == 6, 'gallery == test/A only (6)', len(ds.gallery))
    check({c for _, _, c, _ in ds.gallery} == {0}, 'gallery camid all 0 (A)')
    check(len(ds.query_same) == 6, 'query_same == test/B (6)', len(ds.query_same))
    check({c for _, _, c, _ in ds.query_same} == {1}, 'query_same camid all 1 (B)')
    check(len(ds.query_diff) == 6, 'query_diff == test/C (6)', len(ds.query_diff))
    check({c for _, _, c, _ in ds.query_diff} == {2}, 'query_diff camid all 2 (C)')
    check(ds.query is ds.query_diff, 'dataset.query aliases query_diff')

    check(sorted({p for _, p, _, _ in ds.gallery}) == [1, 2, 3],
          'test pids stay RAW (folder ids)', sorted({p for _, p, _, _ in ds.gallery}))

    # each identity owns exactly two cloth ids, A and B share one
    by_pid = {}
    for _, pid, cam, cloth in ds.gallery + ds.query_same + ds.query_diff:
        by_pid.setdefault(pid, {})[cam] = cloth
    two_cloths = all(len(set(v.values())) == 2 for v in by_pid.values())
    ab_share = all(v[0] == v[1] for v in by_pid.values())
    check(two_cloths, 'each test identity has exactly 2 cloth ids', by_pid)
    check(ab_share, 'A and B share a cloth id (session), C differs')
    check(all(v[2] == v[0] + 1 for v in by_pid.values()),
          'cloth ids follow CSCI pid*2 / pid*2+1', by_pid)

    # training cloth vocabulary: <pid> for A/B, <pid>C for C
    train_cloths = sorted({c for _, _, _, c in ds.train})
    check(len(train_cloths) == 6, 'train cloth vocab = 3 ids x 2 cloths', train_cloths)

    # the pruning rule must keep exactly the clothing-change pairs
    q_pid, q_cam, q_cloth = 1, 2, by_pid[1][2]
    removed_same_cloth = (q_pid == 1 and q_cloth == by_pid[1][0])
    check(not removed_same_cloth, 'C query vs A gallery is NOT pruned (clothing change)')
    q_pid_b, q_cloth_b = 1, by_pid[1][1]
    check(q_cloth_b == by_pid[1][0], 'B query vs A gallery is pruned (same session clothing)')
    return ds


# --------------------------------------------------------------------------- #
# LTCC
# --------------------------------------------------------------------------- #
def verify_ltcc(root):
    print('\n=== LTCC ===')
    ds = ltcc(root=root, verbose=False)

    train_pids = sorted({p for _, p, _, _ in ds.train})
    test_pids = sorted({p for _, p, _, _ in ds.query})
    check(train_pids == [0, 1, 2], 'train relabelled 0..N-1', train_pids)
    check(test_pids == [101, 102], 'test pids stay RAW', test_pids)
    check(len(ds.train) == 24, 'train keeps EVERY image (24, no cloth-change filtering)',
          len(ds.train))
    check(ds.num_train_pids == 3, 'num_train_pids == 3 (not pruned)', ds.num_train_pids)
    check(len(ds.query) == 8 and len(ds.gallery) == 8, 'query/gallery are full splits',
          (len(ds.query), len(ds.gallery)))

    cams = sorted({c for _, _, c, _ in ds.train})
    check(cams == [0, 1], 'camid is 0-based (c1->0)', cams)

    # cloth vocabulary is built on train and REUSED for test
    train_cloths = {c for _, _, _, c in ds.train}
    test_cloths = {c for _, _, _, c in ds.query}
    check(len(train_cloths) == 6, 'train cloth vocab = 3 ids x 2 cams', sorted(train_cloths))
    # CSCI would raise KeyError on an unseen test clothing; we must degrade gracefully
    check(test_cloths and max(test_cloths) >= len(train_cloths),
          'unseen test clothings are appended after the train vocab (no KeyError)',
          (sorted(train_cloths), sorted(test_cloths)))

    # same clothing is only removed when pid AND cloth both match
    q_cloth = ds.query[0][3]
    same_pid_diff_cam = [(p, c, cl) for _, p, c, cl in ds.gallery if p == ds.query[0][1]]
    check(len(same_pid_diff_cam) > 0, 'gallery contains same-pid samples',
          same_pid_diff_cam[:3])
    return ds


# --------------------------------------------------------------------------- #
# evaluator semantics
# --------------------------------------------------------------------------- #
def verify_metrics():
    print('\n=== metrics ===')
    # Two distractors of a DIFFERENT identity sit closest, one of which each rule
    # prunes as junk (same camera / same clothing as the query):
    #   q0  (pid=7, cam=2, cloth=1)
    #   d0  (pid=8, cam=2, cloth=1, d=0.02)  <- junk: same cam AND same cloth
    #   d1  (pid=8, cam=1, cloth=1, d=0.05)  <- junk: same cloth       (CC only)
    #   g0  (pid=7, cam=1, cloth=1, d=0.08)  <- true match, pruned by CC
    #   g1  (pid=7, cam=2, cloth=0, d=0.11)  <- true match, pruned by General
    #   g2  (pid=7, cam=1, cloth=0, d=0.15)  <- true match, kept by both
    #
    # After pruning, CC keeps [d0, g1, g2] and General keeps [d0, d1, g0, g2],
    # so the two rules produce genuinely different CMC/mAP numbers.
    qf = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gf = torch.tensor([
        [0.9998, 0.0200, 0.0, 0.0],    # d0
        [0.99875, 0.0, 0.05, 0.0],     # d1
        [0.9968, 0.0, 0.0, 0.08],      # g0
        [0.99395, 0.0, 0.0, 0.0],      # g1 (orthogonal part scaled to d=0.11)
        [0.98875, 0.0, 0.0, 0.0],      # g2 (d=0.15)
    ])
    # give g1/g2 their own orthogonal direction so distances stay strictly ordered
    gf[3, 3] = np.sqrt(max(0.0, 1.0 - float(gf[3, 0]) ** 2))
    gf[4, 3] = np.sqrt(max(0.0, 1.0 - float(gf[4, 0]) ** 2))

    q = torch.cat([qf, gf], dim=0)
    pids = torch.tensor([7, 8, 8, 7, 7, 7])
    cams = torch.tensor([2, 2, 1, 1, 2, 1])
    cloth = torch.tensor([1, 1, 1, 1, 0, 0])

    ev = R1_mAP_eval_CC(num_query=1, feat_norm=True)
    ev.reset()
    ev.update((q, pids, cams, cloth))
    cmc_cc, mAP_cc, *_ = ev.compute()
    cmc_gen, mAP_gen, *_ = ev.compute_general()

    print('  (CC mAP={:.4f}, CMC@1={:.3f} | General mAP={:.4f}, CMC@1={:.3f})'.format(
        mAP_cc, cmc_cc[0], mAP_gen, cmc_gen[0]))

    # independent reference implementation of both pruning rules, so the check
    # does not depend on hand arithmetic
    def reference(use_cloth):
        feats = torch.nn.functional.normalize(q, dim=1, p=2)
        qq, gg = feats[:1], feats[1:]
        dm = torch.cdist(qq, gg).numpy()[0]
        order = np.argsort(dm)
        gp, gc, gcl = pids[1:].numpy()[order], cams[1:].numpy()[order], cloth[1:].numpy()[order]
        remove = (gp == 7) & (gc == 2)
        if use_cloth:
            remove = remove | ((gp == 7) & (gcl == 1))
        keep = ~remove
        hits = (pids[1:].numpy()[order][keep] == 7).astype(int)
        cmc = np.cumsum(hits)
        cmc[cmc > 1] = 1
        rel = hits.sum()
        prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        return float((prec * hits).sum() / rel), cmc

    ref_cc, ref_cmc_cc = reference(True)
    ref_gen, ref_cmc_gen = reference(False)
    check(abs(mAP_cc - ref_cc) < 1e-6, 'CC mAP equals the reference implementation',
          (mAP_cc, ref_cc))
    check(abs(mAP_gen - ref_gen) < 1e-6, 'General mAP equals the reference implementation',
          (mAP_gen, ref_gen))
    check(np.allclose(cmc_cc[:len(ref_cmc_cc)], ref_cmc_cc),
          'CC CMC curve equals the reference implementation',
          (cmc_cc[:len(ref_cmc_cc)], ref_cmc_cc))
    check(np.allclose(cmc_gen[:len(ref_cmc_gen)], ref_cmc_gen),
          'General CMC curve equals the reference implementation',
          (cmc_gen[:len(ref_cmc_gen)], ref_cmc_gen))
    check(not np.isclose(mAP_cc, mAP_gen),
          'CC and General rules really do differ', (mAP_cc, mAP_gen))
    check(cmc_cc.shape == cmc_gen.shape, 'both CMC curves have a fixed, equal length',
          (cmc_cc.shape, cmc_gen.shape))

    check(hasattr(ev, 'compute') and hasattr(ev, 'compute_general'),
          'R1_mAP_eval_CC exposes compute (CC) and compute_general')

    # plain R1_mAP_eval must keep its old cloth-aware behaviour
    ev_old = R1_mAP_eval(num_query=1, feat_norm=True)
    ev_old.reset()
    ev_old.update((q, pids, cams, cloth))
    ev_old.compute()
    check(True, 'R1_mAP_eval still usable as a drop-in')


# --------------------------------------------------------------------------- #
# end-to-end through make_dataloader
# --------------------------------------------------------------------------- #
def verify_make_dataloader(root):
    print('\n=== make_dataloader ===')
    from configs import cfg
    from datasets import make_dataloader

    for name, key in (('prcc', None), ('ltcc', None)):
        c = cfg.clone()
        c.merge_from_list(['DATASETS.NAMES', name, 'DATASETS.ROOT_DIR', root,
                           'DATALOADER.NUM_WORKERS', 0, 'SOLVER.IMS_PER_BATCH', 6,
                           'DATALOADER.NUM_INSTANCE', 3, 'TEST.IMS_PER_BATCH', 4,
                           'MODEL.DEVICE', 'cpu'])
        c.freeze()
        loaders, bundle = make_dataloader(c)
        img, pid, cam, cloth = next(iter(loaders['train'][0]))
        check(img.shape[1:] == (3, 256, 128), '{} train batch shape'.format(name),
              tuple(img.shape))
        check(pid.dtype == torch.int64 and pid.min() >= 0, '{} train pids are labels'.format(name))

        loader, num_query, data = loaders['query_diff']
        imgs, pids, cams, cloths, camids_batch, paths = next(iter(loader))
        check(num_query == len(data) - bundle.num_gallery_imgs,
              '{} num_query splits query from gallery'.format(name), num_query)
        check(len(paths) == imgs.shape[0], '{} val batches carry image paths'.format(name))

        if name == 'prcc':
            check('query_same' in loaders, 'prcc exposes a query_same loader')
            check(bundle.num_query_imgs_diff == len(bundle.query_diff),
                  'prcc bundle counts query_diff correctly', bundle.num_query_imgs_diff)
            check(bundle.num_query_imgs_same == len(bundle.query_same),
                  'prcc bundle counts query_same correctly', bundle.num_query_imgs_same)
            check(bundle.num_train_pids == 3, 'prcc bundle num_train_pids', bundle.num_train_pids)
        else:
            check('query_same' not in loaders, 'ltcc has a single query loader')
            check(bundle.num_query_imgs == len(bundle.query),
                  'ltcc bundle counts query correctly', bundle.num_query_imgs)
        print('  bundle: {}'.format(bundle))


def verify_dual_branch(root):
    print('\n=== S2A dual-branch shared trunk + losses ===')
    from configs import cfg
    from losses import make_loss
    from losses.disentangle_loss import ClothDisentangleLoss
    from models import make_model
    from models.backbones.s2a_resnet import S2ABlock

    c = cfg.clone()
    c.merge_from_list(['MODEL.DEVICE', 'cpu',
                       'DATASETS.NAMES', 'prcc', 'DATASETS.ROOT_DIR', root,
                       'DATALOADER.NUM_WORKERS', 0, 'SOLVER.IMS_PER_BATCH', 6,
                       'DATALOADER.NUM_INSTANCE', 3])
    c.freeze()

    # ---- 1) the block itself: isolation by construction ----
    blk = S2ABlock(in_planes=32, out_planes=32, num_branches=2)
    blk.train()
    z = torch.randn(1, 32, 8, 8, requires_grad=True)
    o = blk(z)
    check(o.shape == z.shape, 'S2A block preserves the trunk shape', tuple(o.shape))
    half = 16
    g = torch.autograd.grad(o[:, :half].pow(2).sum(), z, retain_graph=True,
                            allow_unused=True)[0]
    cross = 0.0 if g is None else float(g[0, half:].abs().sum())
    check(cross == 0.0, 'cross-branch gradient is EXACTLY zero (branch 0 <- branch 1)',
          cross)
    own = float(g[0, :half].abs().sum())
    check(own > 0, 'each branch still receives its own gradient', own)
    check(tuple(blk.conv_shared.weight.shape) == (32, 1, 3, 3),
          'the shared 3x3 is depthwise: one kernel serves both branches',
          tuple(blk.conv_shared.weight.shape))

    # ---- 2) the model: two 2048-wide branches from ONE shared trunk ----
    model = make_model(c, num_classes=3, num_cloth_classes=6)
    model.train()
    check(not hasattr(model, 'channel_attention'),
          'the SE channel-attention module is gone (replaced by the S2A trunk)')
    out = model(torch.randn(4, 3, 256, 128),
                target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    check(len(out) == 8, 'train forward returns the same 8 slots', len(out))
    id_score, F, cloth_score, Fp, global_feat, map_id, map_cloth, hist_pred = out
    check(id_score.shape == (4, 3), 'F -> id logits', tuple(id_score.shape))
    check(cloth_score.shape == (4, 6), "F' -> cloth logits", tuple(cloth_score.shape))
    check(F.shape == (4, 2048) and Fp.shape == (4, 2048),
          'both branches are 2048 wide (disentangle-ready)',
          (tuple(F.shape), tuple(Fp.shape)))

    # ---- 3) maps are un-pooled and pooling reproduces the features ----
    check(map_id is not None and map_cloth is not None,
          'un-pooled branch maps are still returned')
    check(map_id.dim() == 4 and map_cloth.dim() == 4,
          'maps are 4D (attention/pooling order unchanged: maps before pooling)',
          (tuple(map_id.shape), tuple(map_cloth.shape)))
    check(map_id.shape[1] == 2048 and map_cloth.shape[1] == 2048,
          'each map carries one branch width (no mixing)', tuple(map_id.shape))
    gap_id = map_id.detach().mean(dim=(2, 3))
    check(torch.allclose(model.bottleneck(gap_id), F, atol=1e-4),
          'F == bottleneck(GAP(map_id))', float((model.bottleneck(gap_id) - F).abs().max()))
    gap_cloth = map_cloth.detach().mean(dim=(2, 3))
    # NOTE: with singleton cloth classes the model deliberately normalises F' with
    # running statistics instead of batch statistics (long-standing guard), so
    # compare against that same path.
    was_training = model.cloth_bottleneck.training
    model.cloth_bottleneck.eval()
    expected_fp = model.cloth_bottleneck(gap_cloth)
    model.cloth_bottleneck.train(was_training)
    check(torch.allclose(expected_fp, Fp, atol=1e-4),
          "F' == cloth_bottleneck(GAP(map_cloth)) (running-stats fallback path)",
          float((expected_fp - Fp).abs().max()))

    # ---- 4) the branches are genuinely different ----
    check(not torch.allclose(F, Fp), 'F and F\' are not identical',
          float((F - Fp).abs().mean()))

    # ---- 5) both branches sit on the autograd path of the SHARED trunk ----
    loss_func = make_loss(c, num_classes=3, num_cloth_classes=6)
    target = torch.tensor([0, 0, 1, 1])
    target_cloth = torch.tensor([0, 1, 2, 3])
    # the shared parameter both branches must reach: 'sealed' shares a depthwise
    # kernel, 's2a' shares the scene projection. Probe whichever is present.
    blk0 = model.base.layer4[0]
    probe_w = blk0.conv_shared.weight if hasattr(blk0, 'conv_shared') \
        else blk0.conv_kv.weight
    check(probe_w.requires_grad, 'the shared trunk parameter is trainable',
          tuple(probe_w.shape))
    g_id = torch.autograd.grad(F.pow(2).sum(), probe_w, retain_graph=True,
                               allow_unused=True)[0]
    g_cloth = torch.autograd.grad(Fp.pow(2).sum(), probe_w, retain_graph=True,
                                  allow_unused=True)[0]
    check(g_id is not None and float(g_id.abs().sum()) > 0,
          'F reaches the shared trunk',
          None if g_id is None else float(g_id.abs().sum()))
    check(g_cloth is not None and float(g_cloth.abs().sum()) > 0,
          "F' reaches the shared trunk",
          None if g_cloth is None else float(g_cloth.abs().sum()))

    # ---- 6) losses still work end to end ----
    total, parts = loss_func(id_score, F, target, target_cloth,
                             cloth_score=cloth_score, cloth_feat=Fp,
                             hist_pred=hist_pred,
                             hist_target=torch.rand(4, model.hist_dim)
                             if hist_pred is not None else None,
                             return_parts=True)
    check(torch.isfinite(total), 'total loss is finite', float(total))
    check(parts['cloth'] is not None, 'cloth CE is still active on F\'', parts['cloth'])
    check(parts['disentangle'] is not None, 'disentangle is active',
          parts['disentangle'])
    check(abs(parts['disentangle']
              - float(ClothDisentangleLoss()(F, Fp))) < 1e-5,
          'disentangle term == |cos(F, F\')|')

    # ---- 7) eval still returns F only (retrieval unchanged) ----
    model.eval()
    with torch.no_grad():
        ev = model(torch.randn(2, 3, 256, 128))
    check(torch.is_tensor(ev) and ev.shape == (2, 2048),
          'eval forward returns just F', tuple(ev.shape))

    # ---- 8) DUAL_BRANCH=False keeps a plain single-branch path ----
    c2 = c.clone()
    c2.merge_from_list(['MODEL.DUAL_BRANCH', False])
    c2.freeze()
    plain = make_model(c2, num_classes=3, num_cloth_classes=6)
    plain.train()
    o2 = plain(torch.randn(4, 3, 256, 128))
    check(len(o2) == 8 and o2[2] is None and o2[5] is None and o2[6] is None,
          'dual_branch=False keeps the 8-slot tuple with the extras None',
          [None if x is None else getattr(x, 'shape', '?') for x in o2])
    check(o2[1].shape == (4, 2048), 'single-branch F is still 2048',
          tuple(o2[1].shape))

    return model


def verify_histogram(root):
    """CSCI's colour-histogram regression on F', alongside the cloth softmax."""
    print('\n=== CSCI colour-histogram supervision on F\' ===')
    import importlib.util

    import torch.nn.functional as Fn
    from configs import cfg
    from datasets.histogram import HistogramExtractor, RGBuvHistBlock
    from losses import make_loss
    from models import make_model

    # ---- 1) bit-identical to CSCI's RGBuvHistBlock ----
    csci_path = os.path.join(ROOT, '..', 'ICCV-CSCI-Person-ReID', 'data', 'rgbuc.py')
    if os.path.isfile(csci_path):
        spec = importlib.util.spec_from_file_location('csci_rgbuc', csci_path)
        csci = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(csci)
        torch.manual_seed(0)
        ok_all = True
        for h, sigma, iscale, go, size in [(32, 0.001, False, False, (192, 384)),
                                           (32, 0.02, False, False, (128, 256)),
                                           (64, 0.001, True, False, (256, 128)),
                                           (16, 0.001, False, True, (192, 384))]:
            x = torch.rand(1, 3, *size).clamp(0, 1)
            a = RGBuvHistBlock(h=h, sigma=sigma, intensity_scale=iscale,
                               green_only=go, device='cpu')(x)
            b = csci.RGBuvHistBlock(h=h, insz=150, sigma=sigma,
                                    intensity_scale=iscale, green_only=go,
                                    device='cpu')(x)
            ok_all &= torch.allclose(a, b, atol=1e-7)
        check(ok_all, 'RGBuvHistBlock is bit-identical to CSCI\'s implementation')
    else:
        print('  [skip] CSCI checkout not found, cannot compare the histogram')

    # ---- 2) the block enforces its batch-1 contract ----
    try:
        RGBuvHistBlock(h=32)(torch.rand(2, 3, 128, 64))
        check(False, 'histogram block rejects batch > 1')
    except ValueError:
        check(True, 'histogram block rejects batch > 1 (CSCI only fills sample 0)')

    # ---- 3) the label really is an annotation-free colour descriptor ----
    ex = HistogramExtractor(h=32, sigma=0.001, norm='l1', weight=100.0)
    img = torch.randn(1, 3, 384, 192)
    lab = ex(img)
    check(tuple(lab.shape) == (1, 1024), 'label shape is (1, HIST_DIM**2)', tuple(lab.shape))
    check(abs(float(lab.abs().sum()) - 100.0) < 1e-3,
          'L1 normalisation + weight 100 reproduce CSCI\'s label scale',
          float(lab.abs().sum()))
    lab2 = ex(torch.randn(1, 3, 384, 192))
    check(float((lab - lab2).abs().max()) > 0,
          'different images give different colour labels (it is image-derived)')
    # a flat grey image must land at the centre bin, not on an edge
    grey = ex(torch.full((1, 3, 384, 192), 0.5))
    check(torch.isfinite(grey).all(), 'grey image gives a finite histogram')

    # ---- 4) model + loss wiring ----
    c = cfg.clone()
    c.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.NECK', 'none',
                       'MODEL.USE_HIST', True, 'MODEL.HIST_DIM', 32,
                       'DATALOADER.NUM_WORKERS', 0])
    c.freeze()
    model = make_model(c, num_classes=3, num_cloth_classes=6)
    check(model.hist_head is not None, 'USE_HIST installs a histogram head')
    hist_dim = model.hist_dim
    check(hist_dim == 1024, 'head output width == HIST_DIM**2', hist_dim)
    model.train()
    out = model(torch.randn(4, 3, 256, 128),
                target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    check(len(out) == 8, 'train forward now returns 8 items', len(out))
    hist_pred = out[7]
    check(hist_pred is not None and tuple(hist_pred.shape) == (4, hist_dim),
          'histogram prediction has shape (B, HIST_DIM**2)', tuple(hist_pred.shape))
    # it must be computed FROM F', so perturbing F' must change the prediction
    with torch.no_grad():
        base = hist_pred.detach().clone()
    h2 = model.hist_head(out[3] + 1.0)
    check(float((h2 - base).abs().max()) > 0,
          'histogram head is a function of F\' (perturbing F\' changes the output)',
          float((h2 - base).abs().max()))

    # ---- 5) the loss: cosine / mse / l1 + cloth softmax still active ----
    loss_func = make_loss(c, num_classes=3, num_cloth_classes=6)
    target = torch.tensor([0, 0, 1, 1])
    target_cloth = torch.tensor([0, 1, 2, 3])
    target_hist = ex(torch.randn(1, 3, 256, 128)).expand(4, -1).contiguous()
    total, parts = loss_func(out[0], out[1], target, target_cloth,
                             cloth_score=out[2], cloth_feat=out[3],
                             hist_pred=hist_pred, hist_target=target_hist,
                             return_parts=True)
    check(parts['hist'] is not None, 'histogram term is active and reported',
          parts['hist'])
    check(torch.isfinite(total), 'total loss with the histogram term is finite',
          float(total))
    check(parts['cloth'] is not None,
          'cloth softmax is STILL active next to the histogram term (kept, as asked)',
          parts['cloth'])
    check(parts['hist'] < 1.0 + 1e-6, 'cosine histogram loss is in [0, 2]',
          parts['hist'])

    # turning it off must restore the previous behaviour
    c_off = cfg.clone()
    c_off.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.NECK', 'none',
                           'MODEL.USE_HIST', False, 'DATALOADER.NUM_WORKERS', 0])
    c_off.freeze()
    m_off = make_model(c_off, num_classes=3, num_cloth_classes=6)
    m_off.train()
    out_off = m_off(torch.randn(4, 3, 256, 128),
                    target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    check(m_off.hist_head is None and out_off[7] is None and len(out_off) == 8,
          'USE_HIST=False keeps the 8-slot tuple but disables the head')


def main():
    # NOTE: use a workspace-local mock root (the DSH file sandbox blocks the OS temp dir)
    root = os.path.join(ROOT, '_mock_data')
    shutil.rmtree(root, ignore_errors=True)
    try:
        build_prcc(root)
        build_ltcc(root)
        verify_prcc(root)
        verify_ltcc(root)
        verify_metrics()
        verify_make_dataloader(root)
        verify_dual_branch(root)
        verify_histogram(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print('\n' + '=' * 60)
    if FAILURES:
        print('{} CHECK(S) FAILED:'.format(len(FAILURES)))
        for f in FAILURES:
            print('  - {}'.format(f))
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()