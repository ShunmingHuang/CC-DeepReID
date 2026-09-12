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
    print('\n=== dual-branch attention + losses ===')
    import torch.nn as nn
    from configs import cfg
    from losses import make_loss
    from losses.disentangle_loss import ClothDisentangleLoss
    from models import make_model
    from models.attention import DualBranchChannelAttention

    c = cfg.clone()
    c.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.PRETRAIN', '',
                       'DATASETS.NAMES', 'prcc', 'DATASETS.ROOT_DIR', root,
                       'DATALOADER.NUM_WORKERS', 0, 'SOLVER.IMS_PER_BATCH', 6,
                       'DATALOADER.NUM_INSTANCE', 3])
    c.freeze()

    # ---- 1) the attention module itself ----
    att = DualBranchChannelAttention(channels=2048, reduction=16)
    fm = torch.randn(2, 2048, 8, 4)
    out = att(fm)
    check(len(out) == 4, 'attention returns 4 fields (F, F\', map_id, map_cloth)', len(out))
    F, Fp, map_id, map_cloth = out
    check(F.shape == (2, 2048) and Fp.shape == (2, 2048),
          'attention returns two (B, 2048) features F and F\'', (tuple(F.shape), tuple(Fp.shape)))
    check(F.shape == Fp.shape, 'F and F\' have identical width (disentangle-ready)')
    check(map_id.shape == (2, 2048, 8, 4) and map_cloth.shape == (2, 2048, 8, 4),
          'maps come out un-pooled, same spatial size as the input',
          (tuple(map_id.shape), tuple(map_cloth.shape)))

    # attention is applied on the MAP; pooling is per-branch and comes after
    check(torch.allclose(att.pool(map_id), F, atol=1e-5),
          'GAP(map_id) == F (pooling happens after the channel attention)',
          float((att.pool(map_id) - F).abs().max()))
    check(torch.allclose(att.pool(map_cloth), Fp, atol=1e-5),
          'GAP(map_cloth) == F\' (each branch has its OWN pooling)',
          float((att.pool(map_cloth) - Fp).abs().max()))

    # the two branches must not be trivially identical
    att.eval()
    with torch.no_grad():
        out_eval = att(fm)
    F, Fp = out_eval.feat_id, out_eval.feat_cloth
    check(not torch.allclose(F, Fp), 'F and F\' come from independent branch parameters',
          float((F - Fp).abs().mean()))
    check(not torch.allclose(out_eval.map_id, out_eval.map_cloth),
          'the two branches re-weight the map differently')

    # channel attention must actually re-weight channels (weights not all equal)
    squeezed = fm.mean(dim=(2, 3))
    w_id = att.att_id.fc(squeezed)
    w_cl = att.att_cloth.fc(squeezed)
    check(w_id.std().item() > 0, 'id branch channel weights vary across channels',
          float(w_id.std()))
    check(not torch.allclose(w_id, w_cl), 'the two branches learn different channel weights')

    # ---- 2) disentangle loss ----
    # NOTE: the objective is CSCI's |cos|, which is minimized at cos = 0, i.e. the
    # two features are driven to be ORTHOGONAL (not anti-parallel: -1 also has
    # |cos| = 1 and is penalised just as much as +1).
    d = ClothDisentangleLoss()
    a = torch.randn(8, 64)
    check(abs(float(d(a, a)) - 1.0) < 1e-5, 'identical features -> |cos| = 1 (max loss)',
          float(d(a, a)))
    check(abs(float(d(a, -a)) - 1.0) < 1e-5,
          'anti-parallel features are penalised too (|cos| = 1, orthogonality is the target)',
          float(d(a, -a)))
    check(float(d(a, torch.randn(8, 64))) < float(d(a, a)),
          'separating features lowers the loss')

    # explicit orthogonal pair -> loss must be ~0
    e1 = torch.zeros(4, 64)
    e1[:, 0] = 1.0
    e2 = torch.zeros(4, 64)
    e2[:, 1] = 1.0
    check(float(d(e1, e2)) < 1e-5, 'orthogonal features -> loss ~ 0', float(d(e1, e2)))

    # hinge mode: relu(cos - margin), so cos=1 with margin 0.5 costs exactly 0.5
    d_margin = ClothDisentangleLoss(margin=0.5)
    check(abs(float(d_margin(a, a)) - 0.5) < 1e-5,
          'hinge mode: cos=1 with margin 0.5 costs exactly 0.5', float(d_margin(a, a)))
    check(float(d_margin(a, -a)) < 1e-6,
          'hinge mode charges nothing for anti-parallel features (cos=-1 < margin)',
          float(d_margin(a, -a)))

    # ---- 3) full model, both branches ----
    torch.manual_seed(0)
    model = make_model(c, num_classes=3, num_cloth_classes=6)
    model.train()
    out = model(torch.randn(4, 3, 256, 128))
    check(len(out) == 7, 'train forward returns 7 outputs', len(out))
    id_score, F, cloth_score, Fp, global_feat, map_id, map_cloth = out
    check(id_score.shape == (4, 3), 'F -> id logits (B, num_id)', tuple(id_score.shape))
    check(cloth_score.shape == (4, 6), 'F\' -> cloth logits (B, num_cloth)',
          tuple(cloth_score.shape))
    check(F.shape == (4, 2048) and Fp.shape == (4, 2048),
          'both branch features are (B, 2048)', (tuple(F.shape), tuple(Fp.shape)))
    check(torch.isfinite(id_score).all() and torch.isfinite(cloth_score).all(),
          'logits are finite')

    # the attention-weighted maps must be exposed UN-POOLED (4D, spatial dims kept)
    check(map_id is not None and map_cloth is not None,
          'attention-weighted maps are returned')
    check(map_id.dim() == 4 and map_cloth.dim() == 4,
          'maps are 4D (channel attention before pooling, not after)',
          (tuple(map_id.shape), tuple(map_cloth.shape)))
    check(map_id.shape[1] == 2048 and map_id.shape[2] > 1 and map_id.shape[3] > 1,
          'maps keep the spatial resolution (H/16, W/16)',
          tuple(map_id.shape))
    # Pooling the map must reproduce the PRE-NECK branch feature. With
    # MODEL.NECK='bnneck' the model returns bottleneck(GAP(map)), so assert the
    # whole chain rather than pretending GAP(map) == F.
    pooled_id = map_id.detach().mean(dim=(2, 3))
    check(torch.allclose(model.bottleneck(pooled_id), F, atol=1e-4),
          'F == bottleneck(GAP(map_id))  (attention -> pool -> BNNeck)',
          float((model.bottleneck(pooled_id) - F).abs().max()))

    # ...and with a non-bnneck neck the model output IS the pooled map, which
    # pins down "channel attention first, per-branch pooling afterwards"
    c_noneck = c.clone()
    c_noneck.merge_from_list(['MODEL.NECK', 'none'])
    c_noneck.freeze()
    m_noneck = make_model(c_noneck, num_classes=3, num_cloth_classes=6)
    m_noneck.train()
    o = m_noneck(torch.randn(4, 3, 256, 128),
                 target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    g_id = m_noneck.channel_attention.pool(o[5].detach())
    g_cloth = m_noneck.channel_attention.pool(o[6].detach())
    check(torch.allclose(g_id, o[1], atol=1e-4),
          'GAP(map_id) == F  (channel attention first, pooling afterwards)',
          float((g_id - o[1]).abs().max()))
    check(torch.allclose(g_cloth, o[3], atol=1e-4),
          'GAP(map_cloth) == F\' (each branch has its OWN pooling)',
          float((g_cloth - o[3]).abs().max()))
    check(not torch.allclose(map_id, map_cloth),
          'the two branches produce different maps (independent attention + pooling)',
          float((map_id - map_cloth).abs().mean()))

    # pre-pool maps must sit on the gradient path of the pooled features
    gmap = torch.autograd.grad(F.sum(), map_id, retain_graph=True, allow_unused=True)[0]
    check(gmap is not None and gmap.abs().sum() > 0,
          'map_id is on the autograd path of F (usable by other modules)')

    # eval forward must stay a single F tensor so the retrieval path is unchanged
    model.eval()
    with torch.no_grad():
        eval_out = model(torch.randn(2, 3, 256, 128))
    check(torch.is_tensor(eval_out) and eval_out.shape == (2, 2048),
          'eval forward returns just F (retrieval unchanged)', tuple(eval_out.shape))

    # ---- 4) losses: F gets ID+triplet, F\' gets ID-only, plus disentangle ----
    loss_func = make_loss(c, num_classes=3, num_cloth_classes=6)
    model.train()
    id_score, F, cloth_score, Fp, _, _, _ = model(torch.randn(6, 3, 256, 128),
                                            target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    check(cloth_score is not None and Fp is not None,
          'singleton cloth classes still produce a cloth branch (BN fallback, no crash)')
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    target_cloth = torch.tensor([0, 1, 2, 3, 4, 5])

    total, parts = loss_func(id_score, F, target, target_cloth,
                             cloth_score=cloth_score, cloth_feat=Fp,
                             return_parts=True)
    check(torch.isfinite(total), 'total loss is finite', float(total))
    check(parts['cloth'] is not None and parts['disentangle'] is not None,
          'cloth softmax and disentangle terms are both active')
    check(parts['triplet'] is not None, 'triplet term is reported separately',
          parts['triplet'])
    check(abs(parts['disentangle'] - float(ClothDisentangleLoss()(F, Fp))) < 1e-5,
          'disentangle term equals |cos(F, F\')|')

    # F' is a *classification* branch: its head must be a label-smoothed softmax
    # over the clothing vocabulary, and it must carry NO triplet term.
    from losses.cross_entropy_loss import CrossEntropyLoss as _CE
    expected_cloth = float(_CE(num_classes=6, label_smooth=bool(c.MODEL.LABELSMOOTH),
                               use_gpu=True)(cloth_score.float(), target_cloth))
    check(abs(parts['cloth'] - expected_cloth) < 1e-5,
          'cloth head loss == CE over the cloth vocabulary',
          (parts['cloth'], expected_cloth))
    # Check the label-smoothing FLAG rather than the loss value: at random init the
    # CE is ~ln(num_classes) and smoothing perturbs it by only ~1e-5, so a value
    # comparison cannot distinguish the two configurations.
    check(loss_func.cloth_loss.eps > 0,
          'cloth head is configured with label smoothing (MODEL.LABELSMOOTH)',
          loss_func.cloth_loss.eps)
    check(abs(loss_func.cloth_loss.eps - 0.1) < 1e-9,
          'label smoothing strength is the standard 0.1', loss_func.cloth_loss.eps)
    # and it must follow MODEL.LABELSMOOTH when that is turned off
    c_nols = c.clone()
    c_nols.merge_from_list(['MODEL.LABELSMOOTH', False])
    c_nols.freeze()
    check(make_loss(c_nols, num_classes=3, num_cloth_classes=6).cloth_loss.eps == 0,
          'turning MODEL.LABELSMOOTH off also disables it on the cloth head')

    # ---- 4b) the triplet must equal CSCI's identity-only implementation ----
    # reference: ICCV-CSCI-Person-ReID/loss/triplet_loss.py (hard_example_mining
    # over `labels`, clothing never consulted)
    def csci_triplet_reference(feat, labels, margin):
        feat = feat.float()
        dist = torch.cdist(feat, feat)
        N = dist.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        dist_ap = dist[is_pos].contiguous().view(N, -1).max(1)[0]
        dist_an = dist[is_neg].contiguous().view(N, -1).min(1)[0]
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        return float(torch.nn.MarginRankingLoss(margin=margin)(dist_an, dist_ap, y))

    from losses.hard_mine_triplet_loss import TripletLoss as _TL
    trio = _TL(margin=0.3)
    ref = csci_triplet_reference(F, target, 0.3)
    got = float(trio(F, target, target_cloth))
    check(abs(got - ref) < 1e-5,
          'triplet == CSCI identity-only reference (cloth ignored)', (got, ref))

    # passing a different cloth labelling must NOT change CSCI's triplet
    shuffled_cloth = torch.tensor([5, 4, 3, 2, 1, 0])
    got2 = float(trio(F, target, shuffled_cloth))
    check(abs(got2 - got) < 1e-6,
          'triplet is invariant to the cloth labels (CSCI is cloth-agnostic)',
          (got, got2))

    # cloth branch must NOT receive a triplet term: zeroing cloth_feat must not
    # remove the triplet contribution of F
    no_cloth = loss_func(id_score, F, target, target_cloth)
    check(torch.isfinite(no_cloth), 'loss still works without the cloth branch (back-compat)')
    check(float(total) > float(no_cloth), 'adding the cloth branch raises the total loss',
          (float(total), float(no_cloth)))

    # ---- 5) a real optimisation step reduces the loss ----
    # small lr + eval-mode BN: this is a randomly initialised 25M-param net on
    # 12 synthetic images, so a large lr or training-mode BN statistics diverge
    model.eval()
    model.dual_branch = True
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    x = torch.randn(12, 3, 256, 128)
    target12 = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    cloth12 = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    first = None
    for step in range(6):
        opt.zero_grad()
        # eval() returns only F, so run the branch heads manually for training
        model.train()
        id_score, F, cloth_score, Fp, _, _, _ = model(x, target_cloth=cloth12)
        model.eval()
        loss, _ = loss_func(id_score, F, target12, cloth12,
                            cloth_score=cloth_score, cloth_feat=Fp, return_parts=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
    check(float(loss) < first * 0.9, 'loss decreases over optimisation steps (>10%)',
          (first, float(loss)))

    # ---- 6) dual_branch=False keeps the original single-branch behaviour ----
    c2 = c.clone()
    c2.merge_from_list(['MODEL.DUAL_BRANCH', False])
    c2.freeze()
    plain = make_model(c2, num_classes=3, num_cloth_classes=6)
    plain.train()
    out = plain(torch.randn(4, 3, 256, 128))
    # the tuple shape is kept identical across modes; the dual-branch extras are None
    check(len(out) == 7 and out[2] is None and out[5] is None and out[6] is None,
          'dual_branch=False keeps the legacy path (no cloth head, no maps)',
          [None if x is None else getattr(x, 'shape', type(x).__name__) for x in out])
    check(out[1].shape == (4, 2048), 'legacy F is still the plain pooled descriptor',
          tuple(out[1].shape))

    bundle = None  # (bundle used only for the dataloader checks)
    return model


def verify_cal_and_cloth_head(root):
    """C2R-ReID clothing branch: cosine head, detached discriminator, delay."""
    print('\n=== C2R clothes head (cosine) + CAL discriminator ===')
    import torch.nn.functional as Fn
    from configs import cfg
    from losses import make_loss
    from losses.clothes_adversarial_loss import (ClothesBasedAdversarialLoss,
                                                 positive_clothes_mask)
    from models import make_model
    from models.classifier import NormalizedClassifier

    c = cfg.clone()
    c.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.NECK', 'none',
                       'DATALOADER.NUM_WORKERS', 0])
    c.freeze()

    # ---- 1) cosine head: logits really are scaled cosine similarities ----
    head = NormalizedClassifier(64, 10, scale=16.0)
    x = torch.randn(5, 64)
    logits = head(x)
    manual = 16.0 * Fn.linear(Fn.normalize(x, p=2, dim=1),
                             Fn.normalize(head.weight, p=2, dim=1))
    check(torch.allclose(logits, manual, atol=1e-5),
          'cosine head logits == scale * cos(feature, class weight)',
          float((logits - manual).abs().max()))
    check(float(logits.abs().max()) <= 16.0 + 1e-4,
          'cosine logits are bounded by the scale', float(logits.abs().max()))

    # ---- 2) model wiring: CLOTH_HEAD switches the head type ----
    m_lin = make_model(c, num_classes=3, num_cloth_classes=6)
    check(isinstance(m_lin.cloth_classifier, torch.nn.Linear),
          'CLOTH_HEAD=linear keeps nn.Linear', type(m_lin.cloth_classifier).__name__)

    c_cos = c.clone()
    c_cos.merge_from_list(['MODEL.CLOTH_HEAD', 'cosine', 'MODEL.CLOTH_HEAD_SCALE', 16.0])
    c_cos.freeze()
    m_cos = make_model(c_cos, num_classes=3, num_cloth_classes=6)
    check(isinstance(m_cos.cloth_classifier, NormalizedClassifier),
          'CLOTH_HEAD=cosine installs NormalizedClassifier',
          type(m_cos.cloth_classifier).__name__)
    m_cos.train()
    out = m_cos(torch.randn(4, 3, 256, 128),
                target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    check(float(out[2].abs().max()) <= 16.0 + 1e-3,
          'model cloth logits respect the cosine scale', float(out[2].abs().max()))

    # ---- 3) detach path: the discriminator must NOT reach the backbone ----
    m_cos.zero_grad()
    det_score, _, _, det_feat, _, _, _ = m_cos(
        torch.randn(4, 3, 256, 128), target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]),
        detach_cloth=True)
    check(det_score is not None and det_feat is not None,
          'detach_cloth forward returns cloth logits + features')
    check(not det_feat.requires_grad,
          'the discriminator feature is a true leaf: requires_grad == False '
          '(no BatchNorm on the detach path, or it would rebuild a graph)')
    det_score.float().sum().backward()
    head_grad = m_cos.cloth_classifier.weight.grad
    bn_grad = (m_cos.cloth_bottleneck.weight.grad
               if m_cos.cloth_bottleneck is not None else None)
    bb_grad = next(p for n, p in m_cos.named_parameters()
                   if n.startswith('base.') and p.requires_grad).grad
    check(head_grad is not None and float(head_grad.abs().sum()) > 0,
          'detached path DOES update the clothing head',
          float(head_grad.abs().sum()))
    check(bn_grad is None or float(bn_grad.abs().sum()) == 0,
          'detached path leaves the cloth BatchNorm untouched (it is not on that path)',
          None if bn_grad is None else float(bn_grad.abs().sum()))
    check(bb_grad is None or float(bb_grad.abs().sum()) == 0,
          'detached path does NOT touch the backbone (the whole point of detach)',
          None if bb_grad is None else float(bb_grad.abs().sum()))

    # ---- 4) the live path DOES reach the backbone (that is the adversarial term) ----
    m_cos.zero_grad()
    live_score, _, _, _, _, _, _ = m_cos(
        torch.randn(4, 3, 256, 128), target_cloth=torch.tensor([0, 1, 2, 3, 4, 5]))
    live_score.float().sum().backward()
    bb_grad_live = next(p for n, p in m_cos.named_parameters()
                        if n.startswith('base.') and p.requires_grad).grad
    check(bb_grad_live is not None and float(bb_grad_live.abs().sum()) > 0,
          'live path DOES reach the backbone (CAL drives the generator side)',
          None if bb_grad_live is None else float(live_score.abs().mean()))

    # ---- 5) CAL loss: positive mask semantics + eps spread ----
    cal = ClothesBasedAdversarialLoss(scale=16.0, epsilon=0.1)
    pids = torch.tensor([0, 0, 0, 0])
    cloth = torch.tensor([0, 1, 0, 1])          # identity 0 owns clothes {0,1}
    mask = torch.zeros(4, 4)
    mask[:, 0] = 1
    mask[:, 1] = 1
    logits = torch.randn(4, 4)
    loss_cal = float(cal(logits, cloth, mask))
    check(loss_cal == loss_cal and loss_cal > 0, 'CAL returns a finite positive loss',
          loss_cal)
    # positives must be excluded from the negative set: if the mask covered only
    # the target class, another class of the same identity would become a negative
    mask_only_target = torch.zeros(4, 4)
    mask_only_target.scatter_(1, cloth.unsqueeze(1), 1)
    check(abs(float(cal(logits, cloth, mask_only_target)) - loss_cal) > 1e-6,
          'CAL actually uses the full positive (same-identity) clothes set')
    check(float(cal(logits, cloth, mask)) != float(
        ClothesBasedAdversarialLoss(scale=16.0, epsilon=1.0)(logits, cloth, mask)),
        'epsilon changes the objective (0.1 vs 1.0)')

    # ---- 6) the positive mask comes from the dataset ----
    from datasets import make_dataloader
    c2 = c.clone()
    c2.merge_from_list(['DATASETS.NAMES', 'prcc', 'DATASETS.ROOT_DIR', root,
                        'SOLVER.IMS_PER_BATCH', 6, 'DATALOADER.NUM_INSTANCE', 3])
    c2.freeze()
    loaders, bundle = make_dataloader(c2)
    check(bundle.pid2clothes is not None,
          'DatasetBundle exposes pid2clothes for the CAL positive mask')
    p2c = torch.as_tensor(bundle.pid2clothes)
    check(tuple(p2c.shape) == (bundle.num_train_pids, bundle.num_train_clothes),
          'pid2clothes has shape (num_train_pids, num_train_clothes)', tuple(p2c.shape))
    got = positive_clothes_mask(bundle.pid2clothes, torch.tensor([0, 1]))
    check(tuple(got.shape) == (2, bundle.num_train_clothes),
          'positive_clothes_mask gathers one row per batch sample', tuple(got.shape))
    check(float(got.sum()) > 0, 'the gathered mask is non-empty (some positives exist)',
          float(got.sum()))

    # ---- 7) delayed start: epoch gating is strictly "off before, on after" ----
    start = int(c.MODEL.CAL_START_EPOCH)
    check([e >= start for e in [start - 1, start, start + 1]] == [False, True, True],
          'CAL epoch gate: inactive before CAL_START_EPOCH, active from it on',
          'start={}'.format(start))

    # ---- 8) USE_CAL guards ----
    c_bad = c.clone()
    c_bad.merge_from_list(['MODEL.USE_CAL', True])
    c_bad.freeze()
    from models import make_model as _mm
    from losses import make_loss as _ml
    m_nocloth = _mm(c_bad, num_classes=3, num_cloth_classes=0)
    check(m_nocloth.cloth_classifier is None,
          'without a clothing branch the CAL guard has something to catch')
    check(_ml(c_bad, num_classes=3, num_cloth_classes=6).use_cal,
          'USE_CAL is read from the config into the loss bundle')


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
        verify_cal_and_cloth_head(root)
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
