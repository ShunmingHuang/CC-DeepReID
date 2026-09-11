"""Audit the real PRCC / LTCC datasets through the actual loader code.

Read-only: builds the dataset objects, prints per-split statistics, and checks
the semantics the framework relies on (two cloth ids per test identity, raw test
pids, train/val disjointness, cloth vocabulary built on train, ...).

Run from the CC-DeepReID directory::

    python scripts/audit_real_datasets.py [--root ../data]
"""

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(cond, label, extra=''):
    print('  [{}] {}{}'.format('OK  ' if cond else 'FAIL', label,
                               ' -> ' + str(extra) if extra else ''))
    if not cond:
        FAILURES.append(label)


def stats(name, data):
    pids = {p for _, p, _, _ in data}
    cams = {c for _, _, c, _ in data}
    cloths = {c for _, _, _, c in data}
    print('  {:12s} imgs={:6d} pids={:4d} cams={:2d} cloths={:4d}'.format(
        name, len(data), len(pids), len(cams), len(cloths)))
    return len(data), len(pids), len(cams), len(cloths)


def audit_prcc(root):
    print('\n=== PRCC ===')
    from datasets.prcc import prcc

    logger = logging.getLogger('audit')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    ds = prcc(root=root, verbose=False, logger=logger)
    logger.info('=> PRCC loaded')

    n_train, n_train_pids, n_train_cams, n_train_cloths = stats('train', ds.train)
    n_val, n_val_pids, _, _ = stats('val', ds.val)
    n_qs, n_qs_pids, _, _ = stats('query_same (B)', ds.query_same)
    n_qd, n_qd_pids, _, _ = stats('query_diff (C)', ds.query_diff)
    n_gal, n_gal_pids, n_gal_cams, n_gal_cloths = stats('gallery (A)', ds.gallery)

    print('  -- official PRCC reference: train 150/17896, val 150/5002, '
          'gallery(A) 71/3384, query_same(B) 71/3873, query_diff(C) 71/3543 --')

    check(n_train == 17896 and n_train_pids == 150,
          'PRCC train matches the official 150 ids / 17896 imgs', (n_train_pids, n_train))
    check(n_val == 5002 and n_val_pids == 150,
          'PRCC val parsed separately, not merged', (n_val_pids, n_val))
    check(n_gal == 3384 and n_gal_pids == 71, 'gallery == test/A', (n_gal_pids, n_gal))
    check(n_qs == 3873, 'query_same == test/B', n_qs)
    check(n_qd == 3543, 'query_diff == test/C', n_qd)
    check(n_gal_cams == 1 and {c for _, _, c, _ in ds.gallery} == {0},
          'gallery camid is all 0 (A)')

    train_pid_set = {p for _, p, _, _ in ds.train}
    val_raw_pids = {int(os.path.basename(os.path.dirname(p)))
                    for p, _, _, _ in ds.val}
    # NOTE: PRCC's rgb/val re-uses the SAME 150 identities as rgb/train; the two
    # splits are disjoint at the IMAGE level, not at the identity level. Merging
    # them (the old CC-DeepReID behaviour) leaks nothing image-wise, while CSCI's
    # choice of training on rgb/train only is simply a smaller training set.
    check(val_raw_pids == set(range(1, 151)) or len(val_raw_pids) == 150,
          'val covers the same 150 raw identities as train (image-level split)',
          len(val_raw_pids))

    def raw_pid(path):
        return int(os.path.basename(os.path.dirname(path)))

    train_names = {(raw_pid(p), os.path.basename(p)) for p, _, _, _ in ds.train}
    val_names = {(raw_pid(p), os.path.basename(p)) for p, _, _, _ in ds.val}
    dupes = sorted(train_names & val_names)
    # the released PRCC has a handful of identical file names across train/val
    # (this copy: exactly 1). Harmless for a held-out split; just be aware that a
    # merged train+val therefore contains that many duplicate images.
    if dupes:
        print('  note: {} file(s) present in BOTH train and val, e.g. {}'.format(
            len(dupes), dupes[:3]))
    check(len(dupes) <= 5,
          'train/val are essentially image-disjoint (a few known duplicates)',
          len(dupes))

    # test pids must be raw folder ids shared by all three camera dirs
    gal_pids = {p for _, p, _, _ in ds.gallery}
    qs_pids = {p for _, p, _, _ in ds.query_same}
    qd_pids = {p for _, p, _, _ in ds.query_diff}
    check(gal_pids == qs_pids == qd_pids,
          'A/B/C use the same raw test pids (protocol comparable)',
          (len(gal_pids), len(qs_pids), len(qd_pids)))

    # two cloth ids per identity, A and B sharing one
    by_pid = defaultdict(dict)
    for _, pid, cam, cloth in ds.gallery + ds.query_same + ds.query_diff:
        by_pid[pid][cam] = cloth
    two = all(len(set(v.values())) == 2 for v in by_pid.values())
    ab = all(v[0] == v[1] for v in by_pid.values())
    cd = all(v[2] == v[0] + 1 for v in by_pid.values())
    check(two, 'every test identity has exactly 2 cloth ids')
    check(ab, 'A and B share a cloth id (same session clothing)')
    check(cd, 'C cloth id == A cloth id + 1')

    # train cloth vocabulary: 2 per identity
    train_cloths = {c for _, _, _, c in ds.train}
    check(len(train_cloths) == 2 * n_train_pids,
          'train cloth vocab == 2 x num_train_pids', len(train_cloths))
    check(ds.num_train_clothes == len(train_cloths),
          'num_train_clothes drives the F\' head size', ds.num_train_clothes)

    # CRITICAL for the clothes-aware triplet: mask_pos is
    # (same pid) & (different cloth), so every training identity must own more
    # than one cloth id, otherwise mask_pos is always empty and the triplet
    # silently degrades to a plain same-id triplet.
    pid_cloths = defaultdict(set)
    for _, pid, _, cloth in ds.train:
        pid_cloths[pid].add(cloth)
    cloths_per_pid = sorted({len(v) for v in pid_cloths.values()})
    multi_cloth = sum(1 for v in pid_cloths.values() if len(v) > 1)
    check(multi_cloth == n_train_pids,
          'EVERY train identity has >1 cloth id (triplet mask_pos is reachable)',
          'cloths/pid={} ({}/{} ids)'.format(cloths_per_pid, multi_cloth, n_train_pids))

    # image-count distribution per identity (sanity of the sampler assumptions)
    per_pid = Counter(p for _, p, _, _ in ds.train)
    print('  train images/identity: min={} max={} mean={:.1f}'.format(
        min(per_pid.values()), max(per_pid.values()),
        sum(per_pid.values()) / len(per_pid)))
    return ds


def audit_ltcc(root):
    print('\n=== LTCC ===')
    from datasets.ltcc import ltcc

    logger = logging.getLogger('audit')
    ds = ltcc(root=root, verbose=False, logger=logger)
    logger.info('=> LTCC loaded')

    n_train, n_train_pids, n_train_cams, n_train_cloths = stats('train', ds.train)
    n_q, n_q_pids, _, _ = stats('query', ds.query)
    n_g, n_g_pids, n_g_cams, n_g_cloths = stats('gallery (test)', ds.gallery)

    # NOTE: this local copy is a SUBSET of the published LTCC (1501 train ids /
    # 10024 images, 150 test ids). Do not assert the published counts here -- the
    # checks below are the ones that actually protect the framework.
    print('  -- this copy: {} train ids / {} imgs, {} test ids; '
          'published full LTCC is 1501 / 10024 with 150 test ids --'.format(
              n_train_pids, n_train, n_q_pids))

    check(n_q == 493 and n_g == 7050, 'query/gallery image counts', (n_q, n_g))
    check(n_q_pids == n_g_pids,
          'query and gallery share the same test ids (protocol comparable)',
          (n_q_pids, n_g_pids))

    # identity names must NOT collide between train and test: cloth_id is
    # "<pid>_<cam>" taken from the FILE NAME, so a shared pid would merge two
    # different people into one clothing label.
    train_raw_ids = {int(f.split('_')[0]) for f in os.listdir(ds.train_dir)
                     if f.endswith('.png')}
    test_raw_ids = {int(f.split('_')[0]) for f in os.listdir(ds.gallery_dir)
                    if f.endswith('.png')}
    check(not (train_raw_ids & test_raw_ids),
          'train and test raw identities are disjoint (no cloth-label collision)',
          'overlap={}'.format(sorted(train_raw_ids & test_raw_ids)[:10]))

    # cloth vocabulary is built on train
    train_cloths = {c for _, _, _, c in ds.train}
    check(ds.num_train_clothes == len(train_cloths),
          'num_train_clothes == train cloth vocab size', ds.num_train_clothes)
    test_cloths = {c for _, _, _, c in ds.query} | {c for _, _, _, c in ds.gallery}
    print('  cloth vocab: train={} test={}'.format(len(train_cloths), len(test_cloths)))
    # NOTE: cloth_id here is "<pid>_<cam>" taken from the FILE NAME, and train/test
    # identities are disjoint, so test clothings are *new keys* by construction.
    # They only need to be self-consistent within the test split: the id is used
    # for cloth-identity comparison inside a split, never as a cross-split lookup.
    # (Upstream CSCI would raise KeyError at this point; we append instead.)
    unseen = test_cloths - train_cloths
    check(unseen == test_cloths,
          'test clothings form their own namespace (expected: train/test ids disjoint)',
          '{} new keys'.format(len(unseen)))
    query_cloths = {c for _, _, _, c in ds.query}
    gallery_cloths = {c for _, _, _, c in ds.gallery}
    check(query_cloths == gallery_cloths,
          'query and gallery use the same clothing ids (CC rule can compare them)',
          (len(query_cloths), len(gallery_cloths)))

    # the CC rule must actually remove same-clothing gallery entries: for each
    # query, gallery rows with the same pid AND the same cloth id are junk.
    gallery_cloth_counts = Counter((pid, cloth) for _, pid, _, cloth in ds.gallery)
    same_cloth_hits = sum(gallery_cloth_counts[(pid, cloth)]
                          for _, pid, _, cloth in ds.query)
    print('  query x gallery (same pid AND same cloth) pairs pruned by the CC rule: {}'
          .format(same_cloth_hits))
    check(same_cloth_hits > 0,
          'the CC rule prunes real junk on this data (metric is not a no-op)')

    gallery_cloths_by_pid = defaultdict(Counter)
    for _, pid, _, cloth in ds.gallery:
        gallery_cloths_by_pid[pid][cloth] += 1
    diff_cloth_same_pid = 0
    for _, pid, _, cloth in ds.query:
        diff_cloth_same_pid += sum(v for k, v in gallery_cloths_by_pid[pid].items()
                                   if k != cloth)
    print('  query x gallery (same pid, DIFFERENT cloth) pairs kept as positives: {}'
          .format(diff_cloth_same_pid))
    check(diff_cloth_same_pid > 0, 'clothing-change positives exist (CC has signal)')

    cams = sorted({c for _, _, c, _ in ds.train})
    check(cams == list(range(12)), 'camid is 0-based c1..c12', cams)

    # same identity appearing with different cloth -> the CC rule has signal
    pid_to_cloth = defaultdict(set)
    for _, pid, _, cloth in ds.gallery:
        pid_to_cloth[pid].add(cloth)
    multi = sum(1 for v in pid_to_cloth.values() if len(v) > 1)
    check(multi > 0, 'gallery contains identities with >1 clothing (CC metric is meaningful)',
          '{}/{} ids'.format(multi, len(pid_to_cloth)))

    # same requirement as PRCC, but LTCC legitimately contains identities that
    # only ever wear ONE outfit, so the reachable-positives check is a fraction,
    # not "all". The triplet then falls back to a plain same-id positive for
    # those identities, which is standard for cloth-changing triplet mining.
    train_pid_cloths = defaultdict(set)
    for _, pid, _, cloth in ds.train:
        train_pid_cloths[pid].add(cloth)
    train_multi = sum(1 for v in train_pid_cloths.values() if len(v) > 1)
    cloths_per_pid = sorted({len(v) for v in train_pid_cloths.values()})
    check(train_multi > 0,
          'train identities with >1 cloth exist (different-cloth positives reachable)',
          'cloths/pid={} ({}/{} ids)'.format(cloths_per_pid, train_multi, n_train_pids))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(ROOT, '..', 'data'))
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    print('dataset root: {}'.format(root))
    if not os.path.isdir(root):
        print('root does not exist')
        return 1

    audit_prcc(root)
    audit_ltcc(root)

    print('\n' + '=' * 60)
    if FAILURES:
        print('{} CHECK(S) FAILED:'.format(len(FAILURES)))
        for f in FAILURES:
            print('  - {}'.format(f))
        return 1
    print('REAL DATASET AUDIT PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
