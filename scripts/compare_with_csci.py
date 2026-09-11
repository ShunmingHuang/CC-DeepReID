"""Prove the dataset layer is identical to ICCV-CSCI-Person-ReID, on real data.

This does NOT call CC-DeepReID's dataset classes to decide what is correct.
It re-derives every split, every pid and every cloth id **directly from the file
system using CSCI's own formulas** (transcribed below from
``ICCV-CSCI-Person-ReID/data/datasets/{prcc,ltcc}.py``), then compares that
independent result against what our loaders produce.

    python scripts/compare_with_csci.py [--root ../data]

Checked per split:
  * exact set of image paths
  * pid of every image
  * camid of every image
  * cloth_id **partition** (which images share a cloth id) and the exact label
    numbering for the splits where CSCI's formula pins it down

Known, deliberate deviations (reported, not failures):
  * LTCC test: CSCI does ``clothes2label[cloth]`` on a train-built vocabulary and
    raises KeyError for unseen keys; we append them instead.
  * PRCC val: CSCI parses it by reusing ``_process_dir_train``; we use a separate
    reader, and CSCI never uses ``self.val`` anyway.
"""

import argparse
import glob
import os
import os.path as osp
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILURES = []


def check(cond, label, extra=''):
    print('  [{}] {}{}'.format('OK  ' if cond else 'FAIL', label,
                               ' -> ' + str(extra) if extra else ''))
    if not cond:
        FAILURES.append(label)


# --------------------------------------------------------------------------- #
# CSCI's formulas, transcribed
# --------------------------------------------------------------------------- #
CAM2LABEL = {'A': 0, 'B': 1, 'C': 2}


def csci_prcc_train(train_dir):
    """CSCI prcc.py:96-158 (train = rgb/train)."""
    pdirs = sorted(glob.glob(osp.join(train_dir, '*')))

    pids, clothes = set(), set()
    for pdir in pdirs:
        pids.add(int(osp.basename(pdir)))
        for img in glob.glob(osp.join(pdir, '*.jpg')):
            cam = osp.basename(img)[0]
            if cam in ['A', 'B']:
                clothes.add(osp.basename(pdir))
            else:
                clothes.add(osp.basename(pdir) + osp.basename(img)[0])

    pid2label = {p: l for l, p in enumerate(sorted(pids))}
    clothes2label = {c: l for l, c in enumerate(sorted(clothes))}

    items = []
    for pdir in pdirs:
        pid = int(osp.basename(pdir))
        for img in glob.glob(osp.join(pdir, '*.jpg')):
            cam = osp.basename(img)[0]
            cloth = (osp.basename(pdir) if cam in ['A', 'B']
                     else osp.basename(pdir) + osp.basename(img)[0])
            items.append((img, pid2label[pid], CAM2LABEL[cam], clothes2label[cloth]))
    return items, len(pid2label), len(clothes2label)


def csci_prcc_test(test_dir):
    """CSCI prcc.py:160-245 (test = rgb/test)."""
    pids = sorted({int(osp.basename(d))
                   for d in glob.glob(osp.join(test_dir, 'A', '*'))})
    pid2label = {p: l for l, p in enumerate(pids)}

    gallery, q_same, q_diff = [], [], []
    for cam in ['A', 'B', 'C']:
        for pdir in glob.glob(osp.join(test_dir, cam, '*')):
            pid = int(osp.basename(pdir))
            if cam == 'A' or cam == 'B':
                clothes_id = pid2label[pid] * 2
            else:
                clothes_id = pid2label[pid] * 2 + 1
            target = gallery if cam == 'A' else (q_same if cam == 'B' else q_diff)
            for img in glob.glob(osp.join(pdir, '*.jpg')):
                # CSCI keeps the RAW pid in the tuple
                target.append((img, pid, CAM2LABEL[cam], clothes_id))
    return gallery, q_same, q_diff, len(pid2label)


def _ltcc_parse(img_paths):
    """CSCI ltcc.py:129-130 regexes, applied to the BASENAME (as CSCI does)."""
    import re
    p1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
    p2 = re.compile(r'(\w+)_c')
    out = []
    for p in img_paths:
        pid, _, camid = map(int, p1.search(osp.basename(p)).groups())
        cloth = p2.search(osp.basename(p)).group(1)
        out.append((p, pid, camid - 1, cloth))
    return out


def csci_ltcc(ltcc_dir):
    """CSCI ltcc.py:78-185 -- note the test-time KeyError on unseen clothings."""
    train_dir = osp.join(ltcc_dir, 'train')
    query_dir = osp.join(ltcc_dir, 'query')
    gallery_dir = osp.join(ltcc_dir, 'test')

    tr = _ltcc_parse(sorted(glob.glob(osp.join(train_dir, '*.png'))))
    pid2label = {p: l for l, p in enumerate(sorted({x[1] for x in tr}))}
    clothes2label = {c: l for l, c in enumerate(sorted({x[3] for x in tr}))}
    train = [(p, pid2label[pid], cam, clothes2label[cl]) for p, pid, cam, cl in tr]

    te = _ltcc_parse(sorted(glob.glob(osp.join(query_dir, '*.png'))))
    ga = _ltcc_parse(sorted(glob.glob(osp.join(gallery_dir, '*.png'))))
    unseen = {c for _, _, _, c in te + ga} - set(clothes2label)

    # CSCI does `clothes2label[cloth]` and raises KeyError on an unseen key.
    # For an equivalence comparison we instead assign each unseen key its own
    # label, which is what a KeyError-free run would have to do anyway.
    relabel = dict(clothes2label)
    for c in sorted(unseen):
        relabel[c] = len(relabel)

    def conv(items):
        out = []
        for p, pid, cam, cl in items:
            out.append((p, pid, cam, relabel[cl]))
        return out

    return train, conv(te), conv(ga), unseen


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def key(path):
    return osp.normpath(osp.abspath(path)).lower()


def by_path(items):
    return {key(p): (p, pid, cam, cl) for p, pid, cam, cl in items}


def partition(items):
    """cloth_id equivalence classes: which images share a cloth label."""
    groups = defaultdict(set)
    for p, _, _, cl in items:
        groups[cl].add(key(p))
    return sorted(sorted(v) for v in groups.values())


def compare(name, ours, ref, compare_cloth_labels=True):
    ours_map, ref_map = by_path(ours), by_path(ref)
    check(set(ours_map) == set(ref_map),
          '{}: identical image set'.format(name),
          'ours={} ref={} sym-diff={}'.format(
              len(ours_map), len(ref_map),
              len(set(ours_map) ^ set(ref_map))))

    common = set(ours_map) & set(ref_map)
    pid_bad = [p for p in common if ours_map[p][1] != ref_map[p][1]]
    cam_bad = [p for p in common if ours_map[p][2] != ref_map[p][2]]
    check(not pid_bad, '{}: pid identical for every image'.format(name), pid_bad[:3])
    check(not cam_bad, '{}: camid identical for every image'.format(name), cam_bad[:3])

    ours_part, ref_part = partition(ours), partition(ref)
    same_partition = ours_part == ref_part
    check(same_partition,
          '{}: cloth_id partition identical (same images grouped together)'.format(name),
          '' if same_partition else 'ours has {} groups, ref has {} groups'.format(
              len(ours_part), len(ref_part)))

    if compare_cloth_labels:
        label_bad = [p for p in common if ours_map[p][3] != ref_map[p][3]]
        check(not label_bad,
              '{}: cloth_id LABEL identical for every image'.format(name),
              [(p, ours_map[p][3], ref_map[p][3]) for p in label_bad[:3]])

    return ours_part, ref_part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(ROOT, '..', 'data'))
    args = ap.parse_args()
    root = osp.abspath(args.root)
    print('dataset root: {}'.format(root))

    from datasets.prcc import prcc
    from datasets.ltcc import ltcc

    # ---------------------------- PRCC ---------------------------- #
    print('\n=== PRCC: ours vs CSCI formulas ===')
    prcc_dir = osp.join(root, 'prcc', 'rgb')
    ds = prcc(root=root, verbose=False)

    ref_tr, ref_tr_pids, ref_tr_cloths = csci_prcc_train(osp.join(prcc_dir, 'train'))
    compare('PRCC train', ds.train, ref_tr)
    check(ds.num_train_pids == ref_tr_pids,
          'PRCC train: num_train_pids == CSCI', (ds.num_train_pids, ref_tr_pids))
    check(ds.num_train_clothes == ref_tr_cloths,
          'PRCC train: num_train_clothes == CSCI',
          (ds.num_train_clothes, ref_tr_cloths))

    ref_gal, ref_qs, ref_qd, ref_test_pids = csci_prcc_test(osp.join(prcc_dir, 'test'))
    compare('PRCC gallery (A)', ds.gallery, ref_gal)
    compare('PRCC query_same (B)', ds.query_same, ref_qs)
    compare('PRCC query_diff (C)', ds.query_diff, ref_qd)
    check(ds.num_test_pids == ref_test_pids,
          'PRCC test: num_test_pids == CSCI', (ds.num_test_pids, ref_test_pids))

    # ---------------------------- LTCC ---------------------------- #
    print('\n=== LTCC: ours vs CSCI formulas ===')
    ltcc_dir = osp.join(root, 'LTCC_ReID')
    ds = ltcc(root=root, verbose=False)

    ref_tr, ref_q, ref_g, unseen = csci_ltcc(ltcc_dir)
    compare('LTCC train', ds.train, ref_tr)

    if unseen:
        print('  [note] CSCI would raise KeyError here: {} test clothing keys are'
              ' absent from the train-built vocabulary'.format(len(unseen)))
        print('         our loader appends them to a test-only vocabulary instead;')
        print('         the comparison below ignores label NUMBERING and checks the')
        print('         equivalence structure (which images share a clothing)')
        compare('LTCC query', ds.query, ref_q, compare_cloth_labels=False)
        compare('LTCC gallery', ds.gallery, ref_g, compare_cloth_labels=False)
    else:
        # every test clothing was seen in training: then CSCI's numbering is
        # reproducible and we can demand an exact label match
        compare('LTCC query', ds.query, ref_q)
        compare('LTCC gallery', ds.gallery, ref_g)

    print('\n' + '=' * 62)
    if FAILURES:
        print('{} CHECK(S) FAILED:'.format(len(FAILURES)))
        for f in FAILURES:
            print('  - {}'.format(f))
        return 1
    print('DATASET LAYER IS IDENTICAL TO CSCI (on this data)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
