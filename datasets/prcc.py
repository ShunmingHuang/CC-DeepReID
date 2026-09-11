import glob
import os.path as osp

from .base import (BaseImageDataset, build_pid2label, build_vocab,
                   flat_files)

CAM2LABEL = {'A': 0, 'B': 1, 'C': 2}


class prcc(BaseImageDataset):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view

    Directory layout (same as ICCV-CSCI-Person-ReID)::

        prcc/rgb/train/<pid>/<A|B|C>_cropped_rgb<xxx>.jpg
        prcc/rgb/val/<pid>/<A|B|C>_cropped_rgb<xxx>.jpg
        prcc/rgb/test/{A,B,C}/<pid>/cropped_rgb<xxx>.jpg

    Protocol (aligned with ICCV-CSCI-Person-ReID/data/datasets/prcc.py):

    * ``train``        : rgb/train only, pids relabelled to 0..N-1 (val is NOT merged).
    * ``val``          : rgb/val, parsed and reported but not used for training.
    * ``query_same``   : rgb/test/B  -> Standard / SC protocol query.
    * ``query_diff``   : rgb/test/C  -> Clothes-Changing (CC) protocol query.
    * ``gallery``      : rgb/test/A  -> single gallery shared by both protocols.

    Label semantics (aligned with CSCI):

    * pid   : raw folder id everywhere except ``train``, which is relabelled.
    * cloth : ``<pid>`` for camera A/B and ``<pid>C`` for camera C, i.e. *two
      cloth labels per identity*, then mapped through one train-built
      vocabulary (``pid2cloth``). This keeps CC-DeepReID's cloth-aware triplet
      loss meaningful while matching CSCI's ``pid * 2`` / ``pid * 2 + 1``
      two-cloth-per-identity scheme.

    Cloth id convention used by the reported metrics (single consistent
    convention, documented here because CSCI conflates "cloth identity" and
    "clothing-change")::

        cloth_id = <pid>   -> same clothing as the A-camera reference view of that identity
        cloth_id = <pid>C  -> clothing changed away from that reference view

    so the pruning rule ``same pid AND same cloth_id`` removes A-camera
    gallery entries and keeps precisely the clothing-change pairs:
    A vs C for the CC protocol, and A vs B for the Standard protocol.
    """

    dataset_dir = "prcc"

    def __init__(self, root='', verbose=True, pid_begin=0, logger=None):
        super().__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "rgb/train")
        self.val_dir = osp.join(self.dataset_dir, "rgb/val")
        self.test_dir = osp.join(self.dataset_dir, "rgb/test")
        self.pid_begin = pid_begin
        self._check_before_run()

        # ---- train (CSCI: rgb/train only, own relabel + own clothing vocab) ----
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        # ---- val (CSCI parses it the same way; reported, never trained on) ----
        val, num_val_pids, num_val_imgs, num_val_clothes = \
            self._process_dir_val(self.val_dir)

        query_same, query_diff, gallery, num_test_pids, num_query_imgs_same, \
            num_query_imgs_diff, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.test_dir)

        self.train = train
        self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.query = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_imgs = num_train_imgs
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

        self.num_val_pids = num_val_pids
        self.num_val_imgs = num_val_imgs
        self.num_val_clothes = num_val_clothes

        self.num_test_pids = num_test_pids
        self.num_query_imgs_same = num_query_imgs_same
        self.num_query_imgs_diff = num_query_imgs_diff
        self.num_gallery_imgs = num_gallery_imgs
        self.num_test_clothes = num_test_clothes

        if verbose:
            self.print_dataset_statistics(
                train=self.train, val=self.val, query_same=self.query_same,
                query_diff=self.query_diff, gallery=self.gallery, logger=logger)

    # ------------------------------------------------------------------ #
    # splits
    # ------------------------------------------------------------------ #
    def _person_dirs(self, dir_path):
        pdirs = [d for d in glob.glob(osp.join(dir_path, '*')) if osp.isdir(d)]
        pdirs.sort()
        return pdirs

    def _cloth_key(self, raw_pid, cam):
        """Clothing key, transcribed from ICCV-CSCI-Person-ReID prcc.py:108-111.

        CSCI builds its clothing vocabulary from the raw folder name and the
        camera letter::

            if cam in ['A', 'B']:
                clothes_container.add(osp.basename(pdir))                            # "<pid>"
            else:
                clothes_container.add(osp.basename(pdir) + osp.basename(img_dir)[0])  # "<pid>C"

        ``raw_pid`` MUST be the folder name as a string, leading zeros included
        ("092", not "92"): CSCI adds ``osp.basename(pdir)`` verbatim and then sorts
        the key set, so the string form determines the label numbering. Coercing
        through int() changes the keys and therefore every cloth label.
        """
        raw_pid = str(raw_pid)
        if cam in ['A', 'B']:
            return raw_pid
        return raw_pid + cam

    def _read_train_like(self, dir_path, relabel, pid2label, cloth2label):
        """rgb/train and rgb/val share the <pid>/<cam>_*.jpg structure."""
        dataset = []
        for pdir in self._person_dirs(dir_path):
            raw_pid = osp.basename(pdir)
            pid = int(raw_pid)
            for img_path in flat_files(pdir):
                cam = osp.basename(img_path)[0]
                if cam not in CAM2LABEL:
                    continue
                cloth_key = self._cloth_key(raw_pid, cam)
                cloth2label.setdefault(cloth_key, len(cloth2label))
                label = pid2label[pid] if relabel else pid
                dataset.append((
                    img_path,
                    self.pid_begin + label,
                    CAM2LABEL[cam],
                    cloth2label[cloth_key],
                ))
        return dataset

    def _process_dir_train(self, dir_path):
        """Train split: CSCI builds its own relabel map and clothing vocab here."""
        pdirs = self._person_dirs(dir_path)
        train_pids = sorted({int(osp.basename(pdir)) for pdir in pdirs})
        pid2label = build_pid2label(train_pids)

        # clothing vocabulary is built from the training split, exactly like CSCI
        cloth_keys = []
        for pdir in pdirs:
            raw_pid = osp.basename(pdir)
            for img_path in flat_files(pdir):
                cam = osp.basename(img_path)[0]
                if cam in CAM2LABEL:
                    cloth_keys.append(self._cloth_key(raw_pid, cam))
        cloth2label = build_vocab(cloth_keys)

        self.pid2label = pid2label
        self.cloth2label = cloth2label

        dataset = self._read_train_like(dir_path, True, pid2label, cloth2label)

        num_pids = len(pid2label)
        num_clothes = len(cloth2label)
        num_imgs = len(dataset)

        pid2clothes = [[0] * num_clothes for _ in range(num_pids)]
        for _, label, _, cloth_id in dataset:
            pid2clothes[label][cloth_id] = 1

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_val(self, dir_path):
        """Val split (statistics only, CSCI parses it the same way)."""
        pdirs = self._person_dirs(dir_path)
        val_pids = sorted({int(osp.basename(pdir)) for pdir in pdirs})
        pid2label = build_pid2label(val_pids)

        # reuse the train vocabulary when available so ids stay comparable
        cloth2label = dict(getattr(self, 'cloth2label', {}))
        dataset = self._read_train_like(dir_path, False, pid2label, cloth2label)

        num_pids = len(pid2label)
        num_clothes = len({cloth for _, _, _, cloth in dataset})
        return dataset, num_pids, len(dataset), num_clothes

    def _process_dir_test(self, test_path):
        """Test split, transcribed from CSCI's ``_process_dir_test``.

        * pids: enumerated from ``test/A`` -> ``pid2label = {pid: label}``;
          the tuples keep the RAW pid, exactly like CSCI (which even keeps the
          misleading ``if cam == 'A': clothes_id = pid2label[pid]*2`` naming).
        * cloth ids: ``pid2label[pid] * 2`` for A and B, ``pid2label[pid] * 2 + 1``
          for C (CSCI prcc.py:197-217). CSCI never materialises B's cloth id, but
          its own equivalence test at line 210 is ``cam == 'A'``, so B shares A's
          cloth id -- which is what we assign here, making the Standard protocol
          well-defined as well.
        """
        pid_container = sorted({
            int(osp.basename(pdir))
            for pdir in self._person_dirs(osp.join(test_path, 'A'))
        })
        pid2label = build_pid2label(pid_container)
        num_pids = len(pid2label)

        query_same, query_diff, gallery = [], [], []
        for cam in ['A', 'B', 'C']:
            for pdir in self._person_dirs(osp.join(test_path, cam)):
                pid = int(osp.basename(pdir))
                if cam == 'A' or cam == 'B':
                    clothes_id = pid2label[pid] * 2
                else:
                    clothes_id = pid2label[pid] * 2 + 1

                if cam == 'A':
                    builder = gallery
                elif cam == 'B':
                    builder = query_same
                else:
                    builder = query_diff

                for img_path in flat_files(pdir):
                    builder.append((img_path, pid, CAM2LABEL[cam], clothes_id))

        num_query_imgs_same = len(query_same)
        num_query_imgs_diff = len(query_diff)
        num_gallery_imgs = len(gallery)
        num_clothes = len(pid2label) * 2

        return query_same, query_diff, gallery, num_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, num_clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
