import os.path as osp
import re

from .base import (BaseImageDataset, build_pid2label, build_vocab,
                   flat_files)


class ltcc(BaseImageDataset):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#

    Directory layout (same as ICCV-CSCI-Person-ReID)::

        LTCC_ReID/train/<pid>_<cam>_c<cloth>_<frame>.png
        LTCC_ReID/query/<pid>_<cam>_c<cloth>_<frame>.png
        LTCC_ReID/test/<pid>_<cam>_c<cloth>_<frame>.png

    Protocol (aligned with ICCV-CSCI-Person-ReID/data/datasets/ltcc.py):

    * Every split is loaded in full; **no ID list filtering at read time**
      (CC-DeepReID used to prune ``info/cloth-change_id_{train,test}.txt`` here).
    * ``pid``   : relabelled to 0..N-1 for ``train``, raw directory id for
      ``query``/``gallery`` (both share the same raw pid space).
    * ``camid`` : ``c<id> - 1`` (0-based).
    * ``cloth`` : ``<pid>_<cam>`` key mapped through one ``pid2cloth`` vocabulary
      built from the *training* split (CSCI does exactly this).

    The clothes-changing rule is therefore applied at **evaluation** time
    (``utils/metrics.py::R1_mAP_eval_LTCC``), not by discarding data, and both
    the CC setting and the General setting are reported.
    """

    dataset_dir = "LTCC_ReID"

    def __init__(self, root='', verbose=True, pid_begin=0, logger=None):
        super().__init__()
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "test")
        self.pid_begin = pid_begin
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, \
            num_test_clothes = self._process_dir_test(self.query_dir, self.gallery_dir)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_imgs = num_train_imgs
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

        self.num_test_pids = num_test_pids
        self.num_query_imgs = num_query_imgs
        self.num_gallery_imgs = num_gallery_imgs
        self.num_test_clothes = num_test_clothes

        if verbose:
            self.print_dataset_statistics(
                rows=[('train', self.train), ('query', self.query),
                      ('gallery', self.gallery)],
                logger=logger)
            msg = "=> LTCC loaded (full train set, CC/General rule applied at eval time)"
            logger.info(msg) if logger is not None else print(msg)

    # ------------------------------------------------------------------ #
    # splits
    # ------------------------------------------------------------------ #
    def _parse(self, img_paths):
        """Return [(img_path, pid, camid, cloth_key)] using CSCI's regexes."""
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')
        parsed = []
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(osp.basename(img_path)).groups())
            cloth = pattern2.search(osp.basename(img_path)).group(1)
            parsed.append((img_path, pid, camid - 1, cloth))
        return parsed

    def _process_dir_train(self, dir_path):
        img_paths = flat_files(dir_path)
        parsed = self._parse(img_paths)

        pid2label = build_pid2label([pid for _, pid, _, _ in parsed])
        cloth2label = build_vocab([cloth for _, _, _, cloth in parsed])

        self.pid2label = pid2label
        self.cloth2label = cloth2label

        num_pids = len(pid2label)
        num_clothes = len(cloth2label)

        dataset = []
        pid2clothes = [[0] * num_clothes for _ in range(num_pids)]
        for img_path, pid, camid, cloth in parsed:
            label = pid2label[pid]
            cloth_id = cloth2label[cloth]
            dataset.append((img_path, self.pid_begin + label, camid, cloth_id))
            pid2clothes[label][cloth_id] = 1

        return dataset, num_pids, len(dataset), num_clothes, pid2clothes

    def _process_dir_test(self, query_path, gallery_path):
        query_parsed = self._parse(flat_files(query_path))
        gallery_parsed = self._parse(flat_files(gallery_path))

        # CSCI: test pids stay raw and are shared by query and gallery
        pid_container = sorted({pid for _, pid, _, _ in query_parsed + gallery_parsed})
        num_pids = len(pid_container)

        # reuse the training clothing vocabulary (CSCI builds clothes2label on train).
        # The copy keeps unseen test clothings from leaking into the train vocab.
        cloth2label = dict(getattr(self, 'cloth2label', {}))

        def _build(parsed):
            dataset = []
            for img_path, pid, camid, cloth in parsed:
                cloth2label.setdefault(cloth, len(cloth2label))
                dataset.append((img_path, pid, camid, cloth2label[cloth]))
            return dataset

        query = _build(query_parsed)
        gallery = _build(gallery_parsed)

        num_clothes = len({cloth for _, _, _, cloth in query + gallery})
        return query, gallery, num_pids, len(query), len(gallery), num_clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
