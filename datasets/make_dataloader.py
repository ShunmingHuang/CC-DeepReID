import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .base import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .ltcc import ltcc
from .prcc import prcc

__factory = {
    "ltcc": ltcc,
    "prcc": prcc
}


class DatasetBundle(object):
    """Everything the processor needs, mirroring CSCI's ``dataset`` object.

    ``train`` / ``query_diff`` / ``query_same`` / ``gallery`` hold the raw tuples
    and the ``num_*`` counters are the exact values the evaluators must use.
    """

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

        self.train = dataset.train
        self.val = getattr(dataset, 'val', None)
        self.query_diff = getattr(dataset, 'query_diff', None)
        self.query_same = getattr(dataset, 'query_same', None)
        self.gallery = dataset.gallery

        # LTCC style: a single query set
        self.query = getattr(dataset, 'query', None)

        self.num_train_pids = dataset.num_train_pids
        self.num_train_imgs = dataset.num_train_imgs
        self.num_train_clothes = dataset.num_train_clothes
        self.num_train_cams = dataset.get_imagedata_info(dataset.train)[2]

        self.num_query_imgs_diff = getattr(dataset, 'num_query_imgs_diff', 0)
        self.num_query_imgs_same = getattr(dataset, 'num_query_imgs_same', 0)
        self.num_query_imgs = getattr(dataset, 'num_query_imgs', len(self.query or []))
        self.num_gallery_imgs = len(self.gallery)

        # (num_train_pids, num_train_clothes) binary mask: 1 where a training
        # identity owns a clothing class. This is the positive mask C2R-ReID's
        # clothes-based adversarial loss needs; it is indexed by the *relabelled*
        # pid that the train tuples carry.
        self.pid2clothes = getattr(dataset, 'pid2clothes', None)

    def __repr__(self):
        return ("DatasetBundle(name={}, train={}, query_diff={}, query_same={}, "
                "query={}, gallery={})".format(
                    self.name, len(self.train), self.num_query_imgs_diff,
                    self.num_query_imgs_same,
                    None if self.query is None else len(self.query),
                    len(self.gallery)))


def build_transforms(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    return train_transforms, val_transforms


def _make_val_loader(dataset, transform, cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    return DataLoader(
        ImageDataset(dataset, transform),
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=val_collate_fn,
    )


def make_dataloader(cfg):
    """Build every loader the processor needs.

    Mirrors ICCV-CSCI-Person-ReID's ``build_dataloader`` layout:

    * PRCC  -> ``query_diff`` (test/C, clothes-changing) and ``query_same``
      (test/B, standard), both sharing ``gallery`` = test/A.
    * LTCC  -> a single ``query`` with ``gallery`` = test.

    Returns
    -------
    loaders : dict
        ``{'train', 'query_diff', 'query_same'}`` (+ ``'query'`` for LTCC).
        Each value is ``(loader, num_query, dataset)``; ``num_query`` is ``None``
        for the training loader.
    bundle : DatasetBundle
        Split bookkeeping (counts, pids, raw tuples).
    """
    train_transforms, val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    bundle = DatasetBundle(cfg.DATASETS.NAMES, dataset)

    # ---------------- train ----------------
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'triplet':
        print('using triplet sampler')
        train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None,
                collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=train_collate_fn
        )
    else:
        raise ValueError('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.DATALOADER.SAMPLER))

    # ---------------- test ----------------
    loaders = {'train': (train_loader, None, dataset)}

    def _val_loader(subset, num_query):
        """query first, then gallery -- the evaluators split on ``num_query``."""
        combined = list(subset) + list(dataset.gallery)
        return _make_val_loader(combined, val_transforms, cfg), num_query, combined

    if bundle.query_diff is not None:
        # PRCC: two protocols over the same gallery
        loaders['query_diff'] = _val_loader(bundle.query_diff, bundle.num_query_imgs_diff)
        loaders['query_same'] = _val_loader(bundle.query_same, bundle.num_query_imgs_same)
    else:
        loaders['query_diff'] = _val_loader(bundle.query, bundle.num_query_imgs)

    loaders['gallery'] = (
        _make_val_loader(dataset.gallery, val_transforms, cfg),
        None,
        dataset.gallery,
    )

    return loaders, bundle


def train_collate_fn(batch):
    imgs, pids, cams, clothes, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    cams = torch.tensor(cams, dtype=torch.int64)
    clothes = torch.tensor(clothes, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, cams, clothes


def val_collate_fn(batch):
    imgs, pids, cams, clothes, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids_batch = torch.tensor(cams, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, cams, clothes, camids_batch, img_paths
