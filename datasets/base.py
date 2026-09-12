import glob
import os.path as osp

from PIL import Image, ImageFile

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path, grey_scale=None):
    """Keep reading image until succeed.

    This can avoid IOError incurred by heavy IO process.
    Aligned with ICCV-CSCI-Person-ReID/data/dataset_loader.py::read_image.
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    if grey_scale:
        img = img.convert('L').convert(mode='RGB')
    return img


def build_pid2label(pids):
    """num_pids/debug friendly relabel: sorted unique pids -> 0..N-1."""
    pid_container = sorted(set(pids))
    return {pid: label for label, pid in enumerate(pid_container)}


def build_vocab(keys):
    """sorted unique keys -> 0..N-1, used for clothing vocabularies."""
    return {key: label for label, key in enumerate(sorted(set(keys)))}


def flat_files(dir_path, exts=('*.jpg', '*.jpeg', '*.png')):
    """Deterministically list files directly inside dir_path (CSCI style: glob + sort)."""
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(osp.join(dir_path, ext)))
    img_paths.sort()
    return img_paths


class BaseImageDataset(object):
    """Base class of image reid dataset.

    Data tuples are kept as CC-DeepReID's 4-tuple
    ``(img_path, pid, camid, cloth_id)`` (CSCI additionally carries an
    ``aux_info`` slot which is a 105-dim all-zero placeholder in practice).
    The *splits* and the *label semantics* follow ICCV-CSCI-Person-ReID.
    """

    def get_imagedata_info(self, data):
        pids, cams, clothes = [], [], []

        for _, pid, camera_id, cloth_id in data:
            pids += [pid]
            cams += [camera_id]
            clothes += [cloth_id]
        pids = set(pids)
        cameras = set(cams)
        clothes = set(clothes)
        num_pids = len(pids)
        num_cameras = len(cameras)
        num_clothes = len(clothes)
        num_imgs = len(data)
        return num_imgs, num_pids, num_cameras, num_clothes

    def print_dataset_statistics(self, train=None, val=None, query_same=None,
                                 query_diff=None, gallery=None, logger=None,
                                 query=None, rows=None):
        if rows is None:
            rows = [
                ('train', train),
                ('val', val),
                ('query(same)', query_same),
                ('query(diff)', query_diff),
                ('query', query),
                ('gallery', gallery),
            ]
        lines = []
        lines.append("Dataset statistics:")
        lines.append("  ----------------------------------------------------------")
        lines.append("  subset       | # ids | # images | # cameras | # clothes")
        lines.append("  ----------------------------------------------------------")
        for name, subset in rows:
            if subset is None:
                continue
            num_imgs, num_pids, num_cams, num_clos = self.get_imagedata_info(subset)
            lines.append("  {:12s} | {:5d} | {:8d} | {:9d} | {:9d}".format(
                name, num_pids, num_imgs, num_cams, num_clos))
        lines.append("  ----------------------------------------------------------")

        if logger is not None:
            for line in lines:
                logger.info(line)
        else:
            for line in lines:
                print(line)

    @staticmethod
    def info_tuple(data):
        """(num_imgs, num_pids, num_cams, num_clothes) without instantiating a dataset."""
        pids = {pid for _, pid, _, _ in data}
        cams = {cid for _, _, cid, _ in data}
        clothes = {clo for _, _, _, clo in data}
        return len(data), len(pids), len(cams), len(clothes)


class ImageDataset(Dataset):
    """Wraps a list of ``(img_path, pid, cid, cloth_id)`` tuples.

    When ``hist_extractor`` is given, the colour-histogram label (CSCI's
    annotation-free supervision for the identity-independent branch) is computed
    from the *same augmented tensor* that feeds the network, exactly like
    ``ImageDataset_fixes.w_color`` in CSCI. It comes out as a 5th element, so the
    default 4-tuple collate still works when it is disabled.
    """

    def __init__(self, dataset, transform=None, hist_extractor=None):
        self.dataset = dataset
        self.transform = transform
        self.hist_extractor = hist_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, cid, cloth_id = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.hist_extractor is not None:
            # the extractor only supports batch 1 (see datasets/histogram.py)
            hist = self.hist_extractor(img.unsqueeze(0))[0]
            return img, pid, cid, cloth_id, osp.basename(img_path), hist

        return img, pid, cid, cloth_id, osp.basename(img_path)
