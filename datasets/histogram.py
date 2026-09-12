"""RGB-uv colour histogram feature.

Transcribed from ``ICCV-CSCI-Person-ReID/data/rgbuc.py`` (itself from HistoGAN's
``RGBuvHistBlock``). The histogram is the *supervision target* CSCI regresses its
colour token against -- it needs no annotation, it is computed from the image.

Colour space: ``u = log R - log G``, ``v = log R - log B`` (and the (g,r)/(g,b)
pair), i.e. chroma coordinates that are insensitive to overall brightness. Bins
are filled with an *inverse-quadratic* kernel rather than hard counting, so the
operator is differentiable.

IMPORTANT: like CSCI's implementation, ``forward`` only fills sample 0 of the
batch (see the ``X = X[0]`` in the original). CSCI calls it from the dataloader
on a single image at a time (``histblock(denormalize.unsqueeze(0))``), so this is
correct there. We keep the same signature and **reject batch > 1 loudly** instead
of silently returning zeros for the other samples.
"""

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


class RGBuvHistBlock(nn.Module):
    """Computes the RGB-uv histogram feature of a given image.

    Args:
        h: histogram dimension (bins per axis). Default 32.
        insz: if the input is larger than ``insz`` it is resized to
            ``insz x insz`` before histogramming. Making this fixed means the
            feature is independent of the training resolution.
        method: only ``'inverse-quadratic'`` is implemented (CSCI's default and
            the one its colour profiles use).
        sigma: kernel width for the inverse-quadratic kernel.
        intensity_scale: multiply the counts by the RGB intensity norm.
        hist_boundary: ``[lo, hi]`` range of the u/v axes.
        green_only: keep only the (g,r)/(g,b) plane (single histogram plane).
    """

    def __init__(self, h=32, insz=150, method='inverse-quadratic', sigma=0.001,
                 intensity_scale=False, hist_boundary=None, green_only=False,
                 device='cpu'):
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        # NOTE: ``insz`` is accepted for signature compatibility with CSCI but is
        # NOT used -- CSCI's rgbuc.py never resizes the input (no interpolate call
        # anywhere in that file), so the histogram is computed at the *native*
        # resolution of whatever image it is handed.
        self.insz = insz
        self.method = method
        self.sigma = sigma
        self.intensity_scale = intensity_scale
        self.green_only = green_only
        self.device = device
        if hist_boundary is None:
            hist_boundary = [-3, 3]
        hist_boundary = sorted(hist_boundary)
        self.hist_boundary = hist_boundary
        if method != 'inverse-quadratic':
            raise NotImplementedError(
                "only 'inverse-quadratic' is implemented (got '{}')".format(method))

    @property
    def out_dim(self):
        """Flattened size of one histogram, after the channel mean over planes."""
        return self.h * self.h

    def _kernel(self, delta):
        diff = torch.pow(delta.reshape(-1, self.h), 2) / self.sigma ** 2
        return 1 / (1 + diff)                       # inverse quadratic

    def forward(self, x):
        """x: ``(1, 3, H, W)`` in [0, 1]. Returns ``(1, num_planes, h, h)``."""
        if x.dim() != 4:
            raise ValueError('expected (B, 3, H, W), got {}'.format(tuple(x.shape)))
        if x.shape[0] != 1:
            raise ValueError(
                'RGBuvHistBlock only supports batch size 1 (the original CSCI '
                'implementation fills sample 0 only); call it per sample, got B={}'
                .format(x.shape[0]))

        x = torch.clamp(x, 0, 1)
        I = x[0].reshape(3, -1).t()                 # (H*W, 3)
        II = torch.pow(I, 2)
        if self.intensity_scale:
            Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), 1)
        else:
            Iy = 1

        num_planes = 1 if self.green_only else 3
        hists = torch.zeros((1, num_planes, self.h, self.h), device=x.device)
        grid = torch.unsqueeze(torch.tensor(
            np.linspace(self.hist_boundary[0], self.hist_boundary[1], num=self.h),
            device=x.device), 0)

        def log_ratio(a, b):
            return torch.unsqueeze(torch.log(I[:, a] + EPS) - torch.log(I[:, b] + EPS), 1)

        def fill(idx, ia, ib, ic):
            # u = log(I_ia / I_ib), v = log(I_ia / I_ic) -- transcribed per plane
            # from CSCI (rgbuc.py:50-53, 78-81, 108-111). Note plane 2's v axis is
            # log(I2 / I1), NOT log(I2 / I2): the "second" channel for a plane is
            # the u-axis divisor, i.e. u and v share the same numerator channel.
            du = self._kernel(abs(log_ratio(ia, ib) - grid))
            dv = self._kernel(abs(log_ratio(ia, ic) - grid))
            hists[0, idx, :, :] = torch.mm(torch.t(Iy * du), dv)

        if not self.green_only:
            fill(0, 0, 1, 2)
        fill(0 if self.green_only else 1, 1, 0, 2)
        if not self.green_only:
            fill(2, 2, 0, 1)

        hists = hists / (((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)
        return hists


class HistogramExtractor(nn.Module):
    """Turns an augmented image tensor into CSCI's colour *label*.

    Mirrors ``ImageDataset_fixes.w_color`` (dataset_loader.py:213-235):

        denormalize -> histblock -> mean over planes -> flatten -> normalise -> * wt

    The ``mean over planes`` is what turns the (1, 3, h, h) histogram into the
    ``h*h`` vector CSCI regresses against; ``w_color`` is the default profile.
    """

    def __init__(self, h=32, insz=150, sigma=0.001, intensity_scale=False,
                 norm='l1', norm_p=1, weight=100.0, pixel_mean=None, pixel_std=None,
                 green_only=False, keep_planes=False, device='cpu'):
        super(HistogramExtractor, self).__init__()
        self.histblock = RGBuvHistBlock(
            h=h, insz=insz, method='inverse-quadratic', sigma=sigma,
            intensity_scale=intensity_scale, green_only=green_only, device=device)
        self.norm = norm                 # 'l1' | 'l2' | None
        self.norm_p = norm_p
        self.weight = weight
        self.keep_planes = keep_planes   # w_color3 style (keep all planes)
        mean = torch.tensor(pixel_mean if pixel_mean is not None
                            else [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor(pixel_std if pixel_std is not None
                           else [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean)
        self.register_buffer('pixel_std', std)

    @property
    def out_dim(self):
        planes = 1 if self.histblock.green_only else (3 if self.keep_planes else 1)
        return planes * self.histblock.out_dim

    def forward(self, img):
        """img: ``(1, 3, H, W)`` normalised -> ``(1, out_dim)`` colour label."""
        denorm = img * self.pixel_std + self.pixel_mean      # back to [0, 1]
        hist = self.histblock(denorm)                        # (1, planes, h, h)
        if self.keep_planes:
            hist = hist.reshape(1, -1)
        else:
            hist = hist.mean(1).reshape(1, -1)               # (1, h*h)
        if self.norm == 'l1':
            hist = torch.nn.functional.normalize(hist.float(), p=1, dim=-1)
        elif self.norm == 'l2':
            hist = torch.nn.functional.normalize(hist.float(), p=self.norm_p, dim=-1)
        return hist.float() * self.weight
