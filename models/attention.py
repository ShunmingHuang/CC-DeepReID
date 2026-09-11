"""Dual-branch channel attention for the CC-DeepReID backbone.

The module sits right behind the ResNet50 feature map and splits the pooled
descriptor into two channels-attended branches:

* branch 1 -> ``F``   : identity branch, supervised by ID + triplet (and used at
  test time, so the retrieval protocol is unchanged);
* branch 2 -> ``F'``  : clothing branch, supervised by an ID-style softmax over
  the clothing vocabulary.

``F`` and ``F'`` are additionally pushed towards orthogonality by
``ClothDisentangleLoss`` (CSCI's ``Cosine_Disentangle``).
"""

import torch
import torch.nn as nn


MIN_BOTTLENECK_DIM = 16


def _bottleneck_dim(channels, reduction):
    """Channel bottleneck width, floored at ``MIN_BOTTLENECK_DIM``."""
    return max(int(channels // reduction), MIN_BOTTLENECK_DIM)


class BranchChannelAttention(nn.Module):
    """SE-style channel attention: shared pooling, per-branch squeeze-excite."""

    def __init__(self, channels, reduction=16):
        super(BranchChannelAttention, self).__init__()
        hidden = _bottleneck_dim(channels, reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, pooled, spatial):
        """pooled: (B, C) global average pooling; spatial: (B, C, H, W)."""
        weight = self.fc(pooled).unsqueeze(-1).unsqueeze(-1)
        return spatial * weight


class DualBranchChannelAttention(nn.Module):
    """Two channel-attention branches producing features ``F`` and ``F'``.

    Args:
        channels: backbone output channels (2048 for ResNet50).
        reduction: squeeze-excite reduction ratio.
        cloth_feat_dim: output width of ``F'``; default equals ``channels`` so the
            disentangle loss compares same-width descriptors 1:1.
        projector: optional per-branch MLP ("BNNeck + FC") applied on ``F'``.
            Disabled by default so ``F'`` stays a pure attention response.
    """

    def __init__(self, channels=2048, reduction=16, cloth_feat_dim=None,
                 projector=False):
        super(DualBranchChannelAttention, self).__init__()
        self.channels = channels
        self.cloth_feat_dim = channels if cloth_feat_dim is None else cloth_feat_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att_id = BranchChannelAttention(channels, reduction)
        self.att_cloth = BranchChannelAttention(channels, reduction)

        self.projector = projector
        if projector:
            self.cloth_bn = nn.BatchNorm1d(self.cloth_feat_dim)
            self.cloth_bn.bias.requires_grad_(False)
            self.cloth_fc = nn.Linear(self.cloth_feat_dim, self.cloth_feat_dim, bias=False)

    @staticmethod
    def _flatten(x):
        return x.view(x.shape[0], -1)

    def forward(self, feat_map):
        """feat_map: (B, C, H, W) -> (F, F') both shaped (B, feat_dim)."""
        pooled = self._flatten(self.pool(feat_map))

        id_map = self.att_id(pooled, feat_map)
        cloth_map = self.att_cloth(pooled, feat_map)

        feat_id = self._flatten(self.pool(id_map))
        feat_cloth = self._flatten(self.pool(cloth_map))

        if self.projector:
            feat_cloth = self.cloth_fc(self.cloth_bn(feat_cloth))
            if not self.training:
                feat_cloth = self.cloth_bn(feat_cloth)

        return feat_id, feat_cloth
