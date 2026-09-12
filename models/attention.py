"""Dual-branch channel attention for the CC-DeepReID backbone.

The module sits right behind the ResNet50 feature map and splits it into two
channel-attended branches. **The channel attention is applied on the spatial
feature map and the spatial pooling happens afterwards**, so the
attention-weighted maps ``map_id`` / ``map_cloth`` are exposed and can be
consumed by other modules before ``F`` / ``F'`` are pooled out of them::

    feat_map (B, C, H, W)
      ├─ branch 1: SE re-weight on the map -> map_id    (B, C, H, W) --GAP--> F
      └─ branch 2: SE re-weight on the map -> map_cloth (B, C, H, W) --GAP--> F'

``F`` is the **identity** branch (ID + triplet loss, used at test time).
``F'`` is the **identity-independent** branch: it is meant to carry everything
that is *not* the person's identity, and clothing classification is merely the
supervision currently attached to it. Do not treat "clothing" as the definition
of that branch -- further modules go there later.

``F`` and ``F'`` are pushed apart by ``ClothDisentangleLoss`` (CSCI's
``Cosine_Disentangle``).

Note on the SE gate: squeeze-and-excitation derives its channel weights from a
global average pool *statistic* -- that is inherent to the mechanism (it is the
"squeeze"), and it is not the pooling that produces ``F`` / ``F'``. The maps
handed back below are the un-pooled tensors, i.e. what downstream modules can
attach to.
"""

from collections import namedtuple

import torch
import torch.nn as nn

MIN_BOTTLENECK_DIM = 16

#: Return type of :meth:`DualBranchChannelAttention.forward`.
#:
#: * ``feat_id`` / ``feat_cloth`` -- pooled descriptors ``F`` / ``F'``
#: * ``map_id`` / ``map_cloth``    -- attention-weighted maps *before* pooling
DualBranchOutput = namedtuple(
    'DualBranchOutput',
    ['feat_id', 'feat_cloth', 'map_id', 'map_cloth'])


def _bottleneck_dim(channels, reduction):
    """Channel bottleneck width, floored at ``MIN_BOTTLENECK_DIM``."""
    return max(int(channels // reduction), MIN_BOTTLENECK_DIM)


class BranchChannelAttention(nn.Module):
    """SE-style channel attention applied to a spatial map.

    The excitation weights come from the global average pool statistic (that is
    what SE *is*); the **output stays spatial**:
    ``(B, C, H, W) * sigmoid(gate)``.
    """

    def __init__(self, channels, reduction=16):
        super(BranchChannelAttention, self).__init__()
        hidden = _bottleneck_dim(channels, reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_map):
        """feat_map: (B, C, H, W) -> channel-attended map, still (B, C, H, W)."""
        # squeeze: exists only to produce the excitation weights
        squeezed = feat_map.mean(dim=(2, 3))
        weight = self.fc(squeezed).unsqueeze(-1).unsqueeze(-1)
        # excite: re-weight the spatial map (no pooling on the output path)
        return feat_map * weight


class DualBranchChannelAttention(nn.Module):
    """Two channel-attention branches producing ``F`` / ``F'`` and their maps.

    ``F``       : identity branch.
    ``F'``      : **identity-independent** branch. Clothing is only its current
                  supervision; other modules attach here later.

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

    def forward_maps(self, feat_map):
        """Attention-weighted maps, before any pooling: ``(B, C, H, W)`` each."""
        return self.att_id(feat_map), self.att_cloth(feat_map)

    def pool(self, feat_map):
        """Global average pooling + flatten: ``(B, C, H, W) -> (B, C)``."""
        return self._flatten(
            nn.functional.avg_pool2d(feat_map, feat_map.shape[2:4]))

    def forward(self, feat_map):
        """Returns :class:`DualBranchOutput` (pooled features + pre-pool maps)."""
        map_id, map_cloth = self.forward_maps(feat_map)

        feat_id = self.pool(map_id)
        feat_cloth = self.pool(map_cloth)

        if self.projector:
            feat_cloth = self.cloth_fc(self.cloth_bn(feat_cloth))
            if not self.training:
                feat_cloth = self.cloth_bn(feat_cloth)

        return DualBranchOutput(feat_id, feat_cloth, map_id, map_cloth)
