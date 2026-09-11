"""Clothing / identity feature disentanglement loss.

Mirrors ``Cosine_Disentangle`` in ICCV-CSCI-Person-ReID/loss/custom_loss.py:
the cosine similarity between the identity feature ``F`` and the clothing
feature ``F'`` is driven to zero, i.e. the two descriptors are made orthogonal
("尽量远离").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClothDisentangleLoss(nn.Module):
    """Push ``F`` and ``F'`` apart.

    With ``margin`` unset (default) this is CSCI's objective::

        loss = mean(|cos(F, F')|)

    minimized at ``cos = 0``, i.e. exactly orthogonal features.

    With ``margin`` set the loss becomes a hinge, so features already farther
    apart than the margin cost nothing::

        loss = mean(relu(cos(F, F') - margin))
    """

    def __init__(self, margin=None, eps=1e-8):
        super(ClothDisentangleLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, feat_id, feat_cloth):
        if feat_id.shape != feat_cloth.shape:
            raise ValueError(
                "disentangle needs same-width features, got {} and {}".format(
                    tuple(feat_id.shape), tuple(feat_cloth.shape)))
        cosine = F.cosine_similarity(feat_id, feat_cloth, dim=-1, eps=self.eps)
        if self.margin is None:
            return cosine.abs().mean()
        return F.relu(cosine - self.margin).mean()
