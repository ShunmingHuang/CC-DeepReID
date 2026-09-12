"""Classification heads.

``NormalizedClassifier`` is transcribed from
``C2R-ReID-main/models/classifier.py:24-36``: instead of a plain
``nn.Linear`` (dot product with an unconstrained weight matrix), both the feature
and the class weights are L2-normalised, so the logits are *cosine similarities*
scaled by a constant.

Why this matters for the clothing branch: clothing classes are heavily long-tailed
(one id may own 1..14 outfits), and a cosine head bounds every logit by the scale,
which makes the optimisation far less sensitive to outlier feature norms than an
unconstrained dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedClassifier(nn.Module):
    """Cosine classifier (weight-normalised), as used by C2R-ReID for clothes.

    Args:
        feature_dim: input width.
        num_classes: number of classes.
        scale: logit scale; ``None`` or ``1`` returns raw cosine values.
    """

    def __init__(self, feature_dim, num_classes, scale=None):
        super(NormalizedClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(num_classes, feature_dim))
        # same init as C2R: uniform, then renormalised rows
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(x, w)
        if self.scale is not None and self.scale != 1:
            logits = logits * self.scale
        return logits


class CosineClassifier(NormalizedClassifier):
    """Alias kept for readability at the call sites."""


def build_classifier(feature_dim, num_classes, mode='linear', scale=None):
    """Factory used by the model so the head type is config-driven.

    mode='linear' -> ``nn.Linear`` (dot-product head, no bias), the original
                     CC-DeepReID behaviour.
    mode='cosine' -> :class:`NormalizedClassifier` (C2R-ReID's clothes head).
    """
    if mode == 'linear':
        return nn.Linear(feature_dim, num_classes, bias=False)
    if mode == 'cosine':
        return NormalizedClassifier(feature_dim, num_classes, scale=scale)
    raise ValueError("unknown classifier mode '{}', expected 'linear' or 'cosine'"
                     .format(mode))
