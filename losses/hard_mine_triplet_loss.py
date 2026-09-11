"""Triplet loss with hard example mining.

Transcribed from ``ICCV-CSCI-Person-ReID/loss/triplet_loss.py`` so that the
optimisation target matches CSCI exactly:

* ``euclidean_dist``    -- CSCI lines 16-31
* ``hard_example_mining`` -- CSCI lines 51-104 (``is_pos = labels == labels``)
* ``TripletLoss.__call__`` -- CSCI lines 121-135

Note: CSCI's triplet is **class-agnostic w.r.t. clothing** -- positives are all
samples sharing the *identity* label, the clothing label is never consulted.
``targets_cloth`` is therefore accepted and ignored, so the call sites (which
still pass it) do not have to change.
"""

from __future__ import division, absolute_import

import torch
import torch.nn as nn


def euclidean_dist(x, y):
    """Pairwise euclidean distance, CSCI triplet_loss.py:16-31."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels):
    """Hardest positive / negative per anchor, CSCI triplet_loss.py:51-104.

    ``is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())`` -- identity only.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)[0]
    dist_an = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)[0]
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(nn.Module):
    """Hard-mining triplet loss, CSCI triplet_loss.py:107-135.

    Args:
        margin: if ``None`` CSCI switches to ``nn.SoftMarginLoss`` on
            ``dist_an - dist_ap`` ("soft triplet", CSCI's ``NO_MARGIN``);
            otherwise ``nn.MarginRankingLoss(margin)``.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, targets_cloth=None):
        """
        Args:
            inputs (torch.Tensor): feature matrix, shape (batch_size, feat_dim).
            targets (torch.LongTensor): identity labels, shape (batch_size).
            targets_cloth: accepted for call-site compatibility and IGNORED --
                CSCI's triplet mines positives by identity only.
        """
        # fp32: the squared-sum over 2048 dims is the one place where fp16 could
        # overflow to inf under AMP autocast, poisoning the backward.
        inputs = inputs.float()

        dist_mat = euclidean_dist(inputs, inputs)
        dist_ap, dist_an = hard_example_mining(dist_mat, targets)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
