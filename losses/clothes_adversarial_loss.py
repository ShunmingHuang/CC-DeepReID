"""Clothes-based Adversarial Loss (CAL).

Transcribed from ``C2R-ReID-main/losses/clothes_based_adversarial_loss.py:7-43``
(Gu et al., "Clothes-Changing Person Re-identification with RGB Modality Only",
CVPR 2022), adapted to per-sample positive masks.

For each anchor the *positive* class set is every clothing class owned by the same
identity (``pid2clothes[pid]``); all other clothing classes are negatives::

    negtive_mask = 1 - positive_mask
    log_sum_exp  = log((exp(logits) * negtive_mask).sum(1, keepdim=True) + exp(logits))
    log_prob     = logits - log_sum_exp
    mask         = (1 - eps) * onehot(target) + eps / |pos| * positive_mask
    loss         = (-mask * log_prob).sum(1).mean()

The same objective serves both roles, exactly like C2R:

* trained on **detached** features it trains the clothing *discriminator*;
* trained on **live** features it becomes the term the backbone optimises.

``eps`` controls how much probability mass is spread over the identity's *other*
outfits; C2R uses ``eps=0.1`` for the discriminator and ``eps=1.0`` for its
shuffled-feature variant.
"""

import torch
import torch.nn as nn


class ClothesBasedAdversarialLoss(nn.Module):
    def __init__(self, scale=16.0, epsilon=0.1):
        super(ClothesBasedAdversarialLoss, self).__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: cloth logits ``(B, num_cloth)``.
            targets: cloth labels ``(B,)``.
            positive_mask: ``(B, num_cloth)``, ``1`` for clothing classes belonging
                to the anchor's identity, ``0`` elsewhere.
        """
        # Scale FIRST, then log-softmax: log_prob = s*z - logsumexp(s*z).
        # Scaling after the subtraction would break normalisation and the "loss"
        # could come out negative (a log-probability must be <= 0).
        logits = self.scale * inputs
        positive_mask = positive_mask.to(logits.dtype).to(logits.device)
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(logits.size(), device=logits.device)
        identity_mask.scatter_(1, targets.unsqueeze(1), 1)

        # max-shift for numerical safety (mathematically a no-op for log-softmax)
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        log_sum_exp_pos_and_all_neg = torch.log(
            (exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = logits - log_sum_exp_pos_and_all_neg

        mask = ((1 - self.epsilon) * identity_mask
                + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask)
        return (-mask * log_prob).sum(1).mean()


def positive_clothes_mask(pid2clothes, pids, device=None):
    """Gather the ``(B, num_cloth)`` positive mask for a batch of raw pids."""
    if pid2clothes is None:
        raise ValueError('CAL requires pid2clothes, but the dataset did not provide it')
    if not torch.is_tensor(pid2clothes):
        pid2clothes = torch.as_tensor(pid2clothes)
    mask = pid2clothes[pids.long().cpu()]
    if device is not None:
        mask = mask.to(device)
    return mask
