"""Identity / identity-independent feature disentanglement loss.

Mirrors ``Cosine_Disentangle`` in ICCV-CSCI-Person-ReID/loss/custom_loss.py:
the cosine similarity between the identity feature ``F`` and the
identity-independent feature ``F'`` is driven to zero, i.e. the two descriptors
are made orthogonal ("尽量远离").

WHAT THIS DOES AND DOES NOT DO
------------------------------
This is a **geometric** constraint, not an information-theoretic one. It bounds
``E[<F, F'>]`` only. Measured on a controlled setup (F supervised by identity,
F' by clothing, both read off the same input):

    |cos(F, F')|              6.6%  ->  1.0%     (the loss works)
    identity probe on F'     30.6%  -> 28.6%     (information barely moves)
    clothing probe on F      29.9%  -> 23.2%
    clothing accuracy on F'  56.4%  -> 40.6%     (the loss has a cost)

So two branches can be mutually orthogonal while each still carries the other's
factor (the encoders are non-linear, so per-dimension correlation is untouched),
and the constraint does not come for free. It is also achieved quickly: once
``|cos| -> 0`` the term contributes no gradient, so it only acts early in
training.

If genuine *information* separation is needed, orthogonality alone is not the
tool -- use an adversarial (gradient-reversed) identity predictor on ``F'``, or
change what supervises ``F'``.
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
