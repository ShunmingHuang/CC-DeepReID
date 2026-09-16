"""Loss assembly for the dual-branch model.

Objective (per the CC-DeepReID dual-branch design)::

    L = W_id  * CE_id(F)                 # identity softmax on F
      + W_tri * Triplet(F)               # CSCI's identity-only hard triplet on F
      + W_clo * CE_cloth(F')             # clothing softmax on F'
      + W_dis * Disentangle(F, F')       # push F and F' apart

``F`` is the identity branch; ``F'`` is the **identity-independent** branch.
Clothing classification is merely the supervision currently attached to ``F'``
(plus the optional C2R ``CAL`` term) -- it is not the definition of that branch,
and further modules are expected to be attached to it later.

Alignment notes with ICCV-CSCI-Person-ReID (dataset layer only -- the loss layer
is free to differ and does):

* ``Triplet(F)`` is CSCI's triplet verbatim: hard example mining over *identity*
  labels only, clothing is never consulted.
* ``CE_cloth(F')`` follows the *same* recipe as ``CE_id(F)``, i.e. it is
  label-smoothed when ``MODEL.LABELSMOOTH`` is on. (CSCI uses plain
  ``nn.CrossEntropyLoss()`` for its clothing head; this is a deliberate
  CC-DeepReID deviation.)
* ``F'`` gets **no triplet term at all** -- it is only asked to classify
  clothing, so it cannot absorb identity-discriminative structure.
"""

import torch
import torch.nn as nn

from .cross_entropy_loss import CrossEntropyLoss
from .disentangle_loss import ClothDisentangleLoss
from .hard_mine_triplet_loss import TripletLoss


class CombinedLoss(nn.Module):
    """Callable loss bundle.

    ``forward`` keeps the historical three-argument shape for the ID path and
    accepts the clothing-branch outputs as keywords, so existing call sites that
    only care about ``F`` keep working:
    ``loss_func(cls_score, feat, target, target_cloth_id)``.
    """

    def __init__(self, cfg, num_classes, num_cloth_classes=0, sampler='triplet'):
        super(CombinedLoss, self).__init__()
        self.sampler = sampler
        self.id_weight = cfg.MODEL.ID_LOSS_WEIGHT
        self.triplet_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        self.cloth_weight = getattr(cfg.MODEL, 'CLOTH_LOSS_WEIGHT', 1.0)
        self.disentangle_weight = getattr(cfg.MODEL, 'DISENTANGLE_WEIGHT', 1.0)

        margin = 0 if cfg.MODEL.NO_MARGIN else cfg.SOLVER.MARGIN
        self.triplet_loss = TripletLoss(margin=margin)
        self.id_loss = CrossEntropyLoss(num_classes=num_classes,
                                        label_smooth=bool(cfg.MODEL.LABELSMOOTH),
                                        use_gpu=True)

        self.num_cloth_classes = num_cloth_classes or 0
        if self.num_cloth_classes > 0:
            # F' clothing head: label-smoothed softmax, same setting as the
            # identity head (MODEL.LABELSMOOTH). Note CSCI uses plain
            # nn.CrossEntropyLoss here -- this is a deliberate CC-DeepReID choice.
            self.cloth_loss = CrossEntropyLoss(num_classes=self.num_cloth_classes,
                                               label_smooth=bool(cfg.MODEL.LABELSMOOTH),
                                               use_gpu=True)
        else:
            self.cloth_loss = None

        self.disentangle_loss = ClothDisentangleLoss(
            margin=getattr(cfg.MODEL, 'DISENTANGLE_MARGIN', None))

        # ---- colour-histogram regression (CSCI's annotation-free supervision) ----
        # CSCI regresses its colour token onto an RGB-uv histogram; profile 44 uses
        # Cosine_Similarity (1 - |cos|). Runs ALONGSIDE the cloth softmax.
        self.use_hist = bool(getattr(cfg.MODEL, 'USE_HIST', False))
        self.hist_weight = float(getattr(cfg.MODEL, 'HIST_LOSS_WEIGHT', 1.0))
        self.hist_loss_type = getattr(cfg.MODEL, 'HIST_LOSS', 'cosine')
        if self.hist_loss_type not in ('cosine', 'mse', 'l1'):
            raise ValueError("HIST_LOSS must be 'cosine', 'mse' or 'l1', got '{}'"
                             .format(self.hist_loss_type))

    def histogram_loss(self, hist_pred, hist_target):
        """CSCI eq.: MSE(mean=False) / 1 - |cos| / L1 between prediction and label."""
        if self.hist_loss_type == 'cosine':
            cos = (torch.nn.functional.normalize(hist_pred.float(), p=2, dim=-1)
                   * torch.nn.functional.normalize(hist_target.float(), p=2, dim=-1)
                   ).sum(-1)
            return (1 - cos.abs()).mean()
        if self.hist_loss_type == 'l1':
            return (hist_pred.float() - hist_target.float()).abs().mean()
        return ((hist_pred.float() - hist_target.float()) ** 2).mean()

    def forward(self, id_score, id_feat, target, target_cloth_id=None,
                cloth_score=None, cloth_feat=None,
                hist_pred=None, hist_target=None, return_parts=False):
        id_term = self.id_loss(inputs=id_score, targets=target)
        loss = self.id_weight * id_term

        triplet_term = None
        if self.sampler == 'triplet':
            # CSCI's triplet is identity-only; target_cloth_id is passed through
            # for call-site compatibility and ignored inside TripletLoss.
            triplet_term = self.triplet_loss(
                inputs=id_feat, targets=target, targets_cloth=target_cloth_id)
            loss = loss + self.triplet_weight * triplet_term

        cloth_term = None
        if self.cloth_loss is not None and cloth_score is not None:
            cloth_term = self.cloth_loss(inputs=cloth_score,
                                         targets=target_cloth_id)
            loss = loss + self.cloth_weight * cloth_term

        disentangle_term = None
        if cloth_feat is not None:
            disentangle_term = self.disentangle_loss(id_feat, cloth_feat)
            loss = loss + self.disentangle_weight * disentangle_term

        hist_term = None
        if self.use_hist and hist_pred is not None and hist_target is not None:
            hist_term = self.histogram_loss(hist_pred, hist_target)
            loss = loss + self.hist_weight * hist_term

        if return_parts:
            def _f(x):
                return None if x is None else float(x.detach())
            return loss, {'id': _f(id_term), 'triplet': _f(triplet_term),
                          'cloth': _f(cloth_term),
                          'disentangle': _f(disentangle_term), 'hist': _f(hist_term)}
        return loss


def make_loss(cfg, num_classes, num_cloth_classes=None):
    num_cloth_classes = num_cloth_classes or 0
    return CombinedLoss(cfg, num_classes=num_classes,
                        num_cloth_classes=num_cloth_classes,
                        sampler=cfg.DATALOADER.SAMPLER)
