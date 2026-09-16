import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import resnet50
from .backbones.s2a_resnet import s2a_resnet50
from .classifier import build_classifier


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ResNet(nn.Module):
    """Shared ResNet50 trunk with two channel-isolated branches.

    The trunk is ``models/backbones/s2a_resnet.py``: ``B`` branches live in
    contiguous channel blocks of the shared feature map, and every convolution uses
    ``groups=B``, so **no weight and no gradient crosses the branches at any depth**
    (this replaces the earlier dual SE channel-attention module).

    Forward outputs:

    * ``training`` -> ``(id_score, F, cloth_score, F', global_feat, map_id, map_cloth, hist_pred)``
      ``map_id`` / ``map_cloth`` are the branch maps **before pooling** -- here they
      are simply the two channel blocks of the trunk output, so downstream modules
      still get un-pooled spatial features. ``hist_pred`` is the colour-histogram
      regression off ``F'`` (``None`` when ``MODEL.USE_HIST`` is off).
    * ``eval``     -> ``F`` only (retrieval still uses the identity branch)

    ``F`` is the identity branch. ``F'`` is the **identity-independent** branch:
    clothing classification is only the supervision currently attached to it, not
    its definition.
    """

    def __init__(
        self,
        num_classes,
        cfg,
        num_cloth_classes=None
    ):
        super().__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = num_classes

        self.num_branches = int(getattr(cfg.MODEL, 'S2A_BRANCHES', 2))
        self.branch_width = int(getattr(cfg.MODEL, 'S2A_BRANCH_WIDTH', 2048))
        self.s2a_mode = getattr(cfg.MODEL, 'S2A_MODE', 's2a')
        self.dual_branch = getattr(cfg.MODEL, 'DUAL_BRANCH', True)

        if self.dual_branch:
            # the map is num_branches x branch_width wide; branch 0 is F, 1 is F'
            self.base = s2a_resnet50(num_branches=self.num_branches,
                                     branch_width=self.branch_width,
                                     last_stride=last_stride,
                                     mode=self.s2a_mode)
            self.in_planes = self.branch_width
        else:
            self.base = resnet50(last_stride=last_stride)
            self.in_planes = 2048

        feat_dim = self.in_planes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # ---- clothing branch head (F' -> cloth logits) ----
        # The head type is config-driven: 'linear' keeps the original dot-product
        # head, 'cosine' installs C2R-ReID's NormalizedClassifier (weight-normalised
        # cosine logits), which suits the long-tailed clothing classes better.
        self.num_cloth_classes = num_cloth_classes or 0
        self.cloth_head_mode = getattr(cfg.MODEL, 'CLOTH_HEAD', 'linear')
        self.cloth_head_scale = getattr(cfg.MODEL, 'CLOTH_HEAD_SCALE', 16.0)
        if self.dual_branch and self.num_cloth_classes > 0:
            self.cloth_bottleneck = nn.BatchNorm1d(feat_dim)
            self.cloth_bottleneck.bias.requires_grad_(False)
            self.cloth_bottleneck.apply(weights_init_kaiming)
            self.cloth_classifier = build_classifier(
                feat_dim, self.num_cloth_classes,
                mode=self.cloth_head_mode, scale=self.cloth_head_scale)
            if isinstance(self.cloth_classifier, nn.Linear):
                self.cloth_classifier.apply(weights_init_classifier)
        else:
            self.cloth_bottleneck = None
            self.cloth_classifier = None

        # ---- colour-histogram regression head (CSCI's annotation-free target) ----
        # A second, *additional* supervision on the identity-independent branch:
        # F' is regressed onto the RGB-uv histogram of the image. The cloth
        # classifier above is untouched and keeps running alongside it.
        self.use_hist = bool(getattr(cfg.MODEL, 'USE_HIST', False))
        self.hist_dim = int(getattr(cfg.MODEL, 'HIST_DIM', 32)) ** 2
        self.hist_hidden = int(getattr(cfg.MODEL, 'HIST_HIDDEN', 1024))
        if self.dual_branch and self.use_hist:
            self.hist_head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, self.hist_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.hist_hidden, self.hist_dim),
            )
        else:
            self.hist_head = None

    @staticmethod
    def _pool(z):
        return nn.functional.avg_pool2d(z, z.shape[2:4]).view(z.shape[0], -1)

    def load_parameter(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def _cloth_bn_is_usable(self, feat_cloth_raw, target_cloth):
        """BatchNorm1d over F' needs every cloth class seen at least twice."""
        if feat_cloth_raw.shape[0] < 2:
            return False
        if target_cloth is None:
            return True
        counts = torch.bincount(target_cloth.detach().view(-1))
        return bool(counts.numel() == 0 or counts.min().item() >= 2)

    def forward(self, x, target_cloth=None):
        feat_map = self.base(x)

        if self.dual_branch:
            # the trunk already carries B channel-isolated branches; the first two
            # are the identity and the identity-independent one
            map_id, map_cloth = self.base.split_branches(feat_map)[:2]
            feat_id_raw = self._pool(map_id)
            feat_cloth_raw = self._pool(map_cloth)
        else:
            map_id, map_cloth = None, None
            feat_id_raw = self._pool(feat_map)
            feat_cloth_raw = None

        global_feat = self._pool(feat_map)

        if self.neck == 'bnneck':
            feat = self.bottleneck(feat_id_raw)
        else:
            feat = feat_id_raw

        if self.training:
            cls_score = self.classifier(feat)

            if self.cloth_classifier is not None and feat_cloth_raw is not None:
                if self._cloth_bn_is_usable(feat_cloth_raw, target_cloth):
                    cloth_feat = self.cloth_bottleneck(feat_cloth_raw)
                    cloth_score = self.cloth_classifier(cloth_feat)
                else:
                    # singleton cloth class in this batch: fall back to the running
                    # BatchNorm statistics instead of raising a training error
                    was_training = self.cloth_bottleneck.training
                    self.cloth_bottleneck.eval()
                    cloth_feat = self.cloth_bottleneck(feat_cloth_raw)
                    self.cloth_bottleneck.train(was_training)
                    cloth_score = self.cloth_classifier(cloth_feat)
            else:
                cloth_feat, cloth_score = None, None

            # Training return, in order:
            #   id_score, F, cloth_score, F', global_feat, map_id, map_cloth, hist_pred
            # map_id / map_cloth are the branch maps BEFORE pooling, for any module
            # that wants to work on the spatial features.
            hist_pred = self.hist_head(feat_cloth_raw) if self.hist_head is not None \
                else None
            return (cls_score, feat, cloth_score, cloth_feat, global_feat,
                    map_id, map_cloth, hist_pred)
        else:
            if self.neck_feat:
                return feat
            else:
                return global_feat


def make_model(cfg, num_classes, num_cloth_classes=None):

    if cfg.MODEL.NAME == "resnet":
        model = ResNet(num_classes=num_classes, cfg=cfg,
                       num_cloth_classes=num_cloth_classes)

    return model
