import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import resnet50
from .attention import DualBranchChannelAttention

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
    """ResNet50 backbone + dual-branch channel attention.

    Forward outputs:

    * ``training`` -> ``(id_score, F, cloth_score, F', global_feat, map_id, map_cloth)``
      where ``map_id`` / ``map_cloth`` are the attention-weighted maps **before
      pooling** (``(B, 2048, H/16, W/16)``), available for other modules.
    * ``eval``     -> ``F`` only (retrieval still uses ``F``)

    ``F`` is the identity branch (ID + triplet loss), ``F'`` the clothing branch
    (clothing softmax). ``global_feat`` is the plain pooled backbone descriptor,
    kept for backward compatibility.
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
        self.in_planes = 2048
        self.num_classes = num_classes
        self.base = resnet50(last_stride=last_stride)

        self.dual_branch = getattr(cfg.MODEL, 'DUAL_BRANCH', True)

        if self.dual_branch:
            cloth_feat_dim = getattr(cfg.MODEL, 'CLOTH_FEAT_DIM', -1)
            if cloth_feat_dim is None or cloth_feat_dim <= 0:
                cloth_feat_dim = self.in_planes
            self.channel_attention = DualBranchChannelAttention(
                channels=self.in_planes,
                reduction=getattr(cfg.MODEL, 'ATT_REDUCTION', 16),
                cloth_feat_dim=cloth_feat_dim,
                projector=bool(getattr(cfg.MODEL, 'ATT_PROJECTOR', False)),
            )
            feat_dim = cloth_feat_dim
        else:
            self.channel_attention = None
            feat_dim = self.in_planes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # ---- clothing branch head (F' -> cloth logits) ----
        self.num_cloth_classes = num_cloth_classes or 0
        if self.dual_branch and self.num_cloth_classes > 0:
            self.cloth_bottleneck = nn.BatchNorm1d(feat_dim)
            self.cloth_bottleneck.bias.requires_grad_(False)
            self.cloth_bottleneck.apply(weights_init_kaiming)
            self.cloth_classifier = nn.Linear(feat_dim, self.num_cloth_classes, bias=False)
            self.cloth_classifier.apply(weights_init_classifier)
        else:
            self.cloth_bottleneck = None
            self.cloth_classifier = None

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
        x = self.base(x)

        if self.dual_branch:
            # channel attention first, pooling afterwards; the attention-weighted
            # maps (still B,C,H,W) stay available to downstream modules
            att = self.channel_attention(x)
            feat_id_raw, feat_cloth_raw = att.feat_id, att.feat_cloth
            map_id, map_cloth = att.map_id, att.map_cloth
        else:
            feat_id_raw = nn.functional.avg_pool2d(x, x.shape[2:4])
            feat_id_raw = feat_id_raw.view(feat_id_raw.shape[0], -1)
            feat_cloth_raw = None
            map_id, map_cloth = None, None

        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)

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
            #   id_score, F, cloth_score, F', global_feat, map_id, map_cloth
            # map_id / map_cloth are the attention-weighted maps BEFORE pooling,
            # for any module that wants to work on the spatial features.
            return (cls_score, feat, cloth_score, cloth_feat, global_feat,
                    map_id, map_cloth)
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
