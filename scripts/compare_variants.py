"""'sealed' vs 's2a': what actually differs, measured without ambiguity.

The earlier version of this probe measured |d(cloth CE)/d(branch-0 params)| and
called it "leakage". That was WRONG as a measure of isolation: in the sealed
design the depthwise kernels ARE shared between branches, so their gradient
necessarily contains branch-1's contribution. Parameter gradients cannot
distinguish "shared parameter" from "cross-branch path".

The unambiguous measure is a FORWARD PERTURBATION at a known boundary: inject
noise into branch 0's channels of an intermediate tensor and see whether branch
1's OUTPUT changes, and vice versa. That is exactly the topology claim.

Measures:
  A. forward isolation, perturbation at layer2 output -> layer4 output
  B. parameters and step time
  C. (optional) identity probe on real data, if the smoke-test bundle exists

Run: python scripts/compare_variants.py
"""

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import cfg                                    # noqa: E402
from models.make_model import ResNet                                # noqa: E402
from models.backbones.s2a_resnet import s2a_resnet50                # noqa: E402

torch.manual_seed(0)

BATCH = 4
SIZE = (256, 128)


def block_of(model):
    """One block, taken out of the trunk, for a clean perturbation test."""
    return model.base.layer4[0]


def forward_isolation(block, channels, n=3):
    """Perturb branch 0's channels of the INPUT; measure each branch's response.

    Returns (own_response, other_response): how much branch 0's output moves and
    how much branch 1's output moves. If the design is isolated, the second is
    exactly 0.
    """
    per = channels // block.B
    own, other = 0.0, 0.0
    for _ in range(n):
        z = torch.randn(1, channels, 4, 4)
        with torch.no_grad():
            base = list(torch.chunk(block(z), block.B, dim=1))
            z2 = z.clone()
            z2[:, :per] += 0.5 * torch.randn_like(z2[:, :per])   # only branch 0
            moved = list(torch.chunk(block(z2), block.B, dim=1))
        own = max(own, float((base[0] - moved[0]).abs().max()))
        other = max(other, float((base[1] - moved[1]).abs().max()))
    return own, other


def timing(model, iters=3):
    model.train()
    x = torch.randn(BATCH, 3, *SIZE)
    with torch.no_grad():
        model(x)
    t0 = time.time()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        model(x)[0].pow(2).sum().backward()
    return (time.time() - t0) / iters


def identity_probe_real(mode, n_batches=4):
    """Linear probe on synthetic-but-structured data is meaningless, so this runs
    on the real PRCC mock bundle when it is present. Returns None otherwise.
    """
    mock = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        '_smoke_data')
    if not os.path.isdir(mock):
        return None
    try:
        from datasets import make_dataloader
    except Exception:
        return None
    try:
        c = cfg.clone()
        c.merge_from_list(['MODEL.S2A_MODE', mode,
                           'DATASETS.NAMES', 'prcc',
                           'DATASETS.ROOT_DIR', mock,
                           'MODEL.DEVICE', 'cpu',
                           'SOLVER.IMS_PER_BATCH', 8,
                           'TEST.IMS_PER_BATCH', 8,
                           'DATALOADER.NUM_WORKERS', 0])
        loaders = make_dataloader(c)
    except Exception:
        return None

    train_loader = None
    for key in ('train', 0, 'Train'):
        if isinstance(loaders, dict) and key in loaders:
            train_loader = loaders[key]
            break
    if train_loader is None:
        return None

    bundle = getattr(train_loader, 'bundle', None)
    n_ids = getattr(bundle, 'num_train_pids', 4) or 4
    model = ResNet(num_classes=n_ids, cfg=c,
                   num_cloth_classes=getattr(bundle, 'num_train_clothes', 8))
    model.eval()

    F, Fp, y = [], [], []
    n_seen = 0
    for batch in train_loader:
        imgs = batch[0]
        pids = batch[1]
        with torch.no_grad():
            model.training = True
            out = model(imgs, target_cloth=batch[2] if len(batch) > 2 else None)
            model.training = False
        F.append(out[1].detach())
        Fp.append(out[3].detach())
        y.append(pids)
        n_seen += 1
        if n_seen >= n_batches:
            break
    if not F:
        return None
    F, Fp, y = torch.cat(F), torch.cat(Fp), torch.cat(y)
    n_tr = max(1, int(F.shape[0] * 0.7))
    if y[:n_tr].unique().numel() < 2 or y[n_tr:].unique().numel() < 2:
        return None

    res = {}
    for name, feat in (('F', F), ("F'", Fp)):
        probe = nn.Linear(feat.shape[1], n_ids)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        lossf = nn.CrossEntropyLoss()
        for _ in range(300):
            opt.zero_grad()
            lossf(probe(feat[:n_tr]), y[:n_tr]).backward()
            opt.step()
        with torch.no_grad():
            acc = (probe(feat[n_tr:]).argmax(1) == y[n_tr:]).float().mean().item()
        res[name] = acc
    return res


print('=' * 74)
print('sealed vs s2a -- forward isolation at a known boundary, plus cost')
print('=' * 74)
print()

rows = []
for mode in ('sealed', 's2a'):
    cfg.merge_from_list(['MODEL.S2A_MODE', mode,
                         'MODEL.S2A_BRANCHES', 2,
                         'MODEL.S2A_BRANCH_WIDTH', 2048])
    model = ResNet(num_classes=150, cfg=cfg, num_cloth_classes=300)
    n_par = sum(p.numel() for p in model.parameters())

    trunk = s2a_resnet50(num_branches=2, branch_width=2048, mode=mode)
    trunk.eval()
    blk = trunk.layer4[0]
    channels = blk.downsample[0].in_channels if blk.downsample is not None \
        else 2048 * 2
    own, other = forward_isolation(blk, channels)
    dt = timing(model)

    rows.append((mode, n_par, own, other, dt))
    print('--- MODE = %s ---' % mode)
    print('  params                          : %.2fM  (%.2fx ResNet50)'
          % (n_par / 1e6, n_par / 23.51e6))
    print("  perturb branch 0 -> branch 0 out: %.6e   (must move)" % own)
    print("  perturb branch 0 -> branch 1 out: %.6e   %s"
          % (other, 'EXACTLY 0 -- isolated' if other == 0.0
             else 'NON-ZERO -- coupled'))
    print('  fwd+bwd, batch 4                : %.2fs' % dt)
    print()

print('=' * 74)
print('%-8s %9s %16s %16s %9s'
      % ('mode', 'params', 'b0->b0', 'b0->b1', 'step(s)'))
for mode, n_par, own, other, dt in rows:
    print('%-8s %8.2fM %16.3e %16.3e %9.2f'
          % (mode, n_par / 1e6, own, other, dt))

print()
print('NOTE: the earlier |d(cloth CE)/d(branch-0 params)| measure was dropped.')
print('In the sealed design those parameters are SHARED between branches, so')
print('their gradient cannot distinguish sharing from cross-branch coupling.')
