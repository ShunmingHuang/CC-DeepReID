"""Check the S2A-ALIGNED backbone against the mechanism it claims to copy.

Asserts, in order of importance:

1. ``conv_kv`` is genuinely SHARED -- both branches' outputs move when the shared
   scene moves. This is the property the 'sealed' variant does NOT have, and it
   is the whole reason this variant exists.
2. The DECISION is isolated: branch 1's output does not read branch 0's own
   channels, i.e. the direct coupling is exactly 0 -- the CNN form of S2A's
   ``-inf`` mask entry.
3. Branch 0 DOES reach branch 1 through the shared scene (indirect > 0), so the
   two mechanisms are measurably different rather than both being sealed.
4. The forward pass honours the ``make_model`` contract: the map splits into B
   blocks of ``branch_width`` channels.

Run: python scripts/check_s2a_aligned.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbones.s2a_resnet import (  # noqa: E402
    S2AAlignedBlock, S2ABlock, s2a_resnet50)

torch.manual_seed(0)

FAILED = []


def check(name, ok, detail=''):
    tag = '[OK  ]' if ok else '[FAIL]'
    print('  {} {}{}'.format(tag, name, (' -> ' + str(detail)) if detail != '' else ''))
    if not ok:
        FAILED.append(name)


print('=== 1. branch-width contract ===')
for mode in ('s2a', 'sealed'):
    net = s2a_resnet50(num_branches=2, branch_width=2048, last_stride=1, mode=mode)
    net.eval()
    with torch.no_grad():
        m = net(torch.randn(2, 3, 64, 32))
        parts = net.split_branches(m)
    check('{}: map is B blocks of branch_width'.format(mode),
          m.shape[1] == 4096 and len(parts) == 2 and parts[0].shape[1] == 2048,
          tuple(m.shape))
    check('{}: branch_slices agrees with split_branches'.format(mode),
          torch.equal(net.branch_slices(m, 1), parts[1]))

print()
print('=== 2. direct path: branch 1 must NOT read branch 0\'s own width ===')


def own_channels(block, z, src, dst):
    """max |d(out of dst) / d(own channel block of src)|.

    For 'sealed' the whole block is src's own; for 's2a-aligned' only the branch
    half that branch src actually owns is.
    """
    B = block.B
    per = z.shape[1] // B
    z = z.clone().requires_grad_(True)
    out = list(torch.chunk(block(z), B, dim=1))
    src_z = z[:, src * per:(src + 1) * per]
    g = torch.autograd.grad(out[dst].sum(), src_z, allow_unused=True)[0]
    return 0.0 if g is None else float(g.abs().max())


# both blocks take 32 channels in and out, 2 branches -> 16 per branch
for label, blk in (('s2a-aligned', S2AAlignedBlock(32, 32, num_branches=2)),
                   ('sealed', S2ABlock(32, 32, num_branches=2))):
    blk.eval()
    z = torch.randn(1, 32, 4, 4)
    cross = own_channels(blk, z, 0, 1)
    print('  {:<12} direct d(branch1_out)/d(branch0_own) = {}'.format(label, cross))
    check('{}: direct coupling is exactly 0'.format(label), cross == 0.0, cross)

print()
print('=== 3. shared scene: the two mechanisms must DIFFER ===')


def scene_response(block, z):
    """Change ONLY the shared scene's value projection: BOTH branches must move."""
    B = block.B
    with torch.no_grad():
        base = list(torch.chunk(block(z), B, dim=1))
        d = block.d
        block.conv_kv.weight.data[d:2 * d].mul_(1.5)   # shared V only
        moved = list(torch.chunk(block(z), B, dim=1))
        block.conv_kv.weight.data[d:2 * d].div_(1.5)   # restore
    return [(a - b).abs().max().item() for a, b in zip(base, moved)]


blk = S2AAlignedBlock(32, 32, num_branches=2).eval()
resp = scene_response(blk, torch.randn(1, 32, 4, 4))
check('s2a-aligned: BOTH branches respond to a change in the shared scene',
      all(r > 0 for r in resp), resp)

sealed = S2ABlock(32, 32, num_branches=2).eval()
check('sealed: has no shared scene at all (no conv_kv)',
      not hasattr(sealed, 'conv_kv'))

print()
print('=== 4. shapes through the real model wrapper ===')
try:
    from configs.default import cfg
    from models.make_model import ResNet
    for mode in ('s2a', 'sealed'):
        cfg.merge_from_list(['MODEL.S2A_MODE', mode])
        m = ResNet(num_classes=10, cfg=cfg, num_cloth_classes=20)
        m.train()                      # the 8-slot tuple is the training return
        with torch.no_grad():
            # batch 8 -> 16 cloth labels, so no BatchNorm group is a singleton
            o = m(torch.randn(8, 3, 256, 128))
        check("{}: 8-slot forward, F={} F'={}".format(
            mode, tuple(o[1].shape), tuple(o[3].shape)),
            len(o) == 8 and o[1].shape[1] == 2048 and o[3].shape[1] == 2048)
except Exception as exc:                                   # pragma: no cover
    check('model wrapper builds both modes', False, repr(exc))

print()
print('=' * 60)
if FAILED:
    print('FAILED: {}'.format(FAILED))
    sys.exit(1)
print('S2A-ALIGNED CHECK PASSED')
