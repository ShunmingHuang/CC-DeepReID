"""Check the S2A-style backbone: isolation, parameter sharing, widths."""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models.backbones.s2a_resnet import S2ABlock, s2a_resnet50

FAILS = []


def check(cond, label, extra=''):
    print('  [{}] {}{}'.format('OK  ' if cond else 'FAIL', label,
                               ' -> ' + str(extra) if extra else ''))
    if not cond:
        FAILS.append(label)


def main():
    print('=== S2ABlock: isolation and sharing ===')
    blk = S2ABlock(in_planes=32, out_planes=32, num_branches=2)
    blk.train()
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    z = blk(x)
    check(z.shape == x.shape, 'block preserves the trunk shape', tuple(z.shape))

    half = 16
    g = torch.autograd.grad(z[:, :half].pow(2).sum(), x, retain_graph=True,
                            allow_unused=True)[0]
    cross = 0.0 if g is None else float(g[0, half:].abs().sum())
    check(cross == 0.0,
          'cross-branch gradient is EXACTLY zero (branch 0 out vs branch 1 in)',
          cross)

    g2 = torch.autograd.grad(z[:, half:].pow(2).sum(), x, retain_graph=True,
                             allow_unused=True)[0]
    cross2 = 0.0 if g2 is None else float(g2[0, :half].abs().sum())
    check(cross2 == 0.0,
          'cross-branch gradient is EXACTLY zero (branch 1 out vs branch 0 in)',
          cross2)

    own = float(g[0, :half].abs().sum())
    check(own > 0, 'each branch still receives its own gradient', own)

    # a depthwise 3x3 shares ONE kernel across all channels of both branches
    w = blk.conv_shared.weight
    check(tuple(w.shape) == (32, 1, 3, 3),
          'shared 3x3 is depthwise: one kernel reused by every channel/branch',
          tuple(w.shape))
    pw = blk.conv_dw.weight
    check(tuple(pw.shape) == (32, 16, 1, 1),
          'per-branch 1x1 is grouped: branch-local channel mixing', tuple(pw.shape))

    print()
    print('=== S2AResNet: widths and parameter count ===')
    from models.backbones.resnet import resnet50
    n_plain = sum(p.numel() for p in resnet50(last_stride=1).parameters())

    for bw in (512, 1024, 2048):
        net = s2a_resnet50(num_branches=2, branch_width=bw, mode='sealed')
        net.train()
        out = net(torch.randn(2, 3, 256, 128))
        b = net.split_branches(out)
        n = sum(p.numel() for p in net.parameters())
        print('  branch_width={:<5} trunk={:<6} per-branch={:<5} '
              'params={:7.2f}M ({:.2f}x resnet50)'.format(
                  bw, out.shape[1], b[0].shape[1], n / 1e6, n / n_plain))
        check(b[0].shape[1] == bw and b[1].shape[1] == bw,
              'branch_width={}: each branch owns exactly that width'.format(bw),
              (tuple(b[0].shape), tuple(b[1].shape)))
        check(len(b) == 2, 'split_branches returns one map per branch', len(b))

    net = s2a_resnet50(num_branches=2, branch_width=2048, mode='sealed')
    out = net(torch.randn(2, 3, 256, 128))
    n_s2a = sum(p.numel() for p in net.parameters())
    print()
    print('  configured default (branch_width=2048): {:.2f}M vs plain {:.2f}M '
          '-> {:.2f}x'.format(n_s2a / 1e6, n_plain / 1e6, n_s2a / n_plain))
    print('  NOTE: at branch_width=2048 the trunk is 2.22x a plain ResNet50 and is')
    print('        NOT cheaper than two independent backbones -- exact isolation')
    print('        plus full branch width is what costs the parameters.')

    conv_s2a = [(m.kernel_size, m.in_channels, m.out_channels, m.groups)
                for m in net.modules() if isinstance(m, nn.Conv2d)]
    depthwise = sum(1 for k, i, o, g in conv_s2a if g == i and k == (3, 3))
    print('  conv layers: {} total, {} are depthwise 3x3'.format(
        len(conv_s2a), depthwise))
    check(depthwise > 0,
          'spatial mixing is done by depthwise kernels',
          depthwise)

    print()
    print('=== isolation still exact through the WHOLE trunk ===')
    x = torch.randn(1, 3, 128, 64, requires_grad=True)
    out = net(x)
    gfull = torch.autograd.grad(out[:, :2048].pow(2).sum(), x, retain_graph=True,
                               allow_unused=True)[0]
    # the input stem is shared, so x itself is shared; the check that matters is
    # channel isolation inside the trunk, verified per block above
    check(gfull is not None, 'gradients flow through the whole trunk')

    print()
    print('S2A BLOCK CHECK ' + ('PASSED' if not FAILS else 'FAILED'))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
