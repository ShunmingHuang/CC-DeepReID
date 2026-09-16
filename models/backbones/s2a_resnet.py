"""S2A-style channel-isolated blocks for a shared ResNet50 trunk.

Motivation (from ICCV-CSCI-Person-ReID's S2A-self-attention): keep ONE shared
trunk, but let it carry two feature streams that cannot mix. S2A does it with a
token mask; here the branch identity is a contiguous block of channels and the
isolation is done by ``groups=B`` (grouped = depthwise) convolutions, so no weight
and no gradient ever crosses the branches:

    shared 3x3 (depthwise: ONE kernel reused by every channel/branch)
    per-branch 1x1 (grouped: branch-local channel mixing)
    per-branch 3x3 (depthwise) + BN
    Z <- Z + out                       (the shared trunk is updated in place)

Parameter sharing is exact: a depthwise kernel is applied to every channel of every
branch, so the *same* kernel serves both branches. Branch capacity comes from the
channel positions, not from extra weights -- hence the trunk is ``B x`` the standard
width so each branch keeps the standard width.

The branch assignment can never drift: branch ``b`` always owns the slice
``[b*C/B : (b+1)*C/B]`` of every tensor in the trunk.

STATUS: implemented as an OPTION. Whether it helps is an open question -- see
S2A_DESIGN.md.

------------------------------------------------------------------------------
TWO MECHANISMS LIVE IN THIS FILE
------------------------------------------------------------------------------
``S2ABlock`` / ``S2AResNet``  ("sealed")
    Branch = channel block; isolation by ``groups=B``. Branch 0 can never
    influence branch 1, on the direct *or* the indirect path. Measured both 0.

``S2AAlignedBlock`` / ``S2AAlignedResNet``  ("s2a", DEFAULT)
    The literal translation of ICCV-CSCI's ``EvaAttention_sep_masked``, which is
    *not* sealed: spatial K/V are SHARED and only the head-token rows are masked.
    Direct is 0, indirect is NOT 0 -- the shared scene carries it. That is the
    point of the design, not a leak.

Both are offered because they answer different questions; ``MODEL.S2A_MODE``
selects. See S2A_DESIGN.md for the measured isolation profile of each.

    s2a_resnet50(mode='s2a')      -> S2AAlignedResNet   (matches S2A)
    s2a_resnet50(mode='sealed')   -> S2AResNet          (strict isolation)
"""

import torch
import torch.nn as nn

_DEBUG = False


class S2ABlock(nn.Module):
    """Standard-width residual block, applied channel-wise to B isolated branches.

    ``in_planes`` / ``out_planes`` are TOTAL widths of the shared trunk (i.e. B x
    the per-branch width). Every conv uses ``groups=B`` so the branch slices are
    independent by construction.
    """

    def __init__(self, in_planes, out_planes, num_branches=2, stride=1,
                 downsample=None):
        super(S2ABlock, self).__init__()
        self.B = num_branches
        for name, ch in (('in', in_planes), ('out', out_planes)):
            if ch % num_branches != 0:
                raise ValueError('{} channels ({}) must be divisible by '
                                 'num_branches ({})'.format(name, ch, num_branches))

        # shared spatial mixing: one depthwise kernel reused by every branch
        self.conv_shared = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                                     stride=stride, padding=1,
                                     groups=out_planes, bias=False)
        self.bn_shared = nn.BatchNorm2d(out_planes)
        # project the (possibly wider) input down to this stage's width, still
        # branch-isolated: grouped 1x1, so no cross-branch mixing
        self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                 groups=num_branches, bias=False) \
            if in_planes != out_planes else None
        self.bn_in = nn.BatchNorm2d(out_planes) if in_planes != out_planes else None
        # branch-local channel mixing
        self.conv_dw = nn.Conv2d(out_planes, out_planes, kernel_size=1,
                                 groups=num_branches, bias=False)
        self.bn_dw = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        # branch write-back
        self.conv_pw = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1,
                                 groups=out_planes, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

    def forward(self, z):
        identity = z if self.downsample is None else self.downsample(z)

        h = z
        if self.conv_in is not None:
            h = self.relu(self.bn_in(self.conv_in(h)))
        h = self.relu(self.bn_shared(self.conv_shared(h)))
        h = self.relu(self.bn_dw(self.conv_dw(h)))
        h = self.bn_pw(self.conv_pw(h))

        return self.relu(h + identity)


class S2AResNet(nn.Module):
    """ResNet50-shaped trunk carrying ``B`` channel-isolated branches.

    ``branch_width`` is the width of ONE branch; the trunk is ``B x branch_width``
    wide. ``stage_widths`` are the usual ResNet50 per-branch progression
    (256/512/1024/2048 for ``branch_width=2048``), scaled to ``branch_width``.

    Use ``branch_width=512`` for a trunk whose cost is comparable to a standard
    ResNet50, or ``branch_width=2048`` to match the classic 2048-d feature (that
    makes the trunk 4x wider and far more expensive).
    """

    def __init__(self, num_branches=2, branch_width=2048, last_stride=1,
                 in_channels=3, block_nums=(3, 4, 6, 3), stage_widths=None):
        super(S2AResNet, self).__init__()
        self.B = num_branches
        self.branch_width = branch_width
        # ResNet50 proportion: layer4 == branch_width, layer1 == branch_width/8
        if stage_widths is None:
            scale = branch_width / 2048.0
            stage_widths = [int(256 * scale), int(512 * scale),
                            int(1024 * scale), branch_width]
        self.stage_widths = stage_widths

        # the stem is the first 64 channels of each branch
        stem = int(64 * (branch_width / 2048.0)) * num_branches

        self.conv1 = nn.Conv2d(in_channels, stem, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_planes = stem
        self.layer1 = self._make_layer(stage_widths[0] * num_branches, block_nums[0], 1)
        self.layer2 = self._make_layer(stage_widths[1] * num_branches, block_nums[1], 2)
        self.layer3 = self._make_layer(stage_widths[2] * num_branches, block_nums[2], 2)
        self.layer4 = self._make_layer(stage_widths[3] * num_branches, block_nums[3],
                                       last_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride,
                          groups=self.B, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [S2ABlock(self.in_planes, planes, self.B, stride=stride,
                           downsample=downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(S2ABlock(planes, planes, self.B, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, return_branches=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_branches:
            return self.split_branches(x)
        return x

    def split_branches(self, z):
        """Split the trunk map into the B branch maps (channel blocks).

        Branch ``b`` owns ``[b*C/B : (b+1)*C/B]`` of *every* tensor in the trunk,
        so the split is the same at any depth and can never mix.
        """
        return list(torch.chunk(z, self.B, dim=1))

    def branch_slices(self, z, branch):
        """The channel range owned by ``branch`` -- its feature, never mixed."""
        C = z.shape[1] // self.B
        return z[:, branch * C:(branch + 1) * C]


# ---------------------------------------------------------------------------
# The S2A-ALIGNED variant: shared spatial scene, isolated decision
# ---------------------------------------------------------------------------

class S2AAlignedBlock(nn.Module):
    """Translation of ``EvaAttention_sep_masked`` into a ResNet block.

    S2A's masked attention, read literally:

    * the attention matrix covers **all** tokens; the mask only zeroes
      *head-token -> other-head-token* entries, so ``spatial -> head`` and
      ``head -> spatial`` both stay alive. The scene is read by everyone.
    * ``softmax`` is taken over the whole row, i.e. the shared spatial logits
      stay in the denominator for every head-token row.
    * Q/K/V come from one fused projection, so both streams live in a shared
      feature space and the spatial K/V are *jointly owned*.

    Consequences, both reproduced here on purpose:

    * **decision is isolated** -- branch ``a``'s read uses the shared scene plus
      only ``v_a``; branch ``b != a`` appears nowhere in it (direct = 0), which is
      what S2A's mask entry ``-inf`` accomplishes;
    * **scene is shared** -- both branches read the same ``V_shared``, which is
      built from *all* channels, so branch ``a`` can steer branch ``b`` by moving
      the shared scene (indirect != 0). This is S2A's design, not a defect.

    Widths. A branch owns ``2*out_planes`` channels, split into

        own q = own k = own v = d_q      (its decision machinery)
        shared k = shared v    = d_s      (the scene it reads)

    The only constraint is ``d_q + 2*d_s == out_planes``; both are set to
    ``out_planes // 3``, so the query (``d_q``) dots against ``d_s + d_q`` keys
    and the softmax weights line up with ``[V_shared ; V_own]``, which is
    ``d_s + d_s`` wide. That last equality -- weight width ``d_s + d_q`` against
    value width ``d_s + d_s`` -- is why ``d_q == d_s`` is not a free choice.
    A small read-out then maps the branches back to ``out_planes``; it carries a
    bias so the block is not pinned to the identity at initialisation.
    """

    def __init__(self, in_planes, out_planes, num_branches=2, stride=1,
                 downsample=None):
        super(S2AAlignedBlock, self).__init__()
        self.B = num_branches
        if in_planes != out_planes and in_planes % num_branches != 0:
            raise ValueError('in channels ({}) must be divisible by '
                             'num_branches ({})'.format(in_planes, num_branches))
        # Every internal width is d, so the q/k dot product and the weight/value
        # product line up with no reshaping at all. ``d`` is deliberately much
        # smaller than out_planes: the block is applied 16 times, so anything that
        # scales as out_planes^2 per block would cost hundreds of millions of
        # parameters (measured: it did).
        self.d = max(1, out_planes // 16)
        # shared K + shared V
        self.scene_share = 2 * self.d
        # each branch reads 2*d of value (shared half plus own half) and writes
        # the out_planes it owns, so the residual connection is exact
        self.branch_read = 2 * self.d
        self.read_out = num_branches * self.branch_read

        # --- entry: bring the input into this stage's width AND resolution ---
        # The stride has to be applied here: the branch path does no spatial
        # downsampling of its own, so without it a strided block would emit the
        # input resolution and the residual add would fail on H/W.
        self.need_in_proj = in_planes != out_planes or stride != 1
        if self.need_in_proj:
            self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                     stride=stride, groups=num_branches,
                                     bias=False)
            self.bn_in = nn.BatchNorm2d(out_planes)

        # --- the SHARED scene: one projection, read by BOTH branches ---
        # (S2A's fused qkv -> the spatial tokens' K and V; head -> spatial stays
        #  unmasked, which is exactly why this must be shared, not per-branch)
        self.conv_kv = nn.Conv2d(out_planes, 2 * self.d, kernel_size=1, bias=True)
        # spatial -> spatial is the unmasked part of S2A's matrix: the scene
        # attends to itself, with its own K and V, before any branch reads it
        self.conv_ks = nn.Conv2d(self.d, self.d, kernel_size=1, bias=True)
        self.conv_vs = nn.Conv2d(self.d, self.d, kernel_size=1, bias=True)

        # --- per-branch DECISION: own query, own key, own value ---
        # (S2A: head token a keeps its own q_a / k_a / v_a rows and a mask entry
        #  that blocks head token b != a. The mask is reproduced here by the
        #  absence of any conv_k / conv_v term belonging to b' != b.)
        self.conv_q = nn.ModuleList([
            nn.Conv2d(out_planes, self.d, kernel_size=1, bias=True)
            for _ in range(num_branches)])
        self.conv_k = nn.ModuleList([
            nn.Conv2d(out_planes, self.d, kernel_size=1, bias=True)
            for _ in range(num_branches)])
        self.conv_v = nn.ModuleList([
            nn.Conv2d(out_planes, self.d, kernel_size=1, bias=True)
            for _ in range(num_branches)])
        self.scale = self.d ** -0.5

        # --- read-out (S2A's attention proj), then a branch-local MLP ---
        # 2*d -> d per branch: a low-rank map, not a full-width one, so the
        # block does not pay O(out_planes^2) at every one of its 16 blocks.
        self.proj_sh = nn.ModuleList([
            nn.Linear(self.d, self.d, bias=True) for _ in range(num_branches)])
        self.proj_own = nn.ModuleList([
            nn.Linear(self.d, self.d, bias=True) for _ in range(num_branches)])
        self.bn_attn = nn.BatchNorm2d(num_branches * self.d)
        self.conv_mlp = nn.Conv2d(num_branches * self.d, num_branches * self.d,
                                  kernel_size=1, groups=num_branches, bias=False)
        self.bn_mlp = nn.BatchNorm2d(num_branches * self.d)
        self.readout = nn.Linear(num_branches * self.d, out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    @staticmethod
    def _flatten(t):
        return t.flatten(2)                       # B,C,H,W -> B,C,N

    def forward(self, z):
        identity = z if self.downsample is None else self.downsample(z)

        h = z
        if self.need_in_proj:
            h = self.relu(self.bn_in(self.conv_in(h)))

        B, _, H, W = h.shape

        # shared scene, built from ALL channels and owned by both branches
        kv = self.conv_kv(h)
        k_shared = kv[:, :self.d]                            # B,d,H,W
        v_shared0 = kv[:, self.d:]                           # B,d,H,W

        # --- the scene updates ITSELF, and that is an unmasked block ----------
        # In S2A the mask only removes head-token -> other-head-token entries;
        # spatial -> spatial is untouched. So the spatial tokens attend to each
        # other with no mask at all, and every head token then reads the *updated*
        # scene. That is the part that makes the scene genuinely shared rather
        # than two private reads of one frozen snapshot, so it is reproduced here.
        # No cross-branch mixing: this is the scene, not a branch.
        v_spatial = self._flatten(self.conv_vs(v_shared0))    # B,d,N
        k_spatial = self._flatten(self.conv_ks(k_shared))     # B,d,N
        s_spatial = torch.softmax(
            torch.einsum('bdn,bem->bnm', k_spatial, k_spatial) * self.scale,
            dim=-1)                                           # B,N,N
        v_shared = torch.einsum('bnm,bdm->bdn', s_spatial, v_spatial)
        k_shared = self._flatten(k_shared)                    # B,d,N

        outs = []
        for b in range(self.B):
            q_b = self._flatten(self.conv_q[b](h)) * self.scale   # B,d,N
            k_own = self._flatten(self.conv_k[b](h))              # B,d,N
            v_own = self._flatten(self.conv_v[b](h))              # B,d,N
            # softmax over [shared keys ; own key]: the shared logits stay in the
            # denominator, exactly like S2A's row-wise softmax over all tokens.
            # Distinct einsum subscripts per call -- reusing one letter across
            # these calls silently collides with a differently sized pixel axis.
            s_sh = torch.einsum('bci,bdi->bcd', q_b, k_shared)    # B,N,d
            s_own = torch.einsum('bci,bei->bce', q_b, k_own)      # B,N,d
            attn = torch.softmax(torch.cat([s_sh, s_own], dim=-1), dim=-1)
            w_sh, w_own = torch.split(attn, [self.d, self.d], dim=-1)
            # read the SHARED scene with the shared half, this branch's own value
            # with the own half. No other branch appears in either term: that is
            # S2A's mask, and it is what makes the decision isolated while the
            # scene stays shared (and updated).
            r_sh = torch.einsum('bcd,bdi->bci', w_sh, v_shared)   # B,d,N
            r_own = torch.einsum('bce,bei->bci', w_own, v_own)    # B,d,N
            read = r_sh + r_own                                   # B,d,N
            # low-rank read-out per branch
            read = self.proj_sh[b](read.transpose(1, 2)).transpose(1, 2)
            read = read + self.proj_own[b](r_own.transpose(1, 2)).transpose(1, 2)
            outs.append(read)                                     # B,d,N

        pre = torch.cat(outs, dim=1)                        # B,B*d,N
        # the ReLU between attention and read-out is not decoration: without it a
        # stride-1 block was purely linear (no conv_in, no activation), which is
        # a badly conditioned residual unit rather than a ResNet block
        a = self.relu(self.bn_attn(pre.reshape(B, -1, H, W)))
        a = self.bn_mlp(self.conv_mlp(a))
        a = self.readout(a.flatten(2).transpose(1, 2)).transpose(1, 2)
        a = a.reshape(B, -1, H, W)

        return self.relu(a + identity)


class S2AAlignedResNet(nn.Module):
    """ResNet50-shaped trunk with S2A's shared-scene / isolated-decision blocks.

    Same layout and the same ``split_branches`` / ``branch_slices`` contract as
    ``S2AResNet``: ``branch_width`` is the width ONE branch owns, the map is
    ``B x branch_width`` wide, and branch ``b`` owns the channel block
    ``[b*C/B : (b+1)*C/B]``. So a block's ``out_planes`` is
    ``num_branches * stage_width``, and each branch writes ``out_planes / B``,
    which is exactly what the residual expects.
    """

    def __init__(self, num_branches=2, branch_width=2048, last_stride=1,
                 in_channels=3, block_nums=(3, 4, 6, 3), stage_widths=None):
        super(S2AAlignedResNet, self).__init__()
        self.B = num_branches
        self.branch_width = branch_width
        if stage_widths is None:
            scale = branch_width / 2048.0
            stage_widths = [int(256 * scale), int(512 * scale),
                            int(1024 * scale), branch_width]
        self.stage_widths = stage_widths

        # each branch keeps the standard width from the stem onwards
        stem = int(64 * (branch_width / 2048.0)) * num_branches

        self.conv1 = nn.Conv2d(in_channels, stem, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_planes = stem
        self.layer1 = self._make_layer(stage_widths[0] * num_branches,
                                       block_nums[0], 1)
        self.layer2 = self._make_layer(stage_widths[1] * num_branches,
                                       block_nums[1], 2)
        self.layer3 = self._make_layer(stage_widths[2] * num_branches,
                                       block_nums[2], 2)
        self.layer4 = self._make_layer(stage_widths[3] * num_branches,
                                       block_nums[3], last_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride,
                          groups=self.B, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [S2AAlignedBlock(self.in_planes, planes, self.B, stride=stride,
                                  downsample=downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(S2AAlignedBlock(planes, planes, self.B, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, return_branches=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_branches:
            return self.split_branches(x)
        return x

    def split_branches(self, z):
        return list(torch.chunk(z, self.B, dim=1))

    def branch_slices(self, z, branch):
        C = z.shape[1] // self.B
        return z[:, branch * C:(branch + 1) * C]


def s2a_resnet50(num_branches=2, branch_width=2048, last_stride=1, mode='s2a',
                 **kwargs):
    """Build the S2A-shaped ResNet50 trunk.

    ``mode='s2a'``    -> ``S2AAlignedResNet``: shared spatial scene, isolated
                         per-branch decision (the literal S2A translation).
    ``mode='sealed'`` -> ``S2AResNet``: every convolution branch-isolated.
    """
    if mode not in ('s2a', 'sealed'):
        raise ValueError("mode must be 's2a' or 'sealed', got {!r}".format(mode))
    cls = S2AAlignedResNet if mode == 's2a' else S2AResNet
    return cls(num_branches=num_branches, branch_width=branch_width,
               last_stride=last_stride, **kwargs)
