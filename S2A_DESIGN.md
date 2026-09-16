# S2A-style branch design —status and open questions

Recorded while working on CC-DeepReID. This file exists so the reasoning does not
have to be reconstructed (or misremembered) later.

## Decisions log

| date | decision | note |
|---|---|---|
| 2026-09-12 | **CAL removed entirely** (code, not just a flag) | Observation after a run: CAL "感觉效果更差了". **This is an impression from a run, not a controlled A/B.** What was deleted: `losses/clothes_adversarial_loss.py`, the `detach_cloth` discriminator path in `make_model.py`, the separate discriminator optimizer + `CAL_START_EPOCH` gate in `processor.py`, the `adv` loss term and its log column, the `exclude=` plumbing in `build_optimizer`, and all `USE_CAL` / `CAL_*` config keys. The clothing branch is back to a single label-smoothed CE head. |
| 2026-09-12 | S2A-style block implemented as an **option**, not the default path | `models/backbones/s2a_resnet.py`, verified by `scripts/check_s2a.py`. Effectiveness unvalidated (see below). |
| 2026-09-12 | S2A block **replaces** the SE channel attention in the model | `models/attention.py` deleted; `make_model.py` now builds `s2a_resnet50` and takes the two branches as channel blocks (`S2A_BRANCHES=2`, `S2A_BRANCH_WIDTH`). `F`, `F'`, the cloth head, the histogram head and the 8-slot forward signature all kept |
| 2026-09-12 | **Strict isolation is the intended design, not a defect to fix** | Asked directly whether the branches are "isolated like the two tokens in S2A"; measured answer is that they are isolated *more* strictly than S2A. Chosen deliberately: keep `direct = 0` **and** `indirect = 0`. |
| 2026-09-12 | **A second backbone added that DOES copy S2A's mechanism, and it is now the default** | `S2AAlignedBlock` / `S2AAlignedResNet` in the same file, selected by the new `MODEL.S2A_MODE = 's2a'` (the sealed variant stays available as `'sealed'` and nothing was deleted). This is what "对齐 S2A" required: a **shared spatial scene** (both branches read one `V_shared`, built from *all* channels) with only the **decision** isolated. It is a different mechanism from the earlier entry above, which is why both are kept. |

## Mistakes made along the way (recorded so they are not repeated)

**Conflating branch COUNT with branch WIDTH.** The trunk width is
`B x branch_width`, but the two were treated as one knob. Reasoning "we need 2048
per branch, so B=16" was wrong twice over: it was derived under the assumption
`layer4 == 512`, and it produces *16 branches*, not two wide ones. Measured:

| B | branch_width | trunk | branches | per branch | params |
|---|---|---|---|---|---|
| **2** | **2048** | 4096 | **2** | 2048 | **52.09M** |
| 2 | 512 | 1024 | 2 | 512 | 3.40M |
| 16 | 2048 | 32768 | 16 | 2048 | 416.72M |
| 16 | 512 | 8192 | 16 | 512 | 27.21M |

`make_model` only uses the first two branches (F and F'), so extra branches are
pure waste. Isolation is **exactly zero for every B** (it comes from `groups=B`,
not from the value of B), so there is never a reason to raise B just to change
width -- that is what `S2A_BRANCH_WIDTH` is for.

Also measured and recorded: at `branch_width=2048` the trunk is **2.22x a plain
ResNet50 (52.09M)**, i.e. NOT cheaper than two independent backbones, and one CPU
train step was ~20x slower than the plain model. `branch_width=512` costs 3.40M.

## What we wanted

Imitate CSCI's **S2A-self-attention**. The reading of S2A changed once its code was
read properly, so this file records both attempts:

* first reading -- "two feature streams which cannot leak into each other, with the
  isolation present in **every** ResNet block". That produced the sealed variant.
* correct reading -- S2A masks only *head-token -> other-head-token*; spatial K/V
  stay shared and readable by every token. So the mechanism is **a shared spatial
  scene with an isolated decision**, which is what `S2A_MODE: s2a` now implements.

The second reading is the one that matches the code, and it is the default.

## Why the existing channel attention does not do it

The current `models/attention.py` applies two SE gates to the **same** feature map
and pools afterwards. Measured facts:

* a linear probe still reads **28.6 %** identity information out of `F'`
  (chance 5 %) and **23.2 %** clothing out of `F` (chance 1.25 %);
* S2A is **not** exactly isolated once a whole block is measured (see the corrected
  table below), so the two are not the same kind of mechanism.

## What was chosen (original "plan A")

`models/backbones/s2a_resnet.py`:

* branch identity = a contiguous channel block of the shared trunk;
* **every** convolution uses `groups=B`, so no weight and no gradient crosses the
  branches (verified: cross-branch gradient is exactly `0`);
* parameter sharing is exact and cheap —a *depthwise* kernel is applied to every
  channel of every branch, so the same kernel serves both branches;
* the trunk is `B x` the standard width, so each branch keeps the standard width.

Two things this deliberately does **not** have (recorded so they are not
re-introduced by accident):

1. **No indirect exchange.** Each branch reads only its own channels, so branch 0
   cannot influence branch 1 at all —on either the direct or the indirect path
   (measured, see below). A variant where both branches read the *full* shared trunk
   ("shared read") was tried; it produces a large indirect path, but it was **not
   adopted**.
2. **No cross-branch parameter sharing beyond the depthwise kernels.**

## Correction: is this "isolated like the two tokens in S2A"?

This was asked directly, and the first answer given was wrong. Measured on the same
probe (2 stacked blocks, gradient of branch 1's output w.r.t. branch 0's input):

| mechanism | direct | indirect |
|---|---|---|
| `S2A` (token mask, shared spatial) | **0.0029** | 0.0000 |
| ours (channel blocks, `groups=B`) | **0.0000** | **0.0000** |

Three corrections this table forces:

1. **S2A's isolation is not exact at block level.** The earlier "exactly zero"
   (`d(t0)/d(x1) == 0`) was measured on `EvaAttention_sep_masked` *alone*. A full
   `EvaBlock` also applies a shared `self.mlp` / `self.norm2` on top of the attention:

   ```python
   x = x + drop_path(self.attn(self.norm1(x)))   # mask lives here
   x = x + drop_path(self.mlp(self.norm2(x)))    # shared, unmasked -> residual coupling
   ```

   So S2A isolates the *attention*, not the *block*. Our figure supersedes the
   single-layer one; the single-layer figure should not be quoted as S2A's behaviour.

2. **The earlier `~1.6e3` indirect reading was not S2A.** It came from the
   "shared read" variant tried in between. Real S2A measures `indirect = 0.0000`:
   head token 0 writes only its own `v_nth` into the concatenated attention output,
   so it has no channel through which to reach head token 1. The shared-spatial
   pathway is real in the *forward pass* but does not appear in that gradient pair.

3. **The two designs differ in kind, not in degree.** S2A keeps a shared spatial
   scene (both tokens read the same K/V) and isolates only the decision; we give each
   branch its own channels, so there is no shared *representation* at all —only
   shared *parameters* (the same depthwise kernel acting on both). Ours is strictly
   stronger isolation, and it is deliberately so.

## The two mechanisms, measured side by side

Both live in `models/backbones/s2a_resnet.py` and are selected by `MODEL.S2A_MODE`.
Numbers below are from `scripts/check_s2a_aligned.py` (2 stacked blocks) and from
counting parameters at `branch_width=2048`, `num_branches=2`:

| | `'s2a'` (S2AAlignedBlock) | `'sealed'` (S2ABlock) |
|---|---|---|
| branch identity | channel block | channel block |
| spatial scene | **shared** (`conv_kv`), and **updated** each block | absent |
| decision | per-branch `conv_q`/`conv_k`/`conv_v` | per-branch grouped convs |
| direct coupling | 0 | 0 |
| indirect coupling | **> 0** (the point of the design) | 0 |
| both branches move when the scene changes | **yes** | no |
| params | 64.01M (2.72x ResNet50) | 52.09M (2.22x) |
| config | `S2A_MODE: s2a` (default) | `S2A_MODE: sealed` |

The mapping from S2A's code to the CNN block, which is the whole point of the
exercise:

| S2A (`EvaAttention_sep_masked`) | `S2AAlignedBlock` |
|---|---|
| token index | spatial position |
| embedding channel | feature channel |
| `self.qkv` (one fused projection) | `conv_q` / `conv_k` / `conv_v` / `conv_kv`, all 1x1 |
| head token `a`'s mask row (`-inf` on the other head tokens) | branch `a`'s read uses no `conv_k`/`conv_v` term belonging to `b != a` |
| **spatial -> spatial is unmasked** | the scene attends to itself (`conv_ks`/`conv_vs`) and every branch reads the **updated** scene |
| head -> spatial stays unmasked | both branches read `V_shared` |
| row-wise softmax over all tokens | softmax over `[shared keys ; own key]`, so the shared logits stay in the denominator |
| shared `norm2` / `mlp` | `bn_mlp` / `conv_mlp` (kept branch-local, i.e. deliberately *not* copying S2A's leak) |

`conv_kv` is what makes the scene shared and it is a single module, so branch 0's
channels reach branch 1's output through it -- that is the measured `indirect > 0`
row above, and it is deliberate. The scene-update step matters for a second reason:
without it, "shared scene" would only mean two branches reading one frozen
snapshot, whereas S2A updates the spatial tokens as part of the same block.
Widths: every internal width is `d = out_planes // 16`, which is what keeps
`conv_kv` and the low-rank read-out from costing `O(out_planes^2)` at each of the
16 blocks.

## Mistakes made along the way, part 2 (the S2A-aligned variant)

Five separate defects, each caught only by running it. Recorded because four of
them were *reasoning* errors, not typos:

1. **An inline line break silently re-bound a function argument.** In

   ```python
   attn = torch.softmax((self._flatten(self.conv_q[b](h))
                         * self.scale).transpose(1, 2) @ k_cat,
                        dim=-1)
   ```

   the value of `k_cat` lands in `softmax`'s `dim` slot, and torch then reports a
   nonsensical *matmul* shape error. (Confirmed in isolation: `f(a @ b.T,\n b)`
   receives `b` as the second positional parameter.) Fixed by splitting it into
   statements; there is a comment at that line so it is not "tidied" back.
2. **Forcing every width to be equal.** Four consecutive shape errors came from
   assuming the shared key, the branch key, the query and the read-out all had to
   be the same width. They do not: `shared_k` must equal `query` (the dot
   product), and `w_own` must equal `v_own` (the weight/value product). Nothing
   else is constrained. Adding a low-rank read-out removed the last constraint.
3. **Reusing one einsum subscript across calls.** `'bcn,bdn->bcd'` was applied to
   two tensors with *different* channel counts, so the second call silently
   collided the pixel axis with the channel axis. Every call now uses its own
   letters.
4. **`need_in_proj` did not carry the stride.** The strided block therefore kept
   the input resolution and the residual add failed on H/W. Caught only because
   `layer2`'s stride was exercised, not `layer1`'s.
5. **Parameter explosion from a full-width attention.** With every width equal to
   `out_planes`, `conv_kv` alone was `O(out_planes^2)` per block; measured 1493M
   parameters at one point, then 634M. The internal width is now `out_planes // 16`
   and the read-out is low-rank: 63.38M.

Also fixed while here: the S2A-aligned block originally had **no nonlinearity**
between attention and read-out, so a stride-1 block was a purely linear residual
map. There is now a ReLU there.



## Open question (the reason this is not a conclusion)

**Whether any of this actually helps is unknown.** The isolation is exact by
construction, but:

* exact isolation is not the same as useful separation —the two branches still
  have to be *driven* apart by their losses;
* the earlier measurement that orthogonality does not remove information
  (`|cos|` 6.6 % → 0.97 % while the probes barely moved) is a warning that
  structural separation may not translate into the effect we want;
* two branches of half-width each are not obviously better than one branch of full
  width plus a head;
* the isolation forces **depthwise** convolutions (`groups=B`), so spatial mixing
  capacity per channel is far lower (9 params/channel instead of `9 x C`). This is an
  inherent cost of the design, not an implementation slip. At `branch_width=512` the
  whole model is 3.40M against 23.51M for a plain ResNet50 (0.145x), so the saving is
  paid for in per-channel capacity; at the **currently configured** `branch_width=2048`
  there is no saving at all (52.09M, 2.22x) —that configuration buys width, not
  efficiency.

Treat this as an **unvalidated hypothesis**, not a validated design. The right next
step is a controlled comparison against the plain dual-branch model (`SUPERVISED`
on identical data/schedule), reporting the same probes (identity probe on the
identity-independent branch, clothing probe on the identity branch) plus the
retrieval metrics.

## How the two variants actually differ (analysis + measurement)

### The single structural difference

**What is shared, and at which level.** Both designs share *parameters*; only one
shares a *representation*.

* `sealed`: the shared object is `conv_shared` / `conv_dw` / `conv_pw` (a
  depthwise kernel and grouped 1x1s) plus the BN statistics. Each branch has its
  own channel block and the branch *slices* of those kernels. Sharing is at the
  level of "the same numbers are used by both", so nothing is transmitted in the
  forward pass.
* `s2a`: the shared object is a **scene** of `d` channels per block
  (`conv_kv(h)`, updated by its own unmasked self-attention in
  `conv_ks`/`conv_vs`), which both branches **read** through their own queries.
  Sharing is at the level of "both branches consume the same tensor", so
  information *is* transmitted, every block, in the forward pass.

Everything else follows from this one difference. The read-out
(`readout`, a `Linear` over all branch outputs, plus the grouped `conv_mlp`) is
where the two halves meet again; in `sealed` there is nothing to meet.

### A measurement error worth recording

The first version of `scripts/compare_variants.py` measured
`|d(cloth CE) / d(branch-0 params)|` and read a non-zero value as "leakage". That
is **not a valid measure of isolation** in the sealed design: those depthwise
kernels are shared between branches, so their gradient necessarily contains
branch-1's contribution. A parameter gradient cannot tell "shared weight" apart
from "cross-branch path". The measure was dropped.

The valid measure is a **forward perturbation at a known boundary** (perturb
branch 0's channels of an intermediate tensor, watch branch 1's output), which
tests the topology claim directly. Measured on `layer4[0]`, same seed:

| mode | params (whole model) | perturb b0 -> b0 out | perturb b0 -> b1 out | fwd+bwd, batch 4 |
|---|---|---|---|---|
| `sealed` | 53.02M | 1.649e+00 | **0.000e+00** (exactly 0) | 4.28s |
| `s2a` | 64.94M | 1.774e+00 | **1.411e+00** (coupled) | 28.44s |

(Whole-model params include the id/cloth heads; trunk-only is 52.09M / 64.01M.)

So the difference is exactly what the design intends, and it is categorical rather
than gradual: `sealed` transmits **nothing**, `s2a` transmits **almost as much as
a branch reads about itself** (1.41 vs 1.77 -- branch 1's output is about 80 % as
sensitive to branch 0's content as branch 0's own output is).

### What "sharing the scene" buys and costs

`sealed` gives each branch a **clean, separable optimisation problem**. Its
identity branch has no reason to model clothing, because no clothing-carrying
evidence can reach it; the disentanglement loss becomes a regulariser on the
read-out rather than the load-bearing mechanism. Practical consequence: with
`sealed`, removing the cloth loss does **not** change `F`'s features, so the two
losses can be ablated independently. That is a real experimental advantage.

`s2a` is the opposite trade. The scene is a **shared bottleneck both branches can
write into**, which is precisely what creates a *soft* coupling: the only channel
through which the cloth loss can improve `F'` while damaging `F` is the scene,
and the only channel through which the identity loss can push back is also the
scene (plus the read-out). This is coordination, not leakage for its own sake, and
it is what CSCI relies on. The cost is that the identity branch's features now
depend on the cloth loss: ablating the cloth loss changes them.

A subtlety in favour of `s2a`: with `sealed`, the *only* coordination between
branches is the shared kernels' gradients and their BN statistics. Those are weak
and indirect signals, and the shared depthwise kernel can only receive a single
gradient per channel, so it cannot host a branch-dependent behaviour. With `s2a`
there is an explicit place for shared structure to live.

A subtlety against `s2a`: the scene is built from **all** channels, so `conv_kv`
is an information firehose. "Shared" here does not mean "safe to share" -- it
means `F'` can pull raw identity-bearing evidence into the scene and `F` reads it
back. The `|cos|` orthogonality measurement already warned that structural
separation does not automatically remove information.

### Capacity and cost

* `sealed` is cheap **because** isolation forces grouped/depthwise convolutions:
  spatial mixing is 9 params per channel instead of `9 x C`. `s2a` pays for a
  learned scene instead, so it is wider (64.94M vs 53.02M at the same nominal
  branch width) and, on this CPU box, **6.6x slower per step** (28.44s vs 4.28s).
  Part of that is the depthwise kernels being extremely cheap, and part is the
  scene's `N x N` self-attention, whose cost grows with image size.
* `sealed`'s spatial mixing is one fixed 3x3 kernel per channel; `s2a`'s is
  content-dependent and global. The latter has more expressive headroom but is
  quadratic and has no locality prior, which matters because CC-ReID datasets
  are not large.
* Branch width is a *design choice* in `s2a` (a learned read-out restores
  `out_planes`), whereas in `sealed` it is forced by the channel-block budget.

### Choosing between them

* Want a **guarantee** that clothing evidence cannot reach the identity branch,
  and the ability to ablate the two losses independently? `sealed`.
* Want the **closer transcription of CSCI**, and a deliberate shared-scene
  coordination channel? `s2a`.
* The honest summary: `sealed` is the stronger *structural* claim and the cheaper
  model; `s2a` is the more faithful *mechanism* and the more expensive one. Which
  one produces better CC-ReID numbers is still unmeasured -- neither variant has
  been trained to convergence, and the table above is a topology/cost comparison,
  not an accuracy comparison.
