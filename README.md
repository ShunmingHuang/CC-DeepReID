# CC-DeepReID

Clothes-changing ReID framework: ResNet50 backbone + **dual-branch channel
attention**, with the data layer and evaluation protocol aligned to
`ICCV-CSCI-Person-ReID`.

This document records what the framework *is*, which conventions are frozen, and
which knobs exist — so later experiments do not silently break the plumbing.

---

## 1. Architecture

```
image
  └─ ResNet50 (last_stride=1)                        → (B, 2048, H/16, W/16)
       └─ DualBranchChannelAttention                  models/attention.py
            ├─ SE branch 1 (re-weights the map) ────→ map_id    (B, 2048, H/16, W/16)
            │                                            └─ GAP → F   (B, 2048)
            └─ SE branch 2 (re-weights the map) ────→ map_cloth (B, 2048, H/16, W/16)
                                                         └─ GAP → F'  (B, 2048)
                 F  → BNNeck → classifier_id     → id logits
                 F' → BNNeck → classifier_cloth  → cloth logits
```

> **Naming.** `F` is the **identity** branch. `F'` is the **identity-independent**
> branch — it exists to carry whatever is *not* the person's identity, and
> clothing classification is only the supervision currently attached to it
> (`CLOTH_HEAD`, optional `CAL`). Read every `cloth*` identifier as "the module
> currently hanging off the identity-independent branch", not as the branch's
> definition: further modules will be attached there, and `map_cloth` is exposed
> precisely so they can consume the un-pooled spatial features.

**Channel attention comes first, pooling comes after**, and **the pooling is
per-branch** — each branch pools its own attention-weighted map:

* `map_id` / `map_cloth` are the attention-weighted maps *before* pooling. They
  are returned by the training forward so other modules can attach to the
  spatial features; they sit on the autograd path of `F` / `F'`.
* The SE gate needs a global pool *statistic* to compute its channel weights —
  that is inherent to squeeze-and-excitation (the "squeeze") and is not the
  pooling that produces `F` / `F'`.

Each `SE branch` is `mean(C) → Linear(C→C/r) → ReLU → Linear(C/r→C) → Sigmoid`
applied as per-channel weights on the spatial map. The branches have
**independent attention parameters and independent pooling**, so they can
specialise. `GAP(map_id) == F` (up to `MODEL.NECK`) and `GAP(map_cloth) == F'`
are asserted in `scripts/verify_alignment.py`.

Training forward returns, in order::

    id_score, F, cloth_score, F', global_feat, map_id, map_cloth

The tuple shape is identical when `MODEL.DUAL_BRANCH=False`; the dual-branch
extras are then `None`.

### Loss

The **dataset layer must match CSCI exactly; the loss layer is free to differ.**
Current objective:

```
L = W_id  · CE_id(F)              # label-smoothed identity softmax     (F)
  + W_tri · Triplet(F)            # CSCI's identity-only hard triplet   (F)
  + W_clo · CE_cloth(F')          # label-smoothed clothing softmax     (F')
  + W_dis · |cos(F, F')|          # push F and F' apart
```

* `Triplet(F)` is CSCI's triplet **verbatim** (`losses/hard_mine_triplet_loss.py`
  is transcribed from `ICCV-CSCI-Person-ReID/loss/triplet_loss.py`):
  `hard_example_mining` mines positives by *identity* only — **clothing is never
  consulted**. `targets_cloth` is still accepted by the signature and ignored.
  An earlier CC-DeepReID version used a *cloth-aware* triplet
  (`mask_pos = same_id & different_cloth`); that was experiment residue and has
  been removed. (`scripts/verify_alignment.py` asserts bit-for-bit equality with
  a re-implementation of CSCI's triplet.)
* `CE_cloth(F')` follows the same recipe as `CE_id(F)` — label-smoothed when
  `MODEL.LABELSMOOTH` is on. Note this is a **deliberate deviation**: CSCI uses
  plain `nn.CrossEntropyLoss()` for its clothing head.
* `F'` gets **no triplet term at all** — it is only asked to classify clothing,
  so it cannot absorb identity-discriminative structure.
* The separation term is CSCI's `Cosine_Disentangle`: `|cos|` is minimized at
  `cos = 0`, i.e. the two features are driven **orthogonal**. Note `|cos|` also
  penalises anti-parallel features (cos = -1); that is intentional and matches
  CSCI. Set `MODEL.DISENTANGLE_MARGIN` to a float to switch to a hinge
  `relu(cos - m)` instead.
* `F` is the feature used at test time; the retrieval path is unchanged by the
  dual branch.

> Where strict CSCI alignment **is** required — and is enforced by
> `scripts/audit_real_datasets.py` against the real data: the split protocol, and
> where every `cloth_id` comes from (§2).

---

## 2. Data layer and evaluation protocol (aligned to CSCI)

Entry point: `datasets/make_dataloader(cfg) -> (loaders, bundle)`.

Data tuples are **4-tuples** `(img_path, pid, camid, cloth_id)` (CSCI carries an
extra all-zero `aux_info` slot that is unused here).

### PRCC — two protocols over one gallery

| split | source | note |
|---|---|---|
| `train` | `rgb/train` | pids relabelled `0..N-1`; `val` is **not** merged |
| `val` | `rgb/val` | parsed and reported, never trained on |
| `query_diff` | `rgb/test/C` | clothes-changing (CC) protocol |
| `query_same` | `rgb/test/B` | standard (SC) protocol |
| `gallery` | `rgb/test/A` | shared by both |

* File naming differs between splits (this is how the PRCC release is built):
  `train`/`val` use `<cam>_cropped_rgb###.jpg` inside `<pid>/`, `test` uses
  `cropped_rgb###.jpg` inside `<cam>/<pid>/`.
* Cloth semantics: `<pid>` for A/B, `<pid>C` for C → **two cloth labels per
  identity** in train, matching CSCI's `pid*2` / `pid*2+1`.
* In test, A and B **share** a cloth id and C differs. This is what makes
  `same pid AND same cloth` pruning keep exactly the clothing-change pairs:
  A↔C for CC, A↔B for Standard.

### LTCC — one split, rule applied at eval time

| split | source |
|---|---|
| `train` | `train/` (full, **no cloth-change ID filtering at read time**) |
| `query` | `query/` |
| `gallery` | `test/` |

* `cloth_id = <pid>_<cam>` (CSCI's `(\w+)_c` regex), vocabulary built on train
  and reused for test. Unseen test clothings are appended instead of raising
  `KeyError` (upstream CSCI would crash there).
* The clothes-changing rule lives in the metric (`R1_mAP_eval_CC.compute`), not
  in the loader, so the CC and General numbers are computed from the same
  features.

### What gets reported

`MODEL`-agnostic; controlled by config:

* `TEST.MODE = 'both'` → PRCC: `CC` (test/C) **and** `Standard` (test/B);
  LTCC: `CC` **and** `General`.
* `TEST.MODE = 'cc'` → only the clothes-changing number.
* `TEST.EVAL_PROTOCOL = 'CC'` → which protocol picks the best checkpoint.

---

## 3. Configuration

Data/protocol (`configs/default.py`, per-dataset in `configs/datasets/*/*.yaml`):

| key | meaning |
|---|---|
| `DATASETS.NAMES` | `prcc` \| `ltcc` |
| `DATASETS.ROOT_DIR` | root containing `prcc/` and `LTCC_ReID/` |
| `TEST.MODE` | `both` \| `cc` |
| `TEST.EVAL_PROTOCOL` | `CC` \| `Standard` \| `General` |
| `DATALOADER.SAMPLER` | `triplet` \| `softmax` |
| `DATALOADER.NUM_INSTANCE` | instances per identity per batch |
| `SOLVER.IMS_PER_BATCH` | must be `num_ids_per_batch * NUM_INSTANCE` |

Model (`MODEL.*`):

| key | default | meaning |
|---|---|---|
| `DUAL_BRANCH` | `True` | `False` restores the single-branch baseline |
| `ATT_REDUCTION` | `16` | SE bottleneck ratio |
| `CLOTH_FEAT_DIM` | `-1` | `-1` keeps `F'` at 2048 (same width as `F`) |
| `ATT_PROJECTOR` | `False` | optional BN+FC head on `F'` |
| `CLOTH_LOSS_WEIGHT` | `1.0` | weight of `CE_cloth(F')` |
| `DISENTANGLE_WEIGHT` | `1.0` | weight of the separation term |
| `DISENTANGLE_MARGIN` | `None` | `None` → `\|cos\|` (CSCI); float → hinge |
| `NO_MARGIN` | `False` | triplet margin = 0 when `True` |
| `LABELSMOOTH` | `True` | label smoothing for both softmax heads |
| `CLOTH_HEAD` | `linear` | `linear` = dot-product head; `cosine` = C2R-ReID's `NormalizedClassifier` |
| `CLOTH_HEAD_SCALE` | `16.0` | logit scale, used only by the cosine head |
| `USE_CAL` | `False` | enable C2R's clothes-based adversarial loss |
| `CAL_WEIGHT` | `1.0` | weight of the adversarial term the backbone optimises |
| `CAL_SCALE` / `CAL_EPSILON` | `16.0` / `0.1` | CAL temperature / positive-class spread |
| `CAL_START_EPOCH` | `25` | **1-based epoch at which CAL switches on** |
| `CAL_LR` | `3.5e-4` | learning rate of the discriminator's own optimizer |
| `USE_HIST` | `False` | add CSCI's colour-histogram regression on `F'` |
| `HIST_DIM` | `32` | bins per axis; the label/prediction length is `HIST_DIM**2` |
| `HIST_HIDDEN` | `1024` | hidden width of the regression head |
| `HIST_SIGMA` | `0.001` | inverse-quadratic kernel width (CSCI profile 44) |
| `HIST_INTENSITY_SCALE` | `False` | multiply counts by the RGB intensity norm |
| `HIST_NORM` / `HIST_NORM_P` | `l1` / `1` | label normalisation |
| `HIST_WEIGHT_SCALE` | `100.0` | multiply the label (CSCI's `wt`) |
| `HIST_LOSS` | `cosine` | `cosine` (`1-\|cos\|`, CSCI 44) \| `mse` \| `l1` |
| `HIST_LOSS_WEIGHT` | `1.0` | weight of the histogram term |

### Colour-histogram supervision (optional, CSCI)

`MODEL.USE_HIST=True` attaches a **second, annotation-free** supervision to `F'`,
**alongside** the cloth softmax (which is kept unchanged):

```
augmented image ─┬─► backbone ─► F' ─► hist_head ─► hist_pred ─┐
                 │                                             ├─► 1-|cos|
                 └─► RGBuvHistBlock ─► L1 norm ×100 ─► label ───┘
```

* The target is an **RGB-uv histogram** (`datasets/histogram.py`, transcribed from
  CSCI's `data/rgbuc.py`), computed from the *same augmented tensor* that feeds the
  network -- no clothing label involved.
* The block is **bit-identical to CSCI's** (verified to `~1e-9`); see
  `scripts/verify_alignment.py`, which imports CSCI's class and compares directly.
* Two behaviours of CSCI's implementation are reproduced deliberately rather than
  "fixed", and both are easy to get wrong:
  * `insz` is **never used** -- CSCI has no resize call, so the histogram is
    computed at the image's native resolution;
  * the `v` axis is not uniform across planes: plane 2 uses `log(I2/I1)`, i.e. `u`
    and `v` share the same numerator channel. Making `v = log(I_a/I2)` everywhere
    silently collapses plane 2 (`log(I2/I2) = 0`).
* The block only supports **batch size 1** (CSCI fills sample 0 only); the loader
  calls it per sample and it raises on `B > 1` instead of silently zeroing.
* The cloth softmax keeps running next to it -- the two supervisions are additive.

### C2R-ReID clothes branch (optional)

Two independent switches, both off by default.

**1. `MODEL.CLOTH_HEAD='cosine'`** — replaces `F'`'s dot-product classifier with
C2R's `NormalizedClassifier` (`models/classifier.py`): the feature *and* the class
weights are L2-normalised, so the logits are `scale * cos(...)`. Motivated by the
long-tailed clothing classes (LTCC: 1..14 outfits per identity).

**2. `MODEL.USE_CAL=True`** — adds C2R's clothes-based adversarial loss (CAL). The
**same** clothing head is used twice per iteration, on two different graphs:

```
backbone step : cloth_score = head(bottleneck(F'))               # LIVE   -> CAL -> loss
discriminator : cloth_score = head(pool(map_cloth.detach()))     # FROZEN -> CAL -> optimizer_cc
```

* the discriminator has its **own optimizer** and runs **after** the backbone
  step, so its gradient never reaches the backbone;
* the discriminator path deliberately **skips `cloth_bottleneck`** — a
  parameterised normalisation there would build its own `grad_fn` and the branch
  would stop being a pure "frozen feature" probe;
* both are gated by `MODEL.CAL_START_EPOCH`: before it, neither the discriminator
  nor the adversarial term runs (C2R's `START_EPOCH_CC`/`START_EPOCH_ADV`);
* **`cloth_classifier` is excluded from the main optimizer** when `USE_CAL=True`
  (`train.py` passes `exclude=['cloth_classifier']` to `build_optimizer`). In C2R
  the clothes classifier is a separate module the main optimizer never sees; here
  it is a submodule of the model, so it has to be removed explicitly -- otherwise
  it is stepped twice per iteration and its effective lr is the sum of
  `SOLVER.BASE_LR` and `CAL_LR`. Note the *backbone* still receives the cloth
  softmax gradient (only the head's own weights are excluded), and `cloth_bottleneck`
  is still in the main optimizer, exactly as `F'`'s normalisation is part of `F'`;
* the positive mask is C2R's `pid2clothes[pids]` — every clothing class owned by
  the anchor's identity. `DatasetBundle` exposes it as
  `(num_train_pids, num_train_clothes)`, built by the dataset classes.

> The adversarial term is **not** an attribute-removal loss. It is a weighted
> negative log-likelihood that spreads probability over the identity's *own*
> outfits, and the backbone **minimises** it. Whether that makes the backbone
> clothes-agnostic is an empirical question, not a consequence of the code — see
> the note in `losses/clothes_adversarial_loss.py`.

`MODEL.PRETRAIN` exists in the config but is **not used by the training code**
(`train.py`/`make_model` never call `load_parameter`). The backbone is trained
from scratch. `test.py` does load a checkpoint via `load_parameter`.

### How to launch training

`train.py` is the only training entry point. Its `opts` argument is the yacs
**positional remainder**, not a `--opts` flag:

```bash
cd CC-DeepReID

# PRCC (config already carries NAMES=prcc, ROOT_DIR=../data)
python train.py --config_file configs/datasets/prcc/resnet.yaml

# LTCC
python train.py --config_file configs/datasets/ltcc/resnet.yaml

# override anything on the fly, e.g. a different data root or batch size
python train.py --config_file configs/datasets/prcc/resnet.yaml \
    DATASETS.ROOT_DIR /your/data SOLVER.IMS_PER_BATCH 64
```

`ROOT_DIR` is relative to the CWD, so run from `CC-DeepReID` when leaving it at
`../data`. Outputs land in `<OUTPUT_DIR>/<EXP_NAME>/` (`outputs/prcc/prcc_resnet/`)
and logs in `<LOG_DIR>/<EXP_NAME>/`.

Verified end to end on the real data through this exact entry point: config merge,
`../data` resolution, real splits (PRCC 150 ids / 17,896 images; 300 cloth
classes), model build, one full epoch of the real loop, both evaluation protocols,
`resnet_1.pth` + `resnet_best.pth` written with the dual-branch heads present
(334 state-dict keys).

---

## 4. Constraints worth knowing before changing things

1. **`F` and `F'` must have the same width.** The separation loss compares them
   1:1. If you set `CLOTH_FEAT_DIM != 2048`, `DisentangleLoss` raises rather
   than silently projecting — add an explicit projector if you need asymmetry.
2. **`MODEL.NECK` applies to `F` only.** `F'` always goes through its own
   `cloth_bottleneck` (BatchNorm1d).
3. **`F'`'s BatchNorm falls back to running statistics** when a batch contains a
   cloth class exactly once (BatchNorm1d cannot handle a singleton). With
   `IMS_PER_BATCH = num_ids * NUM_INSTANCE` this is rare; if the cloth branch
   accuracy stays near chance, check this first.
4. **The cloth vocabulary is per-dataset and built from the train split.** Its
   size is the cloth head's class count; test-time clothings never affect it.
5. **`torch.autocast` is used on CUDA (`GradScaler`).** The triplet distance is
   forced to fp32 inside `TripletLoss` because the 2048-dim squared sum is the
   one place fp16 could overflow.
6. **Only `F` is returned in `eval()` mode.** Anything you want measured at test
   time must either come from `F` or be exposed through the `.training` path.
7. **Protocol changes must stay in the datasets + metrics**, not in the model, so
   that CC/General remain comparable across experiments.

---

## 5. Verification scripts

All three are offline (synthetic data, no dataset download required) and run on
CPU; they exist so that a change to the plumbing fails loudly here rather than
halfway through a GPU run.

| script | what it proves |
|---|---|
| `scripts/compare_with_csci.py` | **the definitive check**: re-derives every split, pid, camid and cloth id directly from the filesystem using CSCI's own formulas, then compares against our loaders — per-image, and reports whether the cloth *label numbering* matches, not just the grouping |
| `scripts/audit_real_datasets.py` | split/label semantics, official PRCC counts, CC-rule effectiveness, triplets' reachable positives (runs on the real data) |
| `scripts/real_data_cpu_check.py` | real batches + full real query/gallery evaluation through the processor (CPU, slow) |
| `scripts/verify_alignment.py` | dual-branch shapes, losses, CC vs General metric, triplet == CSCI reference (26+ assertions) |
| `scripts/train_smoke_test.py` | the real `do_train` runs end to end: train → dual-protocol eval → checkpoint |
| `scripts/infer_smoke_test.py` | `load_parameter` + `do_inference` reproduces the training-time metrics |
| `scripts/preflight_3090.py` | **run this on the GPU host**: CUDA build/visibility, AMP fwd+bwd on GPU, peak VRAM, dataset layout, one real train+eval step |

```bash
python scripts/compare_with_csci.py      # dataset layer == CSCI, on the real data
python scripts/audit_real_datasets.py    # split/label sanity + official counts
python scripts/verify_alignment.py       # model/loss/metric assertions
python scripts/train_smoke_test.py       # end-to-end training
python scripts/infer_smoke_test.py       # run after train_smoke_test
```

Last verified result on the local data: `DATASET LAYER IS IDENTICAL TO CSCI` —
PRCC (train/val/test-A/test-B/test-C) and LTCC (train/query/test) are per-image
identical in image set, `pid`, `camid` and `cloth_id`, **including the cloth
label numbering**.

> **Trap this check caught:** PRCC's clothing key is
> `osp.basename(pdir)` *as a string* (`"092"`, leading zeros kept) for A/B and
> `basename + cam` for C. Passing the pid through `int()` first produces `"92"`,
> which changes the key set, and therefore renumbers **all 300** clothing labels
> even though the grouping stays identical. Keep the folder name as a string.

> Second trap: LTCC's `cloth_id` key is `<pid>_<cam>`, which shares a namespace
> with the identity ids. CSCI looks the test key up in the train-built
> vocabulary, so a dataset whose train and test ids are disjoint makes the
> pristine loader raise `KeyError` (on the local copy: 221 unseen keys). We append
> unseen keys to a test-only vocabulary instead; the equivalence structure is
> unchanged because `cloth_id` is only ever compared for equality inside a split.

---

## 6. Expected dataset layout

```
<ROOT>/
├── prcc/rgb/
│   ├── train/<pid>/<A|B|C>_cropped_rgb###.jpg
│   ├── val/<pid>/<A|B|C>_cropped_rgb###.jpg
│   └── test/{A,B,C}/<pid>/cropped_rgb###.jpg
└── LTCC_ReID/
    ├── train/<pid>_<cam>_c<cloth>_<frame>.png
    ├── query/<pid>_<cam>_c<cloth>_<frame>.png
    └── test/<pid>_<cam>_c<cloth>_<frame>.png
```

Use `python check_dataset.py` to print per-split `(imgs, pids, cams, clothes)`
before trusting a new dataset copy.
