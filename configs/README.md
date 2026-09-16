# Configs

Every file here is complete and runnable: `train.py --config_file <path>` loads it
through a utf-8-safe `merge_from_file`, so each yaml carries the full schema
(MODEL / INPUT / DATASETS / DATALOADER / SOLVER / TEST) and can be read on its own
without cross-referencing `configs/default.py`.

Validate all of them at once (loads, builds the model, forwards and backwards,
checks the batch recipe against the real dataset):

```
python scripts/check_configs.py
```

## Files

| config | dataset | `MODEL.S2A_MODE` | batch | notes |
|---|---|---|---|---|
| `datasets/prcc/resnet.yaml` | PRCC | `s2a` | 64 / 16 inst | the reference config; same as `prcc/s2a.yaml` |
| `datasets/prcc/s2a.yaml` | PRCC | `s2a` | 64 / 16 inst | CSCI's mechanism: shared scene, isolated decision |
| `datasets/prcc/sealed.yaml` | PRCC | `sealed` | 64 / 16 inst | strict `groups=B` isolation, provably zero forward coupling |
| `datasets/ltcc/resnet.yaml` | LTCC | `s2a` | 32 / 8 inst | the reference config; same as `ltcc/s2a.yaml` |
| `datasets/ltcc/s2a.yaml` | LTCC | `s2a` | 32 / 8 inst | CSCI's mechanism |
| `datasets/ltcc/sealed.yaml` | LTCC | `sealed` | 32 / 8 inst | strict isolation |

LTCC uses 32/8 rather than PRCC's 64/16 because LTCC rarely has 16 images of one
identity in a batch and its splits are ~2.3x larger.

## The one setting that matters most

`MODEL.S2A_MODE` selects how the two branches are kept apart. The two options are
**different mechanisms**, not two settings of one:

* `s2a` — a **shared spatial scene** (`conv_kv`, updated every block by its own
  unmasked self-attention) that both branches read, with only the **decision**
  isolated. Measured forward coupling branch 0 -> branch 1: **non-zero**.
* `sealed` — every convolution uses `groups=B`, so branch 0 cannot reach branch 1
  on the forward path at all. Measured forward coupling: **exactly 0**.

Measured at `layer4[0]`, same seed (`python scripts/compare_variants.py`):

| mode | params (whole model) | perturb b0 -> b1 | fwd+bwd batch 4 (CPU) |
|---|---|---|---|
| `sealed` | 53.02M | 0.000e+00 | 4.28s |
| `s2a` | 64.94M | 1.411e+00 | 28.44s |

`sealed` is cheaper and lets the two losses be ablated independently (the cloth
loss does not change `F`'s features). `s2a` is the faithful transcription and has
a deliberate shared-scene coupling. Full analysis in `../S2A_DESIGN.md`.

## Running

From the repository root:

```
# PRCC, CSCI's mechanism (default)
python train.py --config_file configs/datasets/prcc/s2a.yaml \
    DATASETS.ROOT_DIR /path/to/data

# PRCC, strict isolation
python train.py --config_file configs/datasets/prcc/sealed.yaml \
    DATASETS.ROOT_DIR /path/to/data

# LTCC
python train.py --config_file configs/datasets/ltcc/s2a.yaml \
    DATASETS.ROOT_DIR /path/to/data
```

`opts` are **positional** overrides (there is no `--opts` flag), applied after the
file, e.g.:

```
python train.py --config_file configs/datasets/prcc/s2a.yaml \
    DATASETS.ROOT_DIR /data MODEL.DEVICE cuda SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR outputs/prcc_s2a EXP_NAME prcc_s2a
```

Every knob in the yaml can be overridden this way without editing the file.

## Turning on the extra supervision

The colour-histogram regression on `F'` (CSCI's profile-44 recipe) is present in
all configs but disabled. Enable it per run rather than by editing files:

```
python train.py --config_file configs/datasets/prcc/s2a.yaml \
    DATASETS.ROOT_DIR /data MODEL.USE_HIST True
```

Its weight is `MODEL.HIST_LOSS_WEIGHT` (default 1.0) and it runs **alongside** the
cloth softmax, which stays active. Verified by `scripts/verify_alignment.py`.

## Adding a config

Copy the closest file and change only what differs — the loader merges onto
`configs/default.py`, so a partial yaml also works, but a complete one is easier
to audit. Then add the path to `CONFIGS` in `scripts/check_configs.py`.
