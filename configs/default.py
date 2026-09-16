from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.LAST_STRIDE = 1
cfg.MODEL.PRETRAIN = r"pretrain/resnet50-pretrained.pth"
cfg.MODEL.NECK = 'bnneck'
cfg.MODEL.ID_LOSS_WEIGHT = 1.0
cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
cfg.MODEL.FEAT_DIM = 2048
cfg.MODEL.NO_MARGIN = False
cfg.MODEL.LABELSMOOTH = True

# -----------------------------------------------------------------------------
# Dual-branch shared trunk (S2A-style)
#   F  = identity branch               : ID (softmax) + triplet loss
#   F' = identity-independent branch   : cloth softmax (+ optional colour histogram)
#   F vs F'                            : pushed apart (orthogonal)
# The two branches are contiguous channel blocks of ONE shared feature map.
#
# S2A_MODE picks HOW the isolation is done (see models/backbones/s2a_resnet.py):
#   's2a'    : CSCI's mechanism -- SHARED spatial K/V (both branches read one
#              scene, so a branch can steer the other through it) and an isolated
#              per-branch query. Isolation of the decision only.
#   'sealed' : every convolution uses groups=B, so branch 0 cannot influence
#              branch 1 on the direct OR the indirect path. Strictly stronger
#              isolation than CSCI, and a different mechanism.
# -----------------------------------------------------------------------------
cfg.MODEL.DUAL_BRANCH = True
cfg.MODEL.S2A_MODE = 's2a'            # 's2a' (matches CSCI) | 'sealed'
cfg.MODEL.S2A_BRANCHES = 2            # B; branch 0 = F (identity), 1 = F' (indep.)
cfg.MODEL.S2A_BRANCH_WIDTH = 2048     # width of ONE branch's features; map is B x this
cfg.MODEL.CLOTH_LOSS_WEIGHT = 1.0     # weight of the F' clothing softmax
cfg.MODEL.DISENTANGLE_WEIGHT = 1.0    # weight of the F / F' separation term
cfg.MODEL.DISENTANGLE_MARGIN = None   # None -> |cos| (CSCI); float -> hinge

# ---- clothing classification head ----
# 'linear' : plain dot-product head (original CC-DeepReID)
# 'cosine' : C2R-ReID's NormalizedClassifier -- L2-normalised features AND class
#            weights, so the logits are cosine similarities
cfg.MODEL.CLOTH_HEAD = 'linear'
cfg.MODEL.CLOTH_HEAD_SCALE = 16.0     # only used when CLOTH_HEAD == 'cosine'

# ---- colour-histogram regression on F' (CSCI's annotation-free supervision) ----
# A second supervision that runs ALONGSIDE the cloth softmax (which is kept).
# The target is an RGB-uv histogram computed from the same augmented image, so it
# needs no label at all. Profile 44 in CSCI == histblock + L1 + wt=100 + cosine loss.
cfg.MODEL.USE_HIST = False
cfg.MODEL.HIST_DIM = 32               # bins per axis; label length is HIST_DIM**2
cfg.MODEL.HIST_HIDDEN = 1024          # hidden width of the regression head
cfg.MODEL.HIST_SIGMA = 0.001          # inverse-quadratic kernel width (CSCI 44)
cfg.MODEL.HIST_INTENSITY_SCALE = False
cfg.MODEL.HIST_NORM = 'l1'            # 'l1' | 'l2' | None
cfg.MODEL.HIST_NORM_P = 1
cfg.MODEL.HIST_WEIGHT_SCALE = 100.0   # multiply the label (CSCI's wt)
cfg.MODEL.HIST_LOSS = 'cosine'        # 'cosine' (1-|cos|, CSCI 44) | 'mse' | 'l1'
cfg.MODEL.HIST_LOSS_WEIGHT = 1.0


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

cfg.INPUT = CN()
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PROB = 0.5
cfg.INPUT.RE_PROB = 0.5
cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
cfg.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------

cfg.DATASETS = CN()
cfg.DATASETS.NAMES = 'ltcc'
cfg.DATASETS.ROOT_DIR = r'../data'

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.DATALOADER.SAMPLER = 'triplet'
cfg.DATALOADER.NUM_INSTANCE = 16

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------

cfg.SOLVER = CN()
cfg.SOLVER.OPTIMIZER_NAME = "AdamW"
cfg.SOLVER.MAX_EPOCHS = 100
cfg.SOLVER.BASE_LR = 3e-4
cfg.SOLVER.SEED = 1234
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.MARGIN = 0.3

cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

cfg.SOLVER.WARMUP_EPOCHS = 5
cfg.SOLVER.WARMUP_TYPE = 'linear'
cfg.SOLVER.COSINE_MARGIN = 0.5
cfg.SOLVER.COSINE_SCALE = 30


cfg.SOLVER.LR_SCHEDULER = 'cosine'
cfg.SOLVER.STEPSIZE = [40, 70]
cfg.SOLVER.GAMMA = 0.1

cfg.SOLVER.CHECKPOINT_PERIOD = 10
cfg.SOLVER.LOG_PERIOD = 10
cfg.SOLVER.EVAL_PERIOD = 10
cfg.SOLVER.IMS_PER_BATCH = 64

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------

cfg.TEST = CN()
cfg.TEST.IMS_PER_BATCH = 128
cfg.TEST.WEIGHT = ""
cfg.TEST.NECK_FEAT = 'after'
cfg.TEST.FEAT_NORM = True

# 'cc'   -> report only the clothes-changing protocol (CSCI's published tables)
# 'both' -> additionally report the Standard (PRCC) / General (LTCC) protocol
cfg.TEST.MODE = 'both'
# which protocol drives best-checkpoint selection
cfg.TEST.EVAL_PROTOCOL = 'CC'

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
cfg.EXP_NAME = ""
cfg.LOG_DIR= "logs"
cfg.OUTPUT_DIR = "outputs"
