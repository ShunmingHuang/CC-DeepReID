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
# Dual-branch channel attention (behind the ResNet50 backbone)
#   F  = identity branch  : ID (softmax) + triplet loss, used at test time
#   F' = clothing branch  : ID-style softmax over the clothing vocabulary
#   F vs F'               : pushed apart (orthogonal) by ClothDisentangleLoss
# -----------------------------------------------------------------------------
cfg.MODEL.DUAL_BRANCH = True
cfg.MODEL.ATT_REDUCTION = 16          # squeeze-excite bottleneck ratio
cfg.MODEL.CLOTH_FEAT_DIM = -1         # -1 keeps F' at 2048 (same width as F)
cfg.MODEL.ATT_PROJECTOR = False       # optional BN+FC head on F'
cfg.MODEL.CLOTH_LOSS_WEIGHT = 1.0     # weight of the F' clothing softmax
cfg.MODEL.DISENTANGLE_WEIGHT = 1.0    # weight of the F / F' separation term
cfg.MODEL.DISENTANGLE_MARGIN = None   # None -> |cos| (CSCI); float -> hinge

# ---- clothing classification head ----
# 'linear' : plain dot-product head (original CC-DeepReID)
# 'cosine' : C2R-ReID's NormalizedClassifier -- L2-normalised features AND class
#            weights, so the logits are cosine similarities
cfg.MODEL.CLOTH_HEAD = 'linear'
cfg.MODEL.CLOTH_HEAD_SCALE = 16.0     # only used when CLOTH_HEAD == 'cosine'

# ---- C2R-ReID clothes-based adversarial loss (CAL) ----
# The clothing discriminator is trained on DETACHED features with its own
# optimizer, and only from MODEL.CAL_START_EPOCH onwards; the backbone sees the
# same loss through live (non-detached) features.
cfg.MODEL.USE_CAL = False
cfg.MODEL.CAL_WEIGHT = 1.0
cfg.MODEL.CAL_SCALE = 16.0
cfg.MODEL.CAL_EPSILON = 0.1
cfg.MODEL.CAL_START_EPOCH = 25        # 1-based epoch at which CAL switches on
cfg.MODEL.CAL_LR = 3.5e-4             # lr of the discriminator's own optimizer


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
