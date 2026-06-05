"""Single source of truth for the VIP5 popularity-mimicry attack.

All paths are relative to the repo root (scripts chdir there via common.py).
Edit here to match your trained run.
"""

# ---- model / data (must match your training run) ----
SPLIT          = "toys"
BACKBONE       = "t5-small"
FEATURE_TYPE   = "vitb32"            # CLIP "ViT-B/32", feat_dim 512
SIZE_RATIO     = 2                   # n_vis_tokens = image_feature_size_ratio
REDUCTION      = 8
CKPT           = "snap/toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
MAX_TEXT_LEN   = 1024
BATCH_SIZE     = 21
DEVICE         = "cuda:0"

# ---- data / asset paths (relative to repo root) ----
FEAT_DIR       = "features/{ft}_features/{split}".format(ft=FEATURE_TYPE, split=SPLIT)
PHOTOS_ZIP     = "vip5/vip5_photos.zip"        # where gdown put it; change if different
PHOTOS_DIR     = "vip5/vip5_photos"            # unzip target
ITEM2IMG       = "data/{split}/item2img_dict.pkl".format(split=SPLIT)
SEQDATA        = "data/{split}/sequential_data.txt".format(split=SPLIT)
DATAMAPS       = "data/{split}/datamaps.json".format(split=SPLIT)

# ---- attack hyper-params ----
K_POPULAR      = 20                  # top-K popular items -> centroid
N_NEG          = 20                  # negatives per user (1 pos + 20 neg = 21)
EPSILON        = 16.0 / 255.0        # L-inf budget on pixels
PGD_STEPS      = 200
PGD_STEP_SIZE  = 2.0 / 255.0         # alpha
PGD_LOSS       = "cosine"            # "cosine" or "l2"
N_TEST_USERS   = 500                 # subsample for speed; None = all
SEED           = 2022

# feature-space ablation (no images needed)
ABLATION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # interpolate clean -> centroid

# ---- output dirs ----
OUT_DIR            = "attack/out"
POISONED_FEAT_DIR  = "attack/out/poisoned_features/{split}".format(split=SPLIT)
CLEAN_FEAT_DIR     = "attack/out/clean_features/{split}".format(split=SPLIT)   # re-extracted clean (confounder control)
PERTURBED_IMG_DIR  = "attack/out/perturbed_images"
RESULTS_DIR        = "attack/out/results"
CENTROID_PATH      = "attack/out/centroid.npy"
EXTRACTION_CHECK   = "attack/out/extraction_check.json"

# CLIP feature normalization: resolved empirically by clip_extract.py and persisted
# in EXTRACTION_CHECK; read via common.get_clip_norm(). None until resolved.
