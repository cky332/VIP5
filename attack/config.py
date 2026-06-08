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
K_POPULAR      = 20                  # "mean" 模式:取 top-K 热门求平均
CENTROID_MODE  = "mean"              # 攻击目标:"mean"=top-K 热门平均质心 ; "top1"=单个最热门商品的特征
TARGET_ITEM    = None                # 指定具体商品ID(str/int)作目标;非 None 时覆盖 CENTROID_MODE
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


# ===========================================================================
# X-Transfer (arXiv 2505.05528) — black-box, super-transferable, single-target
# popularity mimicry. Purely ADDITIVE: the single-CLIP PGD attack above is
# unaffected; all XT artifacts live under attack/out/xtransfer/.
# ===========================================================================
XT_TARGETED       = True              # True: rank-up toward popular centroid; False: degrade (ablation)
XT_USE_OPEN_CLIP  = True              # True: diverse open_clip pool; False: OpenAI-clip-only fallback
XT_INCLUDE_VICTIM = False             # True: also add OpenAI ViT-B/32 to the pool (white-box upper bound)

XT_EPS            = 16.0 / 255.0      # L-inf budget on cover pixels
XT_ALPHA          = 16.0 / 255.0 / 5.0  # sign-gradient step (eps/5, transfer-attack style)
XT_STEPS          = 200               # optimization steps j
XT_K_SELECT       = 4                 # surrogates selected per step (UCB)
XT_BATCH          = 8                 # minibatch of D' covers per step (universal mode)
XT_UCB_C          = 1.0               # UCB exploration constant
XT_MOMENTUM       = 0.9               # reward EMA factor m
XT_DPRIME_MODE    = "single"          # "single" (per-target, chosen) | "universal" (one delta for a sample)
XT_DPRIME_SIZE    = 64                # universal mode only: #covers sampled into D'

# Attack TARGET embedding (what each surrogate's (cover+delta) is pulled toward):
XT_CENTROID_MODE  = "top1"            # "top1": single MOST-popular item's image (default) | "mean": top-K avg
XT_TARGET_ITEM    = None              # specific item id (str/int); overrides XT_CENTROID_MODE when set

# Black-box surrogate search space (MUST exclude the victim OpenAI ViT-B/32).
# entry = {"backend": "open_clip"|"openai_clip", "name": <model>, "pretrained": <tag|None>}
XT_SEARCH_SPACE_OPENCLIP = [   # timm-free (ViT/RN only) so `pip install --no-deps` open_clip works
    {"backend": "open_clip", "name": "ViT-B-16",           "pretrained": "laion2b_s34b_b88k"},
    {"backend": "open_clip", "name": "ViT-L-14",           "pretrained": "laion2b_s32b_b82k"},
    {"backend": "open_clip", "name": "ViT-B-32",           "pretrained": "laion2b_s34b_b79k"},
    {"backend": "open_clip", "name": "ViT-B-16",           "pretrained": "datacomp_xl_s13b_b90k"},
    {"backend": "open_clip", "name": "ViT-L-14",           "pretrained": "datacomp_xl_s13b_b90k"},
    {"backend": "open_clip", "name": "ViT-B-16-quickgelu", "pretrained": "metaclip_400m"},
    {"backend": "open_clip", "name": "ViT-B-32-quickgelu", "pretrained": "metaclip_400m"},
    {"backend": "open_clip", "name": "RN50",               "pretrained": "yfcc15m"},
    # need timm / more GPU mem; enable if available:
    # {"backend": "open_clip", "name": "convnext_base_w",  "pretrained": "laion2b_s13b_b82k"},
    # {"backend": "open_clip", "name": "ViT-H-14",         "pretrained": "laion2b_s32b_b79k"},
]
XT_SEARCH_SPACE_OPENAI = [            # no new deps; cross-arch but single (WIT) pretrain distribution
    {"backend": "openai_clip", "name": "ViT-B/16",       "pretrained": None},
    {"backend": "openai_clip", "name": "ViT-L/14",       "pretrained": None},
    {"backend": "openai_clip", "name": "ViT-L/14@336px", "pretrained": None},
    {"backend": "openai_clip", "name": "RN50",           "pretrained": None},
    {"backend": "openai_clip", "name": "RN101",          "pretrained": None},
    {"backend": "openai_clip", "name": "RN50x4",         "pretrained": None},
]
XT_VICTIM_ENTRY = {"backend": "openai_clip", "name": "ViT-B/32", "pretrained": None}  # held-out victim


def xt_search_space():
    """Active surrogate list (victim appended iff XT_INCLUDE_VICTIM)."""
    base = list(XT_SEARCH_SPACE_OPENCLIP if XT_USE_OPEN_CLIP else XT_SEARCH_SPACE_OPENAI)
    if XT_INCLUDE_VICTIM:
        base.append(dict(XT_VICTIM_ENTRY))
    return base


# ---- X-Transfer output dirs (under attack/out/xtransfer/, never clobber PGD outputs) ----
XT_OUT_DIR         = "attack/out/xtransfer"
XT_CENTROID_DIR    = "attack/out/xtransfer/centroids"
XT_DELTA_DIR       = "attack/out/xtransfer/delta"
XT_POIS_FEAT_DIR   = "attack/out/xtransfer/poisoned_features/{split}".format(split=SPLIT)
XT_CLEAN_FEAT_DIR  = "attack/out/xtransfer/clean_features/{split}".format(split=SPLIT)
XT_PERT_IMG_DIR    = "attack/out/xtransfer/perturbed_images"
XT_RESULTS_DIR     = "attack/out/xtransfer/results"
XT_RESULTS_JSON    = "attack/out/xtransfer/results/xtransfer_pointwise.json"
XT_VICTIM_CENTROID = "attack/out/xtransfer/centroids/__victim__.npy"
