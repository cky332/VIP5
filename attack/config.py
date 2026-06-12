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
XT_USE_OPEN_CLIP  = False             # False: OpenAI-clip-only pool (no new deps, DEFAULT); True: diverse open_clip pool (needs --no-deps install)
XT_INCLUDE_VICTIM = False             # True: also add OpenAI ViT-B/32 to the pool (white-box upper bound)

XT_EPS            = 16.0 / 255.0      # L-inf budget on cover pixels
XT_ALPHA          = 16.0 / 255.0 / 5.0  # sign-gradient step (eps/5, transfer-attack style)
XT_STEPS          = 200               # optimization steps j
XT_K_SELECT       = 4                 # surrogates selected per step (UCB)
XT_GPU_RESIDENT   = True              # keep all surrogates on GPU (fast); False = per-step CPU<->GPU swap (low VRAM)
XT_BATCH          = 8                 # minibatch of D' covers per step (universal mode)
XT_UCB_C          = 1.0               # UCB exploration constant
XT_MOMENTUM       = 0.9               # reward EMA factor m
XT_DPRIME_MODE    = "single"          # "single" (per-target, chosen) | "universal" (one delta for a sample)
XT_DPRIME_SIZE    = 64                # universal mode only: #covers sampled into D'

# Attack TARGET embedding (what each surrogate's (cover+delta) is pulled toward):
XT_CENTROID_MODE  = "top1"            # "top1": single MOST-popular item's image (default) | "mean": top-K avg
XT_TARGET_ITEM    = None              # specific item id (str/int); overrides XT_CENTROID_MODE when set

# M-Attack-style local matching (arXiv 2503.10635): random-resized-crop the adversarial
# cover each step before matching -> less overfit to surrogates, better black-box transfer.
XT_CROP        = False                # enable random-crop local matching
XT_CROP_SCALE  = (0.5, 1.0)          # crop area fraction range (M-Attack uses [0.5,1.0])
XT_CROP_RATIO  = (0.75, 1.3333)      # crop aspect-ratio range

# Black-box surrogate search space (MUST exclude the victim OpenAI ViT-B/32).
# entry = {"backend": "open_clip"|"openai_clip", "name": <model>, "pretrained": <tag|None>}
XT_SEARCH_SPACE_OPENCLIP = [   # timm-free; tags valid in open_clip 2.20 (unknown tags auto-skipped)
    {"backend": "open_clip", "name": "ViT-B-16", "pretrained": "laion2b_s34b_b88k"},
    {"backend": "open_clip", "name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
    {"backend": "open_clip", "name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    {"backend": "open_clip", "name": "ViT-B-16", "pretrained": "datacomp_l_s1b_b8k"},
    {"backend": "open_clip", "name": "ViT-B-16", "pretrained": "commonpool_l_clip_s1b_b8k"},
    {"backend": "open_clip", "name": "ViT-B-16", "pretrained": "laion400m_e32"},
    {"backend": "open_clip", "name": "ViT-B-32", "pretrained": "laion400m_e32"},
    {"backend": "open_clip", "name": "RN50",     "pretrained": "yfcc15m"},
    # newer/heavier (need open_clip>=2.24 or timm); auto-skipped if unavailable:
    # {"backend": "open_clip", "name": "ViT-B-16", "pretrained": "datacomp_xl_s13b_b90k"},
    # {"backend": "open_clip", "name": "ViT-B-16-quickgelu", "pretrained": "metaclip_400m"},
    # {"backend": "open_clip", "name": "convnext_base_w", "pretrained": "laion2b_s13b_b82k"},
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


# ===========================================================================
# AnyAttack (arXiv 2410.05346) — self-contained lightweight port. A target-
# conditioned noise generator G(z_target)->delta, trained self-supervised on the
# toys catalog against the SAME surrogate ensemble (XT_SEARCH_SPACE), victim held
# out. Reuses XT_* pool / EPS. Tests whether an AMORTIZED generator transfers
# better than per-image PGD (full LAION-400M pretraining is out of scope).
# ===========================================================================
AA_EPS         = XT_EPS               # L-inf budget on cover pixels (reuse)
AA_EPOCHS      = 5                    # passes over the catalog covers
AA_BATCH       = 4                    # images per step (targets; sources = shuffled batch); lower if OOM
AA_K           = 4                    # surrogates used per step (random subset; memory/diversity)
AA_LR          = 1e-4                 # Adam lr for the generator
AA_MAX_ITEMS   = None                 # cap #catalog covers used for training (None = all resolvable)
AA_GEN_CH      = 512                  # generator base channels
AA_SEED        = SEED

AA_OUT_DIR        = "attack/out/anyattack"
AA_GEN_PATH       = "attack/out/anyattack/generator.pt"
AA_POIS_FEAT_DIR  = "attack/out/anyattack/poisoned_features/{split}".format(split=SPLIT)
AA_CLEAN_FEAT_DIR = "attack/out/anyattack/clean_features/{split}".format(split=SPLIT)
AA_PERT_IMG_DIR   = "attack/out/anyattack/perturbed_images"
AA_RESULTS_DIR    = "attack/out/anyattack/results"
AA_RESULTS_JSON   = "attack/out/anyattack/results/anyattack_pointwise.json"


# ===========================================================================
# SPAF-style attack (CIKM'24: "Attacking VARS with Transferable and Imperceptible
# Adversarial Styles"). Content-preserving *style* perturbation: per-channel affine
# (gain/bias) + a smooth low-res color/contrast field (bilinearly upsampled) -- NOT
# an L-inf pixel budget and NOT high-freq noise. Optimized WHITE-BOX through the
# victim's real CLIP toward the SAME popular centroid as pgd_attack.py, so we can
# measure how much the *style axis* moves VIP5 vs full-freedom pixel PGD. Naturalness
# (SPAF's imperceptibility) is enforced via TV + magnitude regs + a color-range cap;
# the low-dim/smooth parameterization keeps the change a plausible "restyle".
# ===========================================================================
STYLE_GRID      = 7          # low-res color field resolution (k x k), bilinearly upsampled to 224
STYLE_STEPS     = 200        # Adam steps (match PGD_STEPS for a fair comparison)
STYLE_LR        = 0.02       # Adam lr on the style params
STYLE_REG       = 5e-3       # L2 reg pulling params toward identity (tightened for naturalness)
STYLE_TV        = 0.2        # total-variation reg on the realized recolor (smooth, anti-blotch)
STYLE_MAG       = 2.0        # magnitude reg mean(delta^2) on the recolor (anti-oversaturation)
STYLE_DELTA_CAP = 0.12       # per-pixel |x'-x0| cap in [0,1] (~30/255); None = unbounded (old run)
STYLE_LOSS      = "cosine"   # match pgd target objective ("cosine" or "l2")

STYLE_OUT_DIR        = "attack/out/style"
STYLE_POIS_FEAT_DIR  = "attack/out/style/poisoned_features/{split}".format(split=SPLIT)
STYLE_PERT_IMG_DIR   = "attack/out/style/perturbed_images"
STYLE_RESULTS_DIR    = "attack/out/style/results"
STYLE_RESULTS_JSON   = "attack/out/style/results/style_pointwise.json"


# ===========================================================================
# AUV-Fusion (arXiv 2507.22880) — adaptation to VIP5. Ports its two genuinely
# portable pieces (the diffusion generator needs SD weights, swapped for our proven
# smooth latent generator; addable behind `diffusers` later):
#   1) high-order USER-PREFERENCE target (GCN-lite: engagement-weighted CLIP centroid
#      from partial interaction data) instead of plain popularity, and
#   2) composite stealth loss  L_align + L_CLIP-fidelity + L_SSIM,
# optimized WHITE-BOX through VIP5's real CLIP (encoder-blind transfer hits the known
# cross-encoder wall, so white-box is the informative setting on VIP5).
# ===========================================================================
AUV_TARGET_MODES  = ["preference", "popular"]   # ablation: same generator+loss, only target differs
AUV_STEPS         = 200
AUV_LR            = 0.02
AUV_LAMBDA_ALIGN  = 1.0            # pull toward the target (effectiveness)
AUV_LAMBDA_CLIP   = 0.5            # semantic preservation: stay near the ORIGINAL CLIP feat
AUV_LAMBDA_SSIM   = 0.3            # structural similarity to the original cover
AUV_TV            = 0.2            # smoothness on the recolor (reuse style)
AUV_DELTA_CAP     = 0.12           # per-pixel |x'-x0| cap in [0,1] (~30/255); None = unbounded
AUV_GRID          = 7             # low-res color field resolution (k x k)
AUV_REG           = 5e-3          # L2 reg pulling style params toward identity

AUV_OUT_DIR        = "attack/out/auv"          # per-mode subdirs: attack/out/auv/<mode>/...
AUV_PREF_IMG_DIR   = "attack/out/auv/preference/perturbed_images"
AUV_POP_IMG_DIR    = "attack/out/auv/popular/perturbed_images"
AUV_RESULTS_DIR    = "attack/out/auv/results"
AUV_RESULTS_JSON   = "attack/out/auv/results/auv_pointwise.json"


# ===========================================================================
# AUV-Fusion FAITHFUL diffusion generator (arXiv 2507.22880, Module 2). Real pretrained
# Stable Diffusion: VAE-encode -> forward-diffuse -> inject latent delta -> DDIM-reverse
# (empty text) -> VAE-decode, with composite loss L_align + L_CLIP + L_SSIM. delta is the
# 4x28x28 latent perturbation, optimized per image (deviation: paper uses an amortized MLP
# from a LightGCN user embedding). Needs `pip install diffusers transformers accelerate
# safetensors`; weights auto-download via from_pretrained (China: export HF_ENDPOINT=
# https://hf-mirror.com). AUV_DIFF_SD_MODEL may be a local path.
# ===========================================================================
AUV_DIFF_SD_MODEL   = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # HF id (vae/unet/scheduler) or local path
AUV_DIFF_MODE       = "ddim"      # "ddim": faithful encode->diffuse->inject->DDIM-reverse->decode | "vae": fast (no denoise loop)
AUV_DIFF_STEPS      = 20          # DDIM inference steps
AUV_DIFF_INJECT     = 10          # inject at this step index; reverse runs from here to 0 (fewer steps = less memory/time)
AUV_DIFF_ETA        = 1.0         # latent perturbation scale
AUV_DIFF_OPT_STEPS  = 30          # Adam steps on the latent delta, per image
AUV_DIFF_LR         = 0.05
AUV_DIFF_LAT_REG    = 1e-2        # L2 on the latent delta (budget/stealth)
AUV_DIFF_PIX_CAP    = 0.12        # per-pixel |x_adv-x0| L-inf cap in [0,1] (~30/255, matches style); None = unbounded
AUV_DIFF_LAMBDA_ALIGN = 1.0       # pull toward target (effectiveness)
AUV_DIFF_LAMBDA_CLIP  = 0.5       # semantic preservation (stay near original CLIP feat)
AUV_DIFF_LAMBDA_SSIM  = 0.3       # structural similarity to original
AUV_DIFF_ALIGNER    = "clip"      # "clip": white-box VIP5 encoder (informative ceiling). resnet (encoder-blind) = future
AUV_DIFF_TARGET_MODE = "top1"     # "top1": single MOST-popular item's embedding (default) | "mean": top-K centroid | "preference"

AUV_DIFF_OUT_DIR       = "attack/out/auv_diff"
AUV_DIFF_TARGETS_JSON  = "attack/out/auv_diff/targets.json"   # exported {asin: image_path} for the standalone generator
AUV_DIFF_TARGET_PATH   = "attack/out/auv_diff/target.npy"     # exported attack TARGET embedding (top1/mean/preference) the generator pulls toward
AUV_DIFF_POIS_FEAT_DIR = "attack/out/auv_diff/poisoned_features/{split}".format(split=SPLIT)
AUV_DIFF_PERT_IMG_DIR  = "attack/out/auv_diff/perturbed_images"
AUV_DIFF_RESULTS_DIR   = "attack/out/auv_diff/results"
AUV_DIFF_RESULTS_JSON  = "attack/out/auv_diff/results/auv_diff_pointwise.json"
