"""Feature loaders for the X-Transfer outputs (mirror of feature_source.py).

X-Transfer writes VICTIM-space (ViT-B/32, 512-d) features into attack/out/xtransfer/...
so the existing eval harness (run_pointwise / scorer) consumes them unchanged, without
clobbering the single-CLIP PGD outputs in attack/out/{poisoned,clean}_features/.
"""
import os
import numpy as np

import config as C

_XT_DIRS = (C.XT_OUT_DIR, C.XT_CENTROID_DIR, C.XT_DELTA_DIR,
            C.XT_POIS_FEAT_DIR, C.XT_CLEAN_FEAT_DIR, C.XT_PERT_IMG_DIR, C.XT_RESULTS_DIR)


def ensure_xt_dirs():
    """Create all XT output dirs (keeps common.py untouched)."""
    for d in _XT_DIRS:
        os.makedirs(d, exist_ok=True)


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path + " (run the upstream xt stage first)")
    return np.load(path).astype("float32").reshape(-1)


def poisoned(asin):
    return _load(os.path.join(C.XT_POIS_FEAT_DIR, asin + ".npy"))


def clean(asin):
    return _load(os.path.join(C.XT_CLEAN_FEAT_DIR, asin + ".npy"))


def has_poisoned(asin):
    return os.path.isfile(os.path.join(C.XT_POIS_FEAT_DIR, asin + ".npy"))
