"""Where a candidate's image feature comes from.

The primary evals (eval_pointwise / eval_listwise / ablation) build batches
in-process and feed features explicitly, so no DataLoader monkeypatching is
needed. These helpers just resolve the right .npy for a given kind:

    shipped     : the original feature shipped in features/<type>_features/<split>/
    clean_pgd   : clean feature re-extracted through the SAME CLIP pipeline as the
                  attack (written by pgd_attack.py) -- use this as the clean
                  baseline for the PIXEL attack so the CLIP-pipeline confounder cancels
    poisoned    : adversarial feature (written by pgd_attack.py)
"""
import os
import numpy as np

import common
import config as C


def shipped(dataset, item_id_str):
    return common.load_shipped(common.asin_of(dataset, item_id_str))


def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path + " (run the upstream step first)")
    return np.load(path).astype("float32").reshape(-1)


def poisoned(asin):
    return _load(os.path.join(C.POISONED_FEAT_DIR, asin + ".npy"))


def clean_pgd(asin):
    return _load(os.path.join(C.CLEAN_FEAT_DIR, asin + ".npy"))


def has_poisoned(asin):
    return os.path.isfile(os.path.join(C.POISONED_FEAT_DIR, asin + ".npy"))
