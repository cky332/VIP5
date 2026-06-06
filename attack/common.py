"""Shared setup for attack scripts: path wiring, model/dataset building (reusing
evaluate_vip5.py + src/), feature path helpers, seeding.

Import this FIRST in every attack script:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common
"""
import os
import sys
import json
import zlib
import random


def stable_hash(s):
    """Process-independent hash (Python's str hash is randomized per process)."""
    return zlib.crc32(str(s).encode("utf-8")) & 0x7FFFFFFF

# ---- path wiring: make src/, notebooks/, repo-root importable; run from repo root ----
ATTACK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ATTACK_DIR)
for _p in (os.path.join(ROOT, "notebooks"), os.path.join(ROOT, "src"), ROOT, ATTACK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)  # so 'data/...' and 'features/...' relative paths resolve

import numpy as np
import torch

import config as C


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    for d in (C.OUT_DIR, C.POISONED_FEAT_DIR, C.CLEAN_FEAT_DIR,
              C.PERTURBED_IMG_DIR, C.RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# CLIP-feature normalization flag (resolved by clip_extract.verify_against_shipped)
# ---------------------------------------------------------------------------
def get_clip_norm():
    """Return True/False if resolved (persisted in EXTRACTION_CHECK), else None."""
    if os.path.isfile(C.EXTRACTION_CHECK):
        try:
            return bool(json.load(open(C.EXTRACTION_CHECK))["clip_norm"])
        except Exception:
            return None
    return None


def save_clip_norm(flag, extra=None):
    ensure_dirs()
    payload = {"clip_norm": bool(flag)}
    if extra:
        payload.update(extra)
    json.dump(payload, open(C.EXTRACTION_CHECK, "w"), indent=2)


# ---------------------------------------------------------------------------
# Model + dataset (reuse evaluate_vip5.py and src/)
# ---------------------------------------------------------------------------
class Ctx:
    """Holds the loaded model, the data dataset (for tokenizer / id maps / helpers)."""
    def __init__(self, model, args, dataset, tokenizer, device):
        self.model = model
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device


def _make_cli():
    from argparse import Namespace
    return Namespace(
        split=C.SPLIT, load=C.CKPT, backbone=C.BACKBONE,
        image_feature_type=C.FEATURE_TYPE, image_feature_size_ratio=C.SIZE_RATIO,
        reduction_factor=C.REDUCTION, max_text_length=C.MAX_TEXT_LEN,
        batch_size=C.BATCH_SIZE, num_workers=0, gpu=0,
        tasks="direct", first_template_only=False,
    )


def load_context(need_model=True, mode="test"):
    """Build args, the test-mode VIP5_Dataset (for tokenizer/id maps/helpers) and,
    if need_model, the VIP5 model with the checkpoint loaded."""
    import evaluate_vip5 as E
    from data import VIP5_Dataset
    from tokenization import P5Tokenizer
    from all_templates import all_tasks

    set_seed(C.SEED)
    cli = _make_cli()
    args = E.build_args(cli)
    if torch.cuda.is_available():
        torch.cuda.set_device(C.DEVICE)

    tokenizer = P5Tokenizer.from_pretrained(
        args.backbone, max_length=args.max_text_length, do_lower_case=args.do_lower_case)

    task_list = {"sequential": ["A-1"], "direct": ["B-1"], "explanation": ["C-1"]}
    sample_numbers = {"sequential": (1, 1), "direct": (1, 1), "explanation": 1}
    dataset = VIP5_Dataset(all_tasks, task_list, tokenizer, args, sample_numbers,
                           mode=mode, split=C.SPLIT)

    model = None
    if need_model:
        model = E.build_model(args)
        E.load_ckpt(model, C.CKPT)
        model.eval()
    return Ctx(model=model, args=args, dataset=dataset, tokenizer=tokenizer, device=C.DEVICE)


# ---------------------------------------------------------------------------
# Test-user enumeration + per-user candidate sampling (deterministic)
# ---------------------------------------------------------------------------
def iter_test_users(dataset, n_users=C.N_TEST_USERS, seed=C.SEED):
    """Yield (user_id:str, items:list[int]) for test users (subsampled, deterministic)."""
    lines = dataset.sequential_data
    idx = list(range(len(lines)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if n_users is not None:
        idx = idx[:n_users]
    for i in idx:
        user, items = lines[i].strip().split(" ", 1)
        items = [int(x) for x in items.split(" ")]
        if len(items) < 2:
            continue
        yield user, items


def sample_negatives(dataset, user_id, items, n_neg=C.N_NEG, seed=C.SEED):
    """Sample n_neg negative item-id strings not in the user's history. Deterministic
    per (user, seed) so clean and attacked runs share identical candidate sets."""
    seen = set(items)
    rng = np.random.RandomState((stable_hash(user_id) ^ seed) & 0x7FFFFFFF)
    all_item = dataset.all_item
    negs = []
    while len(negs) < n_neg:
        cand = int(rng.choice(all_item))
        if cand not in seen:
            negs.append(str(cand))
            seen.add(cand)
    return negs


# ---------------------------------------------------------------------------
# Feature path helpers
# ---------------------------------------------------------------------------
def asin_of(dataset, item_id_str):
    return dataset.id2item[str(item_id_str)]


def shipped_feature_path(asin):
    return os.path.join(C.FEAT_DIR, asin + ".npy")


def load_shipped(asin):
    return np.load(shipped_feature_path(asin)).astype("float32").reshape(-1)
