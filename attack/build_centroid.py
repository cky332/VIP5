"""(2) Build the "popular centroid": average CLIP embedding of the top-K
most-interacted items.

Source of per-item features:
  - 'shipped'  : use the shipped .npy (no images/CLIP needed; good for the
                 feature-space ablation and as a fallback)
  - 'reextract': re-extract from raw images via CLIP (consistent space with the
                 PGD attack; preferred for the pixel attack)
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C


def item_interaction_counts(dataset):
    """Mirror src/data.py:78-85 -> {item_int: count}."""
    counts = defaultdict(int)
    for line in dataset.sequential_data:
        _, items = line.strip().split(" ", 1)
        for it in items.split(" "):
            counts[int(it)] += 1
    return counts


def topk_popular(dataset, k=C.K_POPULAR):
    counts = item_interaction_counts(dataset)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [(str(it), c) for it, c in ranked[:k]]


def resolve_norm_flag():
    """Use resolved CLIP_NORM if available, else auto-detect from shipped L2 norms."""
    f = common.get_clip_norm()
    if f is not None:
        return f
    return None  # caller decides per-source


def _shipped_norm_autodetect(dataset, sample=64):
    asins = [a for a in dataset.id2item.values()
             if os.path.isfile(common.shipped_feature_path(a))][:sample]
    norms = [float(np.linalg.norm(common.load_shipped(a))) for a in asins]
    return bool(abs(float(np.mean(norms)) - 1.0) < 0.05) if norms else False


def _feature_for(dataset, item_str, source, normalize, item2img=None, clip_loaded=False):
    asin = common.asin_of(dataset, item_str)
    if source == "shipped":
        f = common.load_shipped(asin)
        if normalize:
            f = f / (np.linalg.norm(f) + 1e-8)
        return f
    # reextract
    import clip_extract as CE
    ip = CE.resolve_image_path(asin, item2img)
    if ip is None:
        # fallback to shipped if the image can't be resolved
        f = common.load_shipped(asin)
        return f / (np.linalg.norm(f) + 1e-8) if normalize else f
    return CE.extract_feature(ip, normalize=normalize)


def build_centroid(dataset, source="shipped", k=C.K_POPULAR, mode=None, target_item=None):
    """attack target written to centroid.npy:
       - mode='mean' (default): average CLIP feature of the top-K popular items
       - mode='top1'          : the single MOST-popular item's CLIP feature
       - target_item=<id>     : a specific item's CLIP feature (overrides mode)
    pgd_attack / eval read centroid.npy unchanged regardless of mode."""
    common.ensure_dirs()
    mode = mode if mode is not None else getattr(C, "CENTROID_MODE", "mean")
    target_item = target_item if target_item is not None else getattr(C, "TARGET_ITEM", None)
    flag = resolve_norm_flag()
    if flag is None:
        flag = _shipped_norm_autodetect(dataset) if source == "shipped" else True

    # 选择构成"目标特征"的商品集合
    if target_item is not None:
        pops = [(str(target_item), -1)]            # 指定的某个具体商品
    elif mode == "top1":
        pops = topk_popular(dataset, 1)            # 仅 #1 最热门商品
    else:
        pops = topk_popular(dataset, k)            # top-K 求平均(默认)

    item2img = None
    if source == "reextract":
        import clip_extract as CE
        item2img = CE.load_item2img()
    vecs = [_feature_for(dataset, item_str, source, flag, item2img) for item_str, _ in pops]
    M = np.stack(vecs, axis=0).astype("float32")
    centroid = M.mean(axis=0)                       # 单个商品时 = 该商品特征本身
    if flag:
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    np.save(C.CENTROID_PATH, centroid.astype("float32"))
    eff_mode = "target_item" if target_item is not None else mode
    meta = {"source": source, "normalize": bool(flag), "mode": eff_mode,
            "k": (1 if eff_mode != "mean" else k),
            "items": [{"item": i, "count": c} for i, c in pops],
            "centroid_l2": float(np.linalg.norm(centroid))}
    json.dump(meta, open(C.CENTROID_PATH.replace(".npy", "_meta.json"), "w"), indent=2)
    print("[build_centroid] saved %s | mode=%s source=%s normalize=%s | items=%s"
          % (C.CENTROID_PATH, eff_mode, source, flag, [i for i, _ in pops]))
    return centroid


if __name__ == "__main__":
    # 用法: python attack/build_centroid.py [shipped|reextract] [mean|top1]
    src = sys.argv[1] if len(sys.argv) > 1 else "shipped"
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    ctx = common.load_context(need_model=False)
    build_centroid(ctx.dataset, source=src, mode=mode)
