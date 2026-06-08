"""Per-surrogate attack TARGET embedding: for each surrogate encoder, the image
embedding (in that encoder's OWN space) of the chosen popular item(s).

Default (XT_CENTROID_MODE="top1"): the single MOST-popular item's cover -> "look like
the hottest product". Alternatives: "mean" (top-K average centroid) or a specific
XT_TARGET_ITEM. When the target is a single item the "centroid" is just that item's
embedding (mean over 1).

Reuses build_centroid.topk_popular (popularity is encoder-agnostic) and clip_extract
for image resolution + the VICTIM pipeline (used only for the sanity centroid).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common
import config as C
import build_centroid as BC
import clip_extract as CE
import surrogates as SUR
import feature_source_xt as FXT


def load_cover_224(asin, item2img):
    """asin -> (3,224,224) float tensor in [0,1], or None if the image is missing."""
    from PIL import Image
    ip = CE.resolve_image_path(asin, item2img)
    if ip is None:
        return None
    return CE.preprocess_to_224(Image.open(ip))


def _stack_covers(dataset, pops, item2img):
    covers, used = [], []
    for item_str, _cnt in pops:
        asin = common.asin_of(dataset, item_str)
        x = load_cover_224(asin, item2img)
        if x is not None:
            covers.append(x)
            used.append(asin)
    if not covers:
        raise RuntimeError("no popular-item covers resolved; check photos / item2img")
    return torch.stack(covers, 0), used   # (M,3,224,224) on CPU


def _centroid_path(sid):
    return os.path.join(C.XT_CENTROID_DIR, sid + ".npy")


def _target_items(dataset, mode, target_item, k):
    """Item set whose covers define the target embedding (mirrors
    build_centroid.build_centroid): a specific item, the single hottest item, or
    the top-K set (averaged)."""
    if target_item is not None:
        return [(str(target_item), -1)]
    if mode == "top1":
        return BC.topk_popular(dataset, 1)
    return BC.topk_popular(dataset, k)


def build_all_centroids(dataset, k=C.K_POPULAR, mode=None, target_item=None,
                        batch=16, device=None):
    """Build + cache one target embedding per surrogate (native dim), plus a victim
    ViT-B/32 target (sanity metric only). One surrogate is GPU-resident at a time.

    mode/target_item default to config (XT_CENTROID_MODE / XT_TARGET_ITEM): 'top1'
    uses the single most-popular item, 'mean' averages the top-K, and a set
    target_item pins one specific item."""
    FXT.ensure_xt_dirs()
    device = device or C.DEVICE
    mode = mode if mode is not None else getattr(C, "XT_CENTROID_MODE", "top1")
    target_item = target_item if target_item is not None else getattr(C, "XT_TARGET_ITEM", None)
    pops = _target_items(dataset, mode, target_item, k)
    item2img = CE.load_item2img()
    covers, used = _stack_covers(dataset, pops, item2img)
    covers = covers.to(device)

    eff_mode = "target_item" if target_item is not None else mode
    meta = {"mode": eff_mode, "k": (1 if eff_mode != "mean" else k), "n_covers": len(used),
            "items": [{"item": i, "count": c} for i, c in pops],
            "surrogates": {}}
    print("[xt-centroid] target = %s | items = %s" % (eff_mode, [i for i, _ in pops]))

    space = SUR.build_search_space(device)
    for s in space:
        s.to_gpu()
        feats = []
        with torch.no_grad():
            for j in range(0, covers.size(0), batch):
                feats.append(s.encode(covers[j:j + batch]))
        centroid = torch.cat(feats, 0).mean(0).detach().cpu().numpy().astype("float32")
        np.save(_centroid_path(s.sid), centroid)
        meta["surrogates"][s.sid] = {"dim": int(centroid.shape[0]),
                                     "centroid_l2": float(np.linalg.norm(centroid))}
        print("[xt-centroid] %-44s dim=%d" % (s.sid, centroid.shape[0]))
        s.to_cpu()

    # victim ViT-B/32 centroid (used only for the transfer-probe sanity metric)
    norm = common.get_clip_norm()
    if norm is not None:
        with torch.no_grad():
            vf = [CE.encode_pixels(covers[j:j + batch], normalize=norm, device=device)
                  for j in range(0, covers.size(0), batch)]
            vfeat = torch.cat(vf, 0).mean(0).detach().cpu().numpy().astype("float32")
        np.save(C.XT_VICTIM_CENTROID, vfeat)
        meta["victim"] = {"dim": int(vfeat.shape[0]), "normalize": bool(norm),
                          "centroid_l2": float(np.linalg.norm(vfeat))}
        print("[xt-centroid] victim ViT-B/32 centroid dim=%d" % vfeat.shape[0])
    else:
        print("[xt-centroid][warn] CLIP_NORM unresolved -> skipped victim centroid; "
              "run `python attack/run_all.py clip` first to enable the sanity metric.")

    json.dump(meta, open(os.path.join(C.XT_CENTROID_DIR, "_meta.json"), "w"), indent=2)
    print("[xt-centroid] saved %d surrogate centroids -> %s" % (len(space), C.XT_CENTROID_DIR))
    return meta


def load_centroid(sid, device=None):
    device = device or C.DEVICE
    c = np.load(_centroid_path(sid)).astype("float32")
    return torch.from_numpy(c).to(device).view(1, -1)


def load_victim_centroid(device=None):
    if not os.path.isfile(C.XT_VICTIM_CENTROID):
        return None
    device = device or C.DEVICE
    c = np.load(C.XT_VICTIM_CENTROID).astype("float32")
    return torch.from_numpy(c).to(device).view(1, -1)


if __name__ == "__main__":
    ctx = common.load_context(need_model=False)
    build_all_centroids(ctx.dataset)
