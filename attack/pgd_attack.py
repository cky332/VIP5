"""(3) PGD pixel attack on the public CLIP ViT-B/32 encoder.

Perturb a target item's cover (L-inf <= EPSILON in [0,1] pixel space) so its CLIP
embedding approaches the popular centroid. Writes, per target asin:
  attack/out/clean_features/<split>/<asin>.npy     (clean feat, SAME pipeline)
  attack/out/poisoned_features/<split>/<asin>.npy  (adversarial feat)
  attack/out/perturbed_images/<asin>.png           (for visual inspection)

Using the SAME in-loop CLIP pipeline for both clean and poisoned features means
the only difference is the adversarial delta -> the CLIP-pipeline confounder
cancels in the attack-vs-clean comparison.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import clip_extract as CE


def _centroid_tensor(device):
    c = np.load(C.CENTROID_PATH).astype("float32")
    return torch.from_numpy(c).to(device).view(1, -1)


def pgd_attack_pixels(x0, centroid, normalize, device,
                      eps=C.EPSILON, steps=C.PGD_STEPS, alpha=C.PGD_STEP_SIZE,
                      loss=C.PGD_LOSS):
    """x0: (3,224,224) in [0,1]. Returns x_adv (3,224,224) in [0,1]."""
    x0 = x0.to(device)
    delta = (torch.rand_like(x0) * 2 - 1) * eps
    delta = (torch.clamp(x0 + delta, 0, 1) - x0).detach().requires_grad_(True)
    for _ in range(steps):
        x = (x0 + delta).unsqueeze(0)
        feat = CE.encode_pixels(x, normalize=normalize, device=device)
        if loss == "l2":
            L = ((feat - centroid) ** 2).sum()
        else:  # cosine: minimize (1 - cos) == maximize cos
            L = 1.0 - F.cosine_similarity(feat, centroid).mean()
        grad, = torch.autograd.grad(L, delta)
        with torch.no_grad():
            delta -= alpha * grad.sign()                 # minimize L
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x0 + delta, 0, 1) - x0)
        delta.requires_grad_(True)
    return torch.clamp(x0 + delta, 0, 1).detach()


def _save_png(x_chw, path):
    from PIL import Image
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def attack_item(asin, image_path, centroid, normalize, device=None):
    device = device or C.DEVICE
    from PIL import Image
    x0 = CE.preprocess_to_224(Image.open(image_path)).to(device)   # (3,224,224) [0,1]
    with torch.no_grad():
        clean_feat = CE.encode_pixels(x0.unsqueeze(0), normalize=normalize, device=device)[0]
    x_adv = pgd_attack_pixels(x0, centroid, normalize, device)
    with torch.no_grad():
        pois_feat = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=device)[0]
        cos_before = F.cosine_similarity(clean_feat.view(1, -1), centroid).item()
        cos_after = F.cosine_similarity(pois_feat.view(1, -1), centroid).item()
        linf = (x_adv - x0).abs().max().item()
    np.save(os.path.join(C.CLEAN_FEAT_DIR, asin + ".npy"),
            clean_feat.cpu().numpy().astype("float32"))
    np.save(os.path.join(C.POISONED_FEAT_DIR, asin + ".npy"),
            pois_feat.cpu().numpy().astype("float32"))
    _save_png(x_adv, os.path.join(C.PERTURBED_IMG_DIR, asin + ".png"))
    return {"asin": asin, "cos_before": cos_before, "cos_after": cos_after, "linf": linf}


def attack_targets(dataset, target_item_strs):
    """target_item_strs: iterable of item-id strings whose covers to poison."""
    common.ensure_dirs()
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run clip_extract.verify_against_shipped first.")
    if not os.path.isfile(C.CENTROID_PATH):
        raise RuntimeError("centroid missing; run build_centroid.py (source=reextract) first.")
    centroid = _centroid_tensor(C.DEVICE)
    item2img = CE.load_item2img()
    CE.load_clip(C.DEVICE)

    seen, rows, skipped = set(), [], 0
    for it in target_item_strs:
        asin = common.asin_of(dataset, it)
        if asin in seen:
            continue
        seen.add(asin)
        ip = CE.resolve_image_path(asin, item2img)
        if ip is None:
            skipped += 1
            continue
        r = attack_item(asin, ip, centroid, normalize)
        rows.append(r)
        if len(rows) % 25 == 0:
            print("[pgd] %d done | last cos %.3f->%.3f linf %.4f"
                  % (len(rows), r["cos_before"], r["cos_after"], r["linf"]))
    summary = {"n_attacked": len(rows), "n_skipped_no_image": skipped,
               "mean_cos_before": float(np.mean([r["cos_before"] for r in rows])) if rows else None,
               "mean_cos_after": float(np.mean([r["cos_after"] for r in rows])) if rows else None,
               "max_linf": float(np.max([r["linf"] for r in rows])) if rows else None}
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(C.RESULTS_DIR, "pgd_summary.json"), "w"), indent=2)
    print("[pgd] summary:", json.dumps(summary, indent=2))
    return summary


def test_positive_items(dataset, n_users=C.N_TEST_USERS, seed=C.SEED):
    return [str(items[-1]) for _u, items in common.iter_test_users(dataset, n_users, seed)]


if __name__ == "__main__":
    ctx = common.load_context(need_model=False)
    targets = test_positive_items(ctx.dataset)
    attack_targets(ctx.dataset, targets)
