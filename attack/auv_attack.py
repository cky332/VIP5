"""Attack (5): AUV-Fusion (arXiv 2507.22880) adapted to VIP5 -- TARGET ablation.

AUV-Fusion is a black-box, item-representation attack on visually-aware recommenders
that (1) models high-order USER PREFERENCE from interaction data (LightGCN) to build a
target, and (2) generates a visually-plausible adversarial cover via a diffusion model,
trained with L_align + L_CLIP + L_SSIM. Its victims are classic MF VARS (VBPR/DVBPR/AMR).

VIP5 is a generative LLM recommender consuming a pooled CLIP ViT-B/32 feature, with no
dot-product user/item score -- and AUV-Fusion's encoder-blind transfer hits the same
cross-encoder wall we measured for X-Transfer/AnyAttack. So we port the two transferable
pieces and run them WHITE-BOX through VIP5's REAL CLIP, with the SAME smooth latent
generator (affine + bilinear color field + color-range cap) and the SAME composite loss
L_align + L_CLIP-fidelity + L_SSIM -- the only thing that changes is the TARGET:

  * "preference": GCN-lite engagement-weighted CLIP centroid over items real users
    interacted with (AUV-Fusion's idea), vs
  * "popular":    the top-K popularity centroid (what pgd/style aim at).

Running both isolates AUV-Fusion's target contribution on VIP5. The real diffusion
generator is the one heavy piece left out (needs SD weights; swappable behind `diffusers`).

Run (after the `clip` + `centroid` stages):
    python attack/auv_attack.py
Outputs per mode under attack/out/auv/<mode>/ ; one combined table:
    clean | pgd | style | auv_preference | auv_popular
"""
import os
import sys
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import clip_extract as CE
import pgd_attack as PGD          # reuse test_positive_items + centroid loader
import eval_pointwise as EP       # reuse run_pointwise / _print_table
import feature_source as FS       # reuse clean_pgd / poisoned (clean & PGD columns)


def _mode_paths(mode):
    base = os.path.join(C.AUV_OUT_DIR, mode)
    return {"base": base,
            "target": os.path.join(base, "target.npy"),
            "pois": os.path.join(base, "poisoned_features", C.SPLIT),
            "img": os.path.join(base, "perturbed_images")}


# ---------------------------------------------------------------------------
# target builders
# ---------------------------------------------------------------------------
def _preference_target(ctx, normalize, device, cache_path):
    """Engagement-weighted centroid of CLIP features over interacted items (cached)."""
    if os.path.isfile(cache_path):
        c = np.load(cache_path).astype("float32")
        return torch.from_numpy(c).to(device).view(1, -1)
    CE.load_clip(device)
    item2img = CE.load_item2img()
    freq = Counter()
    for _u, items in common.iter_test_users(ctx.dataset, C.N_TEST_USERS):
        for it in items[:-1]:                       # histories (exclude held-out positive)
            freq[str(it)] += 1
    acc, wsum, used = None, 0.0, 0
    for it, w in freq.items():
        asin = common.asin_of(ctx.dataset, it)
        ip = CE.resolve_image_path(asin, item2img)
        if ip is None:
            continue
        x = CE.preprocess_to_224(Image.open(ip)).to(device)
        with torch.no_grad():
            f = CE.encode_pixels(x.unsqueeze(0), normalize=normalize, device=device)[0]
        acc = (f * w) if acc is None else (acc + f * w)
        wsum += w
        used += 1
        if used % 250 == 0:
            print("[auv] preference target: aggregated %d / %d items" % (used, len(freq)))
    if acc is None or wsum == 0:
        raise RuntimeError("preference target empty (no resolvable history images)")
    tgt = acc / wsum
    tgt = tgt / tgt.norm().clamp_min(1e-8)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, tgt.cpu().numpy().astype("float32"))
    print("[auv] preference target built from %d unique items -> %s" % (used, cache_path))
    return tgt.view(1, -1)


def build_target(ctx, normalize, mode, paths, device=None):
    device = device or C.DEVICE
    if mode == "popular":
        c = np.load(C.CENTROID_PATH).astype("float32")
        return torch.from_numpy(c).to(device).view(1, -1)
    return _preference_target(ctx, normalize, device, paths["target"])


# ---------------------------------------------------------------------------
# composite-loss white-box attack through VIP5's real CLIP
# ---------------------------------------------------------------------------
def _tv(d):
    return ((d[:, 1:, :] - d[:, :-1, :]).abs().mean()
            + (d[:, :, 1:] - d[:, :, :-1]).abs().mean())


def _ssim(x, y, win=11, c1=0.01 ** 2, c2=0.03 ** 2):
    """Mean SSIM between two (C,H,W) images in [0,1] (uniform window via avg_pool)."""
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    pad = win // 2
    mu_x = F.avg_pool2d(x, win, 1, pad)
    mu_y = F.avg_pool2d(y, win, 1, pad)
    mx2, my2, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.avg_pool2d(x * x, win, 1, pad) - mx2
    sy = F.avg_pool2d(y * y, win, 1, pad) - my2
    sxy = F.avg_pool2d(x * y, win, 1, pad) - mxy
    ssim = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))
    return ssim.mean()


def auv_attack_image(x0, target, clean_feat, normalize, device,
                     grid=C.AUV_GRID, steps=C.AUV_STEPS, lr=C.AUV_LR,
                     la=C.AUV_LAMBDA_ALIGN, lc=C.AUV_LAMBDA_CLIP, ls=C.AUV_LAMBDA_SSIM,
                     tv=C.AUV_TV, reg=C.AUV_REG, cap=C.AUV_DELTA_CAP):
    """x0: (3,224,224) [0,1]. Returns x_adv (3,224,224) [0,1]."""
    x0 = x0.to(device)
    ch, hw = x0.shape[0], x0.shape[-2:]
    gamma = torch.ones(ch, 1, 1, device=device, requires_grad=True)
    beta = torch.zeros(ch, 1, 1, device=device, requires_grad=True)
    gfield = torch.zeros(1, ch, grid, grid, device=device, requires_grad=True)
    bfield = torch.zeros(1, ch, grid, grid, device=device, requires_grad=True)
    opt = torch.optim.Adam([gamma, beta, gfield, bfield], lr=lr)
    cf = clean_feat.view(1, -1)

    def styled():
        gup = F.interpolate(gfield, size=hw, mode="bilinear", align_corners=False)[0]
        bup = F.interpolate(bfield, size=hw, mode="bilinear", align_corners=False)[0]
        d = (gamma + gup) * x0 + (beta + bup) - x0
        if cap is not None:
            d = d.clamp(-cap, cap)
        return (x0 + d).clamp(0, 1)

    for _ in range(steps):
        opt.zero_grad()
        x = styled()
        feat = CE.encode_pixels(x.unsqueeze(0), normalize=normalize, device=device)
        L_align = 1.0 - F.cosine_similarity(feat, target).mean()
        L_clip = ((feat - cf) ** 2).mean()
        L_ssim = 1.0 - _ssim(x, x0)
        d = x - x0
        Lp = ((gamma - 1) ** 2).sum() + (beta ** 2).sum() + (gfield ** 2).sum() + (bfield ** 2).sum()
        L = la * L_align + lc * L_clip + ls * L_ssim + tv * _tv(d) + reg * Lp
        L.backward()
        opt.step()
    with torch.no_grad():
        return styled().detach()


def _save_png(x_chw, path):
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# generation (one mode)
# ---------------------------------------------------------------------------
def generate_mode(ctx, normalize, mode):
    paths = _mode_paths(mode)
    for d in (paths["pois"], paths["img"], C.AUV_RESULTS_DIR, C.CLEAN_FEAT_DIR):
        os.makedirs(d, exist_ok=True)
    target = build_target(ctx, normalize, mode, paths)
    if os.path.isfile(C.CENTROID_PATH):
        pop = PGD._centroid_tensor(C.DEVICE)
        print("[auv:%s] cos(target, popular_centroid) = %.3f"
              % (mode, F.cosine_similarity(target, pop).item()))
    CE.load_clip(C.DEVICE)
    item2img = CE.load_item2img()

    targets = PGD.test_positive_items(ctx.dataset)
    seen, rows, skipped = set(), [], 0
    for it in targets:
        a = common.asin_of(ctx.dataset, it)
        if a in seen:
            continue
        seen.add(a)
        ip = CE.resolve_image_path(a, item2img)
        if ip is None:
            skipped += 1
            continue
        x0 = CE.preprocess_to_224(Image.open(ip)).to(C.DEVICE)
        with torch.no_grad():
            clean_feat = CE.encode_pixels(x0.unsqueeze(0), normalize=normalize, device=C.DEVICE)[0]
        x_adv = auv_attack_image(x0, target, clean_feat, normalize, C.DEVICE)
        with torch.no_grad():
            pois_feat = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=C.DEVICE)[0]
            ca = F.cosine_similarity(pois_feat.view(1, -1), target).item()
            cb = F.cosine_similarity(clean_feat.view(1, -1), target).item()
            dd = (x_adv - x0).abs()
            linf, meanabs = dd.max().item() * 255, dd.mean().item() * 255
        np.save(os.path.join(C.CLEAN_FEAT_DIR, a + ".npy"), clean_feat.cpu().numpy().astype("float32"))
        np.save(os.path.join(paths["pois"], a + ".npy"), pois_feat.cpu().numpy().astype("float32"))
        _save_png(x_adv, os.path.join(paths["img"], a + ".png"))
        rows.append({"asin": a, "cos_before": cb, "cos_after": ca,
                     "linf_/255": linf, "meanabs_/255": meanabs})
        if len(rows) % 50 == 0:
            print("[auv:%s] %d done | last cos %.3f->%.3f | linf %.0f mean %.1f"
                  % (mode, len(rows), cb, ca, linf, meanabs))

    summ = {"mode": mode, "n": len(rows), "skipped_no_image": skipped,
            "mean_cos_before": float(np.mean([r["cos_before"] for r in rows])) if rows else None,
            "mean_cos_after": float(np.mean([r["cos_after"] for r in rows])) if rows else None,
            "mean_linf_/255": float(np.mean([r["linf_/255"] for r in rows])) if rows else None,
            "mean_meanabs_/255": float(np.mean([r["meanabs_/255"] for r in rows])) if rows else None}
    json.dump({"summary": summ, "rows": rows},
              open(os.path.join(C.AUV_RESULTS_DIR, "auv_generate_%s.json" % mode), "w"), indent=2)
    print("[auv:%s] generation done:" % mode, json.dumps(summ, indent=2))
    return summ


# ---------------------------------------------------------------------------
# evaluation -> clean | pgd | style (if present) | auv_<mode> for each present mode
# ---------------------------------------------------------------------------
def _loader(pois_dir):
    return lambda asin, clean: FS._load(os.path.join(pois_dir, asin + ".npy"))


def _dir_has_npy(d):
    return os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d))


def evaluate(ctx):
    os.makedirs(C.AUV_RESULTS_DIR, exist_ok=True)
    have_pgd = _dir_has_npy(C.POISONED_FEAT_DIR)
    have_style = _dir_has_npy(C.STYLE_POIS_FEAT_DIR)
    modes = [m for m in C.AUV_TARGET_MODES if _dir_has_npy(_mode_paths(m)["pois"])]

    attacked = {}
    if have_pgd:
        attacked["pgd"] = lambda asin, clean: FS.poisoned(asin)
    if have_style:
        attacked["style"] = _loader(C.STYLE_POIS_FEAT_DIR)
    for m in modes:
        attacked["auv_%s" % m] = _loader(_mode_paths(m)["pois"])

    def require(asin):
        ok = all(os.path.isfile(os.path.join(_mode_paths(m)["pois"], asin + ".npy")) for m in modes)
        if have_pgd:
            ok = ok and FS.has_poisoned(asin)
        if have_style:
            ok = ok and os.path.isfile(os.path.join(C.STYLE_POIS_FEAT_DIR, asin + ".npy"))
        return ok

    agg, n_skip = EP.run_pointwise(ctx, FS.clean_pgd, attacked, require=require)
    order = (["clean"] + (["pgd"] if have_pgd else []) + (["style"] if have_style else [])
             + ["auv_%s" % m for m in modes])
    print("\n=== AUV-Fusion target ablation (same generator+composite loss) — B-1, n=%d ==="
          % C.N_TEST_USERS)
    EP._print_table(agg, order=order)
    json.dump(agg, open(C.AUV_RESULTS_JSON, "w"), indent=2)
    print("[auv] users=%d skipped=%d | saved %s"
          % (agg.get("clean", {}).get("n", 0), n_skip, C.AUV_RESULTS_JSON))


def main():
    ctx = common.load_context(need_model=True)
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    if not os.path.isfile(C.CENTROID_PATH):
        raise RuntimeError("centroid missing; run build_centroid.py first.")
    for mode in C.AUV_TARGET_MODES:
        generate_mode(ctx, normalize, mode)
    evaluate(ctx)


if __name__ == "__main__":
    main()
