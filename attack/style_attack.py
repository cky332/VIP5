"""Attack (4): SPAF-style content-preserving STYLE perturbation (CIKM'24 idea).

Instead of an L-inf pixel budget (pgd_attack.py) or arbitrary high-freq noise, we
perturb only the *style* of the cover:

    x' = clamp( (gamma + Gfield_up) * x + (beta + Bfield_up),  0, 1 )

  - gamma/beta: per-channel affine gain/bias (global color & contrast)
  - Gfield/Bfield: smooth k x k color/contrast fields, bilinearly upsampled to 224
    (spatially-varying recolor, but low-frequency -> no high-freq content edits)

So the change is content-preserving and visually plausible (like SPAF restyling a
product) and UNBOUNDED in pixel L-inf. We optimize the style params WHITE-BOX
through the victim's REAL CLIP toward the SAME popular centroid as pgd_attack.py,
then evaluate with the SAME pointwise B-1 eval -- and put PGD and STYLE in one
table. This quantifies how much the *style axis* can move VIP5 vs full-freedom
pixel PGD (expectation: CLIP is fairly style-invariant, so weaker but stealthier).

Run (after the `clip` + `centroid` stages, exactly like pgd_attack.py):
    python attack/style_attack.py
Outputs:
    attack/out/clean_features/<split>/<asin>.npy           (re-extracted clean; shared w/ pgd)
    attack/out/style/poisoned_features/<split>/<asin>.npy
    attack/out/style/perturbed_images/<asin>.png
    attack/out/style/results/style_pointwise.json  (+ style_generate.json)
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import clip_extract as CE
import pgd_attack as PGD          # reuse centroid + test_positive_items
import eval_pointwise as EP       # reuse run_pointwise / _print_table
import feature_source as FS       # reuse clean_pgd / poisoned (for the PGD column)


def _dirs():
    for d in (C.STYLE_POIS_FEAT_DIR, C.STYLE_PERT_IMG_DIR, C.STYLE_RESULTS_DIR, C.CLEAN_FEAT_DIR):
        os.makedirs(d, exist_ok=True)


def style_attack_image(x0, centroid, normalize, device,
                       grid=C.STYLE_GRID, steps=C.STYLE_STEPS, lr=C.STYLE_LR,
                       reg=C.STYLE_REG, loss=C.STYLE_LOSS):
    """x0: (3,224,224) in [0,1]. Optimize style params -> return x_adv (3,224,224) [0,1]."""
    x0 = x0.to(device)
    ch = x0.shape[0]
    hw = x0.shape[-2:]
    gamma = torch.ones(ch, 1, 1, device=device, requires_grad=True)
    beta = torch.zeros(ch, 1, 1, device=device, requires_grad=True)
    gfield = torch.zeros(1, ch, grid, grid, device=device, requires_grad=True)
    bfield = torch.zeros(1, ch, grid, grid, device=device, requires_grad=True)
    opt = torch.optim.Adam([gamma, beta, gfield, bfield], lr=lr)

    def styled():
        gup = F.interpolate(gfield, size=hw, mode="bilinear", align_corners=False)[0]
        bup = F.interpolate(bfield, size=hw, mode="bilinear", align_corners=False)[0]
        return ((gamma + gup) * x0 + (beta + bup)).clamp(0, 1)

    for _ in range(steps):
        opt.zero_grad()
        feat = CE.encode_pixels(styled().unsqueeze(0), normalize=normalize, device=device)
        if loss == "l2":
            L = ((feat - centroid) ** 2).sum()
        else:  # cosine: minimize (1 - cos) == maximize cos
            L = 1.0 - F.cosine_similarity(feat, centroid).mean()
        reg_t = ((gamma - 1) ** 2).sum() + (beta ** 2).sum() + (gfield ** 2).sum() + (bfield ** 2).sum()
        (L + reg * reg_t).backward()
        opt.step()
    with torch.no_grad():
        return styled().detach()


def _save_png(x_chw, path):
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate(ctx):
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    if not os.path.isfile(C.CENTROID_PATH):
        raise RuntimeError("centroid missing; run build_centroid.py (source=reextract) first.")
    _dirs()
    centroid = PGD._centroid_tensor(C.DEVICE)
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
        x_adv = style_attack_image(x0, centroid, normalize, C.DEVICE)
        with torch.no_grad():
            pois_feat = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=C.DEVICE)[0]
            cb = F.cosine_similarity(clean_feat.view(1, -1), centroid).item()
            ca = F.cosine_similarity(pois_feat.view(1, -1), centroid).item()
            d = (x_adv - x0).abs()
            linf, meanabs = d.max().item() * 255, d.mean().item() * 255
        np.save(os.path.join(C.CLEAN_FEAT_DIR, a + ".npy"), clean_feat.cpu().numpy().astype("float32"))
        np.save(os.path.join(C.STYLE_POIS_FEAT_DIR, a + ".npy"), pois_feat.cpu().numpy().astype("float32"))
        _save_png(x_adv, os.path.join(C.STYLE_PERT_IMG_DIR, a + ".png"))
        rows.append({"asin": a, "cos_before": cb, "cos_after": ca,
                     "linf_/255": linf, "meanabs_/255": meanabs})
        if len(rows) % 25 == 0:
            print("[style] %d done | last cos %.3f->%.3f | linf %.0f/255 mean %.1f/255"
                  % (len(rows), cb, ca, linf, meanabs))

    summ = {"n": len(rows), "skipped_no_image": skipped,
            "mean_cos_before": float(np.mean([r["cos_before"] for r in rows])) if rows else None,
            "mean_cos_after": float(np.mean([r["cos_after"] for r in rows])) if rows else None,
            "mean_linf_/255": float(np.mean([r["linf_/255"] for r in rows])) if rows else None,
            "mean_meanabs_/255": float(np.mean([r["meanabs_/255"] for r in rows])) if rows else None}
    json.dump({"summary": summ, "rows": rows},
              open(os.path.join(C.STYLE_RESULTS_DIR, "style_generate.json"), "w"), indent=2)
    print("[style] generation done:", json.dumps(summ, indent=2))
    return summ


# ---------------------------------------------------------------------------
# evaluation (reuse eval_pointwise.run_pointwise unchanged) -> clean | pgd | style
# ---------------------------------------------------------------------------
def _style_fn(asin, clean):
    return FS._load(os.path.join(C.STYLE_POIS_FEAT_DIR, asin + ".npy"))


def _style_has(asin):
    return os.path.isfile(os.path.join(C.STYLE_POIS_FEAT_DIR, asin + ".npy"))


def evaluate(ctx):
    os.makedirs(C.STYLE_RESULTS_DIR, exist_ok=True)
    have_pgd = os.path.isdir(C.POISONED_FEAT_DIR) and any(
        f.endswith(".npy") for f in os.listdir(C.POISONED_FEAT_DIR))

    attacked = {}
    if have_pgd:                                      # show PGD alongside, same users
        attacked["pgd"] = lambda asin, clean: FS.poisoned(asin)
    attacked["style"] = _style_fn

    def require(asin):
        return _style_has(asin) and (FS.has_poisoned(asin) if have_pgd else True)

    agg, n_skip = EP.run_pointwise(ctx, FS.clean_pgd, attacked, require=require)
    order = ["clean"] + (["pgd"] if have_pgd else []) + ["style"]
    print("\n=== SPAF-style STYLE perturbation vs %sclean — direct recommendation (B-1, n=%d) ==="
          % ("PGD & " if have_pgd else "", C.N_TEST_USERS))
    EP._print_table(agg, order=order)
    json.dump(agg, open(C.STYLE_RESULTS_JSON, "w"), indent=2)
    print("[style] users=%d skipped=%d | saved %s"
          % (agg.get("clean", {}).get("n", 0), n_skip, C.STYLE_RESULTS_JSON))


def main():
    ctx = common.load_context(need_model=True)
    generate(ctx)
    evaluate(ctx)


if __name__ == "__main__":
    main()
