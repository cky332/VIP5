"""Attack (6): AUV-Fusion FAITHFUL diffusion generator (arXiv 2507.22880), adapted to VIP5.

This implements AUV-Fusion's Module-2 image generator faithfully -- the part the ablation
showed is the DOMINANT factor on VIP5 (~5 rank places) -- using a real pretrained Stable
Diffusion model, following the paper's steps (Eq 12-13):

    z0   = VAE_enc(x0)                                  # encode cover to latent
    z_t  = sqrt(a_t) z0 + sqrt(1-a_t) eps               # forward diffuse to step t
    z~_t = z_t + eta * delta                            # inject adversarial latent perturbation
    z~0  = DDIM_reverse(z~_t, empty-text)               # deterministic reverse (UNet, h=empty)
    x_adv= VAE_dec(z~0)                                  # decode to the adversarial cover

and the composite loss (Eq 14-17): L = la*L_align + lc*L_CLIP + ls*L_SSIM, where L_align pulls
toward a target (popular centroid by default, the best target per our ablation) and L_CLIP/L_SSIM
preserve the original. delta (the 4x28x28 latent perturbation) is optimized per image.

Deviations from the paper (documented honestly):
  * delta is optimized PER IMAGE (Adam), not produced by an amortized MLP from a LightGCN user
    embedding. (Target still comes from interaction data via auv_attack.build_target.)
  * L_align uses VIP5's REAL CLIP (white-box) by default to measure the generator's ceiling on
    VIP5; the paper is encoder-blind (ResNet). Set AUV_DIFF_ALIGNER accordingly.
  * "empty text" h is approximated by zeros(1,77,cross_attn_dim) so no text encoder is needed.
  * AUV_DIFF_MODE="vae" skips the denoise loop (encode->inject->decode) -- fast sanity path;
    "ddim" is the faithful full sequence above.

WEIGHTS: needs a pretrained SD checkpoint (vae+unet+scheduler subfolders). It auto-downloads via
`from_pretrained` on first run. For a China network set:  export HF_ENDPOINT=https://hf-mirror.com
and `pip install diffusers transformers accelerate safetensors`. AUV_DIFF_SD_MODEL may also be a
local path. The VIP5/CLIP/centroid stages must already have run (like auv_attack.py).

    python attack/auv_diffusion.py
Outputs under attack/out/auv_diff/ ; table: clean | pgd | style | auv | auv_diff
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
import pgd_attack as PGD
import eval_pointwise as EP
import feature_source as FS
import auv_attack as AUV          # reuse _ssim + build_target (preference/popular)

SCALE = 0.18215                    # SD VAE latent scaling


# ---------------------------------------------------------------------------
# model loading (real pretrained Stable Diffusion)
# ---------------------------------------------------------------------------
def load_sd(device=None):
    device = device or C.DEVICE
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    except Exception as e:
        raise RuntimeError("diffusers not installed -- `pip install diffusers transformers "
                           "accelerate safetensors` (and `export HF_ENDPOINT=https://hf-mirror.com` "
                           "on a China network).") from e
    mid = C.AUV_DIFF_SD_MODEL
    try:
        vae = AutoencoderKL.from_pretrained(mid, subfolder="vae")
        sch = DDIMScheduler.from_pretrained(mid, subfolder="scheduler")
        unet = None
        if C.AUV_DIFF_MODE == "ddim":
            unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet")
    except Exception as e:
        raise RuntimeError(
            "could not load SD '%s'. Set AUV_DIFF_SD_MODEL to a local path or a mirror-available "
            "id, and `export HF_ENDPOINT=https://hf-mirror.com`. Original error: %s" % (mid, e)) from e
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if unet is not None:
        unet = unet.to(device).eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        try:
            unet.enable_gradient_checkpointing()      # keep DDIM backprop memory in check
        except Exception:
            pass
    return vae, unet, sch


# ---------------------------------------------------------------------------
# faithful generator (testable: pass any vae/unet/scheduler)
# ---------------------------------------------------------------------------
def decode_from_delta(z_base, delta, models, rev_ts, h, eta):
    """z~ = z_base + eta*delta -> (optional DDIM reverse over rev_ts) -> VAE decode -> x in [0,1]."""
    vae, unet, scheduler = models
    z = z_base + eta * delta
    if unet is not None and rev_ts is not None:        # faithful DDIM reverse (empty text h)
        for t in rev_ts:
            eps = unet(z, t, encoder_hidden_states=h).sample
            z = scheduler.step(eps, t, z).prev_sample
    x = vae.decode(z / SCALE).sample
    return (x / 2 + 0.5).clamp(0, 1)                    # (1,3,H,W) in [0,1]


def attack_image(x0, target, clean_feat, models, normalize, device):
    """Optimize the latent delta with the composite loss; return x_adv (3,224,224) [0,1]."""
    vae, unet, scheduler = models
    x0 = x0.to(device)
    with torch.no_grad():
        z0 = vae.encode((x0.unsqueeze(0) * 2 - 1)).latent_dist.mean * SCALE
        rev_ts, h, z_base = None, None, z0
        if C.AUV_DIFF_MODE == "ddim":
            scheduler.set_timesteps(C.AUV_DIFF_STEPS, device=device)
            inj = min(C.AUV_DIFF_INJECT, len(scheduler.timesteps) - 1)
            t_inject = scheduler.timesteps[inj]
            noise = torch.randn_like(z0)
            z_base = scheduler.add_noise(z0, noise, t_inject)
            rev_ts = scheduler.timesteps[inj:]
            h = torch.zeros(1, 77, unet.config.cross_attention_dim, device=device, dtype=z0.dtype)
        cf = clean_feat.view(1, -1)
    delta = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=C.AUV_DIFF_LR)

    for _ in range(C.AUV_DIFF_OPT_STEPS):
        opt.zero_grad()
        x = decode_from_delta(z_base, delta, models, rev_ts, h, C.AUV_DIFF_ETA)
        feat = CE.encode_pixels(x, normalize=normalize, device=device)
        L_align = 1.0 - F.cosine_similarity(feat, target).mean()
        L_clip = ((feat - cf) ** 2).mean()
        L_ssim = 1.0 - AUV._ssim(x[0], x0)
        L = (C.AUV_DIFF_LAMBDA_ALIGN * L_align + C.AUV_DIFF_LAMBDA_CLIP * L_clip
             + C.AUV_DIFF_LAMBDA_SSIM * L_ssim + C.AUV_DIFF_LAT_REG * (delta ** 2).mean())
        L.backward()
        opt.step()
    with torch.no_grad():
        x = decode_from_delta(z_base, delta, models, rev_ts, h, C.AUV_DIFF_ETA)
    return x[0].detach()


# ---------------------------------------------------------------------------
# generation + eval
# ---------------------------------------------------------------------------
def _target(ctx, normalize):
    if C.AUV_DIFF_TARGET_MODE == "popular":
        c = np.load(C.CENTROID_PATH).astype("float32")
        return torch.from_numpy(c).to(C.DEVICE).view(1, -1)
    paths = AUV._mode_paths("preference")
    return AUV.build_target(ctx, normalize, "preference", paths)


def _save_png(x_chw, path):
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def generate(ctx):
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    if not os.path.isfile(C.CENTROID_PATH):
        raise RuntimeError("centroid missing; run build_centroid.py first.")
    for d in (C.AUV_DIFF_POIS_FEAT_DIR, C.AUV_DIFF_PERT_IMG_DIR, C.AUV_DIFF_RESULTS_DIR, C.CLEAN_FEAT_DIR):
        os.makedirs(d, exist_ok=True)
    print("[auv_diff] loading SD '%s' (mode=%s)..." % (C.AUV_DIFF_SD_MODEL, C.AUV_DIFF_MODE))
    models = load_sd(C.DEVICE)
    target = _target(ctx, normalize)
    if os.path.isfile(C.CENTROID_PATH):
        print("[auv_diff] cos(target, popular_centroid)=%.3f"
              % F.cosine_similarity(target, PGD._centroid_tensor(C.DEVICE)).item())
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
        x_adv = attack_image(x0, target, clean_feat, models, normalize, C.DEVICE)
        with torch.no_grad():
            pois_feat = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=C.DEVICE)[0]
            ca = F.cosine_similarity(pois_feat.view(1, -1), target).item()
            cb = F.cosine_similarity(clean_feat.view(1, -1), target).item()
            dd = (x_adv - x0).abs()
            linf, meanabs = dd.max().item() * 255, dd.mean().item() * 255
        np.save(os.path.join(C.CLEAN_FEAT_DIR, a + ".npy"), clean_feat.cpu().numpy().astype("float32"))
        np.save(os.path.join(C.AUV_DIFF_POIS_FEAT_DIR, a + ".npy"), pois_feat.cpu().numpy().astype("float32"))
        _save_png(x_adv, os.path.join(C.AUV_DIFF_PERT_IMG_DIR, a + ".png"))
        rows.append({"asin": a, "cos_before": cb, "cos_after": ca, "linf_/255": linf, "meanabs_/255": meanabs})
        if len(rows) % 25 == 0:
            print("[auv_diff] %d done | cos %.3f->%.3f | linf %.0f mean %.1f"
                  % (len(rows), cb, ca, linf, meanabs))

    summ = {"n": len(rows), "skipped_no_image": skipped, "mode": C.AUV_DIFF_MODE,
            "target_mode": C.AUV_DIFF_TARGET_MODE,
            "mean_cos_before": float(np.mean([r["cos_before"] for r in rows])) if rows else None,
            "mean_cos_after": float(np.mean([r["cos_after"] for r in rows])) if rows else None,
            "mean_linf_/255": float(np.mean([r["linf_/255"] for r in rows])) if rows else None,
            "mean_meanabs_/255": float(np.mean([r["meanabs_/255"] for r in rows])) if rows else None}
    json.dump({"summary": summ, "rows": rows},
              open(os.path.join(C.AUV_DIFF_RESULTS_DIR, "auv_diff_generate.json"), "w"), indent=2)
    print("[auv_diff] generation done:", json.dumps(summ, indent=2))
    return summ


def _dir_has_npy(d):
    return os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d))


def _loader(pois_dir):
    return lambda asin, clean: FS._load(os.path.join(pois_dir, asin + ".npy"))


def evaluate(ctx):
    os.makedirs(C.AUV_DIFF_RESULTS_DIR, exist_ok=True)
    have_pgd = _dir_has_npy(C.POISONED_FEAT_DIR)
    have_style = _dir_has_npy(C.STYLE_POIS_FEAT_DIR)
    auv_pref = AUV._mode_paths("preference")["pois"]
    have_auv = _dir_has_npy(auv_pref)

    attacked = {}
    if have_pgd:
        attacked["pgd"] = lambda asin, clean: FS.poisoned(asin)
    if have_style:
        attacked["style"] = _loader(C.STYLE_POIS_FEAT_DIR)
    if have_auv:
        attacked["auv"] = _loader(auv_pref)
    attacked["auv_diff"] = _loader(C.AUV_DIFF_POIS_FEAT_DIR)

    def require(asin):
        ok = os.path.isfile(os.path.join(C.AUV_DIFF_POIS_FEAT_DIR, asin + ".npy"))
        if have_pgd:
            ok = ok and FS.has_poisoned(asin)
        if have_style:
            ok = ok and os.path.isfile(os.path.join(C.STYLE_POIS_FEAT_DIR, asin + ".npy"))
        if have_auv:
            ok = ok and os.path.isfile(os.path.join(auv_pref, asin + ".npy"))
        return ok

    agg, n_skip = EP.run_pointwise(ctx, FS.clean_pgd, attacked, require=require)
    order = (["clean"] + (["pgd"] if have_pgd else []) + (["style"] if have_style else [])
             + (["auv"] if have_auv else []) + ["auv_diff"])
    print("\n=== AUV-Fusion faithful diffusion generator vs baselines — B-1, n=%d ===" % C.N_TEST_USERS)
    EP._print_table(agg, order=order)
    json.dump(agg, open(C.AUV_DIFF_RESULTS_JSON, "w"), indent=2)
    print("[auv_diff] users=%d skipped=%d | saved %s"
          % (agg.get("clean", {}).get("n", 0), n_skip, C.AUV_DIFF_RESULTS_JSON))


def main():
    ctx = common.load_context(need_model=True)
    generate(ctx)
    evaluate(ctx)


if __name__ == "__main__":
    main()
