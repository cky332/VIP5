"""X-Transfer (arXiv 2505.05528) ported to VIP5: a black-box, super-transferable,
single-target popularity-mimicry attack.

Craft a pixel perturbation delta on an ENSEMBLE of surrogate CLIP encoders (UCB
surrogate scaling), pulling each surrogate's embedding of (cover + delta) toward that
surrogate's popular centroid -- WITHOUT touching the victim. Then re-extract the
perturbed cover with the VICTIM (OpenAI ViT-B/32) to obtain the 512-d feature VIP5
consumes. delta lives in the 224x224 [0,1] deployable-cover space.

Threat model (per the approved plan): victim ViT-B/32 is held out of the surrogate
pool; transfer is proven by the victim-space "transfer probe" (poisoned cos-to-centroid
> clean) and by the downstream rank rise measured in eval_xtransfer.py.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import common
import config as C
import clip_extract as CE
import surrogates as SUR
import xtransfer_centroid as XC
import feature_source_xt as FXT
import pgd_attack as P   # reuse test_positive_items


def _cos_np(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def xtransfer_delta(X, space, centroids, eps=C.XT_EPS, alpha=C.XT_ALPHA,
                    steps=C.XT_STEPS, k=C.XT_K_SELECT, momentum=C.XT_MOMENTUM,
                    c=C.XT_UCB_C, targeted=C.XT_TARGETED, batch=C.XT_BATCH,
                    device=None, log_every=25):
    """Optimize one delta over D'.

    X         : (B,3,224,224) in [0,1] (single-target: B=1; universal: B=|D'|).
    centroids : list[(1,d_i)] aligned to `space`.
    Returns (delta (3,224,224) cpu float32, trace dict).
    """
    device = device or C.DEVICE
    X = X.to(device)
    if X.dim() == 3:
        X = X.unsqueeze(0)
    bandit = SUR.UCB(len(space), c=c, momentum=momentum)
    delta = ((torch.rand(X.shape[1:], device=device) * 2.0 - 1.0) * eps).detach()
    delta.requires_grad_(True)

    trace = {"loss": [], "cos": []}
    for step in range(steps):
        idx = bandit.select(k)
        if X.size(0) > batch:
            sel = torch.randint(0, X.size(0), (batch,), device=X.device)
            xb = X.index_select(0, sel)
        else:
            xb = X
        x = torch.clamp(xb + delta, 0, 1)               # delta broadcasts over the batch
        total, per_loss, per_cos = 0.0, {}, []
        for i in idx:
            s = space[i].to_gpu()
            feat = s.encode(x)                          # (B, d_i)
            cos = F.cosine_similarity(feat, centroids[i].to(device)).mean()
            if targeted:
                Li = 1.0 - cos                          # pull toward popular centroid
            else:
                with torch.no_grad():
                    cf = s.encode(torch.clamp(xb, 0, 1))  # degrade: push away from clean
                Li = F.cosine_similarity(feat, cf).mean()
            total = total + Li
            per_loss[i] = float(Li.detach())
            per_cos.append(float(cos.detach()))
        L = total / max(len(idx), 1)
        grad, = torch.autograd.grad(L, delta)
        with torch.no_grad():
            delta -= alpha * grad.sign()                # minimize
            delta.clamp_(-eps, eps)                     # L-inf projection
        delta.requires_grad_(True)
        for i in idx:
            bandit.update(i, per_loss[i])
        for i in idx:
            space[i].to_cpu()
        trace["loss"].append(float(L.detach()))
        trace["cos"].append(float(np.mean(per_cos)))
        if log_every and (step + 1) % log_every == 0:
            print("[xt] step %d/%d  L=%.4f  mean_cos_to_centroid=%.4f"
                  % (step + 1, steps, trace["loss"][-1], trace["cos"][-1]))
    trace["ucb"] = bandit.snapshot()
    return delta.detach().cpu(), trace


def _save_png(x_chw, path):
    from PIL import Image
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def attack_item_xt(asin, image_path, space, centroids, normalize,
                   victim_centroid=None, shared_delta=None, device=None):
    """Craft (or apply) delta for one target, then materialize VICTIM-space features."""
    device = device or C.DEVICE
    from PIL import Image
    x0 = CE.preprocess_to_224(Image.open(image_path)).to(device)   # (3,224,224) [0,1]
    if shared_delta is not None:                                    # universal: apply precomputed
        delta = shared_delta.to(device)
    else:                                                           # single-target: optimize here
        delta, _trace = xtransfer_delta(x0.unsqueeze(0), space, centroids, device=device)
        delta = delta.to(device)
    x_adv = torch.clamp(x0 + delta, 0, 1)

    with torch.no_grad():
        clean_feat = CE.encode_pixels(x0.unsqueeze(0), normalize=normalize, device=device)[0]
        pois_feat = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=device)[0]
    clean_np = clean_feat.cpu().numpy().astype("float32")
    pois_np = pois_feat.cpu().numpy().astype("float32")

    np.save(os.path.join(C.XT_CLEAN_FEAT_DIR, asin + ".npy"), clean_np)
    np.save(os.path.join(C.XT_POIS_FEAT_DIR, asin + ".npy"), pois_np)
    _save_png(x_adv, os.path.join(C.XT_PERT_IMG_DIR, asin + ".png"))

    row = {"asin": asin, "linf": float((x_adv - x0).abs().max().item())}
    if victim_centroid is not None:
        vc = victim_centroid.detach().cpu().numpy()
        row["victim_cos_before"] = _cos_np(clean_np, vc)
        row["victim_cos_after"] = _cos_np(pois_np, vc)
    return row


def _sample_catalog_covers(dataset, item2img, device, n=None):
    """Universal mode: a deterministic sample of catalog covers as D'."""
    import random as _r
    n = n or C.XT_DPRIME_SIZE
    ids = list(dataset.id2item.keys())
    _r.Random(C.SEED).shuffle(ids)
    covers = []
    for iid in ids:
        if len(covers) >= n:
            break
        x = XC.load_cover_224(common.asin_of(dataset, iid), item2img)
        if x is not None:
            covers.append(x)
    if not covers:
        raise RuntimeError("no catalog covers resolved for universal D'")
    return torch.stack(covers, 0).to(device)


def attack_targets_xt(dataset, target_item_strs, device=None):
    """Per target: craft a black-box transferable delta and write victim-space feats."""
    FXT.ensure_xt_dirs()
    device = device or C.DEVICE
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")

    space = SUR.build_search_space(device)
    centroids = []
    for s in space:
        if not os.path.isfile(XC._centroid_path(s.sid)):
            raise RuntimeError("missing centroid for %s; run `python attack/run_all.py xt-centroid` first."
                               % s.sid)
        centroids.append(XC.load_centroid(s.sid, device))
    victim_centroid = XC.load_victim_centroid(device)
    item2img = CE.load_item2img()
    CE.load_clip(device)   # warm the victim encoder

    shared_delta = None
    if C.XT_DPRIME_MODE == "universal":
        Dp = _sample_catalog_covers(dataset, item2img, device)
        shared_delta, trace = xtransfer_delta(Dp, space, centroids, device=device)
        np.save(os.path.join(C.XT_DELTA_DIR, "uap.npy"),
                shared_delta.numpy().astype("float32"))
        json.dump({"loss_tail": trace["loss"][-5:], "cos_tail": trace["cos"][-5:],
                   "ucb": trace["ucb"]},
                  open(os.path.join(C.XT_DELTA_DIR, "uap_meta.json"), "w"), indent=2)
        print("[xt] universal delta crafted over |D'|=%d covers" % Dp.size(0))

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
        row = attack_item_xt(asin, ip, space, centroids, normalize,
                             victim_centroid=victim_centroid, shared_delta=shared_delta,
                             device=device)
        rows.append(row)
        if len(rows) % 10 == 0:
            msg = "[xt] %d done | last linf %.4f" % (len(rows), row["linf"])
            if "victim_cos_after" in row:
                msg += " | victim cos %.3f->%.3f" % (row["victim_cos_before"], row["victim_cos_after"])
            print(msg)

    summary = {"n_attacked": len(rows), "n_skipped_no_image": skipped,
               "mode": C.XT_DPRIME_MODE, "targeted": C.XT_TARGETED,
               "n_surrogates": len(space), "k": C.XT_K_SELECT, "steps": C.XT_STEPS,
               "max_linf": float(np.max([r["linf"] for r in rows])) if rows else None}
    if rows and "victim_cos_after" in rows[0]:
        summary["mean_victim_cos_before"] = float(np.mean([r["victim_cos_before"] for r in rows]))
        summary["mean_victim_cos_after"] = float(np.mean([r["victim_cos_after"] for r in rows]))
        # transfer-probe GO/NO-GO gate: poisoned must sit closer to the popular centroid
        summary["transfer_probe_ok"] = bool(
            summary["mean_victim_cos_after"] > summary["mean_victim_cos_before"])
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(C.XT_RESULTS_DIR, "xt_attack_summary.json"), "w"), indent=2)
    print("[xt] summary:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ctx = common.load_context(need_model=False)
    targets = P.test_positive_items(ctx.dataset)
    attack_targets_xt(ctx.dataset, targets)
