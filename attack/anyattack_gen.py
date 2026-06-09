"""AnyAttack (arXiv 2410.05346) — self-contained lightweight port for VIP5.

Train a target-conditioned noise generator G(z_target) -> delta, self-supervised on
the toys catalog against the SAME surrogate CLIP ensemble used by X-Transfer (victim
OpenAI ViT-B/32 held out). AnyAttack's idea: delta is produced from the TARGET image's
embedding and added to ANY source image so the source's embedding aligns to the target.

Here the deployment target is the hottest item (XT_CENTROID_MODE), so at attack time we
emit ONE delta = G(E0(hottest cover)) and add it to every candidate cover (a learned,
amortized universal perturbation toward "popular"). Then re-extract with the VICTIM
ViT-B/32 and evaluate rank -- exactly the same probe/eval as X-Transfer.

NOTE: full AnyAttack pre-trains on LAION-400M; that scale (its main transfer edge) is
out of scope here. This trains on the toys catalog only, so expect transfer on the order
of the X-Transfer ensemble PGD. Runs comfortably on a single 24GB GPU.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import config as C
import clip_extract as CE
import surrogates as SUR
import xtransfer_centroid as XC


def _ensure_aa_dirs():
    for d in (C.AA_OUT_DIR, C.AA_POIS_FEAT_DIR, C.AA_CLEAN_FEAT_DIR,
              C.AA_PERT_IMG_DIR, C.AA_RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


def _cos_np(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# generator: target embedding -> delta in [-eps, eps], 224x224
# ---------------------------------------------------------------------------
class NoiseGenerator(nn.Module):
    def __init__(self, in_dim, eps, base=512):
        super().__init__()
        self.eps = float(eps)
        self.base = int(base)
        self.fc = nn.Linear(in_dim, self.base * 4 * 4)

        def block(ci, co):
            return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        b = self.base
        self.net = nn.Sequential(
            block(b, b // 2),        # 4 -> 8
            block(b // 2, b // 4),   # 8 -> 16
            block(b // 4, b // 8),   # 16 -> 32
            block(b // 8, b // 16),  # 32 -> 64
            block(b // 16, b // 32), # 64 -> 128
            nn.ConvTranspose2d(b // 32, 3, 4, 2, 1),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, self.base, 4, 4)
        x = self.net(x)                                              # (B,3,256,256) in [-1,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x * self.eps                                         # delta in [-eps, eps]


def _norm(z):
    return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# training data: catalog covers (uint8 CPU cache)
# ---------------------------------------------------------------------------
def _load_catalog_covers(dataset, item2img, max_items=None):
    import random as _r
    ids = list(dataset.id2item.keys())
    _r.Random(C.AA_SEED).shuffle(ids)
    covers = []
    for iid in ids:
        if max_items and len(covers) >= max_items:
            break
        x = XC.load_cover_224(common.asin_of(dataset, iid), item2img)   # (3,224,224) [0,1] or None
        if x is not None:
            covers.append((x.clamp(0, 1) * 255.0).round().to(torch.uint8))
    if not covers:
        raise RuntimeError("no catalog covers resolved for AnyAttack training")
    return torch.stack(covers, 0)   # (N,3,224,224) uint8 on CPU


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_generator(dataset, device=None):
    _ensure_aa_dirs()
    device = device or C.DEVICE
    common.set_seed(C.AA_SEED)
    space = SUR.build_search_space(device)
    for s in space:
        s.to_gpu()
    E0 = space[0]                                   # conditioning encoder
    item2img = CE.load_item2img()
    X = _load_catalog_covers(dataset, item2img, C.AA_MAX_ITEMS)
    N = X.size(0)
    print("[aa] training covers: %d | surrogates: %d | cond=%s" % (N, len(space), E0.sid))

    with torch.no_grad():
        d0 = E0.encode((X[:1].float() / 255.0).to(device)).shape[-1]
    G = NoiseGenerator(d0, C.AA_EPS, base=C.AA_GEN_CH).to(device)
    G.train()
    opt = torch.optim.Adam(G.parameters(), lr=C.AA_LR)
    rng = np.random.RandomState(C.AA_SEED)
    spe = max(1, N // C.AA_BATCH)

    for ep in range(C.AA_EPOCHS):
        perm = rng.permutation(N)
        run = 0.0
        for b in range(spe):
            idx = perm[b * C.AA_BATCH:(b + 1) * C.AA_BATCH]
            if len(idx) < 2:
                continue
            tgt = (X[idx].float() / 255.0).to(device)              # targets (B,3,224,224)
            src = torch.roll(tgt, 1, dims=0)                       # sources = shuffled (different imgs)
            with torch.no_grad():
                z0 = _norm(E0.encode(tgt))                         # conditioning embedding
            delta = G(z0)                                          # (B,3,224,224) per-target
            adv = torch.clamp(src + delta, 0, 1)
            # memory-bounded: accumulate grad wrt `adv` ONE surrogate at a time (each
            # surrogate's forward graph is freed by autograd.grad), then a single
            # backward adv->delta->G. Avoids holding all k CLIP graphs at once (OOM).
            adv_d = adv.detach().requires_grad_(True)
            sel = rng.choice(len(space), size=min(C.AA_K, len(space)), replace=False)
            adv_grad = torch.zeros_like(adv_d)
            step_loss = 0.0
            for i in sel:
                s = space[int(i)]
                with torch.no_grad():
                    zt = s.encode(tgt)
                za = s.encode(adv_d)
                Li = (1.0 - F.cosine_similarity(za, zt)).mean() / len(sel)
                gi, = torch.autograd.grad(Li, adv_d)              # frees this surrogate's graph
                adv_grad = adv_grad + gi
                step_loss += float(Li.detach())
            opt.zero_grad()
            adv.backward(adv_grad)                                 # adv -> delta -> G (once)
            opt.step()
            run += step_loss
            if b % 50 == 0:
                print("[aa] ep %d step %d/%d loss %.4f" % (ep + 1, b, spe, step_loss))
        print("[aa] epoch %d done | mean loss %.4f" % (ep + 1, run / max(1, spe)))

    torch.save({"state_dict": G.state_dict(), "in_dim": d0,
                "eps": C.AA_EPS, "base": C.AA_GEN_CH, "cond_sid": E0.sid}, C.AA_GEN_PATH)
    print("[aa] saved generator ->", C.AA_GEN_PATH)
    return G


def load_generator(device=None):
    device = device or C.DEVICE
    ck = torch.load(C.AA_GEN_PATH, map_location=device)
    G = NoiseGenerator(ck["in_dim"], ck["eps"], base=ck["base"]).to(device)
    G.load_state_dict(ck["state_dict"])
    G.eval()
    return G


# ---------------------------------------------------------------------------
# attack: one delta toward the hottest item, applied to every candidate
# ---------------------------------------------------------------------------
def _hottest_asin(dataset):
    pops = XC._target_items(dataset, getattr(C, "XT_CENTROID_MODE", "top1"),
                            getattr(C, "XT_TARGET_ITEM", None), C.K_POPULAR)
    return common.asin_of(dataset, pops[0][0])


def _save_png(x_chw, path):
    from PIL import Image
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def attack_targets_aa(dataset, target_item_strs, device=None):
    from PIL import Image
    _ensure_aa_dirs()
    device = device or C.DEVICE
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    if not os.path.isfile(C.AA_GEN_PATH):
        raise RuntimeError("no generator; run `python attack/run_all.py aa-train` first.")

    space = SUR.build_search_space(device)
    E0 = space[0].to_gpu()
    G = load_generator(device)
    item2img = CE.load_item2img()
    CE.load_clip(device)

    tgt_asin = _hottest_asin(dataset)
    tgt_cover = XC.load_cover_224(tgt_asin, item2img).to(device)
    with torch.no_grad():
        z0 = _norm(E0.encode(tgt_cover.unsqueeze(0)))
        delta = G(z0)[0].detach()                                          # (3,224,224) universal
        victim_tgt = CE.encode_pixels(tgt_cover.unsqueeze(0), normalize=normalize,
                                      device=device)[0].cpu().numpy().astype("float32")
    np.save(os.path.join(C.AA_OUT_DIR, "delta.npy"), delta.cpu().numpy().astype("float32"))
    print("[aa] hottest target asin=%s | delta linf=%.4f" % (tgt_asin, float(delta.abs().max())))

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
        x0 = CE.preprocess_to_224(Image.open(ip)).to(device)
        x_adv = torch.clamp(x0 + delta, 0, 1)
        with torch.no_grad():
            clean = CE.encode_pixels(x0.unsqueeze(0), normalize=normalize, device=device)[0].cpu().numpy().astype("float32")
            pois = CE.encode_pixels(x_adv.unsqueeze(0), normalize=normalize, device=device)[0].cpu().numpy().astype("float32")
        np.save(os.path.join(C.AA_CLEAN_FEAT_DIR, asin + ".npy"), clean)
        np.save(os.path.join(C.AA_POIS_FEAT_DIR, asin + ".npy"), pois)
        _save_png(x_adv, os.path.join(C.AA_PERT_IMG_DIR, asin + ".png"))
        row = {"asin": asin, "linf": float((x_adv - x0).abs().max().item()),
               "victim_cos_before": _cos_np(clean, victim_tgt),
               "victim_cos_after": _cos_np(pois, victim_tgt)}
        rows.append(row)
        if len(rows) % 10 == 0:
            print("[aa] %d done | linf %.4f | victim cos %.3f->%.3f"
                  % (len(rows), row["linf"], row["victim_cos_before"], row["victim_cos_after"]))

    summary = {"n_attacked": len(rows), "n_skipped_no_image": skipped,
               "hottest_asin": tgt_asin, "n_surrogates": len(space),
               "epochs": C.AA_EPOCHS, "max_linf": float(np.max([r["linf"] for r in rows])) if rows else None}
    if rows:
        summary["mean_victim_cos_before"] = float(np.mean([r["victim_cos_before"] for r in rows]))
        summary["mean_victim_cos_after"] = float(np.mean([r["victim_cos_after"] for r in rows]))
        summary["transfer_probe_ok"] = bool(summary["mean_victim_cos_after"] > summary["mean_victim_cos_before"])
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(C.AA_RESULTS_DIR, "aa_attack_summary.json"), "w"), indent=2)
    print("[aa] summary:", json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# eval (reuse run_pointwise/scorer)
# ---------------------------------------------------------------------------
def _aa_clean(asin):
    return np.load(os.path.join(C.AA_CLEAN_FEAT_DIR, asin + ".npy")).astype("float32").reshape(-1)


def _aa_poisoned(asin):
    return np.load(os.path.join(C.AA_POIS_FEAT_DIR, asin + ".npy")).astype("float32").reshape(-1)


def _aa_has(asin):
    return os.path.isfile(os.path.join(C.AA_POIS_FEAT_DIR, asin + ".npy"))


def eval_aa():
    import eval_pointwise as EP
    ctx = common.load_context(need_model=True)
    _ensure_aa_dirs()
    agg, n_skip = EP.run_pointwise(ctx, _aa_clean,
                                   {"anyattack": lambda asin, clean: _aa_poisoned(asin)},
                                   require=_aa_has)
    print("[aa-eval] users scored:", agg.get("clean", {}).get("n", 0), "| skipped:", n_skip)
    EP._print_table(agg, order=["clean", "anyattack"])
    os.makedirs(C.AA_RESULTS_DIR, exist_ok=True)
    json.dump(agg, open(C.AA_RESULTS_JSON, "w"), indent=2)
    print("[aa-eval] saved", C.AA_RESULTS_JSON)


if __name__ == "__main__":
    import pgd_attack as P
    ctx = common.load_context(need_model=False)
    train_generator(ctx.dataset)
    attack_targets_aa(ctx.dataset, P.test_positive_items(ctx.dataset))
