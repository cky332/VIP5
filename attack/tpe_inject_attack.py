"""Attack (7): black-box, SCORE-BASED typographic injection with TPE (Bayesian) search.

Implements the spec: overlay a FIXED IPI injection text on the item's cover, but do NOT
hand-tune the typography -- use Optuna's TPE to search {position, font size, color (bg-mean
or fixed RGB), opacity, stroke, wrap} to MAXIMIZE the item's mean recommendation score over
N user contexts, MINUS a visibility penalty. Pure black-box, score-based, no gradients.

    obj(theta) = mean_u Score(VIP5 | user_u, render(x; theta)) - lambda * Visibility(x, render)

Score hook (VIP5): P("yes") from the B-1 yes/no template (scorer.score_user) for the item as
a candidate in user_u's context, using the rendered cover's CLIP feature.

HONEST EXPECTATION: VIP5 consumes only a pooled CLIP embedding and never decodes image text,
so the injection SEMANTICS are wasted and this is expected to be weak (same reason
text_attack's typographic variants failed). The right target for this attack is a
text-reading multimodal LLM (MLLM-MSR). It is implemented here as a rigorous black-box test
("does TPE-optimized typography beat the fixed inject_bgadapt variant on VIP5?").

Run (after the `clip` stage; needs the trained model):
    pip install optuna            # optional; falls back to random search if missing
    python attack/tpe_inject_attack.py
Outputs under attack/out/tpe_inject/ ; eval table: clean | tpe_inject [| inject_fixed].
"""
import os
import sys
import json
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import clip_extract as CE
import scorer
import pgd_attack as PGD
import eval_pointwise as EP
import feature_source as FS
import text_attack as TA          # reuse _font / _wrap / _text_w / _to_pil / _to_tensor01


# ---------------------------------------------------------------------------
# rendering: overlay `text` with typography theta on a 224x224 RGB PIL image
# ---------------------------------------------------------------------------
def render_inject(clean_pil, text, th):
    base = clean_pil.convert("RGBA")
    W, H = base.size
    base_np = np.asarray(clean_pil.convert("RGB"))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = TA._font(int(th["font_px"]))
    max_w = max(8, int(th["wrap_frac"] * W))
    lines = TA._wrap(d, text, font, max_w)
    asc, desc = font.getmetrics()
    line_h = asc + desc + 2
    line_ws = [TA._text_w(d, ln, font) for ln in lines]
    block_w = max(line_ws) if line_ws else 0
    block_h = line_h * len(lines)
    x0 = th["x_frac"] * max(0, W - block_w)
    y0 = th["y_frac"] * max(0, H - block_h)

    if th["color_mode"] == "bg":                       # blend with local background + brightness offset
        x1, y1 = int(max(0, x0)), int(max(0, y0))
        x2, y2 = int(min(W, x0 + block_w)), int(min(H, y0 + block_h))
        region = base_np[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else base_np
        m = region.reshape(-1, 3).mean(0)
        col = tuple(int(np.clip(c + th["bright_off"], 0, 255)) for c in m)
    else:                                              # fixed RGB
        col = (int(th["r"]), int(th["g"]), int(th["b"]))

    op = int(th["opacity"])
    sw = int(th["stroke"])
    fill = (*col, op)
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    stroke_col = (0, 0, 0, op) if lum > 127 else (255, 255, 255, op)   # auto contrast
    for i, ln in enumerate(lines):
        x = x0 + (block_w - line_ws[i]) / 2
        y = y0 + i * line_h
        d.text((x, y), ln, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke_col)
    return Image.alpha_composite(base, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# visibility:  LPIPS -> 1-SSIM(gray, global) -> RMSE   (larger = more visible)
# ---------------------------------------------------------------------------
_LPIPS = None


def _lpips_model():
    global _LPIPS
    if _LPIPS == "off":
        return None
    if _LPIPS is None:
        if C.TPE_VIS not in ("auto", "lpips"):
            _LPIPS = "off"; return None
        try:
            import lpips
            _LPIPS = lpips.LPIPS(net="alex").to(C.DEVICE).eval()
        except Exception:
            _LPIPS = "off"; return None
    return _LPIPS


def visibility(clean01, adv01):
    m = _lpips_model()
    if m is not None:
        with torch.no_grad():
            a = (clean01 * 2 - 1).unsqueeze(0).to(C.DEVICE)
            b = (adv01 * 2 - 1).unsqueeze(0).to(C.DEVICE)
            return float(m(a, b).item())
    a = clean01.cpu().numpy().transpose(1, 2, 0)
    b = adv01.cpu().numpy().transpose(1, 2, 0)
    if C.TPE_VIS == "rmse":
        return float(np.sqrt(((a - b) ** 2).mean()))
    w = np.array([0.299, 0.587, 0.114], dtype="float32")
    ga, gb = a @ w, b @ w
    mua, mub = ga.mean(), gb.mean()
    va, vb = ga.var(), gb.var()
    cov = ((ga - mua) * (gb - mub)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mua * mub + c1) * (2 * cov + c2)) / ((mua ** 2 + mub ** 2 + c1) * (va + vb + c2))
    return float(1.0 - ssim)


# ---------------------------------------------------------------------------
# Score hook: mean P("yes") for the item across N user contexts (black-box query)
# ---------------------------------------------------------------------------
def _mean_pyes(ctx, yes_id, no_id, user_ids, item_str, img01, normalize):
    feat = CE.encode_pixels(img01.unsqueeze(0), normalize=normalize,
                            device=C.DEVICE)[0].detach().cpu().numpy().astype("float32")
    ss = [scorer.score_user(ctx.model, ctx.dataset, u, [item_str], [feat],
                            yes_id, no_id, C.DEVICE)[0] for u in user_ids]
    return float(np.mean(ss)), feat


# ---------------------------------------------------------------------------
# TPE search space
# ---------------------------------------------------------------------------
def _suggest(trial, H):
    cm = trial.suggest_categorical("color_mode", ["bg", "rgb"])
    th = {"color_mode": cm,
          "x_frac": trial.suggest_float("x_frac", 0.0, 1.0),
          "y_frac": trial.suggest_float("y_frac", 0.0, 1.0),
          "font_px": trial.suggest_int("font_px", 10, max(11, int(0.18 * H))),
          "opacity": trial.suggest_int("opacity", 40, 255),
          "stroke": trial.suggest_int("stroke", 0, 3),
          "wrap_frac": trial.suggest_float("wrap_frac", 0.30, 0.95)}
    if cm == "bg":
        th["bright_off"] = trial.suggest_int("bright_off", -40, 40)
    else:
        th["r"] = trial.suggest_int("r", 0, 255)
        th["g"] = trial.suggest_int("g", 0, 255)
        th["b"] = trial.suggest_int("b", 0, 255)
    return th


def _rand_theta(rng, H):
    cm = rng.choice(["bg", "rgb"])
    th = {"color_mode": cm, "x_frac": rng.random(), "y_frac": rng.random(),
          "font_px": rng.randint(10, max(11, int(0.18 * H))),
          "opacity": rng.randint(40, 255), "stroke": rng.randint(0, 3),
          "wrap_frac": rng.uniform(0.30, 0.95)}
    if cm == "bg":
        th["bright_off"] = rng.randint(-40, 40)
    else:
        th["r"], th["g"], th["b"] = rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)
    return th


def optimize_item(ctx, item_str, clean_pil, user_ids, yes_id, no_id, normalize):
    W, H = clean_pil.size
    clean01 = TA._to_tensor01(clean_pil)
    s_clean, _ = _mean_pyes(ctx, yes_id, no_id, user_ids, item_str, clean01, normalize)
    text = C.TPE_INJECT_TEXT
    best = {"obj": -1e9}

    def eval_theta(th):
        adv = render_inject(clean_pil, text, th)
        adv01 = TA._to_tensor01(adv)
        s, feat = _mean_pyes(ctx, yes_id, no_id, user_ids, item_str, adv01, normalize)
        v = visibility(clean01, adv01)
        return s - C.TPE_LAMBDA * v, s, v, adv, feat

    def keep(th):
        obj, s, v, adv, feat = eval_theta(th)
        if obj > best["obj"]:
            best.update({"obj": obj, "s": s, "v": v, "adv": adv, "feat": feat, "theta": th})
        return obj

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=C.TPE_SEED))
        study.optimize(lambda tr: keep(_suggest(tr, H)), n_trials=C.TPE_TRIALS)
        best["search"] = "optuna-tpe"
    except ImportError:
        rng = random.Random(C.TPE_SEED)
        for _ in range(C.TPE_TRIALS):
            keep(_rand_theta(rng, H))
        best["search"] = "random"
    best["s_clean"] = s_clean
    return best


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate(ctx):
    normalize = common.get_clip_norm()
    if normalize is None:
        raise RuntimeError("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    for d in (C.TPE_POIS_FEAT_DIR, C.TPE_PERT_IMG_DIR, C.TPE_RESULTS_DIR, C.CLEAN_FEAT_DIR):
        os.makedirs(d, exist_ok=True)
    CE.load_clip(C.DEVICE)
    item2img = CE.load_item2img()
    yes_id, no_id = scorer.resolve_yes_no_ids(ctx.dataset.tokenizer)

    users_all = [u for u, _ in common.iter_test_users(ctx.dataset)]
    random.Random(C.TPE_SEED).shuffle(users_all)
    user_ids = users_all[:C.TPE_N_USERS]

    seen, pairs = set(), []
    if getattr(C, "TPE_ITEMS", None):                  # attack exactly these asins (e.g. the showcase covers)
        id2 = json.load(open(C.DATAMAPS))["item2id"]
        for a in C.TPE_ITEMS:
            if a in id2:
                pairs.append((id2[a], a))
            else:
                print("[tpe] TPE_ITEMS asin not in datamaps (skipped):", a)
    else:
        for it in PGD.test_positive_items(ctx.dataset):
            a = common.asin_of(ctx.dataset, it)
            if a in seen:
                continue
            seen.add(a); pairs.append((it, a))
        if C.TPE_N_ITEMS:
            pairs = pairs[:C.TPE_N_ITEMS]

    manifest, done, skipped = [], 0, 0
    for it, a in pairs:
        ip = CE.resolve_image_path(a, item2img)
        if ip is None:
            skipped += 1; continue
        clean224 = CE.preprocess_to_224(Image.open(ip))
        clean_pil = TA._to_pil(clean224)
        best = optimize_item(ctx, it, clean_pil, user_ids, yes_id, no_id, normalize)
        with torch.no_grad():
            clean_feat = CE.encode_pixels(clean224.unsqueeze(0), normalize=normalize, device=C.DEVICE)[0]
        np.save(os.path.join(C.CLEAN_FEAT_DIR, a + ".npy"), clean_feat.cpu().numpy().astype("float32"))
        np.save(os.path.join(C.TPE_POIS_FEAT_DIR, a + ".npy"), best["feat"])
        best["adv"].save(os.path.join(C.TPE_PERT_IMG_DIR, a + ".png"))
        manifest.append({"asin": a, "item": it, "s_clean": round(best["s_clean"], 4),
                         "s_adv": round(best["s"], 4), "delta_s": round(best["s"] - best["s_clean"], 4),
                         "visibility": round(best["v"], 4), "obj": round(best["obj"], 4),
                         "search": best["search"], "theta": best["theta"]})
        done += 1
        print("[tpe] %d/%d %s | P(yes) %.3f->%.3f (Δ%.3f) | vis %.3f | %s"
              % (done, len(pairs), a, best["s_clean"], best["s"],
                 best["s"] - best["s_clean"], best["v"], best["theta"]["color_mode"]))

    mean_ds = float(np.mean([m["delta_s"] for m in manifest])) if manifest else 0.0
    json.dump({"mean_delta_pyes": round(mean_ds, 4), "n": done, "skipped": skipped,
               "trials": C.TPE_TRIALS, "n_users": C.TPE_N_USERS, "lambda": C.TPE_LAMBDA,
               "rows": manifest}, open(C.TPE_MANIFEST_JSON, "w"), indent=2)
    print("[tpe] generation done: %d items, mean ΔP(yes) over contexts = %+.4f" % (done, mean_ds))
    return manifest


# ---------------------------------------------------------------------------
# evaluation: clean | tpe_inject  (+ inject_fixed if text_attack's variant exists)
# ---------------------------------------------------------------------------
def _loader(d):
    return lambda asin, clean: FS._load(os.path.join(d, asin + ".npy"))


def evaluate(ctx):
    os.makedirs(C.TPE_RESULTS_DIR, exist_ok=True)
    inj_fixed_dir = os.path.join(C.OUT_DIR, "text", "inject_bgadapt", "poisoned", C.SPLIT)
    have_fixed = os.path.isdir(inj_fixed_dir) and any(f.endswith(".npy") for f in os.listdir(inj_fixed_dir))

    attacked = {"tpe_inject": _loader(C.TPE_POIS_FEAT_DIR)}
    if have_fixed:
        attacked["inject_fixed"] = _loader(inj_fixed_dir)

    def require(asin):
        ok = os.path.isfile(os.path.join(C.TPE_POIS_FEAT_DIR, asin + ".npy"))
        if have_fixed:
            ok = ok and os.path.isfile(os.path.join(inj_fixed_dir, asin + ".npy"))
        return ok

    agg, n_skip = EP.run_pointwise(ctx, FS.clean_pgd, attacked, require=require)
    order = ["clean", "tpe_inject"] + (["inject_fixed"] if have_fixed else [])
    print("\n=== TPE typographic injection (black-box score-based) vs clean — B-1, posonly ===")
    EP._print_table(agg, order=order)
    json.dump(agg, open(C.TPE_RESULTS_JSON, "w"), indent=2)
    print("[tpe] users=%d skipped=%d | saved %s"
          % (agg.get("clean", {}).get("n", 0), n_skip, C.TPE_RESULTS_JSON))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "showcase":                       # attack exactly the 6 showcase covers (no config edit needed)
        C.TPE_ITEMS = C.SHOWCASE_ASINS
    elif arg:                                   # or an explicit comma-separated asin list
        C.TPE_ITEMS = arg.split(",")
    ctx = common.load_context(need_model=True)
    generate(ctx)
    evaluate(ctx)


if __name__ == "__main__":
    main()
