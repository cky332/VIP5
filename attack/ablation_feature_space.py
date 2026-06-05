"""(6) Feature-space ablation -- the UPPER BOUND, runnable immediately (no images,
no CLIP). Move the positive's CLEAN feature toward the popular centroid by alpha
and re-score with the pointwise P("yes") harness.

If even alpha=1 (pure centroid) barely raises P("yes")/rank, the model is robust
to the image channel and the pixel PGD attack cannot beat this bound -- an
important guardrail / (negative) result.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import build_centroid as BC
import eval_pointwise as EP


def interpolate(feat, centroid, alpha, normalize):
    f = (1.0 - alpha) * feat + alpha * centroid
    if normalize:
        f = f / (np.linalg.norm(f) + 1e-8)
    return f.astype("float32")


def main():
    ctx = common.load_context(need_model=True)
    common.ensure_dirs()

    if not os.path.isfile(C.CENTROID_PATH):
        BC.build_centroid(ctx.dataset, source="shipped")
    centroid = np.load(C.CENTROID_PATH).astype("float32")

    flag = common.get_clip_norm()
    if flag is None:
        flag = BC._shipped_norm_autodetect(ctx.dataset)

    clean_pos_fn = common.load_shipped                      # by asin (shipped)
    variants = {}
    for a in C.ABLATION_ALPHAS:
        variants["alpha=%.2f" % a] = (
            lambda asin, clean, _a=a: interpolate(clean, centroid, _a, flag))

    agg, n_skip = EP.run_pointwise(ctx, clean_pos_fn, variants)
    order = ["clean"] + ["alpha=%.2f" % a for a in C.ABLATION_ALPHAS]
    print("[ablation] normalize=%s | users:" % flag, agg.get("clean", {}).get("n", 0))
    EP._print_table(agg, order=order)
    json.dump({"normalize": bool(flag), "alphas": C.ABLATION_ALPHAS, "agg": agg},
              open(os.path.join(C.RESULTS_DIR, "ablation.json"), "w"), indent=2)
    print("[ablation] saved", os.path.join(C.RESULTS_DIR, "ablation.json"))


if __name__ == "__main__":
    main()
