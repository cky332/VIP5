"""X-Transfer evaluation -- clean vs the black-box transferable attack, reusing the
pointwise harness (run_pointwise / scorer) unchanged.

Clean baseline = the XT clean re-extract (same victim pipeline as the XT poisoned
feature), so the only difference vs. poisoned is the adversarial delta (the CLIP-pipeline
confounder cancels). For a head-to-head table against the single-CLIP PGD baseline and
the alpha=1 feature bound, run those existing stages (`pgd`, `ablation`) on the SAME
target set and compare the JSONs.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import config as C
import eval_pointwise as EP
import feature_source_xt as FXT


def main():
    ctx = common.load_context(need_model=True)
    FXT.ensure_xt_dirs()
    clean_pos_fn = FXT.clean
    attacked = {"xtransfer": lambda asin, clean: FXT.poisoned(asin)}
    agg, n_skip = EP.run_pointwise(ctx, clean_pos_fn, attacked, require=FXT.has_poisoned)
    print("[eval_xtransfer] users scored:", agg.get("clean", {}).get("n", 0),
          "| skipped (no xt feat):", n_skip)
    EP._print_table(agg, order=["clean", "xtransfer"])
    os.makedirs(C.XT_RESULTS_DIR, exist_ok=True)
    json.dump(agg, open(C.XT_RESULTS_JSON, "w"), indent=2)
    print("[eval_xtransfer] saved", C.XT_RESULTS_JSON)


if __name__ == "__main__":
    main()
