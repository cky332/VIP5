"""(4) PRIMARY attack eval -- faithful MLLM-MSR analog.

Per test user: 1 positive + N_NEG negatives, scored independently by P("yes") via
the B-1 yes/no template. Only the positive's cover feature is swapped (clean ->
poisoned); negatives are held fixed. Reports positive mean-rank, HR@10, NDCG@10,
and mean P("yes") before/after.

Confounder control: the CLEAN positive feature is the re-extracted clean feature
from the SAME CLIP pipeline as the attack (attack/out/clean_features), so the only
difference vs. the poisoned feature is the adversarial perturbation.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import scorer
import feature_source as FS


def _aggregate(metric_list, ks):
    out = {"n": len(metric_list),
           "mean_rank": float(np.mean([m["rank"] for m in metric_list])),
           "mean_p_yes": float(np.mean([m["p_yes"] for m in metric_list]))}
    for k in ks:
        out["HR@%d" % k] = float(np.mean([m["hit@%d" % k] for m in metric_list]))
        out["NDCG@%d" % k] = float(np.mean([m["ndcg@%d" % k] for m in metric_list]))
    return out


def run_pointwise(ctx, clean_pos_fn, attacked_variants, neg_fn=None,
                  ks=(5, 10), n_users=C.N_TEST_USERS, require=None):
    """clean_pos_fn(pos_asin)->feat ; attacked_variants {label: fn(pos_asin, clean_feat)->feat}
       neg_fn(neg_item_str)->feat (default: shipped) ; require(pos_asin)->bool (optional filter)."""
    model, dataset, device = ctx.model, ctx.dataset, ctx.device
    yes_id, no_id = scorer.resolve_yes_no_ids(dataset.tokenizer)
    neg_fn = neg_fn or (lambda n: FS.shipped(dataset, n))

    per = {"clean": []}
    for lbl in attacked_variants:
        per[lbl] = []
    n_skip = 0

    for user_id, items in common.iter_test_users(dataset, n_users):
        pos = str(items[-1])
        pos_asin = common.asin_of(dataset, pos)
        if require is not None and not require(pos_asin):
            n_skip += 1
            continue
        try:
            clean_pos = clean_pos_fn(pos_asin)
        except FileNotFoundError:
            n_skip += 1
            continue
        negs = common.sample_negatives(dataset, user_id, items)
        neg_feats = [neg_fn(n) for n in negs]
        s = scorer.score_user(model, dataset, user_id, [pos] + negs,
                              [clean_pos] + neg_feats, yes_id, no_id, device)
        pos_clean, neg_scores = s[0], s[1:]
        per["clean"].append(scorer.user_metrics(pos_clean, neg_scores, ks))
        for lbl, fn in attacked_variants.items():
            try:
                af = fn(pos_asin, clean_pos)
            except FileNotFoundError:
                continue
            pa = scorer.score_user(model, dataset, user_id, [pos], [af],
                                   yes_id, no_id, device)[0]
            per[lbl].append(scorer.user_metrics(pa, neg_scores, ks))

    agg = {lbl: _aggregate(ms, ks) for lbl, ms in per.items() if ms}
    return agg, n_skip


def _print_table(agg, order):
    ks_cols = ["mean_rank", "mean_p_yes", "HR@10", "NDCG@10"]
    print("\n%-14s %8s %9s %7s %8s %5s" % ("condition", "rank", "P(yes)", "HR@10", "NDCG@10", "n"))
    for lbl in order:
        if lbl not in agg:
            continue
        a = agg[lbl]
        print("%-14s %8.3f %9.4f %7.3f %8.4f %5d"
              % (lbl, a["mean_rank"], a["mean_p_yes"], a["HR@10"], a["NDCG@10"], a["n"]))


def main():
    ctx = common.load_context(need_model=True)
    common.ensure_dirs()
    # PIXEL attack: clean = re-extracted clean (same CLIP pipeline), attacked = poisoned
    clean_pos_fn = FS.clean_pgd
    attacked = {"attacked": lambda asin, clean: FS.poisoned(asin)}
    agg, n_skip = run_pointwise(ctx, clean_pos_fn, attacked,
                                require=FS.has_poisoned)
    print("[eval_pointwise] users scored:", agg.get("clean", {}).get("n", 0),
          "| skipped (no poisoned/clean feat):", n_skip)
    _print_table(agg, order=["clean", "attacked"])
    json.dump(agg, open(os.path.join(C.RESULTS_DIR, "pointwise.json"), "w"), indent=2)
    print("[eval_pointwise] saved", os.path.join(C.RESULTS_DIR, "pointwise.json"))


if __name__ == "__main__":
    main()
