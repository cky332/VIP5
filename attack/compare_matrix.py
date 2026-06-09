"""One-shot comparison of the X-Transfer attack across two axes, no config editing:

    target embedding :  top1 (single hottest item)   |  mean (top-K centroid)
    threat model     :  black-box (victim held out)   |  white-box (victim in pool)

For each combo it runs the full chain end-to-end on a SMALL subsample (fast):
    build per-surrogate centroids  ->  per-target delta + victim re-extract  ->  pointwise eval
then prints ONE comparison table (+ a combined JSON), reusing run_pointwise/scorer.

It sets the two axes on the in-memory config (read at call-time by build_all_centroids /
xt_search_space), and passes `steps`/`n_users` explicitly (those are import-time defaults
elsewhere), so nothing on disk is edited between runs.

Usage (from repo root, after `python attack/run_all.py clip` has resolved CLIP_NORM):
    python attack/compare_matrix.py                          # 50 users, 100 steps, all 4 combos
    python attack/compare_matrix.py --users 100 --steps 200  # closer to full-scale
    python attack/compare_matrix.py --combos white-top1,white-mean
    python attack/compare_matrix.py --pool open_clip         # use the diverse pool (needs install)
"""
import os
import sys
import json
import time
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import xtransfer_centroid as XC
import xtransfer_attack as XA
import eval_pointwise as EP
import feature_source_xt as FXT
import pgd_attack as P

COMBOS = {
    "black-top1": dict(include_victim=False, mode="top1"),
    "black-mean": dict(include_victim=False, mode="mean"),
    "white-top1": dict(include_victim=True,  mode="top1"),
    "white-mean": dict(include_victim=True,  mode="mean"),
}


def _reset_xt_feature_dirs():
    """Wipe per-combo centroids + poisoned/clean feats so combos never read stale files."""
    for d in (C.XT_CENTROID_DIR, C.XT_POIS_FEAT_DIR, C.XT_CLEAN_FEAT_DIR):
        shutil.rmtree(d, ignore_errors=True)
    FXT.ensure_xt_dirs()


def run_combo(ctx, name, n_users, steps, targets):
    cfg = COMBOS[name]
    C.XT_INCLUDE_VICTIM = cfg["include_victim"]   # read live by config.xt_search_space()
    C.XT_CENTROID_MODE = cfg["mode"]              # read live by build_all_centroids()
    C.XT_TARGET_ITEM = None
    pool = C.xt_search_space()
    print("\n" + "=" * 74)
    print("[matrix] %s | victim_in_pool=%s | target=%s | surrogates=%d | users=%d steps=%d"
          % (name, cfg["include_victim"], cfg["mode"], len(pool), n_users, steps))
    print("=" * 74)
    t0 = time.time()
    _reset_xt_feature_dirs()
    XC.build_all_centroids(ctx.dataset)                          # -> centroids/ (mode + pool aware)
    XA.attack_targets_xt(ctx.dataset, targets, steps=steps)      # -> poisoned/clean feats
    agg, n_skip = EP.run_pointwise(
        ctx, FXT.clean, {"xtransfer": lambda asin, clean: FXT.poisoned(asin)},
        require=FXT.has_poisoned, n_users=n_users)
    summ = json.load(open(os.path.join(C.XT_RESULTS_DIR, "xt_attack_summary.json")))["summary"]
    return {"combo": name, **cfg, "n_surrogates": len(pool), "n_skip": n_skip,
            "secs": round(time.time() - t0, 1), "agg": agg,
            "transfer": {k: summ.get(k) for k in
                         ("mean_victim_cos_before", "mean_victim_cos_after", "transfer_probe_ok")}}


def _row(label, a, vcos_delta=None, probe=None):
    vc = "%+8.3f" % vcos_delta if vcos_delta is not None else "%8s" % "-"
    pr = "%s" % probe if probe is not None else "-"
    return "%-12s %6.3f %8.4f %8.3f %8.4f %s   %s" % (
        label, a["mean_rank"], a["mean_p_yes"], a["HR@10"], a["NDCG@10"], vc, pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=50, help="subsample size (targets) per combo")
    ap.add_argument("--steps", type=int, default=100, help="PGD steps per target")
    ap.add_argument("--combos", default=",".join(COMBOS), help="comma list from: " + ",".join(COMBOS))
    ap.add_argument("--pool", choices=["openai", "open_clip"], default=None,
                    help="surrogate pool (default: whatever XT_USE_OPEN_CLIP is in config)")
    ap.add_argument("--crop", action="store_true",
                    help="M-Attack random-crop local matching (sets XT_CROP=True)")
    args = ap.parse_args()
    names = [c.strip() for c in args.combos.split(",") if c.strip() in COMBOS]
    if not names:
        raise SystemExit("no valid combos in --combos; choose from " + ",".join(COMBOS))
    if args.pool is not None:
        C.XT_USE_OPEN_CLIP = (args.pool == "open_clip")
    if args.crop:
        C.XT_CROP = True

    C.N_TEST_USERS = args.users
    ctx = common.load_context(need_model=True)
    if common.get_clip_norm() is None:
        raise SystemExit("CLIP_NORM unresolved; run `python attack/run_all.py clip` first.")
    targets = P.test_positive_items(ctx.dataset, n_users=args.users)   # explicit (def-time default)

    results = [run_combo(ctx, n, args.users, args.steps, targets) for n in names]

    # ---------------- combined table ----------------
    pool = "open_clip" if C.XT_USE_OPEN_CLIP else "openai"
    print("\n" + "#" * 74)
    print("X-TRANSFER COMPARISON  (n=%d users, %d steps, pool=%s, crop=%s)"
          % (args.users, args.steps, pool, C.XT_CROP))
    hdr = "%-12s %6s %8s %8s %8s %8s   %s" % (
        "combo", "rank", "P(yes)", "HR@10", "NDCG@10", "vcosD", "probe")
    print("#" * 74)
    print(hdr)
    print("-" * len(hdr))
    if results:                                  # clean is identical across combos -> print once
        print(_row("clean", results[0]["agg"]["clean"]))
    for r in results:
        a = r["agg"].get("xtransfer")
        if not a:
            print("%-12s  (no xtransfer rows; n_skip=%d)" % (r["combo"], r["n_skip"]))
            continue
        tr = r["transfer"]
        vd = (tr["mean_victim_cos_after"] - tr["mean_victim_cos_before"]
              if tr.get("mean_victim_cos_after") is not None else None)
        print(_row(r["combo"], a, vcos_delta=vd, probe=tr.get("transfer_probe_ok")))
    print("\nlower rank = better for the attacker;  vcosD = victim cos(after)-cos(before);  "
          "compare against clean.")
    out = os.path.join(C.XT_RESULTS_DIR, "compare_matrix.json")
    os.makedirs(C.XT_RESULTS_DIR, exist_ok=True)
    json.dump({"users": args.users, "steps": args.steps, "pool": pool, "results": results},
              open(out, "w"), indent=2)
    print("[matrix] saved", out)


if __name__ == "__main__":
    main()
