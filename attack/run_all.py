"""End-to-end orchestrator for the VIP5 popularity-mimicry attack.

Usage (from repo root):
    python attack/run_all.py                 # full pipeline
    python attack/run_all.py ablation        # only the feature-space upper bound (no images)
    python attack/run_all.py clip centroid pgd      # only build poisoned features
    python attack/run_all.py pointwise listwise     # only evals (after pgd)

Stages: ablation | clip | centroid | pgd | pointwise | listwise
        xt-centroid | xt-attack | xt-eval    (X-Transfer: black-box transferable attack)

X-Transfer usage (after `clip` has resolved CLIP_NORM):
    python attack/run_all.py xt-centroid xt-attack xt-eval
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C

STAGES = ["ablation", "clip", "centroid", "pgd", "pointwise", "listwise",
          "xt-centroid", "xt-attack", "xt-eval",
          "aa-train", "aa-attack", "aa-eval"]


def preflight():
    miss = []
    for p in (os.path.join("data", C.SPLIT), C.FEAT_DIR, C.CKPT):
        if not os.path.exists(p):
            miss.append(p)
    if miss:
        raise SystemExit("[run_all] missing required assets (place them first, see DEPLOY.md):\n  "
                         + "\n  ".join(miss))
    common.ensure_dirs()
    print("[run_all] preflight OK")


def main(stages):
    preflight()
    for st in stages:
        print("\n" + "=" * 60 + "\n[run_all] STAGE:", st, "\n" + "=" * 60)
        if st == "ablation":
            import ablation_feature_space as A
            A.main()
        elif st == "clip":
            import clip_extract as CE
            ctx = common.load_context(need_model=False)
            CE.unzip_photos()
            CE.verify_against_shipped(ctx.dataset)
        elif st == "centroid":
            import build_centroid as BC
            ctx = common.load_context(need_model=False)
            BC.build_centroid(ctx.dataset, source="reextract")
        elif st == "pgd":
            import pgd_attack as P
            ctx = common.load_context(need_model=False)
            P.attack_targets(ctx.dataset, P.test_positive_items(ctx.dataset))
        elif st == "pointwise":
            import eval_pointwise as EP
            EP.main()
        elif st == "listwise":
            import eval_listwise as EL
            EL.main()
        elif st == "xt-centroid":
            import xtransfer_centroid as XC
            ctx = common.load_context(need_model=False)
            XC.build_all_centroids(ctx.dataset)
        elif st == "xt-attack":
            import xtransfer_attack as XA
            import pgd_attack as P
            ctx = common.load_context(need_model=False)
            XA.attack_targets_xt(ctx.dataset, P.test_positive_items(ctx.dataset))
        elif st == "xt-eval":
            import eval_xtransfer as EX
            EX.main()
        elif st == "aa-train":
            import anyattack_gen as AA
            ctx = common.load_context(need_model=False)
            AA.train_generator(ctx.dataset)
        elif st == "aa-attack":
            import anyattack_gen as AA
            import pgd_attack as P
            ctx = common.load_context(need_model=False)
            AA.attack_targets_aa(ctx.dataset, P.test_positive_items(ctx.dataset))
        elif st == "aa-eval":
            import anyattack_gen as AA
            AA.eval_aa()
        else:
            print("[run_all] unknown stage:", st)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in STAGES]
    main(args or STAGES)
