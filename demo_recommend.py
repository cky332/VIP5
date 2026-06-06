"""Showcase VIP5's flagship task: SEQUENTIAL RECOMMENDATION (template A-9).

For a few test users it prints, concretely:
  - the user's purchase history (item ids + titles + each item carries its cover image)
  - the held-out true "next item"
  - VIP5's top-k generated recommendations (beam search) with titles
  - whether/where the true next item appears

This is the most representative demo of "how VIP5 recommends": everything is cast
as text-to-text generation over item ids, and the multimodal prompt feeds each
history item's CLIP image at its <extra_id_0> placeholders.

Run from repo root:
    python demo_recommend.py            # 3 users, top-10
    python demo_recommend.py 5 20       # 5 users, top-20 beams
"""
import os
import sys

import numpy as np
import torch

# reuse the (tested) model+dataset setup from attack/common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack"))
import common          # noqa: sets paths + chdir(ROOT)
import config as C


def title_of(ds, item_id):
    try:
        asin = ds.id2item[str(item_id)]
        m = ds.meta_data[ds.meta_dict[asin]]
        t = m.get("title", "") if isinstance(m, dict) else ""
        return (t[:60] + "…") if len(t) > 60 else (t or "(no title)")
    except Exception:
        return "(no title)"


def build_a9_batch(ds, user_desc, history):
    """Construct one A-9 example exactly like src/data.py:233-243 (test branch)."""
    ratio = C.SIZE_RATIO
    tmpl = ds.all_tasks["sequential"]["A-9"]
    hist_str = " {}, ".format("<extra_id_0> " * ratio).join(history) + " <extra_id_0>" * ratio
    source = tmpl["source"].format(user_desc, hist_str)
    tok = ds.tokenizer
    input_ids = tok.encode(source, padding=True, truncation=True, max_length=ds.args.max_text_length)
    tokenized = tok.tokenize(source)
    wwids = ds.calculate_whole_word_ids(tokenized, input_ids)
    cat_ids = [1 if t == 32099 else 0 for t in input_ids]
    feats = np.stack([common.load_shipped(common.asin_of(ds, h)) for h in history], axis=0).astype("float32")
    entry = {
        "input_ids": torch.LongTensor(input_ids), "input_length": len(input_ids),
        "whole_word_ids": torch.LongTensor(wwids), "category_ids": torch.LongTensor(cat_ids),
        "target_ids": torch.LongTensor(tok.encode("0")), "target_length": 1,
        "source_text": source, "tokenized_text": tokenized, "target_text": "0", "task": "sequential",
        "vis_feats": torch.from_numpy(feats), "vis_feat_length": feats.shape[0], "loss_weight": 1.0,
    }
    return ds.collate_fn([entry]), source


@torch.no_grad()
def recommend(model, ds, batch, device, num_beams=20):
    out = model.generate(
        input_ids=batch["input_ids"].to(device),
        whole_word_ids=batch["whole_word_ids"].to(device),
        category_ids=batch["category_ids"].to(device),
        vis_feats=batch["vis_feats"].to(device),
        task="sequential", max_length=50, num_beams=num_beams,
        no_repeat_ngram_size=0, num_return_sequences=num_beams, early_stopping=True)
    gen = model.tokenizer.batch_decode(out, skip_special_tokens=True)
    # de-duplicate, keep beam order
    seen, ranked = set(), []
    for g in gen:
        g = g.strip()
        if g and g not in seen:
            seen.add(g); ranked.append(g)
    return ranked


def main(n_users=3, num_beams=20, topk=10):
    ctx = common.load_context(need_model=True)
    ds, model, device = ctx.dataset, ctx.model, ctx.device

    shown = 0
    for user_id, items in common.iter_test_users(ds, n_users=2000, seed=C.SEED):
        if not (3 <= len(items) <= 8):          # pick short, readable histories
            continue
        history = [str(x) for x in items[:-1]]
        target = str(items[-1])
        user_desc = ds.user_id2name.get(user_id, user_id)
        batch, _src = build_a9_batch(ds, user_desc, history)
        recs = recommend(model, ds, batch, device, num_beams=num_beams)

        print("\n" + "=" * 78)
        print("User: %s   (user_id=%s)" % (user_desc, user_id))
        print("-" * 78)
        print("Purchase history (each item also carries its cover image):")
        for h in history:
            print("   [%s] %s" % (h, title_of(ds, h)))
        print("True next item (held out):  [%s] %s" % (target, title_of(ds, target)))
        print("-" * 78)
        print("VIP5 top-%d recommendations (beam search over item ids):" % topk)
        rank = None
        for i, r in enumerate(recs[:topk], 1):
            hit = ""
            if r == target:
                rank = i; hit = "   <== HIT (true next item)"
            print("  %2d. [%s] %s%s" % (i, r, title_of(ds, r), hit))
        full_rank = (recs.index(target) + 1) if target in recs else None
        print("-> true next item rank among beams: %s | HR@10: %s"
              % (full_rank if full_rank else ">%d" % len(recs), "YES" if (rank and rank <= 10) else "no"))
        shown += 1
        if shown >= n_users:
            break


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(n, nb)
