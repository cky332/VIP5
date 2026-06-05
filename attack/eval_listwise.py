"""(5) SECONDARY attack eval -- native VIP5 listwise B-8 (generative beam search).

Builds the B-8 prompt exactly like src/data.py:472-491 (99 sampled negatives + the
target, each with its image), but with the TARGET's image feature controllable
(clean vs poisoned). Runs beam search and measures the target's rank among the
beam outputs, paired (same negatives + shuffle) so only the target image differs.

NOTE (expected): B-8 ranking is generative -- item identity is dominated by the
textual item ids, so a single perturbed image may move the rank only slightly or
not at all. The pointwise eval (eval_pointwise) is the sensitive, primary metric.
"""
import os
import sys
import json
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C
import feature_source as FS

N_CAND = 100   # 99 negatives + 1 target (matches data.py B-8)


def _sample_99(dataset, user_id, items, seed):
    seen = set(items)
    rng = np.random.RandomState((common.stable_hash(user_id) ^ seed ^ 0x9E3779B9) & 0x7FFFFFFF)
    negs = []
    while len(negs) < N_CAND - 1:
        c = int(rng.choice(dataset.all_item))
        if c not in seen:
            negs.append(str(c)); seen.add(c)
    return negs


def _build_b8(dataset, user_id, candidate_samples, feats_matrix):
    ratio = C.SIZE_RATIO
    tok = dataset.tokenizer
    tmpl = dataset.all_tasks["direct"]["B-8"]
    cand_str = (" {}, ".format("<extra_id_0> " * ratio).join(candidate_samples)
                + " <extra_id_0>" * ratio)
    source = tmpl["source"].format(user_id, cand_str)
    input_ids = tok.encode(source, padding=True, truncation=True,
                           max_length=dataset.args.max_text_length)
    tokenized = tok.tokenize(source)
    wwids = dataset.calculate_whole_word_ids(tokenized, input_ids)
    cat_ids = [1 if t == 32099 else 0 for t in input_ids]
    target_ids = tok.encode("0", padding=True, truncation=True, max_length=dataset.args.gen_max_length)
    entry = {
        "input_ids": torch.LongTensor(input_ids), "input_length": len(input_ids),
        "whole_word_ids": torch.LongTensor(wwids), "category_ids": torch.LongTensor(cat_ids),
        "target_ids": torch.LongTensor(target_ids), "target_length": len(target_ids),
        "source_text": source, "tokenized_text": tokenized, "target_text": "0", "task": "direct",
        "vis_feats": torch.from_numpy(feats_matrix.astype("float32")),
        "vis_feat_length": feats_matrix.shape[0], "loss_weight": 1.0,
    }
    return dataset.collate_fn([entry])


@torch.no_grad()
def _beam_rank(model, dataset, batch, target_str, device, num_beams=20):
    out = model.generate(
        input_ids=batch["input_ids"].to(device),
        whole_word_ids=batch["whole_word_ids"].to(device),
        category_ids=batch["category_ids"].to(device),
        vis_feats=batch["vis_feats"].to(device),
        task="direct", max_length=50, num_beams=num_beams,
        no_repeat_ngram_size=0, num_return_sequences=num_beams, early_stopping=True)
    gen = model.tokenizer.batch_decode(out, skip_special_tokens=True)
    seen, rank = set(), None
    pos = 0
    for g in gen:
        g = g.strip()
        if g in seen:
            continue
        seen.add(g)
        pos += 1
        if g == target_str:
            rank = pos
            break
    return rank if rank is not None else 999


def _feats_for(dataset, candidate_samples, target_idx, target_asin, target_kind):
    feats = np.stack([common.load_shipped(common.asin_of(dataset, c))
                      for c in candidate_samples], axis=0).astype("float32")
    feats[target_idx] = (FS.clean_pgd(target_asin) if target_kind == "clean"
                         else FS.poisoned(target_asin))
    return feats


def _agg(ranks, ks=(1, 5, 10)):
    r = np.array(ranks, dtype=float)
    out = {"n": len(ranks), "mean_rank_capped": float(np.mean(np.minimum(r, 21)))}
    for k in ks:
        out["HR@%d" % k] = float(np.mean(r <= k))
        out["NDCG@%d" % k] = float(np.mean([(1.0 / np.log2(x + 1)) if x <= k else 0.0 for x in r]))
    return out


def main(n_users=None):
    ctx = common.load_context(need_model=True)
    common.ensure_dirs()
    dataset, model, device = ctx.dataset, ctx.model, ctx.device
    n_users = n_users or min(C.N_TEST_USERS or 200, 200)

    before, after, n_skip = [], [], 0
    for user_id, items in common.iter_test_users(dataset, n_users):
        target = str(items[-1])
        target_asin = common.asin_of(dataset, target)
        if not FS.has_poisoned(target_asin):
            n_skip += 1
            continue
        try:
            FS.clean_pgd(target_asin)
        except FileNotFoundError:
            n_skip += 1
            continue
        negs = _sample_99(dataset, user_id, items, C.SEED)
        cands = negs + [target]
        random.Random((common.stable_hash(user_id) ^ C.SEED) & 0x7FFFFFFF).shuffle(cands)
        tidx = cands.index(target)
        b_clean = _build_b8(dataset, user_id, cands,
                            _feats_for(dataset, cands, tidx, target_asin, "clean"))
        b_pois = _build_b8(dataset, user_id, cands,
                           _feats_for(dataset, cands, tidx, target_asin, "poisoned"))
        before.append(_beam_rank(model, dataset, b_clean, target, device))
        after.append(_beam_rank(model, dataset, b_pois, target, device))
        if len(before) % 25 == 0:
            print("[listwise] %d users | HR@10 %.3f->%.3f"
                  % (len(before), np.mean(np.array(before) <= 10), np.mean(np.array(after) <= 10)))

    res = {"clean": _agg(before), "attacked": _agg(after), "n_skip": n_skip}
    print("[eval_listwise] clean   :", json.dumps(res["clean"]))
    print("[eval_listwise] attacked:", json.dumps(res["attacked"]))
    json.dump(res, open(os.path.join(C.RESULTS_DIR, "listwise.json"), "w"), indent=2)
    print("[eval_listwise] saved", os.path.join(C.RESULTS_DIR, "listwise.json"))


if __name__ == "__main__":
    main()
