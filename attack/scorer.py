"""Pointwise P("yes") scorer for the B-1 yes/no template -- the faithful
MLLM-MSR analog. Builds one scoring example per (user, candidate) exactly the
way src/data.py encodes B-1, then reads the decoder's first-step logits for the
"yes" vs "no" tokens.
"""
import math
import numpy as np
import torch

import config as C

EXTRA_ID_0 = 32099   # '<extra_id_0>' placeholder token id (see src/data.py:632)


def resolve_yes_no_ids(tokenizer):
    """First decoder-step token id for targets 'yes' / 'no' (SentencePiece -> the
    '▁yes' / '▁no' piece, i.e. encode(...)[0])."""
    yes_id = tokenizer.encode("yes")[0]
    no_id = tokenizer.encode("no")[0]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    assert yes_id != no_id, (yes_id, no_id)
    assert yes_id not in (eos, pad) and no_id not in (eos, pad), (yes_id, no_id, eos, pad)
    return yes_id, no_id


def _b1_template(dataset):
    t = dataset.all_tasks["direct"]["B-1"]
    return t["source"], t["target"]


def _encode_example(dataset, source_text, target_text, feat_vec):
    """Replicates src/data.py:628-655 for a single example. feat_vec: np (dim,)."""
    tok = dataset.tokenizer
    input_ids = tok.encode(source_text, padding=True, truncation=True,
                           max_length=dataset.args.max_text_length)
    tokenized_text = tok.tokenize(source_text)
    whole_word_ids = dataset.calculate_whole_word_ids(tokenized_text, input_ids)
    category_ids = [1 if tid == EXTRA_ID_0 else 0 for tid in input_ids]
    assert len(whole_word_ids) == len(input_ids)
    target_ids = tok.encode(target_text, padding=True, truncation=True,
                            max_length=dataset.args.gen_max_length)
    feats = torch.from_numpy(feat_vec.reshape(1, -1).astype("float32"))
    return {
        "input_ids": torch.LongTensor(input_ids), "input_length": len(input_ids),
        "whole_word_ids": torch.LongTensor(whole_word_ids),
        "category_ids": torch.LongTensor(category_ids),
        "target_ids": torch.LongTensor(target_ids), "target_length": len(target_ids),
        "source_text": source_text, "tokenized_text": tokenized_text,
        "target_text": target_text, "task": "direct",
        "vis_feats": feats, "vis_feat_length": 1, "loss_weight": 1.0,
    }


def build_b1_batch(dataset, user_id, cand_ids, feat_list):
    """cand_ids: list[str] item ids; feat_list: list[np(dim,)] aligned to cand_ids."""
    src_tmpl, tgt_tmpl = _b1_template(dataset)
    placeholder = "<extra_id_0> " * (C.SIZE_RATIO - 1) + "<extra_id_0>"
    entries = []
    for cand, feat in zip(cand_ids, feat_list):
        source = src_tmpl.format(user_id, cand, placeholder)
        target = tgt_tmpl.format("yes")
        entries.append(_encode_example(dataset, source, target, feat))
    return dataset.collate_fn(entries)


@torch.no_grad()
def p_yes(model, batch, yes_id, no_id, device):
    """Return P('yes') per example via 2-way softmax of first decoder-step logits."""
    input_ids = batch["input_ids"].to(device)
    whole_word_ids = batch["whole_word_ids"].to(device)
    category_ids = batch["category_ids"].to(device)
    vis_feats = batch["vis_feats"].to(device)
    pad = model.tokenizer.pad_token_id
    attn = (input_ids != pad).long()
    B = input_ids.size(0)
    dec = torch.full((B, 1), pad, dtype=torch.long, device=device)
    out = model(input_ids=input_ids, whole_word_ids=whole_word_ids,
                category_ids=category_ids, vis_feats=vis_feats,
                attention_mask=attn, decoder_input_ids=dec,
                return_dict=True, task="direct")
    step0 = out.logits[:, 0, :]                       # (B, vocab)
    two = torch.stack([step0[:, yes_id], step0[:, no_id]], dim=-1)   # (B, 2)
    return torch.softmax(two, dim=-1)[:, 0].detach().cpu().numpy()   # P(yes)


def score_user(model, dataset, user_id, cand_ids, feat_list, yes_id, no_id, device,
               batch_size=C.BATCH_SIZE):
    """P(yes) for every candidate of one user (candidates are independent)."""
    scores = []
    for s in range(0, len(cand_ids), batch_size):
        cb = cand_ids[s:s + batch_size]
        fb = feat_list[s:s + batch_size]
        batch = build_b1_batch(dataset, user_id, cb, fb)
        scores.extend(p_yes(model, batch, yes_id, no_id, device).tolist())
    return scores


# ---------------------------------------------------------------------------
# ranking metrics (positive is candidate index 0)
# ---------------------------------------------------------------------------
def rank_of(pos_score, neg_scores):
    """1-based rank of the positive among [pos]+negs (higher score = better).
    Ties: positive placed AFTER equal-scoring negatives (conservative)."""
    better = sum(1 for s in neg_scores if s > pos_score)
    ties = sum(1 for s in neg_scores if s == pos_score)
    return better + ties + 1


def user_metrics(pos_score, neg_scores, ks=(5, 10)):
    r = rank_of(pos_score, neg_scores)
    m = {"rank": r, "p_yes": float(pos_score)}
    for k in ks:
        m["hit@%d" % k] = 1.0 if r <= k else 0.0
        m["ndcg@%d" % k] = (1.0 / math.log2(r + 1)) if r <= k else 0.0
    return m
