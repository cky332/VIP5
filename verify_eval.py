#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIP5 复现验证脚本(独立运行,等价于 notebooks/evaluate_VIP5.ipynb)。

用法示例:
    python3 verify_eval.py \
        --split toys \
        --load snap/toys-vitb32-2-8-5/Epoch05.pth \
        --image_feature_type vitb32 --image_feature_size_ratio 2 --reduction_factor 8 \
        --prompts A-3,A-9

说明:
    - A-* 序列推荐 / B-* 直接推荐:beam search 生成 -> HR@k / NDCG@k
    - C-* 解释生成:greedy 生成 -> BLEU4 / ROUGE1 / ROUGE2 / ROUGEL(百分比)
    - 结尾自动和论文(Toys)目标值并排对比
"""
import argparse
import os
import re
import sys
import random

import numpy as np
import torch
from tqdm import tqdm

# ---------- 路径设置:保证 src / notebooks(evaluate 包)可导入,且 data/、features/ 能被相对路径找到 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)  # data.py 用的是 'data/...' 'features/...' 这种相对路径

from src.tokenization import P5Tokenizer            # noqa: E402
from src.model import VIP5Tuning                    # noqa: E402
from src.data import get_loader                     # noqa: E402
from src.utils import load_state_dict               # noqa: E402
from evaluate.metrics4rec import evaluate_all       # noqa: E402
from evaluate.utils import rouge_score, bleu_score  # noqa: E402

IMAGE_FEATURE_DIM = {"vitb32": 512, "vitb16": 512, "vitl14": 768, "rn50": 1024, "rn101": 512}

# ---------- 论文 Table 2/3/4 的 VIP5 在 Toys 上的目标值(解释生成为百分比) ----------
PAPER_TOYS = {
    "A-3":  {"HR@5": 0.0662, "NDCG@5": 0.0577, "HR@10": 0.0749, "NDCG@10": 0.0604},
    "A-9":  {"HR@5": 0.0641, "NDCG@5": 0.0556, "HR@10": 0.0716, "NDCG@10": 0.0580},
    "B-5":  {"HR@1": 0.0428, "HR@5": 0.1225, "NDCG@5": 0.0833, "HR@10": 0.1906, "NDCG@10": 0.1051},
    "B-8":  {"HR@1": 0.0433, "HR@5": 0.1301, "NDCG@5": 0.0875, "HR@10": 0.2037, "NDCG@10": 0.1110},
    "C-3":  {"BLEU4": 2.3241, "ROUGE1": 15.3006, "ROUGE2": 3.6590, "ROUGEL": 12.0421},
    "C-12": {"BLEU4": 3.9293, "ROUGE1": 28.9225, "ROUGE2": 9.5441, "ROUGEL": 23.3148},
}

PROMPT_GROUP = {"A": "sequential", "B": "direct", "C": "explanation"}


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def build_args(cli):
    """与 evaluate_VIP5.ipynb 中的 args 设置保持一致。"""
    a = DotDict()
    a.distributed = False
    a.multiGPU = True
    a.fp16 = True
    a.split = cli.split
    a.train = cli.split
    a.valid = cli.split
    a.test = cli.split
    a.batch_size = cli.batch_size
    a.optim = "adamw"
    a.warmup_ratio = 0.1
    a.lr = 1e-3
    a.num_workers = cli.num_workers
    a.clip_grad_norm = 5.0
    a.losses = "sequential,direct,explanation"
    a.backbone = "t5-small"
    a.image_feature_type = cli.image_feature_type
    a.image_feature_size_ratio = cli.image_feature_size_ratio
    a.use_adapter = True
    a.reduction_factor = cli.reduction_factor
    a.use_single_adapter = True
    a.use_vis_layer_norm = True
    a.add_adapter_cross_attn = True
    a.use_lm_head_adapter = True
    a.epoch = 20
    a.local_rank = 0
    a.comment = ""
    a.train_topk = -1
    a.valid_topk = -1
    a.dropout = 0.1
    a.tokenizer = "p5"
    a.max_text_length = 1024
    a.gen_max_length = 64
    a.do_lower_case = False
    a.weight_decay = 0.01
    a.adam_eps = 1e-6
    a.gradient_accumulation_steps = 1
    a.whole_word_embed = True
    a.category_embed = True
    a.gpu = cli.gpu
    a.rank = cli.gpu
    a.world_size = 1
    a.seed = cli.seed
    return a


def create_config(args):
    from transformers import T5Config
    from adapters import AdapterConfig

    config = T5Config.from_pretrained(args.backbone)
    for k, v in vars(args).items():
        setattr(config, k, v)

    config.feat_dim = IMAGE_FEATURE_DIM[args.image_feature_type]
    config.n_vis_tokens = args.image_feature_size_ratio
    config.use_vis_layer_norm = args.use_vis_layer_norm
    config.reduction_factor = args.reduction_factor
    config.use_adapter = args.use_adapter
    config.add_adapter_cross_attn = args.add_adapter_cross_attn
    config.use_lm_head_adapter = args.use_lm_head_adapter
    config.use_single_adapter = args.use_single_adapter
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    tasks = re.split("[, ]+", args.losses)
    if args.use_adapter:
        config.adapter_config = AdapterConfig()
        config.adapter_config.tasks = tasks
        config.adapter_config.d_model = config.d_model
        config.adapter_config.use_single_adapter = args.use_single_adapter
        config.adapter_config.reduction_factor = args.reduction_factor
        config.adapter_config.track_z = False
    else:
        config.adapter_config = None
    return config


def evaluate_rec(model, loader, prompt, ks, num_beams, device):
    """A-* / B-* 推荐任务:beam search 生成候选 item id 列表,算 HR@k / NDCG@k。"""
    all_info = []
    for batch in tqdm(loader, desc=f"[{prompt}] generating"):
        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                whole_word_ids=batch["whole_word_ids"].to(device),
                category_ids=batch["category_ids"].to(device),
                vis_feats=batch["vis_feats"].to(device),
                task=batch["task"][0],
                max_length=50,
                num_beams=num_beams,
                no_repeat_ngram_size=0,
                num_return_sequences=num_beams,
                early_stopping=True,
            )
            sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for j, target in enumerate(batch["target_text"]):
            all_info.append({"target": target, "gen": sents[j * num_beams:(j + 1) * num_beams]})

    gt, ui_scores = {}, {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info["target"])]
        pred = {}
        for j, g in enumerate(info["gen"]):
            try:
                pred[int(g)] = -(j + 1)  # beam 排名越靠前分数越高
            except ValueError:
                pass
        ui_scores[i] = pred

    out = {}
    for k in ks:
        _, res = evaluate_all(ui_scores, gt, k)
        out[f"HR@{k}"] = res["hit"]
        out[f"NDCG@{k}"] = res["ndcg"]
    return out


def evaluate_exp(model, loader, prompt):
    """C-* 解释生成:greedy 生成,算 BLEU4 / ROUGE(百分比)。"""
    preds, refs = [], []
    for batch in tqdm(loader, desc=f"[{prompt}] generating"):
        with torch.no_grad():
            results = model.generate_step(batch)
        preds.extend(results)
        refs.extend(batch["target_text"])
    bleu4 = bleu_score(refs, preds, n_gram=4, smooth=False)
    rouge = rouge_score(refs, preds)
    return {
        "BLEU4": bleu4,
        "ROUGE1": rouge["rouge_1/f_score"],
        "ROUGE2": rouge["rouge_2/f_score"],
        "ROUGEL": rouge["rouge_l/f_score"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="toys")
    ap.add_argument("--load", required=True, help="检查点 .pth 路径,如 snap/toys-vitb32-2-8-5/Epoch05.pth")
    ap.add_argument("--prompts", default="A-3,A-9", help="逗号分隔,如 A-3,A-9,B-5,B-8,C-3,C-12")
    ap.add_argument("--image_feature_type", default="vitb32")
    ap.add_argument("--image_feature_size_ratio", type=int, default=2)
    ap.add_argument("--reduction_factor", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2022)
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)
    np.random.seed(cli.seed)

    use_cuda = torch.cuda.is_available()
    device = f"cuda:{cli.gpu}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(cli.gpu)
    print(f"[device] {device}  ({torch.cuda.get_device_name(cli.gpu) if use_cuda else 'CPU'})")

    args = build_args(cli)

    print("[build] config / tokenizer / model ...")
    config = create_config(args)
    tokenizer = P5Tokenizer.from_pretrained(
        args.backbone, max_length=args.max_text_length, do_lower_case=args.do_lower_case
    )
    model = VIP5Tuning.from_pretrained(args.backbone, config=config)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer
    model = model.to(device)

    print(f"[load] {cli.load}")
    state_dict = load_state_dict(cli.load, "cpu")
    result = model.load_state_dict(state_dict, strict=False)
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print(f"[load] missing_keys={n_missing}  unexpected_keys={n_unexpected}")
    if n_missing:
        print("       (前若干 missing:", result.missing_keys[:6], "...)")
    model.eval()

    sample_numbers = {"sequential": (1, 1), "direct": (1, 1), "explanation": 1}
    prompts = [p.strip() for p in cli.prompts.split(",") if p.strip()]

    collected = {}
    for prompt in prompts:
        group = PROMPT_GROUP.get(prompt.split("-")[0].upper())
        if group is None:
            print(f"[skip] 未知 prompt: {prompt}")
            continue
        print(f"\n================= {prompt}  ({group}) =================")
        loader = get_loader(
            args,
            {group: [prompt]},
            sample_numbers,
            split=args.test,
            mode="test",
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=False,
        )
        print(f"[loader] {len(loader)} batches")

        if group == "explanation":
            collected[prompt] = evaluate_exp(model, loader, prompt)
        elif group == "sequential":
            collected[prompt] = evaluate_rec(model, loader, prompt, ks=[5, 10],
                                             num_beams=cli.num_beams, device=device)
        else:  # direct
            collected[prompt] = evaluate_rec(model, loader, prompt, ks=[1, 5, 10],
                                             num_beams=cli.num_beams, device=device)

    # ---------- 与论文 Toys 目标值并排对比 ----------
    print("\n\n########## 结果 vs 论文(Toys)##########")
    is_toys = (args.split == "toys")
    for prompt, metrics in collected.items():
        print(f"\n--- {prompt} ---")
        ref = PAPER_TOYS.get(prompt, {}) if is_toys else {}
        header = f"{'metric':<10}{'got':>12}{'paper(toys)':>16}{'diff':>12}"
        print(header)
        print("-" * len(header))
        for m, v in metrics.items():
            pv = ref.get(m)
            if pv is not None:
                diff = v - pv
                print(f"{m:<10}{v:>12.4f}{pv:>16.4f}{diff:>+12.4f}")
            else:
                print(f"{m:<10}{v:>12.4f}{'-':>16}{'-':>12}")
    if not is_toys:
        print("\n(注:--split 非 toys,论文目标值未内置对比,请对照论文表格)")
    print("\n提示:5 轮检查点欠训练,数字低于论文(10/20 轮)属正常;"
          "A-3/A-9 确定可复现,B-5/B-8 因现场随机负采样会小幅波动。")


if __name__ == "__main__":
    main()
