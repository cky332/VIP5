#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIP5 评估脚本（命令行版，等价于 notebooks/evaluate_VIP5.ipynb）。

在【项目根目录】运行，例如：
    python evaluate_vip5.py \
        --split toys \
        --load snap/toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth \
        --image_feature_type vitb32 --image_feature_size_ratio 2 --reduction_factor 8

常用：
    # 只先跑 explanation（最快，验证流程通不通）
    python evaluate_vip5.py --tasks explanation
    # 显存不够就减小 batch
    python evaluate_vip5.py --batch_size 4
    # 每个任务只跑第一个模板（省时间）
    python evaluate_vip5.py --first_template_only
    # 指定用哪张卡
    CUDA_VISIBLE_DEVICES=1 python evaluate_vip5.py

注意：sequential / direct 用 beam search（num_beams=20），很慢且吃显存，OOM 就调小 --batch_size。
"""
import os
# 国内网络：让 transformers 走 HF 镜像（必须在 import transformers 之前设置）。海外可删这行。
os.environ.setdefault("HUGGINGFACE_CO_RESOLVE_ENDPOINT", "https://hf-mirror.com")

import sys
import re
import random
import argparse

# 让脚本找得到 src/（模型、数据）和 notebooks/（评估指标）里的模块
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "notebooks"))  # evaluate.* 指标
sys.path.insert(0, os.path.join(_ROOT, "src"))        # 优先用 src/ 的模型/数据

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from transformers import T5Config
from model import VIP5Tuning
from tokenization import P5Tokenizer
from utils import load_state_dict
from data import get_loader
from adapters import AdapterConfig
from evaluate.utils import rouge_score, bleu_score
from evaluate.metrics4rec import evaluate_all


IMAGE_FEATURE_DIM = {"vitb32": 512, "vitb16": 512, "vitl14": 768, "rn50": 1024, "rn101": 512}

# 每个任务默认评测的 prompt 模板（与 notebook 一致）
DEFAULT_TEMPLATES = {
    "explanation": ["C-12", "C-3"],
    "sequential": ["A-9", "A-3"],
    "direct": ["B-8", "B-5"],
}
SAMPLE_NUMBERS = {"sequential": (1, 1), "direct": (1, 1), "explanation": 1}


class DotDict(dict):
    def __init__(self, **kw):
        super().__init__()
        self.update(kw)
        self.__dict__ = self


def build_args(cli):
    """构造与训练一致的配置（对应 notebook 的 cell 1）。"""
    args = DotDict()
    args.distributed = False
    args.multiGPU = True
    args.fp16 = True
    args.split = cli.split
    args.train = args.split
    args.valid = args.split
    args.test = args.split
    args.batch_size = cli.batch_size
    args.optim = "adamw"
    args.warmup_ratio = 0.1
    args.lr = 1e-3
    args.num_workers = cli.num_workers
    args.clip_grad_norm = 5.0
    args.losses = "sequential,direct,explanation"
    args.backbone = cli.backbone
    args.image_feature_type = cli.image_feature_type
    args.image_feature_size_ratio = cli.image_feature_size_ratio
    args.use_adapter = True
    args.reduction_factor = cli.reduction_factor
    args.use_single_adapter = True
    args.use_vis_layer_norm = True
    args.add_adapter_cross_attn = True
    args.use_lm_head_adapter = False           # 对齐 train_VIP5.sh（训练时未开）
    args.epoch = 20
    args.local_rank = 0
    args.comment = ""
    args.train_topk = -1
    args.valid_topk = -1
    args.dropout = 0.1
    args.tokenizer = "p5"
    args.max_text_length = cli.max_text_length
    args.gen_max_length = 64
    args.do_lower_case = False
    args.weight_decay = 0.01
    args.adam_eps = 1e-6
    args.gradient_accumulation_steps = 1
    args.seed = 2022
    args.whole_word_embed = True
    args.category_embed = True
    args.gpu = cli.gpu
    args.rank = cli.gpu
    args.world_size = 1
    return args


def create_config(args):
    """对应 notebook 的 create_config()。"""
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
        ac = AdapterConfig()
        ac.tasks = tasks
        ac.d_model = config.d_model
        ac.use_single_adapter = args.use_single_adapter
        ac.reduction_factor = args.reduction_factor
        ac.track_z = False
        config.adapter_config = ac
    else:
        config.adapter_config = None
    return config


def build_model(args):
    config = create_config(args)
    tokenizer = P5Tokenizer.from_pretrained(
        args.backbone, max_length=args.max_text_length, do_lower_case=args.do_lower_case
    )
    model = VIP5Tuning.from_pretrained(args.backbone, config=config)
    model = model.cuda()
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer
    return model


def load_ckpt(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}（用 --load 指定正确路径）")
    state_dict = load_state_dict(path, "cpu")
    res = model.load_state_dict(state_dict, strict=False)
    print(f"[load] {path}")
    print(f"[load] missing_keys={len(res.missing_keys)}  unexpected_keys={len(res.unexpected_keys)}")
    if res.missing_keys:
        print("       e.g. missing:", res.missing_keys[:5])
    if res.unexpected_keys:
        print("       e.g. unexpected:", res.unexpected_keys[:5])
    if len(res.unexpected_keys) > 0:
        print("       [warn] unexpected_keys present: checkpoint may not match the current config")
    model.eval()


def eval_explanation(args, model, template):
    loader = get_loader(args, {"explanation": [template]}, SAMPLE_NUMBERS,
                        split=args.test, mode="test",
                        batch_size=args.batch_size, workers=args.num_workers,
                        distributed=args.distributed)
    preds, refs = [], []
    for batch in tqdm(loader, desc=f"explanation {template}"):
        with torch.no_grad():
            preds.extend(model.generate_step(batch))
            refs.extend(batch["target_text"])
    b1 = bleu_score(refs, preds, n_gram=1, smooth=False)
    b4 = bleu_score(refs, preds, n_gram=4, smooth=False)
    rouge = rouge_score(refs, preds)
    print(f"\n========== explanation [{template}] ==========")
    print(f"BLEU-1 {b1:7.4f}   BLEU-4 {b4:7.4f}")
    for k, v in rouge.items():
        print(f"{k} {v:7.4f}")


def eval_ranking(args, model, task_type, template, ks):
    loader = get_loader(args, {task_type: [template]}, SAMPLE_NUMBERS,
                        split=args.test, mode="test",
                        batch_size=args.batch_size, workers=args.num_workers,
                        distributed=args.distributed)
    all_info = []
    for batch in tqdm(loader, desc=f"{task_type} {template}"):
        with torch.no_grad():
            results = model.generate_step(batch)
            beam_outputs = model.generate(
                input_ids=batch["input_ids"].cuda(),
                whole_word_ids=batch["whole_word_ids"].cuda(),
                category_ids=batch["category_ids"].cuda(),
                vis_feats=batch["vis_feats"].cuda(),
                task=batch["task"][0],
                max_length=50, num_beams=20, no_repeat_ngram_size=0,
                num_return_sequences=20, early_stopping=True,
            )
            gen = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            for j, item in enumerate(zip(results, batch["target_text"], batch["source_text"])):
                all_info.append({"target_item": item[1], "gen_item_list": gen[j * 20:(j + 1) * 20]})
    gt, ui_scores = {}, {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info["target_item"])]
        pred = {}
        for j in range(len(info["gen_item_list"])):
            try:
                pred[int(info["gen_item_list"][j])] = -(j + 1)
            except Exception:
                pass
        if not pred:
            # 模型没生成任何合法 item id：记为一次 miss（放一个必然错误的占位项），
            # 否则空预测会让 evaluate_once 里 topk=0 触发 assert 崩溃。
            pred[-1] = -(10 ** 9)
        ui_scores[i] = pred
    print(f"\n========== {task_type} [{template}] ==========")
    for k in ks:
        evaluate_all(ui_scores, gt, k)


def main():
    p = argparse.ArgumentParser(description="VIP5 评估（命令行版）")
    p.add_argument("--split", default="toys", help="toys/beauty/sports/clothing")
    p.add_argument("--load", default="snap/toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth")
    p.add_argument("--backbone", default="t5-small")
    p.add_argument("--image_feature_type", default="vitb32")
    p.add_argument("--image_feature_size_ratio", type=int, default=2)
    p.add_argument("--reduction_factor", type=int, default=8)
    p.add_argument("--max_text_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--tasks", default="explanation,sequential,direct",
                   help="逗号分隔，从 explanation/sequential/direct 里选")
    p.add_argument("--first_template_only", action="store_true",
                   help="每个任务只跑第一个模板（更快）")
    cli = p.parse_args()

    os.chdir(_ROOT)  # 保证 data/ 和 features/ 的相对路径能找到

    args = build_args(cli)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(f"[init] split={args.split} backbone={args.backbone} feat={args.image_feature_type} "
          f"ratio={args.image_feature_size_ratio} reduction={args.reduction_factor} batch={args.batch_size}")
    model = build_model(args)
    load_ckpt(model, cli.load)

    tasks = [t.strip() for t in cli.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in DEFAULT_TEMPLATES:
            print(f"[skip] 未知任务: {task}")
            continue
        templates = DEFAULT_TEMPLATES[task]
        if cli.first_template_only:
            templates = templates[:1]
        if task in ("sequential", "direct"):
            print(f"\n[note] {task}: beam search (num_beams=20), slow & memory-heavy; lower --batch_size if OOM")
        for tmpl in templates:
            if task == "explanation":
                eval_explanation(args, model, tmpl)
            elif task == "sequential":
                eval_ranking(args, model, "sequential", tmpl, ks=[5, 10])
            elif task == "direct":
                eval_ranking(args, model, "direct", tmpl, ks=[1, 5, 10])

    print("\n[done] 评估完成。")


if __name__ == "__main__":
    main()
