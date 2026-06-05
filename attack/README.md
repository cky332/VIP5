# VIP5 对抗鲁棒性攻击:热门质心模仿(对标 MLLM-MSR)

对 VIP5 做"扰动候选封面像素 → 让其 CLIP 嵌入 ≈ 热门商品质心 → 抬高被推荐概率"的攻击,
用于**学术安全鲁棒性研究**。评测对标 MLLM-MSR 的 pointwise P(Yes)。

> ⚠️ 威胁模型:攻击者控制商品封面图 → 平台用公开 CLIP ViT-B/32 **重提取**特征存为该商品
> `.npy` → VIP5 消费它。因此攻击只需白盒攻击**公开 CLIP**(产生这些特征的同一编码器),
> **不需要对 VIP5 求梯度**。VIP5 推理时不碰像素。

## 前置条件(在你的服务器上,VIP5 根目录下)
- `data/toys/*`、`features/vitb32_features/toys/*.npy`、训练好的 `snap/toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth`
- `t5-small` 已缓存(`export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com`)
- 像素攻击还需原图:`vip5/vip5_photos.zip`(gdown 下下来的那个)+ 装 CLIP:
  `pip install ftfy regex Pillow && pip install git+https://github.com/openai/CLIP.git`
  (github 不通时用镜像,见 `DEPLOY.md` FAQ#10)
- 路径/超参在 `attack/config.py` 里改(默认对应你这次 toys/vitb32/ratio2/reduction8 的训练)。

## 怎么跑

```bash
cd ~/data/cky/VIP5

# A) 先跑特征层面消融(免图片/CLIP,最快,拿"效果上界"+护栏结论)
python attack/run_all.py ablation

# B) 完整像素攻击流水线
python attack/run_all.py            # clip 核验→建质心→PGD→pointwise→listwise
# 或分步:
python attack/run_all.py clip centroid pgd      # 只生成"中毒特征"
python attack/run_all.py pointwise              # 主评测(P(yes) 排序,攻击前后)
python attack/run_all.py listwise               # 辅评测(原生 B-8 生成式排序)
```

结果与产物在 `attack/out/`:
- `results/ablation.json`(α 扫描:P(yes)/名次随"特征→质心"程度)
- `results/pointwise.json`(主:正样本均名次 / HR@10 / NDCG@10 / P(yes),clean vs attacked)
- `results/listwise.json`(辅:B-8 名次,clean vs attacked)
- `centroid.npy`、`poisoned_features/`、`clean_features/`、`perturbed_images/`、`extraction_check.json`

## 设计要点 / 注意
- **主评测=pointwise B-1**(`"Will user_X likely to interact with item_Y [图]?"`→P(yes)),
  逐候选独立打分,只换正样本封面 → 干净对标 MLLM-MSR。
- **混杂控制**:pixel 攻击里"干净"基线用的是与攻击**同一 CLIP 管线重提取**的 clean 特征
  (`clean_features/`),与"中毒"特征只差对抗扰动。负样本两次完全相同。
- **`CLIP_NORM`**(出厂特征是否 L2 归一化)由 `clip_extract.verify_against_shipped` **经验确定**
  并写入 `extraction_check.json`;若它报告 cos<0.9,说明 CLIP 版本/预处理可能不一致,
  绝对值仅供参考,但"攻击 vs 干净"的相对 delta 仍有效。
- **预判**:VIP5 视觉通路只是个小 MLP、且排序偏向文本 item_id,攻击效果可能比 MLLM-MSR 弱。
  先看 ablation 的 `alpha=1.00`:若它都几乎不抬 P(yes),则模型对图像通道鲁棒,像素攻击也超不过——
  这是重要的(负)结论。
- 各脚本也可单独运行,如 `python attack/clip_extract.py`、`python attack/pgd_attack.py`。
