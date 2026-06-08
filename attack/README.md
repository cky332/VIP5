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

## 进阶:X-Transfer 黑盒·超可迁移攻击(对标 arXiv 2505.05528)

上面的 PGD 攻击是**白盒·单 CLIP·单品**:前提是能精确复刻平台那版 CLIP(ViT-B/32)。
X-Transfer 变体去掉这个前提——在一个**替身 CLIP 集成**上构造扰动(每步用 UCB 老虎机挑 k 个替身,
即 surrogate scaling),使「只改图」的扰动能**迁移到未知/黑盒的 CLIP**。目标仍是「热门质心模仿」
(把替身嵌入推向各自的热门质心),因此能**抬高候选排名**;受害者 ViT-B/32 被**留出**,仅在最后用于
重提取被污染特征以证明迁移。默认 `XT_DPRIME_MODE="single"`(单品定向,最贴合「抬高我的某个候选」)。

> 威胁模型:攻击者改封面 → 平台用某个(未知的)公开 CLIP 重提特征 → VIP5 消费。攻击只需白盒一批
> **替身** CLIP,**既不需要 VIP5 梯度,也不需要平台那版 CLIP**。

前置:在 `clip` 阶段解析出 `CLIP_NORM` 之后再跑。替身池默认用 `open_clip`(`pip install open_clip_torch`,
首次会从 HF 下权重,可 `export HF_ENDPOINT=https://hf-mirror.com`);若不便装,设
`attack/config.py` 里 `XT_USE_OPEN_CLIP=False` 退回「仅 OpenAI clip」池(免新依赖)。

```bash
python attack/run_all.py clip                      # 解析 CLIP_NORM(只需一次)
python attack/run_all.py xt-centroid xt-attack xt-eval
cat attack/out/xtransfer/results/xtransfer_pointwise.json
```

- `xt-centroid`:为**每个替身**在其自身空间建热门质心 → `attack/out/xtransfer/centroids/`;
  同时建一个**受害者** ViT-B/32 质心(仅作迁移探针的健全性指标)。
- `xt-attack`:逐目标构造黑盒 δ,用**受害者**重提取 clean/poisoned 特征(512 维,不做 L2 归一化,
  与出厂特征一致)→ `attack/out/xtransfer/{poisoned,clean}_features/<split>/`;
  产物 `xt_attack_summary.json` 含 **transfer_probe_ok**(被污染特征是否比 clean 更贴近受害者质心——
  这是 VIP5 打分前的 GO/NO-GO 闸门)。
- `xt-eval`:复用 `run_pointwise/scorer`,输出 clean vs `xtransfer` 的名次/HR@10/NDCG@10/P(yes)。

对照实验(同一目标集/用户/负样本):`XT_INCLUDE_VICTIM=True` 得白盒上界;已有 `pgd`(单 CLIP 白盒)
与 `ablation`(α=1 特征上界)作参照。关键超参在 `attack/config.py` 的 `XT_*` 段
(`XT_EPS/XT_STEPS/XT_K_SELECT/XT_SEARCH_SPACE_*` 等)。**只适用于 direct 推荐(B 类模板)**,
与上文同一适用边界。
