# VIP5 部署与运行指南（Linux + Anaconda）

本指南给出在 **Linux + Anaconda** 上从零跑通 VIP5 的完整步骤，并标注了几个容易踩的版本坑。

> 关键结论（已在干净环境中实测验证）：
> - 代码依赖**旧版** `transformers`，必须锁定 **`transformers==4.17.0`**。
>   更高版本把 `T5DenseReluDense` / `T5DenseGatedGeluDense` 改了名（4.18 改成
>   `*ActDense`），`BeamSearchScorer` 也在 4.40 被删除，直接装新版会 `ImportError`。
> - 推荐 **Python 3.9 + PyTorch 1.12.1 + CUDA 11.3**。
> - `transformers 4.17.0` 启动时**硬依赖 `sacremoses`**，必须一起装。
> - **训练至少需要 1 张 NVIDIA GPU**（用了 NCCL 后端 + `.cuda()`，纯 CPU 跑不了训练）。
>   纯 CPU 只能用来跑评估 notebook 的部分逻辑。

---

## 0. 硬件 / 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（已在容器里验证依赖兼容性） |
| GPU | 训练必需，≥1 张 NVIDIA GPU（驱动需支持 CUDA 11.3，即 driver ≥ 465） |
| 显存 | backbone 是 `t5-small`，单卡 ≈ 12–16GB 可用 `batch_size=36`；显存小就调小 batch |
| 磁盘 | 数据 + 图像特征较大，建议预留 ≥ 30GB |

查看 GPU / 驱动：`nvidia-smi`

---

## 1. 克隆仓库

```bash
git clone <你的仓库地址> VIP5
cd VIP5
```

---

## 2. 创建 conda 环境

### 方式 A：用 environment.yml（推荐，一步到位）

```bash
conda env create -f environment.yml
conda activate vip5
```

### 方式 B：手动创建（等价，便于排查）

```bash
conda create -n vip5 python=3.9 -y
conda activate vip5

# PyTorch 1.12.1 + CUDA 11.3（官方组合）
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# 其余依赖（版本是硬锁，别随意升级）
pip install -r requirements.txt
```

### 验证环境

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"   # 应为 4.17.0
# 验证关键符号都在（不应报错）
python -c "from transformers.models.t5.modeling_t5 import T5DenseReluDense, T5DenseGatedGeluDense; from transformers import BeamSearchScorer; print('imports OK')"
```

> **没有 GPU？** 把 environment.yml 里的 `cudatoolkit=11.3` 换成 `cpuonly`（或
> `conda install pytorch==1.12.1 cpuonly -c pytorch`）。注意此时只能跑评估 notebook，
> **不能训练**。

---

## 3. 下载数据与图像特征

数据和预提取的图像特征在作者提供的 Google Drive：
<https://drive.google.com/drive/u/1/folders/1AjM8Gx4A3xo8seYFWwNUBHpM9uRbfydR>

用 `gdown` 下载（已在 requirements 里）：

```bash
# 下载整个文件夹（文件多时 gdown 可能限流/限 50 个文件，建议分子目录下载）
gdown --folder "https://drive.google.com/drive/folders/1AjM8Gx4A3xo8seYFWwNUBHpM9uRbfydR" -O downloads

# 把内容解压/放到 data 和 features 目录
mkdir -p data features
# 按下载到的实际文件，把数据放进 data/<split>/，特征放进 features/<类型>_features/<split>/
```

> 若 `gdown --folder` 因为文件过多失败，就到上面的网页里**进入每个子文件夹**复制其
> 文件夹 ID，逐个 `gdown --folder <子文件夹链接>`；或在浏览器手动下载后上传到服务器。

### 需要的目录结构（以 `toys` 为例）

```
VIP5/
├── data/
│   └── toys/                          # 还有 beauty/ sports/ clothing/
│       ├── exp_splits.pkl             # 解释生成任务 train/val/test
│       ├── sequential_data.txt        # 序列推荐数据
│       ├── negative_samples.txt       # 评估用负样本
│       ├── datamaps.json              # user2id / item2id / id2item
│       ├── user_id2name.pkl
│       ├── meta.json.gz               # 物品元信息
│       ├── item2img_dict.pkl          # 物品→图像 映射
│       └── rating_splits_augmented.pkl# 评估 notebook 用
└── features/
    └── vitb32_features/               # 还有 vitb16_/ vitl14_/ rn50_/ rn101_
        └── toys/
            ├── <asin>.npy             # 每个物品一个图像特征文件
            └── ...
```

图像特征类型与维度对应（训练时 `image_feature_type` 参数）：

| 类型 | 维度 |
|------|------|
| vitb32 | 512 |
| vitb16 | 512 |
| vitl14 | 768 |
| rn50 | 1024 |
| rn101 | 512 |

---

## 4. 创建输出目录

```bash
mkdir -p snap log
```

- `snap/`：保存 checkpoint（`BEST_EVAL_LOSS.pth`、`Epoch20.pth` 等）
- `log/`：训练日志

---

## 5. 训练

### 5.1 多 GPU（README 原始示例，4 卡）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_VIP5.sh 4 toys 13579 vitb32 2 8 20
```

`train_VIP5.sh` 的位置参数含义：

| 位置 | 含义 | 示例 |
|------|------|------|
| `$1` | GPU 数（nproc_per_node） | `4` |
| `$2` | 数据集 split | `toys` / `beauty` / `sports` / `clothing` |
| `$3` | master_port（多任务并行时改成不同端口避免冲突） | `13579` |
| `$4` | 图像特征类型 | `vitb32` |
| `$5` | image_feature_size_ratio（视觉 token 数 n_vis_tokens） | `2` |
| `$6` | reduction_factor（adapter 瓶颈维度比） | `8` |
| `$7` | epoch | `20` |

> 脚本里 `--batch_size 36` 是写死的，是**每卡** batch size。

### 5.2 单 GPU（本仓库新增的便捷脚本）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5_single_gpu.sh toys vitb32 2 8 20
# 显存不够就降 batch：
BATCH_SIZE=12 CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5_single_gpu.sh toys
```

（等价地，原脚本传 `1` 也行：`CUDA_VISIBLE_DEVICES=0 bash scripts/train_VIP5.sh 1 toys 13579 vitb32 2 8 20`）

### 5.3 训练注意

- 训练**必须**经由 `torch.distributed.launch` 启动（代码里 `if args.distributed:`
  才会真正跑），且 backend 是 `nccl`，所以**必须有 GPU**。
- 首次运行会从 HuggingFace 下载 `t5-small` 权重，需要联网到 `huggingface.co`。
  若服务器内网无法访问，先在能联网的机器上 `T5Tokenizer/T5ForConditionalGeneration
  .from_pretrained('t5-small')` 缓存好 `~/.cache/huggingface`，再拷贝过去；或设置
  `export TRANSFORMERS_OFFLINE=1` 并提供本地缓存。

---

## 6. 评估（生成推荐 / 解释，算指标）

评估走 notebook：`notebooks/evaluate_VIP5.ipynb`

```bash
conda activate vip5
cd notebooks
jupyter notebook   # 或 jupyter lab，打开 evaluate_VIP5.ipynb
```

在 notebook 里需要设置：
- `args.split`（如 `"toys"`）、`args.backbone='t5-small'`、`args.image_feature_type` 等，
  要与训练时一致；
- `args.load = "../snap/vip5_toys.pth"`：指向训练得到的 checkpoint
  （或作者在 Google Drive 提供的预训练权重，放到 `snap/` 下）。

> notebook 里默认 `model = model.cuda()`，**默认需要 GPU**。纯 CPU 评估需把 `.cuda()`
> 改成 `.to('cpu')` 并把 `args.fp16` 关掉，速度会很慢。

---

## 7. 常见坑（FAQ）

1. **`ImportError: cannot import name 'T5DenseReluDense'`**
   → transformers 版本太新。必须 `pip install transformers==4.17.0`。

2. **`ImportError: cannot import name 'BeamSearchScorer'`**
   → 同上，transformers ≥ 4.40 删了它。用 4.17.0。

3. **`No package metadata was found for sacremoses`**
   → `pip install sacremoses`（4.17.0 启动硬依赖）。

4. **`cannot import name 'cached_download' from 'huggingface_hub'`**
   → huggingface_hub 太新。锁定 `huggingface_hub==0.8.1`（requirements 已锁）。

5. **`numpy` 相关的 ABI / dtype 报错**
   → numpy 2.x 与 torch 1.12 不兼容。锁定 `numpy==1.23.5`（requirements 已锁）。

6. **多卡训练 `--local_rank` 解析不到 / DDP 卡住**
   → 本代码按 torch 1.x 的 `torch.distributed.launch`（传 `--local_rank` 下划线）写的。
   请用 **torch 1.12**；torch 2.x 的 launcher 传的是 `--local-rank`（连字符）会对不上。

7. **`CUDA out of memory`**
   → 调小 batch（单卡脚本用 `BATCH_SIZE=...`），或减小 `image_feature_size_ratio`、
   `max_text_length`，或用更少的 `--losses`。

8. **下载 `t5-small` 失败 / 内网无外网**
   → 见 5.3，预先缓存 + `TRANSFORMERS_OFFLINE=1`。

9. **`gdown --folder` 报错或只下了一部分**
   → Google Drive 文件夹文件多会被限流；改为逐子文件夹下载，或浏览器手动下载再上传。
   注意 `gdown` 走 `drive.google.com`，**国内网络通常需要代理/VPN**才能访问。

10. **建环境时卡在 `git clone https://github.com/openai/CLIP.git ... Failed to connect to github.com`**
    → 这是**网络问题，不是版本问题**：你的机器连不上 `github.com`（国内常见）。
    而且 **CLIP 根本用不到**（`src/`、`notebooks/` 里没有任何 `import clip`，它只用于
    从原图重新提取特征，而你用 Google Drive 的预提取 `.npy` 特征即可）。
    本仓库已把 CLIP 从默认安装里移除，正常 `conda env create -f environment.yml` 不会再触发它。
    若你**确实**要装 CLIP（自行提特征），单独走 github 镜像：
    ```bash
    pip install ftfy regex Pillow
    git config --global url."https://gitclone.com/github.com/".insteadOf "https://github.com/"
    pip install git+https://github.com/openai/CLIP.git
    # 镜像站时有失效，可换成当前可用的 github 加速镜像
    ```

---

## 8. 国内网络：访问 github / huggingface / Google Drive

本项目部署会碰到三个境外服务：**github**（可选，装 CLIP 用）、**huggingface.co**
（必需，下 `t5-small`）、**Google Drive**（必需，下数据/特征）。国内服务器通常都连不上。

### 8.1 优先用「域名镜像」，多数情况不用代理

| 服务 | 用途 | 镜像/办法（无需代理） |
|------|------|----------------------|
| PyPI | 装 Python 包 | 清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`（你已在用） |
| huggingface | 下 `t5-small` 权重 | `export HF_ENDPOINT=https://hf-mirror.com` 后再训练（强烈推荐） |
| github | 装 CLIP（可选） | `git config --global url."https://gitclone.com/github.com/".insteadOf "https://github.com/"` |
| Google Drive | 下数据/特征 | 镜像难，基本只能靠代理或在能联网的机器下好再 `scp` 上传 |

### 8.2 有代理时（最通用，一次解决三者）

git 走 HTTP/SOCKS 代理（把地址换成你的）：
```bash
git config --global http.proxy  http://127.0.0.1:7890     # SOCKS5 用 socks5://127.0.0.1:7891
git config --global https.proxy http://127.0.0.1:7890
# 只给 github 走代理（不影响国内站）：
git config --global http.https://github.com.proxy http://127.0.0.1:7890
# 用完取消：
git config --global --unset http.proxy && git config --global --unset https.proxy
```
其它命令（pip / curl / gdown / huggingface）走代理，用环境变量：
```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7891
```

### 8.3 代理只在本地电脑上？用 SSH 反向隧道把它带到服务器

若代理（clash/v2ray 等，假设本地端口 7890）跑在你**自己的电脑**上，服务器用不了。
从**本地电脑**这样连服务器，把本地代理「借」给服务器：
```bash
# 在本地电脑执行：服务器的 127.0.0.1:7890 会转发回本地的 7890
ssh -R 7890:127.0.0.1:7890 mlsnrs@<服务器IP>
```
然后在**服务器**上 `export https_proxy=http://127.0.0.1:7890`，git/pip/gdown 即可联网。

### 8.4 测试是否通

```bash
curl -I --connect-timeout 10 https://github.com        # 返回 HTTP 200/301 即通
git ls-remote https://github.com/openai/CLIP.git | head # 能列出引用即通
```

> 对本项目而言：**github 非必需**（CLIP 可跳过）；最该解决的是 **huggingface**（用
> `HF_ENDPOINT=https://hf-mirror.com` 通常免代理即可）和 **Google Drive**（多半得靠代理）。

---

## 附：本仓库为部署新增/修改的文件

- `environment.yml`：一键创建 conda 环境（含正确版本锁）。
- `requirements.txt`：pip 依赖（含版本锁与说明）。
- `scripts/train_VIP5_single_gpu.sh`：单卡训练便捷脚本（可调 batch）。
- `DEPLOY.md`：本指南。

> 兼容性验证说明：以上 `transformers==4.17.0` 与本仓库源码的兼容性已在干净环境实测
> ——所有版本敏感的 import 均通过，`VIP5Tuning`（t5-small backbone）可成功构建
> （约 6,380 万参数）。`PyTorch 1.12.1 + CUDA 11.3` 为运行训练/评估的推荐组合。
