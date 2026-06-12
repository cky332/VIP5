"""Standalone AUV-Fusion diffusion generator -- run in a SEPARATE conda env.

Why separate: modern `diffusers` needs newer transformers/huggingface_hub than VIP5's pinned
transformers==4.17 -- they cannot coexist in one env. So we decouple: this script only needs
diffusers + a public CLIP (open_clip) and writes adversarial PNGs; VIP5 never imports diffusers.

It implements AUV-Fusion's faithful diffusion generator (arXiv 2507.22880, Eq 12-17) against the
SAME public CLIP ViT-B/32 that VIP5 consumes (open_clip "ViT-B-32"/"openai"), optimizing a
4x28x28 latent delta toward the popular centroid (built by VIP5) with L_align + L_CLIP + L_SSIM:

    z0   = VAE_enc(x0);  z_t = sqrt(a_t)z0 + sqrt(1-a_t)eps;  z~_t = z_t + eta*delta
    z~0  = DDIM_reverse(z~_t, empty-text);   x_adv = VAE_dec(z~0)

Setup (separate env):
    conda create -n diff python=3.10 -y && conda activate diff
    pip install torch diffusers transformers open_clip_torch safetensors
    export HF_ENDPOINT=https://hf-mirror.com          # China mirror for weight download

Flow:
    # (VIP5 env)        python attack/auv_diffusion.py export
    # (this diff env)   python attack/auv_diffusion_gen.py
    # (VIP5 env)        python attack/auv_diffusion.py eval
Reads  C.AUV_DIFF_TARGETS_JSON + C.CENTROID_PATH ; writes PNGs to C.AUV_DIFF_PERT_IMG_DIR.
config.py is dependency-free so it imports fine here too.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

SCALE = 0.18215
DEVICE = os.environ.get("AUV_DIFF_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _load_sd(device):
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    except Exception as e:
        raise SystemExit("install diffusers in THIS env: pip install diffusers transformers "
                         "safetensors  (export HF_ENDPOINT=https://hf-mirror.com). err: %s" % e)
    mid = C.AUV_DIFF_SD_MODEL
    vae = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(device).eval()
    sch = DDIMScheduler.from_pretrained(mid, subfolder="scheduler")
    unet = None
    if C.AUV_DIFF_MODE == "ddim":
        unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet").to(device).eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        try:
            unet.enable_gradient_checkpointing()
        except Exception:
            pass
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, unet, sch


def _load_clip(device):
    try:
        import open_clip
    except Exception as e:
        raise SystemExit("install open_clip in THIS env: pip install open_clip_torch  (err: %s)" % e)
    model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean, std = _MEAN.to(device), _STD.to(device)

    def feat(x01):                       # x01 (1,3,224,224) in [0,1], differentiable
        return model.encode_image((x01 - mean) / std)
    return feat


def _ssim(x, y, win=11, c1=0.01 ** 2, c2=0.03 ** 2):
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    p = win // 2
    mx = F.avg_pool2d(x, win, 1, p)
    my = F.avg_pool2d(y, win, 1, p)
    sx = F.avg_pool2d(x * x, win, 1, p) - mx * mx
    sy = F.avg_pool2d(y * y, win, 1, p) - my * my
    sxy = F.avg_pool2d(x * y, win, 1, p) - mx * my
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))).mean()


def _decode_from_delta(z_base, delta, vae, unet, scheduler, rev_ts, h, eta):
    z = z_base + eta * delta
    if unet is not None and rev_ts is not None:
        for t in rev_ts:
            eps = unet(z, t, encoder_hidden_states=h).sample
            z = scheduler.step(eps, t, z).prev_sample
    x = vae.decode(z / SCALE).sample
    return (x / 2 + 0.5).clamp(0, 1)


def attack(x0, vae, unet, scheduler, clip_feat, target, device):
    x0 = x0.to(device)
    with torch.no_grad():
        z0 = vae.encode((x0.unsqueeze(0) * 2 - 1)).latent_dist.mean * SCALE
        rev_ts, h, z_base = None, None, z0
        if C.AUV_DIFF_MODE == "ddim":
            scheduler.set_timesteps(C.AUV_DIFF_STEPS, device=device)
            inj = min(C.AUV_DIFF_INJECT, len(scheduler.timesteps) - 1)
            z_base = scheduler.add_noise(z0, torch.randn_like(z0), scheduler.timesteps[inj])
            rev_ts = scheduler.timesteps[inj:]
            h = torch.zeros(1, 77, unet.config.cross_attention_dim, device=device, dtype=z0.dtype)
        clean_feat = clip_feat(x0.unsqueeze(0)).detach()
    delta = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=C.AUV_DIFF_LR)
    for _ in range(C.AUV_DIFF_OPT_STEPS):
        opt.zero_grad()
        x = _decode_from_delta(z_base, delta, vae, unet, scheduler, rev_ts, h, C.AUV_DIFF_ETA)
        feat = clip_feat(x)
        fa = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        L_align = 1.0 - (fa * target).sum(-1).mean()
        L_clip = ((feat - clean_feat) ** 2).mean()
        L_ssim = 1.0 - _ssim(x[0], x0)
        L = (C.AUV_DIFF_LAMBDA_ALIGN * L_align + C.AUV_DIFF_LAMBDA_CLIP * L_clip
             + C.AUV_DIFF_LAMBDA_SSIM * L_ssim + C.AUV_DIFF_LAT_REG * (delta ** 2).mean())
        L.backward()
        opt.step()
    with torch.no_grad():
        x = _decode_from_delta(z_base, delta, vae, unet, scheduler, rev_ts, h, C.AUV_DIFF_ETA)
    return x[0].detach()


def _img01(path):
    arr = np.asarray(Image.open(path).convert("RGB").resize((224, 224))).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _save(x_chw, path):
    arr = (x_chw.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def main():
    if not os.path.isfile(C.AUV_DIFF_TARGETS_JSON):
        raise SystemExit("missing %s -- run `python attack/auv_diffusion.py export` in the VIP5 env first."
                         % C.AUV_DIFF_TARGETS_JSON)
    if not os.path.isfile(C.CENTROID_PATH):
        raise SystemExit("missing centroid %s -- build it in the VIP5 env first." % C.CENTROID_PATH)
    os.makedirs(C.AUV_DIFF_PERT_IMG_DIR, exist_ok=True)
    targets = json.load(open(C.AUV_DIFF_TARGETS_JSON))
    print("[gen] %d targets | SD=%s mode=%s | device=%s" % (len(targets), C.AUV_DIFF_SD_MODEL, C.AUV_DIFF_MODE, DEVICE))
    vae, unet, sch = _load_sd(DEVICE)
    clip_feat = _load_clip(DEVICE)
    c = np.load(C.CENTROID_PATH).astype("float32")
    target = torch.from_numpy(c).to(DEVICE).view(1, -1)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    done = 0
    for asin, path in targets.items():
        if not os.path.isfile(path):
            continue
        x0 = _img01(path)
        x_adv = attack(x0, vae, unet, sch, clip_feat, target, DEVICE)
        _save(x_adv, os.path.join(C.AUV_DIFF_PERT_IMG_DIR, asin + ".png"))
        done += 1
        if done % 25 == 0:
            d = (x_adv.to(DEVICE) - x0.to(DEVICE)).abs()
            print("[gen] %d/%d | last linf %.0f/255 mean %.1f/255"
                  % (done, len(targets), d.max().item() * 255, d.mean().item() * 255))
    print("[gen] done: %d PNGs -> %s  (now run `python attack/auv_diffusion.py eval` in the VIP5 env)"
          % (done, C.AUV_DIFF_PERT_IMG_DIR))


if __name__ == "__main__":
    main()
