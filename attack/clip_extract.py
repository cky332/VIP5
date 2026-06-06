"""(1) Image recovery + CLIP ViT-B/32 extraction + fidelity check.

Reconstructs the public CLIP pipeline that produced VIP5's shipped .npy features,
resolves whether those features are L2-normalized, and exposes a differentiable
encoder used by the PGD attack.

Requires OpenAI CLIP:  pip install git+https://github.com/openai/CLIP.git
(use a github mirror if needed, see DEPLOY.md FAQ #10).
"""
import os
import sys
import json
import zipfile
import pickle
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import config as C

# CLIP ViT-B/32 normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_MODEL = None
_PREPROC_224 = None   # PIL -> [0,1] CHW tensor (resize+centercrop, NO normalize)


def load_clip(device=None):
    """Return the CLIP ViT-B/32 model (eval, frozen) on the given device."""
    global _MODEL, _PREPROC_224
    device = device or C.DEVICE
    if _MODEL is None:
        import clip  # lazy
        import torchvision.transforms as T
        try:
            BICUBIC = T.InterpolationMode.BICUBIC
        except AttributeError:
            from PIL import Image
            BICUBIC = Image.BICUBIC
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL = model
        # CLIP preprocess minus Normalize (we normalize separately, differentiably)
        _PREPROC_224 = T.Compose([
            T.Resize(224, interpolation=BICUBIC),
            T.CenterCrop(224),
            lambda im: im.convert("RGB"),
            T.ToTensor(),   # -> [0,1]
        ])
    return _MODEL


def preprocess_to_224(pil_image):
    """PIL.Image -> float tensor (3,224,224) in [0,1]."""
    load_clip()
    return _PREPROC_224(pil_image)


def _normalize(x):
    mean = torch.tensor(_CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def encode_pixels(x_0to1, normalize, device=None):
    """Differentiable: [0,1] pixels (B,3,224,224) -> CLIP image embedding (B,512).
    `normalize` = L2-normalize the embedding (CLIP_NORM)."""
    device = device or C.DEVICE
    model = load_clip(device)
    x = x_0to1.to(device)
    feats = model.encode_image(_normalize(x)).float()
    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats


def extract_feature(pil_or_path, normalize, device=None):
    """Non-differentiable convenience: image -> np.float32 (512,)."""
    from PIL import Image
    img = Image.open(pil_or_path) if isinstance(pil_or_path, str) else pil_or_path
    x = preprocess_to_224(img).unsqueeze(0)
    with torch.no_grad():
        f = encode_pixels(x, normalize=normalize, device=device)
    return f[0].cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# image recovery
# ---------------------------------------------------------------------------
def unzip_photos(zip_path=None, dest=None):
    zip_path, dest = zip_path or C.PHOTOS_ZIP, dest or C.PHOTOS_DIR
    if os.path.isdir(dest) and os.listdir(dest):
        print("[clip_extract] photos dir already populated:", dest)
        return dest
    os.makedirs(dest, exist_ok=True)
    print("[clip_extract] unzipping", zip_path, "->", dest)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def load_item2img(path=None):
    return pickle.load(open(path or C.ITEM2IMG, "rb"))


_PHOTO_INDEX = None


def _photo_index(photos_dir):
    """basename -> full path, built once by walking photos_dir (bulletproof fallback)."""
    global _PHOTO_INDEX
    if _PHOTO_INDEX is None:
        _PHOTO_INDEX = {}
        for root, _dirs, files in os.walk(photos_dir):
            for f in files:
                _PHOTO_INDEX.setdefault(f, os.path.join(root, f))
        print("[clip_extract] indexed %d image files under %s" % (len(_PHOTO_INDEX), photos_dir))
    return _PHOTO_INDEX


def resolve_image_path(asin, item2img, photos_dir=None):
    """asin -> image file path. item2img values look like 'toys_photos/<file>.jpg'
    and the zip unpacks to <photos_dir>/photos/<cat>_photos/<file>.jpg, so we try
    several base dirs and fall back to a basename index that tolerates any nesting."""
    photos_dir = photos_dir or C.PHOTOS_DIR
    v = item2img.get(asin) if isinstance(item2img, dict) else None
    bases = [photos_dir, os.path.join(photos_dir, "photos"),
             os.path.join(photos_dir, "vip5_photos"), "."]
    cands = []
    base_name = None
    if v is not None:
        v = str(v).split("?")[0]
        base_name = os.path.basename(v)
        for b in bases:
            cands.append(os.path.join(b, v))        # join the relative value
            cands.append(os.path.join(b, base_name))  # join just the filename
    for ext in (".jpg", ".jpeg", ".png"):
        cands.append(os.path.join(photos_dir, asin + ext))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    # bulletproof fallback: look the filename up in the recursive index
    if base_name:
        hit = _photo_index(photos_dir).get(base_name)
        if hit:
            return hit
    return None


# ---------------------------------------------------------------------------
# fidelity check -> resolves CLIP_NORM
# ---------------------------------------------------------------------------
def _cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def verify_against_shipped(dataset, n=30, seed=C.SEED):
    """Compare re-extracted vs shipped features; decide & persist CLIP_NORM."""
    common.ensure_dirs()
    item2img = load_item2img()
    asins = [a for a in dataset.id2item.values()
             if os.path.isfile(common.shipped_feature_path(a))]
    random.Random(seed).shuffle(asins)

    rows, used = [], 0
    for a in asins:
        if used >= n:
            break
        ip = resolve_image_path(a, item2img)
        if ip is None:
            continue
        try:
            re_raw = extract_feature(ip, normalize=False)
        except Exception as e:
            print("[warn] extract failed for", a, e)
            continue
        re_norm = re_raw / (np.linalg.norm(re_raw) + 1e-8)
        shipped = common.load_shipped(a)
        rows.append({"asin": a,
                     "cos_raw": _cos(shipped, re_raw),
                     "cos_norm": _cos(shipped, re_norm),
                     "shipped_norm": float(np.linalg.norm(shipped))})
        used += 1

    if not rows:
        raise RuntimeError("No (asin -> image) resolved. Check PHOTOS_DIR / item2img_dict "
                           "value format / unzip step.")

    mean_cos_raw = float(np.mean([r["cos_raw"] for r in rows]))
    mean_cos_norm = float(np.mean([r["cos_norm"] for r in rows]))
    mean_shipped_norm = float(np.mean([r["shipped_norm"] for r in rows]))
    clip_norm = (abs(mean_shipped_norm - 1.0) < 0.05) and (mean_cos_norm >= mean_cos_raw)
    residual_floor = mean_cos_norm if clip_norm else mean_cos_raw

    extra = {"n_used": used, "mean_cos_raw": round(mean_cos_raw, 4),
             "mean_cos_norm": round(mean_cos_norm, 4),
             "mean_shipped_l2norm": round(mean_shipped_norm, 4),
             "residual_floor_cos": round(residual_floor, 4)}
    common.save_clip_norm(clip_norm, extra=extra)
    print("[clip_extract] verify:", json.dumps(extra, indent=2))
    print("[clip_extract] -> CLIP_NORM =", clip_norm,
          "| clean re-extraction cos floor =", round(residual_floor, 4))
    if residual_floor < 0.9:
        print("[clip_extract][WARN] low cos vs shipped (<0.9): possible CLIP version/"
              "preprocess mismatch. Absolute numbers may be off; relative attack deltas "
              "remain valid because we compare attacked-vs-clean using OUR re-extraction.")
    return {"clip_norm": clip_norm, **extra}


if __name__ == "__main__":
    ctx = common.load_context(need_model=False)
    unzip_photos()
    verify_against_shipped(ctx.dataset)
