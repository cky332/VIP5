"""(viz) Visualize the adversarial perturbation as a labeled montage:

    [ clean | adversarial | perturbation x AMP ]   with a header (column names)
    and a per-row label: "item <id> (<asin>)  max|d|=../255  mean|d|=..".

Reconstructs the clean 224 image from the original cover via the SAME preprocess and
diffs it against the saved adversarial PNG. Works for any of the three attacks
(they all save perturbed PNGs in the same 224x224 [0,1] space):

    pgd : attack/out/perturbed_images/            (single-CLIP white-box PGD)
    xt  : attack/out/xtransfer/perturbed_images/  (X-Transfer black-box transfer)
    aa  : attack/out/anyattack/perturbed_images/  (AnyAttack generator)

PIL + numpy only (no matplotlib).

Run (from repo root):
    python attack/visualize.py                 # pgd, 8 samples, diff x10  (back-compat)
    python attack/visualize.py 12 20 xt        # X-Transfer, 12 samples, diff x20
    python attack/visualize.py 6 10 aa         # AnyAttack, 6 samples, diff x10
Outputs (next to the perturbed images): compare_<asin>.png + grid_compare.png
"""
import os
import sys
import glob
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common          # noqa: sets paths + chdir(ROOT)
import config as C
import clip_extract as CE

CW = 224                       # per-panel width/height
GAP = 8                        # gap between panels
W = CW * 3 + GAP * 2           # full row width = 688
BAND = 22                      # label band height
COL_CX = [CW // 2, CW + GAP + CW // 2, 2 * (CW + GAP) + CW // 2]   # column centers

# attack selector -> directory holding <asin>.png perturbed covers
SRC_DIRS = {
    "pgd": C.PERTURBED_IMG_DIR,
    "xt": C.XT_PERT_IMG_DIR,
    "aa": C.AA_PERT_IMG_DIR,
    "style": C.STYLE_PERT_IMG_DIR,
    "auv": C.AUV_PERT_IMG_DIR,
}


def _font(size=13):
    for p in ("DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _u8(a01):
    return (np.clip(a01, 0, 1) * 255).round().astype("uint8")


def _text_w(draw, text, font):
    try:
        return draw.textlength(text, font=font)
    except Exception:
        return font.getbbox(text)[2]


def _band(text_left=None, centered=None, h=BAND):
    """White strip; optional left-aligned text and/or centered column labels."""
    band = Image.new("RGB", (W, h), (255, 255, 255))
    d = ImageDraw.Draw(band)
    f = _font(13)
    if text_left:
        d.text((4, h // 2 - 7), text_left, fill=(0, 0, 0), font=f)
    if centered:
        for cx, name in zip(COL_CX, centered):
            d.text((cx - _text_w(d, name, f) / 2, h // 2 - 7), name, fill=(0, 0, 0), font=f)
    return np.asarray(band)


def make_row(asin, item2img, item2id, amp, src_dir):
    adv_path = os.path.join(src_dir, asin + ".png")
    if not os.path.isfile(adv_path):
        return None
    ip = CE.resolve_image_path(asin, item2img)
    if ip is None:
        return None
    clean = CE.preprocess_to_224(Image.open(ip)).numpy().transpose(1, 2, 0)
    adv = np.asarray(Image.open(adv_path).convert("RGB")).astype("float32") / 255.0
    delta = adv - clean
    linf = float(np.abs(delta).max() * 255)
    meanabs = float(np.abs(delta).mean() * 255)
    l2 = float(np.sqrt((delta ** 2).sum()))
    diff_vis = np.clip(0.5 + delta * amp, 0, 1)
    gap = np.ones((CW, GAP, 3), dtype="uint8") * 255
    imgs = np.concatenate([_u8(clean), gap, _u8(adv), gap, _u8(diff_vis)], axis=1)
    label = "item %s (%s)   max|d|=%.0f/255   mean|d|=%.2f/255" % (
        item2id.get(asin, "?"), asin, linf, meanabs)
    block = np.concatenate([_band(text_left=label), imgs], axis=0)
    stats = {"asin": asin, "item": item2id.get(asin, "?"),
             "Linf_/255": round(linf, 2), "mean_abs_/255": round(meanabs, 3), "L2": round(l2, 3)}
    return block, stats


def main(n=8, amp=10, which="pgd"):
    if which in SRC_DIRS:
        src_dir = SRC_DIRS[which]
    else:
        # also accept a typographic variant name (text_attack.py writes
        # attack/out/text/<variant>/images/<asin>.png), e.g. inject_bgadapt
        cand = os.path.join(C.OUT_DIR, "text", which, "images")
        if os.path.isdir(cand):
            src_dir = cand
        else:
            raise SystemExit("unknown attack '%s'; choose from %s or a text variant "
                             "with images under attack/out/text/<variant>/images/"
                             % (which, list(SRC_DIRS)))
    item2img = CE.load_item2img()
    item2id = json.load(open(C.DATAMAPS))["item2id"]
    paths = sorted(p for p in glob.glob(os.path.join(src_dir, "*.png"))
                   if not os.path.basename(p).startswith(("compare_", "grid")))
    asins = [os.path.basename(p)[:-4] for p in paths][:n]

    blocks, all_stats = [], []
    for a in asins:
        r = make_row(a, item2img, item2id, amp, src_dir)
        if r is None:
            continue
        block, stats = r
        all_stats.append(stats)
        print(stats)
        Image.fromarray(np.concatenate([_band(centered=["clean", "adversarial", "diff x%d" % amp]),
                                         block], axis=0)).save(
            os.path.join(src_dir, "compare_%s.png" % a))
        blocks.append(block)

    if not blocks:
        print("[viz] no perturbed images found in", src_dir,
              "- run the '%s' attack first." % which)
        return
    header = _band(centered=["clean", "adversarial", "diff x%d" % amp])
    row_gap = np.ones((10, W, 3), dtype="uint8") * 255
    parts = [header]
    for b in blocks:
        parts += [b, row_gap]
    grid = np.concatenate(parts[:-1], axis=0)
    grid_path = os.path.join(src_dir, "grid_compare.png")
    Image.fromarray(grid).save(grid_path)
    print("\n[viz] attack=%s | %d labeled montages + grid -> %s" % (which, len(blocks), grid_path))
    print("[viz] columns: clean | adversarial | perturbation x%d (mid-gray = no change)" % amp)
    print("[viz] mean Linf = %.2f/255 (budget 16/255), mean |delta| = %.2f/255 per channel"
          % (np.mean([s["Linf_/255"] for s in all_stats]),
             np.mean([s["mean_abs_/255"] for s in all_stats])))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    amp = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    which = sys.argv[3] if len(sys.argv) > 3 else "pgd"
    main(n, amp, which)
