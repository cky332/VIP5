"""(viz) Text-overlay attack comparison montage.

Per sampled item, one row of 4 panels:
    [ clean | rank_first_en | rank_first_en_stealth | rank_first_en_stealth_low ]
with column headers and a per-row "item <id> (<asin>)" label.

Reuses the rendered variant images written by text_attack.py
(attack/out/text/<variant>/images/<asin>.png) and reconstructs the clean 224
panel from the original via the SAME preprocess (so it matches the render base).
PIL + numpy only.

Run from repo root:
    python attack/visualize_text.py            # 6 items
    python attack/visualize_text.py 10
Outputs in attack/out/text/:
    compare_text_<asin>.png   (one item, 4 panels)
    grid_text_compare.png     (all sampled items, with header)
"""
import os
import sys
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common          # noqa: paths + chdir(ROOT)
import config as C
import clip_extract as CE
import text_attack as TA

CW, GAP, BAND = 224, 8, 22
COLS = ["clean"] + list(TA.VARIANTS.keys())          # 4 columns
NCOL = len(COLS)
W = CW * NCOL + GAP * (NCOL - 1)
COL_CX = [i * (CW + GAP) + CW // 2 for i in range(NCOL)]
COL_LABELS = ["clean", "rank_first_en", "stealth a96", "stealth_low a64"]


def _font(size=13):
    for p in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_w(d, t, f):
    try:
        return d.textlength(t, font=f)
    except Exception:
        return f.getbbox(t)[2]


def _band(text_left=None, centered=None, h=BAND):
    band = Image.new("RGB", (W, h), (255, 255, 255))
    d = ImageDraw.Draw(band)
    f = _font(13)
    if text_left:
        d.text((4, h // 2 - 7), text_left, fill=(0, 0, 0), font=f)
    if centered:
        for cx, name in zip(COL_CX, centered):
            d.text((cx - _text_w(d, name, f) / 2, h // 2 - 7), name, fill=(0, 0, 0), font=f)
    return np.asarray(band)


def _load224(path):
    return np.asarray(Image.open(path).convert("RGB").resize((CW, CW)))


def make_row(asin, item2img, item2id):
    # clean panel: reconstruct via the same CLIP preprocess used for rendering
    ip = CE.resolve_image_path(asin, item2img)
    if ip is None:
        return None
    clean_pil = TA._to_pil(CE.preprocess_to_224(Image.open(ip)))
    panels = [np.asarray(clean_pil)]
    for v in TA.VARIANTS:
        p = os.path.join(TA._img_dir(v), asin + ".png")
        if not os.path.isfile(p):
            return None
        panels.append(_load224(p))
    gap = np.ones((CW, GAP, 3), dtype="uint8") * 255
    cells = []
    for i, pan in enumerate(panels):
        cells.append(pan.astype("uint8"))
        if i < len(panels) - 1:
            cells.append(gap)
    imgs = np.concatenate(cells, axis=1)
    label = "item %s (%s)" % (item2id.get(asin, "?"), asin)
    return np.concatenate([_band(text_left=label), imgs], axis=0)


def main(n=6):
    import json
    item2img = CE.load_item2img()
    item2id = json.load(open(C.DATAMAPS))["item2id"]
    # sample asins that have rendered images for the first variant
    first = list(TA.VARIANTS.keys())[0]
    paths = sorted(glob.glob(os.path.join(TA._img_dir(first), "*.png")))
    asins = [os.path.basename(p)[:-4] for p in paths][:n]

    header = _band(centered=COL_LABELS)
    rows = []
    for a in asins:
        r = make_row(a, item2img, item2id)
        if r is None:
            continue
        Image.fromarray(np.concatenate([header, r], axis=0)).save(
            os.path.join(TA.TEXT_DIR, "compare_text_%s.png" % a))
        rows.append(r)

    if not rows:
        print("[viz_text] no rendered images found - run `python attack/text_attack.py` first.")
        return
    gap = np.ones((10, W, 3), dtype="uint8") * 255
    parts = [header]
    for r in rows:
        parts += [r, gap]
    grid = np.concatenate(parts[:-1], axis=0)
    out = os.path.join(TA.TEXT_DIR, "grid_text_compare.png")
    Image.fromarray(grid).save(out)
    print("[viz_text] %d items -> %s" % (len(rows), out))
    print("[viz_text] columns:", " | ".join(COL_LABELS))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
