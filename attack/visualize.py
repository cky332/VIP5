"""(viz) Visualize the adversarial perturbation.

For sampled attacked items, makes a side-by-side montage:
    [ clean 224 | attacked 224 | perturbation x AMP ]
and prints the perturbation magnitude (L-inf, L2, mean) per image.

Reuses the perturbed PNGs written by pgd_attack (attack/out/perturbed_images/<asin>.png)
and reconstructs the clean 224 image from the original via the SAME preprocess.
PIL + numpy only (no matplotlib).

Run:
    python attack/visualize.py            # 8 samples, amplify diff x10
    python attack/visualize.py 12 20      # 12 samples, amplify x20
View the PNGs in attack/out/perturbed_images/ (VS Code can open them, or scp down).
"""
import os
import sys
import glob

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common          # noqa: sets paths + chdir(ROOT)
import config as C
import clip_extract as CE


def _u8(a01):
    return (np.clip(a01, 0, 1) * 255).round().astype("uint8")


def make_compare(asin, item2img, amp=10):
    adv_path = os.path.join(C.PERTURBED_IMG_DIR, asin + ".png")
    if not os.path.isfile(adv_path):
        return None
    ip = CE.resolve_image_path(asin, item2img)
    if ip is None:
        return None
    clean = CE.preprocess_to_224(Image.open(ip)).numpy().transpose(1, 2, 0)          # HWC [0,1]
    adv = np.asarray(Image.open(adv_path).convert("RGB")).astype("float32") / 255.0  # HWC [0,1]
    delta = adv - clean
    stats = {"asin": asin,
             "Linf_/255": round(float(np.abs(delta).max() * 255), 2),
             "L2": round(float(np.sqrt((delta ** 2).sum())), 3),
             "mean_abs_/255": round(float(np.abs(delta).mean() * 255), 3)}
    diff_vis = np.clip(0.5 + delta * amp, 0, 1)   # mid-gray = no change; color = delta x amp
    gap = np.ones((224, 8, 3), dtype="uint8") * 255
    row = np.concatenate([_u8(clean), gap, _u8(adv), gap, _u8(diff_vis)], axis=1)
    out = os.path.join(C.PERTURBED_IMG_DIR, "compare_%s.png" % asin)
    Image.fromarray(row).save(out)
    stats["fig"] = out
    return row, stats


def main(n=8, amp=10):
    item2img = CE.load_item2img()
    paths = sorted(p for p in glob.glob(os.path.join(C.PERTURBED_IMG_DIR, "*.png"))
                   if not os.path.basename(p).startswith("compare_")
                   and not os.path.basename(p).startswith("grid"))
    asins = [os.path.basename(p)[:-4] for p in paths][:n]
    rows, all_stats = [], []
    for a in asins:
        r = make_compare(a, item2img, amp=amp)
        if r:
            rows.append(r[0]); all_stats.append(r[1])
            print(r[1])
    if rows:
        gap = np.ones((10, rows[0].shape[1], 3), dtype="uint8") * 255
        stacked = []
        for r in rows:
            stacked.append(r); stacked.append(gap)
        grid = np.concatenate(stacked[:-1], axis=0)
        grid_path = os.path.join(C.PERTURBED_IMG_DIR, "grid_compare.png")
        Image.fromarray(grid).save(grid_path)
        linf = np.mean([s["Linf_/255"] for s in all_stats])
        meanabs = np.mean([s["mean_abs_/255"] for s in all_stats])
        print("\n[viz] %d montages -> %s" % (len(rows), C.PERTURBED_IMG_DIR))
        print("[viz] columns = [clean | attacked | perturbation x%d]" % amp)
        print("[viz] combined grid -> %s" % grid_path)
        print("[viz] mean Linf = %.2f/255 (budget 16/255) | mean |delta| = %.3f/255 per channel"
              % (linf, meanabs))
    else:
        print("[viz] no perturbed images found in", C.PERTURBED_IMG_DIR, "- run pgd first.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    amp = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(n, amp)
