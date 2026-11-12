"""
Command-line interface for prnucf.

Subcommands
-----------
  fingerprint   estimate a camera reference fingerprint from a folder of images
  pce           score a questioned image against a fingerprint (attribution)
  inject        apply a fingerprint-copy to an image (evaluation only)
  triangle      run the triangle test: is a questioned image a copy-forgery?

All image I/O uses Pillow. Install extras with: pip install 'prnucf[io]'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from . import (
    estimate_fingerprint, noise_residual, pce,
    inject_fingerprint, triangle_test,
)


def _load(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _load_folder(folder):
    paths = sorted(
        p for ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
        for p in glob.glob(os.path.join(folder, f"*.{ext}"))
    )
    if not paths:
        sys.exit(f"no images found in {folder}")
    return paths


def _save_npy(arr, path):
    np.save(path, arr)


def cmd_fingerprint(a):
    paths = _load_folder(a.folder)
    K = estimate_fingerprint((_load(p) for p in paths))
    _save_npy(K, a.out)
    print(f"fingerprint from {len(paths)} images -> {a.out}  shape={K.shape}")


def cmd_pce(a):
    img = _load(a.image)
    K = np.load(a.fingerprint)
    val = pce(noise_residual(img), K * img)
    print(f"PCE = {val:.2f}")
    if a.threshold is not None:
        print("ABOVE threshold (same-source)" if val > a.threshold
              else "below threshold")


def cmd_inject(a):
    from PIL import Image
    img = _load(a.image)
    K = np.load(a.fingerprint)
    kw = {}
    if a.target_pce is not None:
        kw.update(match_target_pce=a.target_pce,
                  detector=lambda im: pce(noise_residual(im), K * im))
    else:
        kw.update(alpha=a.alpha)
    res = inject_fingerprint(img, K, **kw)
    Image.fromarray(np.clip(res.image, 0, 255).astype("uint8")).save(a.out)
    print(f"injected (alpha={res.alpha:.3f}, PSNR={res.psnr:.1f} dB) -> {a.out}")
    print("NB: evaluation use only; see ETHICS.md")


def cmd_triangle(a):
    q = _load(a.image)
    pool = [_load(p) for p in _load_folder(a.sources)]
    cohort = [_load(p) for p in _load_folder(a.cohort)] if a.cohort else None
    res = triangle_test(q, pool, reference_images=cohort, z_threshold=a.z)
    print(res.summary())


def build_parser():
    p = argparse.ArgumentParser(prog="prnucf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fingerprint", help="estimate a camera fingerprint")
    f.add_argument("folder")
    f.add_argument("-o", "--out", default="fingerprint.npy")
    f.set_defaults(func=cmd_fingerprint)

    c = sub.add_parser("pce", help="attribution score")
    c.add_argument("image")
    c.add_argument("fingerprint")
    c.add_argument("-t", "--threshold", type=float, default=None)
    c.set_defaults(func=cmd_pce)

    i = sub.add_parser("inject", help="fingerprint-copy (evaluation only)")
    i.add_argument("image")
    i.add_argument("fingerprint")
    i.add_argument("-o", "--out", default="forged.png")
    i.add_argument("--alpha", type=float, default=0.02)
    i.add_argument("--target-pce", type=float, default=None,
                   help="tune alpha to reach this PCE")
    i.set_defaults(func=cmd_inject)

    t = sub.add_parser("triangle", help="detect a copy-forgery")
    t.add_argument("image", help="questioned image")
    t.add_argument("sources", help="folder of candidate source images")
    t.add_argument("--cohort", default=None,
                   help="folder of known-innocent reference images")
    t.add_argument("-z", type=float, default=3.0, help="z threshold to flag")
    t.set_defaults(func=cmd_triangle)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
