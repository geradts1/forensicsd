"""
End-to-end demonstration: the fingerprint-copy attack fools a PCE-only
examiner, and the triangle test catches it.

Runs entirely on SYNTHETIC cameras and scenes -- no real images needed, so it
is safe to run anywhere and demonstrates the method, not a victim-targeting
pipeline. To run on real data, replace the synthetic loaders with your own
over your own cameras.

Usage:  python examples/demo_attack_and_detect.py
"""
from __future__ import annotations

import numpy as np

from prnucf import (
    estimate_fingerprint, noise_residual, pce,
    inject_fingerprint, triangle_test,
)

SHAPE = (256, 256)


def synth_prnu(rng, strength=0.03):
    return rng.standard_normal(SHAPE) * strength


def synth_image(prnu, rng):
    base = rng.uniform(40, 210)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]] / max(SHAPE)
    scene = (base
             + 30 * np.sin(6 * xx + rng.uniform(0, 6))
             + 25 * np.cos(5 * yy + rng.uniform(0, 6)))
    img = scene * (1.0 + prnu) + rng.standard_normal(SHAPE) * 2.0
    return np.clip(img, 0, 255)


def pce_vs(img, K):
    return pce(noise_residual(img), K * img)


def main():
    rng = np.random.default_rng(42)

    prnu_A = synth_prnu(rng)          # victim / target camera
    prnu_B = synth_prnu(rng)          # attacker's own camera

    victim_imgs = [synth_image(prnu_A, rng) for _ in range(12)]
    K_A = estimate_fingerprint(victim_imgs)

    carrier = synth_image(prnu_B, rng)

    # A reference cohort: images from OTHER, unrelated cameras. This is how an
    # examiner calibrates the triangle test (and how its false-positive rate
    # is controlled). None of these came from the victim pool.
    cohort = []
    for _ in range(15):
        prnu_other = synth_prnu(rng)
        cohort.append(synth_image(prnu_other, rng))

    # Forensic decision bar an examiner would use for "same source": PCE ~ 60.
    DECISION_PCE = 60.0
    genuine_A = synth_image(prnu_A, rng)
    print("=== Fingerprint-copy attack ===")
    res = inject_fingerprint(
        carrier, K_A, match_target_pce=DECISION_PCE,
        detector=lambda im: pce_vs(im, K_A),
    )
    forged = res.image
    print(f"  tuned alpha = {res.alpha:.3f}   image PSNR = {res.psnr:.1f} dB "
          f"(visually lossless)\n")

    print("=== PCE-only examiner (the attack target) ===")
    print(f"  decision threshold (same source) : PCE > {DECISION_PCE:.0f}")
    print(f"  genuine A image   PCE vs K_A     : {pce_vs(genuine_A, K_A):12.1f}")
    print(f"  FORGED  B image   PCE vs K_A     : {pce_vs(forged, K_A):12.1f}")
    print("  -> the forgery clears the attribution bar: PCE alone is fooled.\n")

    print("=== Triangle test on the FORGED image (should FLAG) ===")
    print(triangle_test(forged, victim_imgs, reference_images=cohort).summary(), "\n")

    print("=== Triangle test on an INNOCENT un-forged B image (should NOT flag) ===")
    innocent = synth_image(prnu_B, rng)
    print(triangle_test(innocent, victim_imgs, reference_images=cohort).summary())


if __name__ == "__main__":
    main()
