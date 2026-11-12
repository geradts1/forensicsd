import numpy as np

from prnucf import (
    estimate_fingerprint, noise_residual, pce,
    inject_fingerprint, remove_fingerprint,
)


def _pce_vs(img, K):
    return pce(noise_residual(img), K * img)


def test_injection_raises_pce(cameras):
    imgs_a = [cameras["img_a"]() for _ in range(10)]
    K = estimate_fingerprint(imgs_a)
    carrier = cameras["img_b"]()

    before = _pce_vs(carrier, K)
    forged = inject_fingerprint(carrier, K, alpha=0.4).image
    after = _pce_vs(forged, K)

    assert after > before
    assert after > 20.0  # injection is detectable by a PCE examiner


def test_injection_is_visually_lossless(cameras):
    imgs_a = [cameras["img_a"]() for _ in range(8)]
    K = estimate_fingerprint(imgs_a)
    carrier = cameras["img_b"]()
    res = inject_fingerprint(carrier, K, alpha=0.3)
    assert res.psnr > 45.0  # high PSNR => no visible artefact


def test_match_target_pce_search(cameras):
    imgs_a = [cameras["img_a"]() for _ in range(10)]
    K = estimate_fingerprint(imgs_a)
    carrier = cameras["img_b"]()
    target = 60.0
    res = inject_fingerprint(carrier, K, match_target_pce=target,
                             detector=lambda im: _pce_vs(im, K))
    achieved = _pce_vs(res.image, K)
    assert abs(achieved - target) < 0.2 * target


def test_removal_undoes_injection(cameras):
    """flat-field removal with the same fingerprint should neutralise an
    injected copy, dropping PCE back toward the un-injected baseline."""
    imgs_a = [cameras["img_a"]() for _ in range(10)]
    K = estimate_fingerprint(imgs_a)
    carrier = cameras["img_b"]()

    baseline = _pce_vs(carrier, K)
    forged = inject_fingerprint(carrier, K, alpha=0.5).image
    injected = _pce_vs(forged, K)
    cleaned = remove_fingerprint(forged, fingerprint=K, alpha=0.5,
                                 method="flatfield")
    after = _pce_vs(cleaned, K)

    assert injected > baseline
    assert after < injected / 2.0


def test_flatfield_removal_runs(cameras):
    imgs_a = [cameras["img_a"]() for _ in range(8)]
    K = estimate_fingerprint(imgs_a)
    genuine = cameras["img_a"]()
    out = remove_fingerprint(genuine, fingerprint=K, alpha=1.0,
                             method="flatfield")
    assert out.shape == cameras["shape"]
