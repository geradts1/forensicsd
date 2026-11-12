"""
Counter-forensics: PRNU injection ("fingerprint-copy") and removal.

PURPOSE AND SCOPE
-----------------
This module exists to let a forensic examiner *measure how fragile* a PRNU
source-attribution conclusion is -- i.e. to run the attack against their own
pipeline and confirm their detectors (see ``triangle.py``) catch it. It
implements only the additive fingerprint-copy model that is already fully
described in the open literature; it deliberately contains no refinements
aimed at defeating the triangle test or any specific deployed system.

Intended use is on your OWN cameras and images, or on consented test data,
inside a robustness evaluation. Using it to attribute a questioned image to a
camera that did not take it -- i.e. to fabricate source-camera evidence
against a person -- is evidence tampering. Don't. See ETHICS.md.

References
----------
M. Goljan, J. Fridrich, M. Chen, "Defending against fingerprint-copy attack
in sensor-based camera identification", IEEE TIFS 2011 -- which introduces
BOTH the attack modelled here and the triangle-test defence.
"""
from __future__ import annotations

import numpy as np

from .core import noise_residual, to_gray, wavelet_denoise, estimate_fingerprint

__all__ = [
    "inject_fingerprint",
    "remove_fingerprint",
    "InjectionResult",
]


class InjectionResult:
    """Container returned by :func:`inject_fingerprint` for bookkeeping."""

    def __init__(self, image: np.ndarray, alpha: float, psnr: float):
        self.image = image
        self.alpha = alpha
        self.psnr = psnr

    def __repr__(self) -> str:
        return f"InjectionResult(alpha={self.alpha:.4f}, psnr={self.psnr:.1f} dB)"


def _psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def inject_fingerprint(image: np.ndarray, fingerprint: np.ndarray,
                       alpha: float = 0.02,
                       clip: tuple[float, float] = (0.0, 255.0),
                       match_target_pce: float | None = None,
                       detector=None, max_iter: int = 12) -> InjectionResult:
    """
    Additive fingerprint-copy on a grayscale (or luma) image:

        I' = I + alpha * I * K

    This is the published model: the PRNU term is multiplicative in the scene
    content, so the copy is scaled by the local intensity. ``alpha`` trades
    off detectability-by-PCE against visible distortion.

    Parameters
    ----------
    image : HxW or HxWxC array (operates on luma; a 3-channel input is
        processed on its luma and the correction added back equally per channel)
    fingerprint : HxW reference fingerprint K (same shape as the luma plane)
    alpha : injection strength. Values ~0.005-0.05 are typical.
    match_target_pce : if given, ``alpha`` is ignored and the routine does a
        bounded bisection search on alpha so that ``detector(I')`` reaches
        approximately this PCE. Supply a ``detector`` callable that maps an
        image to a PCE float. This is provided so an evaluator can ask "what
        injection strength would it take to reach my decision threshold, and
        is the result still clean enough to fool a human?" -- the answer is
        usually "yes for PCE, but the triangle test still fires".

    Notes
    -----
    Nothing here conceals the statistical link between K and the images it was
    estimated from; that link is exactly what the triangle test exploits.
    """
    img = np.asarray(image, dtype=np.float64)
    gray = to_gray(img)
    K = np.asarray(fingerprint, dtype=np.float64)
    if K.shape != gray.shape:
        raise ValueError(f"fingerprint shape {K.shape} != image luma {gray.shape}")

    def _apply(a: float) -> np.ndarray:
        delta = a * gray * K
        if img.ndim == 2:
            out = gray + delta
        else:
            out = img.copy()
            out[..., :3] += delta[..., None]
        return np.clip(out, clip[0], clip[1])

    if match_target_pce is not None:
        if detector is None:
            raise ValueError("match_target_pce requires a detector callable")
        # grow an upper bound until it brackets the target (PCE is monotone
        # increasing in alpha), capped to avoid destroying the image.
        hi = max(alpha, 0.05)
        for _ in range(20):
            if detector(_apply(hi)) >= match_target_pce or hi > 50.0:
                break
            hi *= 2.0
        lo = 0.0
        best = hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = detector(_apply(mid))
            if val < match_target_pce:
                lo = mid
            else:
                hi = mid
            best = mid
        alpha = best

    out = _apply(alpha)
    return InjectionResult(out, alpha, _psnr(gray, to_gray(out)))


def remove_fingerprint(image: np.ndarray, fingerprint: np.ndarray | None = None,
                       alpha: float = 1.0, method: str = "flatfield",
                       sigma: float = 3.0) -> np.ndarray:
    """
    Suppress PRNU so a residual no longer attributes to the source camera.

    method='flatfield' : inverse of the injection model, I / (1 + alpha*K),
        requires the camera's own fingerprint (the "forger owns the camera"
        case, or an anonymisation case).
    method='denoise'   : fingerprint-free; just strips the high-frequency
        residual via the same wavelet denoiser. Cruder, and conspicuously
        "too clean" -- which is itself a detectable tell.

    Removal is generally easier than injection and easier to flag (a residual
    with no PRNU where one is expected). Included for completeness of the
    robustness evaluation (anonymisation / attribution-denial scenarios).
    """
    img = np.asarray(image, dtype=np.float64)
    gray = to_gray(img)
    if method == "denoise":
        return wavelet_denoise(gray, sigma=sigma)
    if method == "flatfield":
        if fingerprint is None:
            raise ValueError("flatfield removal needs the camera fingerprint")
        K = np.asarray(fingerprint, dtype=np.float64)
        return gray / (1.0 + alpha * K)
    raise ValueError(f"unknown removal method {method!r}")
