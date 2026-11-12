"""
Core PRNU estimation and attribution.

Implements the standard sensor-noise (PRNU) pipeline used in digital image
forensics for source-camera attribution:

  * wavelet-based denoising residual (Lukas, Fridrich & Goljan, 2006)
  * reference fingerprint estimation by (optionally ML-weighted) averaging of
    residuals, with zero-mean + Wiener post-processing to suppress
    non-unique artefacts (Chen, Fridrich, Goljan & Lukas, 2008)
  * Normalised Cross-Correlation (NCC) and Peak-to-Correlation Energy (PCE)
    as the detection statistic for "did camera C take image I?"

These are textbook forensic methods. This module is the *defender's* side of
the package and is useful on its own for ordinary attribution work.

References
----------
J. Lukas, J. Fridrich, M. Goljan, "Digital camera identification from sensor
pattern noise", IEEE TIFS 2006.
M. Chen, J. Fridrich, M. Goljan, J. Lukas, "Determining image origin and
integrity using sensor noise", IEEE TIFS 2008.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import uniform_filter

__all__ = [
    "to_gray",
    "wavelet_denoise",
    "noise_residual",
    "estimate_fingerprint",
    "zero_mean",
    "wiener_in_dft",
    "ncc",
    "pce",
    "crosscorr_2d",
]

EPS = 1.0


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance conversion; accepts HxW or HxWxC float/uint arrays."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            return arr[..., 0]
        # Rec.601 luma on the first three channels; ignore alpha.
        w = np.array([0.299, 0.587, 0.114])
        return arr[..., :3] @ w
    raise ValueError(f"unexpected image shape {arr.shape}")


# --------------------------------------------------------------------------- #
# Denoising residual
# --------------------------------------------------------------------------- #
def wavelet_denoise(image: np.ndarray, sigma: float = 3.0, wavelet: str = "db8",
                    levels: int = 4) -> np.ndarray:
    """
    Wavelet-domain Wiener denoiser from the Lukas-Fridrich-Goljan PRNU work.

    The local variance of each detail coefficient is estimated over several
    window sizes and combined; coefficients are then shrunk toward zero
    assuming additive noise of variance ``sigma**2``.
    """
    image = np.asarray(image, dtype=np.float64)
    # cap decomposition depth to what the image size supports (avoids boundary
    # warnings / degenerate sub-bands on small crops)
    max_lvl = pywt.dwtn_max_level(image.shape, wavelet)
    levels = max(1, min(levels, max_lvl))
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    cA, details = coeffs[0], coeffs[1:]

    var0 = float(sigma) ** 2
    new_details = []
    for (cH, cV, cD) in details:
        new_details.append(tuple(_wiener_shrink(c, var0) for c in (cH, cV, cD)))

    out = pywt.waverec2([cA] + new_details, wavelet)
    # waverec2 can pad by one pixel; crop back.
    return out[: image.shape[0], : image.shape[1]]


def _wiener_shrink(coeff: np.ndarray, var0: float) -> np.ndarray:
    """Local-variance Wiener shrinkage of one detail sub-band."""
    sq = coeff ** 2
    est = None
    for win in (3, 5, 7, 9):
        local = uniform_filter(sq, size=win, mode="mirror")
        local = np.maximum(local - var0, 0.0)
        est = local if est is None else np.minimum(est, local)
    return coeff * est / (est + var0)


def noise_residual(image: np.ndarray, sigma: float = 3.0, **kw) -> np.ndarray:
    """W = I - denoise(I): the per-image noise residual (contains PRNU)."""
    gray = to_gray(image)
    return gray - wavelet_denoise(gray, sigma=sigma, **kw)


# --------------------------------------------------------------------------- #
# Fingerprint estimation
# --------------------------------------------------------------------------- #
def estimate_fingerprint(images, sigma: float = 3.0, weighted: bool = True,
                         post: bool = True) -> np.ndarray:
    """
    Estimate a camera reference fingerprint K from an iterable of images.

    With ``weighted=True`` the maximum-likelihood estimator K = sum(W_i I_i) /
    sum(I_i^2) is used (Chen et al. 2008); otherwise a plain residual average.
    ``post`` applies zero-mean + DFT-Wiener cleaning to remove shared,
    non-unique artefacts (JPEG grid, CFA interpolation, lens).
    """
    num = None
    den = None
    acc = None
    n = 0
    for img in images:
        gray = to_gray(img)
        w = gray - wavelet_denoise(gray, sigma=sigma)
        if weighted:
            if num is None:
                num = np.zeros_like(gray)
                den = np.zeros_like(gray)
            num += w * gray
            den += gray * gray
        else:
            acc = w if acc is None else acc + w
        n += 1
    if n == 0:
        raise ValueError("no images supplied for fingerprint estimation")

    k = num / (den + EPS) if weighted else acc / n
    if post:
        k = wiener_in_dft(zero_mean(k))
    return k


def zero_mean(x: np.ndarray) -> np.ndarray:
    """Remove row and column means (kills column/row FPN & linear pattern)."""
    x = x - x.mean(axis=0, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)
    return x


def wiener_in_dft(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Wiener-filter the magnitude spectrum to suppress periodic artefacts."""
    f = np.fft.fft2(x)
    mag = np.abs(f)
    if sigma is None:
        sigma = np.sqrt(np.mean(mag ** 2))
    mag_clean = np.maximum(mag - sigma, 0.0) * mag / (mag ** 2 + EPS) * mag
    # equivalent local-Wiener on spectrum magnitude
    local = uniform_filter(mag ** 2, size=3, mode="wrap")
    attn = np.maximum(local - sigma ** 2, 0.0) / (local + EPS)
    f_clean = f * attn
    out = np.real(np.fft.ifft2(f_clean))
    return out


# --------------------------------------------------------------------------- #
# Detection statistics
# --------------------------------------------------------------------------- #
def crosscorr_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full circular 2-D cross-correlation via FFT (a star b)."""
    a = a - a.mean()
    b = b - b.mean()
    F = np.fft.fft2(a)
    G = np.fft.fft2(b)
    return np.real(np.fft.ifft2(F * np.conj(G)))


def ncc(residual: np.ndarray, fingerprint_times_image: np.ndarray) -> float:
    """
    Normalised cross-correlation at zero shift between a test residual W and
    the predicted PRNU term (K * I). Inputs should already be the two signals
    to correlate; both are flattened.
    """
    x = (residual - residual.mean()).ravel()
    y = (fingerprint_times_image - fingerprint_times_image.mean()).ravel()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + EPS
    return float(np.sum(x * y) / denom)


def pce(residual: np.ndarray, fingerprint_times_image: np.ndarray,
        neigh: int = 11) -> float:
    """
    Peak-to-Correlation Energy (Goljan 2009): the standard, shift-robust
    attribution statistic. A peak at (0,0) well above the correlation-plane
    energy indicates the residual carries this camera's PRNU.

    Returned value is signed (sign of the peak) so that forced/forged peaks
    and genuine peaks are reported on the same scale.
    """
    cc = crosscorr_2d(residual, fingerprint_times_image)
    h, w = cc.shape
    peak_idx = (0, 0)
    peak = cc[peak_idx]

    # energy of the plane excluding a small neighbourhood around the peak
    mask = np.ones_like(cc, dtype=bool)
    r = neigh // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            mask[(peak_idx[0] + dy) % h, (peak_idx[1] + dx) % w] = False
    energy = np.sum(cc[mask] ** 2) / mask.sum()
    val = (peak ** 2) / (energy + EPS)
    return float(np.sign(peak) * val)
