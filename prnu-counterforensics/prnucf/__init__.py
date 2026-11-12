"""
prnucf -- PRNU counter-forensics evaluation toolkit.

A research toolkit for evaluating the robustness of sensor-noise (PRNU)
source-camera attribution against the published fingerprint-copy attack, and
for detecting that attack with the triangle test.

It pairs an attack with its defence on purpose. The attack is included only
so examiners can test their own pipelines and confirm the detector fires; it
implements the already-published additive model and nothing tuned to evade
detection. See ETHICS.md before using it on anything but your own or
consented data.
"""
from .core import (
    estimate_fingerprint,
    noise_residual,
    wavelet_denoise,
    ncc,
    pce,
    to_gray,
    zero_mean,
    wiener_in_dft,
)
from .counterforensics import inject_fingerprint, remove_fingerprint, InjectionResult
from .triangle import triangle_test, TriangleResult, pairwise_corr

__version__ = "0.1.0"

__all__ = [
    "estimate_fingerprint", "noise_residual", "wavelet_denoise",
    "ncc", "pce", "to_gray", "zero_mean", "wiener_in_dft",
    "inject_fingerprint", "remove_fingerprint", "InjectionResult",
    "triangle_test", "TriangleResult", "pairwise_corr",
    "__version__",
]
