# prnucf — PRNU counter-forensics evaluation toolkit

A research toolkit for testing how robust **PRNU source-camera attribution**
is against the published **fingerprint-copy attack**, and for **detecting**
that attack with the **triangle test**.

It deliberately ships the attack together with its detector. The attack is
here so you can confirm your own attribution pipeline's robustness and show
the detector fires; it implements only the already-published additive model
and nothing tuned to evade detection. **Read [`ETHICS.md`](ETHICS.md) before
using it** — in short: your own or consented data only; never to fabricate
source-camera evidence against a person.

## Why

A high PCE between a questioned image and a camera fingerprint is a strong
source-attribution signal — but Goljan, Fridrich & Chen showed in 2011 that an
adversary can *copy* a fingerprint estimated from a person's public images
into an unrelated photo and reach a passing PCE. The same paper gave the
defence: the **triangle test**, which detects that the questioned image shares
a fingerprint statistically tied to a specific image pool rather than one it
acquired by being taken with the camera. This package implements both, plus
the standard PRNU estimation and PCE scoring, so the whole attack-and-detect
loop can be run and validated.

## Install

```bash
pip install -e '.[dev]'      # package + Pillow + pytest
# or just the library:
pip install -e .
```

Requires Python ≥ 3.10, numpy, scipy, PyWavelets (Pillow for image I/O / CLI).

## Quick start (Python)

```python
import numpy as np
from prnucf import estimate_fingerprint, noise_residual, pce, triangle_test

# 1. estimate a camera fingerprint from a set of its images
K = estimate_fingerprint(my_camera_images)        # iterable of HxW arrays

# 2. attribution: did this camera take the questioned image?
score = pce(noise_residual(questioned), K * questioned)

# 3. robustness check: is the questioned image a fingerprint-copy forgery?
res = triangle_test(questioned, candidate_source_pool,
                    reference_images=unrelated_cohort)
print(res.summary())          # z-score vs cohort, p-value, flag
```

## Quick start (CLI)

```bash
# estimate a fingerprint from a folder of a camera's images
prnucf fingerprint ./cam_A_images -o camA.npy

# attribution score for a questioned image
prnucf pce questioned.jpg camA.npy --threshold 60

# triangle test: questioned image vs a suspected source pool,
# calibrated against a folder of known-unrelated images
prnucf triangle questioned.jpg ./suspected_pool --cohort ./other_cameras
```

The `inject` subcommand (fingerprint-copy) exists for robustness evaluation on
your own data and prints a reminder to that effect. See `ETHICS.md`.

## End-to-end demo

```bash
python examples/demo_attack_and_detect.py
```

Runs on synthetic cameras (no external data) and shows the forgery clearing a
PCE decision threshold while the triangle test flags it (z ≈ +53) and leaves a
genuinely innocent image unflagged (z ≈ −0.6).

## What's in the box

| module | contents |
|---|---|
| `prnucf.core` | wavelet denoising residual, ML fingerprint estimation, zero-mean + DFT-Wiener cleaning, NCC, **PCE** |
| `prnucf.counterforensics` | additive **fingerprint-copy** injection (with optional PCE-targeting), PRNU **removal** / anonymisation |
| `prnucf.triangle` | **triangle test** detector with reference-cohort calibration |
| `prnucf.cli` | `fingerprint` / `pce` / `inject` / `triangle` commands |
| `examples/` | end-to-end synthetic demo |
| `tests/` | pytest suite (attribution separation, injection, detection, removal) |

## Method notes

- **Fingerprint estimation** follows the maximum-likelihood estimator
  K = Σ(Wᵢ Iᵢ)/Σ(Iᵢ²) with zero-mean + spectral Wiener post-processing to
  suppress non-unique (demosaicing / JPEG) artefacts.
- **PCE** is computed as a signed peak-to-correlation-energy ratio over the
  circular correlation plane, the shift-robust statistic standard in the
  field.
- **Triangle test** scores the mean over triples of
  c(Y,Jᵢ)·c(Y,Jⱼ)·c(Jᵢ,Jⱼ) and compares the questioned image against a
  reference cohort (median/MAD by default), so the false-positive rate is
  controlled by a population of known-innocent images — as in casework.

## Where this sits in the current arms race

The additive fingerprint-copy attack this toolkit implements is the *classical*
form (2011). It has since moved to deep generative models, and the defence
side is still catching up — which is the gap a toolkit like this is meant to
let examiners probe.

- **Learned fingerprint injection.** *SpoC* (Cozzolino, Thies, Rössler,
  Nießner & Verdoliva, CVPR 2021 Media-Forensics workshop) uses a GAN to graft
  a target camera's traces into an image and fools CNN camera-model
  identifiers and GAN-image detectors, including ones unseen at training time.
  This is the feature-level successor to the additive paste: no explicit
  fingerprint arithmetic, so the statistical link the triangle test keys on is
  weaker. Later work extends the idea to **PRNU transfer attacks spanning both
  physical and AI-generated images**, which is the setting your lab's deepfake
  work overlaps with most directly.
- **Camera-trace erasing / anonymisation** has likewise gone neural (hybrid-
  loss trace-erasers; conditional adversarial camera-model anonymisation by
  Andrews, Zhang & Griffin), beyond the inverse-flat-field and denoising
  removal included here.
- **Detection is the acknowledged weak point.** The 2024 anti-forensics survey
  (arXiv:2402.02089) notes that implanted sensor fingerprints still leave
  traces a detector can exploit, but that counter-anti-forensics methods remain
  narrow and technique-specific, and that explainability of learned detectors
  is a live admissibility concern for court. In other words: the triangle test
  remains a sound, explainable detector for the *additive* attack, and the open
  problem is an equally defensible detector for the *learned* injection — a
  natural next step for this package and for a paper.

Pointers (verify and pin exact venues/DOIs before citing in a publication):
[SpoC (arXiv)](https://arxiv.org/pdf/1911.12069) ·
[SpoC (CVPRW 2021 PDF)](https://openaccess.thecvf.com/content/CVPR2021W/WMF/papers/Cozzolino_SpoC_Spoofing_Camera_Fingerprints_CVPRW_2021_paper.pdf) ·
[PRNU transfer attacks on physical & AI-generated images (Springer, 2025)](https://link.springer.com/chapter/10.1007/978-3-032-08124-7_17) ·
[Recent advances in image/video forensics, anti-forensics & counter-anti-forensics — survey (arXiv:2402.02089)](https://arxiv.org/pdf/2402.02089) ·
[Individual source camera identification — survey (Sensors, 2025)](https://www.mdpi.com/1424-8220/25/10/3027) ·
[Robust PRNU-based attribution with CNNs (J. Supercomputing, 2024)](https://dl.acm.org/doi/10.1007/s11227-024-06579-8)

## References

- J. Lukáš, J. Fridrich, M. Goljan. *Digital camera identification from sensor
  pattern noise.* IEEE TIFS, 2006.
- M. Chen, J. Fridrich, M. Goljan, J. Lukáš. *Determining image origin and
  integrity using sensor noise.* IEEE TIFS, 2008.
- M. Goljan, J. Fridrich, M. Chen. *Defending against fingerprint-copy attack
  in sensor-based camera identification.* IEEE TIFS, 2011.
- D. Cozzolino, J. Thies, A. Rössler, M. Nießner, L. Verdoliva. *SpoC: Spoofing
  Camera Fingerprints.* CVPR Workshops (Media Forensics), 2021.

## Status

Research code, v0.1.0. Validated on synthetic sensors in the test suite; run
your own validation on real cameras before relying on thresholds in casework.

## License

MIT — see [`LICENSE`](LICENSE).
