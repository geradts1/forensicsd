"""
Triangle test: detecting a fingerprint-copy forgery.

This is the DEFENCE. If an attacker estimated a camera fingerprint K from a
set of the victim's images J_1..J_N and pasted a scaled copy of K into a
forged image Y, then Y necessarily carries a fraction of the *specific
residuals* of those J_i. The triangle test detects that shared component.

Intuition (Goljan, Fridrich & Chen, IEEE TIFS 2011)
---------------------------------------------------
Take the questioned image Y and the pool J_1..J_N the attacker is suspected
to have harvested. For a triple (Y, J_i, J_j) form

    T_ij = c(Y, J_i) * c(Y, J_j) * c(J_i, J_j)

where c() is the normalised correlation of noise residuals.

* If Y was NOT built from the pool, c(Y, J_i) only reflects whatever
  fingerprint Y and J_i genuinely share, and is uncorrelated with the pool's
  internal structure c(J_i, J_j). The mean of T over triples sits at the
  level of a reference cohort of unrelated images.
* If Y WAS forged by copying a fingerprint estimated from the pool, then
  c(Y, J_i) carries an extra component proportional to J_i's contribution to
  K, which is itself tied to c(J_i, J_j). Mean-T is then pushed *above* the
  reference cohort.

So the decision is made by comparing the questioned image's mean-T against a
REFERENCE COHORT of images of known-innocent origin (other cameras, or other
images that did not come from the pool). That cohort is how real examiners
calibrate, and it is what keeps the false-positive rate controlled: an
innocent image lands inside the cohort; a forgery lands in its upper tail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .core import noise_residual, EPS

__all__ = ["TriangleResult", "triangle_test", "pairwise_corr", "mean_triple_stat"]


def _centered_flat(x: np.ndarray) -> np.ndarray:
    x = x.ravel()
    return x - x.mean()


def pairwise_corr(wa: np.ndarray, wb: np.ndarray) -> float:
    """Normalised correlation of two residuals."""
    a = _centered_flat(wa)
    b = _centered_flat(wb)
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + EPS))


def mean_triple_stat(cq: np.ndarray, cij: np.ndarray) -> float:
    """
    Mean over all triples {i<j} of  c(Y,i)*c(Y,j)*c(i,j).

    cq  : length-n vector of c(Y, J_i)
    cij : n x n matrix of c(J_i, J_j)
    """
    n = len(cq)
    outer = np.outer(cq, cq)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(outer[iu] * cij[iu]))


def _pool_matrix(residuals):
    n = len(residuals)
    cij = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        v = pairwise_corr(residuals[i], residuals[j])
        cij[i, j] = cij[j, i] = v
    return cij


@dataclass
class TriangleResult:
    n_sources: int
    questioned_stat: float
    cohort_mean: float
    cohort_std: float
    z_score: float
    p_value: float
    flagged: bool
    cohort_size: int = 0
    per_source_corr: dict = field(default_factory=dict)

    def summary(self) -> str:
        verdict = ("FLAGGED: consistent with a copied fingerprint"
                   if self.flagged else
                   "not flagged: consistent with genuine / unrelated origin")
        return (f"Triangle test over {self.n_sources} candidate source images\n"
                f"  questioned statistic : {self.questioned_stat:+.3e}\n"
                f"  reference cohort     : mean {self.cohort_mean:+.3e} "
                f"sd {self.cohort_std:.3e}  (n={self.cohort_size})\n"
                f"  z-score vs cohort    : {self.z_score:+.2f}\n"
                f"  p-value              : {self.p_value:.4f}\n"
                f"  => {verdict}")


def triangle_test(questioned, source_images, reference_images=None,
                  sigma: float = 3.0, z_threshold: float = 3.0,
                  sigma_robust: bool = True) -> TriangleResult:
    """
    Run the triangle test for one questioned image against a candidate source
    pool, calibrated against a reference cohort of known-innocent images.

    Parameters
    ----------
    questioned : the image Y under examination.
    source_images : iterable of images J_i suspected to be the attacker's
        fingerprint-estimation pool (>= 3; more is stronger).
    reference_images : iterable of images of KNOWN-INNOCENT origin w.r.t. the
        pool (ideally from other cameras). Each is scored the same way as Y to
        build the null distribution of the statistic. If omitted, a
        leave-one-out cohort is built from the pool itself: each J_k is scored
        against the remaining pool. Leave-one-out is weaker (the pool members
        do share the real camera fingerprint) and should be treated as a lower
        bound on separation; supply a real cohort for casework.
    z_threshold : flag if the questioned statistic exceeds the cohort by this
        many (robust) standard deviations.
    sigma_robust : use median / MAD instead of mean / std for the cohort
        location and scale (resistant to a contaminated cohort).

    Returns
    -------
    TriangleResult
    """
    wq = noise_residual(questioned, sigma=sigma)
    srcs = []
    for img in source_images:
        w = noise_residual(img, sigma=sigma)
        if w.shape != wq.shape:
            raise ValueError("all images must share dimensions (align/crop first)")
        srcs.append(w)
    n = len(srcs)
    if n < 3:
        raise ValueError("triangle test needs at least 3 candidate source images")

    cij = _pool_matrix(srcs)

    cq = np.array([pairwise_corr(wq, w) for w in srcs])
    q_stat = mean_triple_stat(cq, cij)

    # ---- reference cohort ----
    cohort_stats = []
    if reference_images is not None:
        for img in reference_images:
            wr = noise_residual(img, sigma=sigma)
            if wr.shape != wq.shape:
                raise ValueError("reference images must match questioned dims")
            cr = np.array([pairwise_corr(wr, w) for w in srcs])
            cohort_stats.append(mean_triple_stat(cr, cij))
    else:
        # leave-one-out over the pool
        for k in range(n):
            idx = [i for i in range(n) if i != k]
            sub = [srcs[i] for i in idx]
            sub_cij = cij[np.ix_(idx, idx)]
            crk = np.array([pairwise_corr(srcs[k], srcs[i]) for i in idx])
            cohort_stats.append(mean_triple_stat(crk, sub_cij))

    cohort = np.array(cohort_stats, dtype=float)
    if sigma_robust:
        loc = float(np.median(cohort))
        scale = 1.4826 * float(np.median(np.abs(cohort - loc)))
    else:
        loc = float(np.mean(cohort))
        scale = float(np.std(cohort))
    scale = max(scale, 1e-9)

    z = (q_stat - loc) / scale
    # one-sided normal tail
    from math import erfc, sqrt
    p = 0.5 * erfc(z / sqrt(2.0))

    return TriangleResult(
        n_sources=n,
        questioned_stat=q_stat,
        cohort_mean=loc,
        cohort_std=scale,
        z_score=z,
        p_value=p,
        flagged=z > z_threshold,
        cohort_size=len(cohort),
        per_source_corr={i: float(cq[i]) for i in range(n)},
    )
