import numpy as np

from prnucf import estimate_fingerprint, inject_fingerprint, triangle_test
from prnucf.triangle import pairwise_corr, mean_triple_stat


def test_mean_triple_stat_matches_bruteforce():
    rng = np.random.default_rng(0)
    n = 6
    cq = rng.standard_normal(n)
    cij = rng.standard_normal((n, n))
    cij = (cij + cij.T) / 2
    np.fill_diagonal(cij, 0)
    # brute force
    from itertools import combinations
    vals = [cq[i] * cq[j] * cij[i, j] for i, j in combinations(range(n), 2)]
    assert np.isclose(mean_triple_stat(cq, cij), np.mean(vals))


def test_triangle_flags_forgery_clears_innocent(cameras):
    pool = [cameras["img_a"]() for _ in range(10)]
    K = estimate_fingerprint(pool)
    cohort = [cameras["img_other"]() for _ in range(15)]

    carrier = cameras["img_b"]()
    forged = inject_fingerprint(
        carrier, K, match_target_pce=60.0,
        detector=lambda im: __import__("prnucf").pce(
            __import__("prnucf").noise_residual(im), K * im),
    ).image

    res_forged = triangle_test(forged, pool, reference_images=cohort)
    res_innocent = triangle_test(cameras["img_b"](), pool, reference_images=cohort)

    assert res_forged.flagged
    assert res_forged.z_score > 5.0
    assert not res_innocent.flagged


def test_triangle_requires_three_sources(cameras):
    import pytest
    pool = [cameras["img_a"]() for _ in range(2)]
    with pytest.raises(ValueError):
        triangle_test(cameras["img_b"](), pool)


def test_leave_one_out_cohort_runs(cameras):
    # without an explicit cohort the test still runs (weaker separation)
    pool = [cameras["img_a"]() for _ in range(8)]
    res = triangle_test(cameras["img_other"](), pool)
    assert res.cohort_size == len(pool)
