import numpy as np
import pytest

SHAPE = (192, 192)


def _prnu(rng, strength=0.03):
    return rng.standard_normal(SHAPE) * strength


def _image(prnu, rng):
    base = rng.uniform(40, 210)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]] / max(SHAPE)
    scene = (base + 30 * np.sin(6 * xx + rng.uniform(0, 6))
             + 25 * np.cos(5 * yy + rng.uniform(0, 6)))
    return np.clip(scene * (1.0 + prnu) + rng.standard_normal(SHAPE) * 2.0, 0, 255)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def cameras(rng):
    """Two cameras A (target) and B (attacker), with image generators."""
    prnu_a = _prnu(rng)
    prnu_b = _prnu(rng)
    return {
        "prnu_a": prnu_a,
        "prnu_b": prnu_b,
        "img_a": lambda: _image(prnu_a, rng),
        "img_b": lambda: _image(prnu_b, rng),
        "img_other": lambda: _image(_prnu(rng), rng),
        "shape": SHAPE,
    }
