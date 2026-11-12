import numpy as np

from prnucf import estimate_fingerprint, noise_residual, pce, to_gray


def test_gray_passthrough_and_rgb():
    g = np.random.rand(10, 10) * 255
    assert np.allclose(to_gray(g), g)
    rgb = np.stack([g, g, g], axis=-1)
    assert np.allclose(to_gray(rgb), g, atol=1e-9)


def test_residual_shape(cameras):
    img = cameras["img_a"]()
    w = noise_residual(img)
    assert w.shape == cameras["shape"]
    # residual should be small relative to the image
    assert np.abs(w).mean() < np.abs(img).mean()


def test_pce_separates_same_vs_different_camera(cameras):
    imgs_a = [cameras["img_a"]() for _ in range(10)]
    K = estimate_fingerprint(imgs_a)

    test_a = cameras["img_a"]()
    test_b = cameras["img_b"]()

    pce_same = pce(noise_residual(test_a), K * test_a)
    pce_diff = pce(noise_residual(test_b), K * test_b)

    # same-camera PCE must dominate the cross-camera PCE by a wide margin
    assert pce_same > 50.0
    assert pce_diff < pce_same / 10.0
