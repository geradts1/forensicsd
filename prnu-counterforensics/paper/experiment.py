"""
Monte-Carlo evaluation for the paper.

For each trial:
  * pick a target camera A and estimate K_A from a pool of its images
    (the "harvested" pool the attacker would use);
  * take a carrier image from a different camera B and inject K_A, tuned so
    the forgery reaches a fixed PCE decision threshold (the attack succeeds
    against a PCE-only examiner);
  * score the forgery with the triangle test against the A-pool, calibrated
    on a cohort of unrelated cameras -> z_forged;
  * score an innocent B image the same way -> z_innocent.

Outputs:
  results.json           summary statistics
  figures/zscores.png    histogram of triangle-test z-scores (forged vs innocent)
  figures/roc.png        ROC of the triangle test as a copy-forgery detector
  figures/pce.png        PCE before/after injection vs the decision threshold
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prnucf import estimate_fingerprint, noise_residual, pce, inject_fingerprint
from prnucf.triangle import triangle_test

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SHAPE = (128, 128)
DECISION_PCE = 60.0
POOL = 10            # images in the attacker's harvested pool
COHORT = 20          # unrelated reference cameras (one image each)
N_TRIALS = 100
PRNU_STRENGTH = 0.03


def make_camera(rng):
    return rng.standard_normal(SHAPE) * PRNU_STRENGTH


def make_image(prnu, rng):
    base = rng.uniform(40, 210)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]] / max(SHAPE)
    scene = (base + 30 * np.sin(6 * xx + rng.uniform(0, 6))
             + 25 * np.cos(5 * yy + rng.uniform(0, 6))
             + 15 * np.sin(11 * xx * yy + rng.uniform(0, 6)))
    return np.clip(scene * (1.0 + prnu) + rng.standard_normal(SHAPE) * 2.0, 0, 255)


def pce_vs(img, K):
    return pce(noise_residual(img), K * img)


def run():
    rng = np.random.default_rng(20261112)
    z_forged, z_innocent = [], []
    pce_before, pce_after, alphas, psnrs = [], [], [], []

    for t in range(N_TRIALS):
        prnu_A = make_camera(rng)
        prnu_B = make_camera(rng)

        pool_imgs = [make_image(prnu_A, rng) for _ in range(POOL)]
        K_A = estimate_fingerprint(pool_imgs)

        cohort_imgs = [make_image(make_camera(rng), rng) for _ in range(COHORT)]

        carrier = make_image(prnu_B, rng)
        before = pce_vs(carrier, K_A)
        res = inject_fingerprint(
            carrier, K_A, match_target_pce=DECISION_PCE,
            detector=lambda im: pce_vs(im, K_A),
        )
        forged = res.image
        after = pce_vs(forged, K_A)

        innocent = make_image(prnu_B, rng)

        rf = triangle_test(forged, pool_imgs, reference_images=cohort_imgs)
        ri = triangle_test(innocent, pool_imgs, reference_images=cohort_imgs)

        z_forged.append(rf.z_score)
        z_innocent.append(ri.z_score)
        pce_before.append(before)
        pce_after.append(after)
        alphas.append(res.alpha)
        psnrs.append(res.psnr)

        if (t + 1) % 10 == 0:
            print(f"  trial {t+1}/{N_TRIALS}")

    z_forged = np.array(z_forged)
    z_innocent = np.array(z_innocent)

    # ---- ROC for the triangle test (forged = positive) ----
    scores = np.concatenate([z_forged, z_innocent])
    labels = np.concatenate([np.ones_like(z_forged), np.zeros_like(z_innocent)])
    order = np.argsort(-scores)
    sl = labels[order]
    tpr = np.cumsum(sl) / max(sl.sum(), 1)
    fpr = np.cumsum(1 - sl) / max((1 - sl).sum(), 1)
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr))

    # detection rate at a fixed operating point z>3
    det_at_z3 = float(np.mean(z_forged > 3.0))
    fp_at_z3 = float(np.mean(z_innocent > 3.0))

    # threshold-calibrated operating points: pick the z cutoff that yields a
    # target FPR on the innocent set, report the detection rate (TPR) there.
    def tpr_at_fpr(target_fpr):
        thr = np.quantile(z_innocent, 1.0 - target_fpr)
        return float(np.mean(z_forged > thr)), float(thr)
    tpr5, thr5 = tpr_at_fpr(0.05)
    tpr1, thr1 = tpr_at_fpr(0.01)

    summary = dict(
        n_trials=N_TRIALS, shape=list(SHAPE), decision_pce=DECISION_PCE,
        pool=POOL, cohort=COHORT, prnu_strength=PRNU_STRENGTH,
        pce_before_mean=float(np.mean(pce_before)),
        pce_after_mean=float(np.mean(pce_after)),
        alpha_mean=float(np.mean(alphas)), alpha_std=float(np.std(alphas)),
        psnr_mean=float(np.mean(psnrs)), psnr_std=float(np.std(psnrs)),
        z_forged_mean=float(z_forged.mean()), z_forged_std=float(z_forged.std()),
        z_forged_min=float(z_forged.min()),
        z_innocent_mean=float(z_innocent.mean()),
        z_innocent_std=float(z_innocent.std()),
        z_innocent_max=float(z_innocent.max()),
        auc=auc, detection_rate_at_z3=det_at_z3, false_positive_at_z3=fp_at_z3,
        tpr_at_fpr5=tpr5, z_thresh_fpr5=thr5,
        tpr_at_fpr1=tpr1, z_thresh_fpr1=thr1,
    )
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- styling ----
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 150})
    C_F, C_I = "#c0392b", "#2c6fbb"

    # z-score histogram
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    lo = min(z_innocent.min(), -4); hi = z_forged.max()
    bins_i = np.linspace(lo, max(z_innocent.max() + 2, 6), 25)
    ax.hist(z_innocent, bins=bins_i, color=C_I, alpha=0.8,
            label="innocent (un-forged)")
    ax.hist(z_forged, bins=25, color=C_F, alpha=0.8, label="fingerprint-copy forgery")
    ax.axvline(3.0, color="k", ls="--", lw=1, label="decision threshold z=3")
    ax.set_xlabel("triangle-test z-score vs cohort")
    ax.set_ylabel("count")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "zscores.png"))
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot(fpr, tpr, color=C_F, lw=2, label=f"triangle test (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "roc.png"))
    plt.close(fig)

    # PCE before/after
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    idx = np.arange(len(pce_before))
    ax.scatter(idx, pce_before, s=14, color=C_I, label="carrier before injection")
    ax.scatter(idx, pce_after, s=14, color=C_F, label="after injection")
    ax.axhline(DECISION_PCE, color="k", ls="--", lw=1,
               label=f"PCE decision threshold = {DECISION_PCE:.0f}")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("trial"); ax.set_ylabel("PCE vs target fingerprint")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pce.png"))
    plt.close(fig)

    print("figures written to", FIG)


if __name__ == "__main__":
    run()
