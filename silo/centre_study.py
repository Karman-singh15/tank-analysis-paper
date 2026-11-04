"""Which centre estimate recovers a tank's true axis? Semi-synthetic study -> table + figure.

Usage: python silo/centre_study.py [points.txt] [out_dir]
Defaults: silo/data/data.txt -> silo/figures/  (centre_study.csv, 10_centre_study.png/pdf)

Test tanks: the real scan's deviation field (1 deg x 100 mm) with its n = 1 harmonic removed per height
(that term is indistinguishable from an axis shift, so the true axis is unambiguous), wrapped on a
cylinder with a known axis: tilt 0 / 4 mm/m x directions 0 / 120 / 250 deg, 3 mm noise, 1 cm rounding.
Three scenarios: perfectly round at the real scan's point positions, deformed with uniform sampling,
deformed at the real scan's point positions. Takes a few minutes.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from reused import load_point_cloud, correct_yz_alignment, azimuthally_balanced_shell_points
from silo_figures import analyse, grid_mean, area_centroid, lsq_centre, save, SLICE_MM


def balance_forced(yz):
    """azimuthally_balanced_shell_points minus its 'good scan -> do nothing' bypass (reused.py:118)."""
    th, r = np.arctan2(yz[:, 1], yz[:, 0]), np.hypot(yz[:, 0], yz[:, 1])
    bins = np.linspace(-np.pi, np.pi, max(24, int(np.sqrt(len(th)))) + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rb, pb = r[(th >= lo) & (th < hi)], yz[(th >= lo) & (th < hi)]
        if len(rb) >= 5 and rb.std() > 0:
            out.append(pb[np.abs(rb - np.percentile(rb, 95)) < rb.std()])
    return np.vstack(out)


def mill_fit(q, a, b, balance=True, loss="soft_l1"):
    """fit_shell_circle_trial9 (reused.py:150) with its two ingredients switchable; same shell mask and start."""
    pts = q[(q[:, 0] >= a) & (q[:, 0] <= b)]
    r = np.hypot(pts[:, 1], pts[:, 2])
    Rc = np.median(r)
    yz = pts[(r >= 0.82 * Rc) & (r <= 1.05 * Rc)][:, 1:]
    yz = {True: azimuthally_balanced_shell_points, False: lambda x: x, "forced": balance_forced}[balance](yz)
    f = lambda p: np.hypot(yz[:, 0] - p[0], yz[:, 1] - p[1]) - p[2]
    return least_squares(f, [0, 0, Rc], loss=loss, f_scale=5.0).x[:2]


METHODS = {   # name -> centre(q, a, b, slice_yz, start)
    "Mill fit as reused (balancing + robust loss)": lambda q, a, b, yz, c0: mill_fit(q, a, b),
    "Mill fit, robust loss off": lambda q, a, b, yz, c0: mill_fit(q, a, b, loss="linear"),
    "Mill fit, balancing forced on, robust loss off": lambda q, a, b, yz, c0: mill_fit(q, a, b, "forced", "linear"),
    "Plain least squares": lambda q, a, b, yz, c0: lsq_centre(yz, c0, np.median(np.hypot(*yz.T))),
    "Area centroid (used)": lambda q, a, b, yz, c0: area_centroid(yz, c0)[0],
}


def tilt_vectors(P):
    """Axis tilt (mm/m, x and y) per method, slicing exactly as analyse() does."""
    q = P[:, [2, 0, 1]].astype(float)
    q[:, 1:] -= q[:, 1:].mean(0)
    q = correct_yz_alignment(q)
    edges = np.arange(q[:, 0].min(), q[:, 0].max() + SLICE_MM, SLICE_MM)
    hs, C = [], {k: [] for k in METHODS}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (q[:, 0] >= a) & (q[:, 0] < b)
        if m.sum() < 500:
            continue
        hs.append(0.5 * (a + b))
        c0 = mill_fit(q, a, b)
        for k, f in METHODS.items():
            C[k].append(f(q, a, b, q[m, 1:], c0))
    return {k: np.polyfit(hs, np.array(v), 1)[0] * 1000 for k, v in C.items()}


def deformation_field(a):
    """Real deviation field per 1 deg x 100 mm, gaps filled along theta, n = 1 harmonic removed."""
    m, _, _, he = grid_mean(a["theta"], a["h"], a["d"])
    t = np.radians(np.arange(360) + 0.5)
    A = np.c_[np.ones(360), np.cos(t), np.sin(t)]
    F = np.empty_like(m)
    for i, row in enumerate(m):
        ok = ~np.isnan(row)
        r = np.interp(np.arange(360), np.arange(360)[ok], row[ok], period=360)
        c = np.linalg.lstsq(A, r, rcond=None)[0]
        F[i] = r - c[1] * np.cos(t) - c[2] * np.sin(t)
    return F, he


def run(P, out):
    a = analyse(P)
    F, he = deformation_field(a)
    H = np.ptp(a["h"]) / 1000
    rng = np.random.default_rng(3)
    scenarios = {"Round tank, real scan positions": (0 * F, True),
                 "Deformed, uniform sampling": (F, False),
                 "Deformed, real scan positions": (F, True)}
    err = {s: {k: [] for k in METHODS} for s in scenarios}
    for s, (field, real) in scenarios.items():
        for tilt in (0.0, 4.0):
            for deg in (0, 120, 250):
                th, h = (a["theta"], a["h"]) if real else (rng.uniform(0, 360, len(P)), rng.uniform(a["h"].min(), a["h"].max(), len(P)))
                r = a["R"] + field[np.clip(np.searchsorted(he, h) - 1, 0, len(F) - 1), th.astype(int) % 360] + rng.normal(0, 3, len(h))
                e, d = tilt * (h - a["h"].min()) / 1000, np.radians(deg)
                T = np.round(np.c_[30000 + e * np.cos(d) + r * np.cos(np.radians(th)),
                                   2000 + e * np.sin(d) + r * np.sin(np.radians(th)), h], -1)
                true = tilt * np.array([np.cos(d), np.sin(d)])
                for k, v in tilt_vectors(T).items():
                    err[s][k].append(np.hypot(*(v - true)) * H)
    real = {k: np.hypot(*v) * H for k, v in tilt_vectors(P).items()}

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "centre_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{s}: {q}_mm" for s in scenarios for q in ("median", "max")] + ["real tank lean_mm"])
        for k in METHODS:
            w.writerow([k] + [f"{fn(err[s][k]):.1f}" for s in scenarios for fn in (np.median, np.max)] + [f"{real[k]:.1f}"])

    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    y = np.arange(len(METHODS))[::-1]
    for (s, col, mk), dy in zip(zip(scenarios, ("#1baf7a", "#eb6834", "#2a78d6"), ("^", "s", "o")), (0.22, 0, -0.22)):
        ax.scatter([np.median(err[s][k]) for k in METHODS], y + dy, s=64, color=col, marker=mk,
                   edgecolor="white", linewidth=1.5, label=s, zorder=3)
    ax.set_yticks(y, [f"{k}\n(real tank: lean {real[k]:.0f} mm)" for k in METHODS], fontsize=8)
    ax.set(xlabel=f"Lean error over {H:.1f} m vs the known axis (mm, median of 6 test tanks)",
           title="Centre estimate vs true axis on test tanks built from this scan")
    ax.grid(axis="x", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, out, "10_centre_study")
    return err, real


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "data.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "figures"
    run(load_point_cloud(src), out)
    print((out / "centre_study.csv").read_text())
