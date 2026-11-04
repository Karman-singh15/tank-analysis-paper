"""How much data does the method need, and how accurate is it? -> tables + figures.

Usage: python silo/robustness_study.py [points.txt] [out_dir]
Defaults: silo/data/data.txt -> silo/figures/

Part 1, real tank (no truth available, so stability against the full-data result): thinning, added
noise, and a missing vertical strip of wall. 11_robustness_real.png/pdf, robustness_real.csv.
Part 2, synthetic tanks (known answers): sweep one factor at a time around a baseline tank.
12_accuracy_synthetic.png/pdf, accuracy_synthetic.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import silo_figures as sf
from reused import load_point_cloud

sf.MIN_SLICE_PTS = 120    # thinned clouds still need slices; fit_shell_circle_trial9 itself needs >= 100
REPEATS, SEEDS = 10, 3


def measure(P):
    """Headline numbers from one analysis run."""
    a = sf.analyse(P)
    sl = a["slices"]
    H = np.ptp(a["h"]) / 1000
    tilt = lambda i: np.array([np.polyfit(sl[:, 0], sl[:, i + j], 1)[0] for j in (0, 1)]) * 1000
    ur = [sf.out_of_roundness(a["theta"][s], a["rho"][s], a["R"])
          for s in (np.abs(a["h"] - hc) < sf.SLICE_MM / 2 for hc in sl[:, 0])]
    return dict(a=a, H=H, t=tilt(1), t_reused=tilt(4), R=a["R"], R_reused=float(np.median(sl[:, 8])), oor=100 * float(np.nanmedian(ur)),
                oor_max=100 * float(np.nanmax(ur)), std=float(a["d"].std()))


def real_tank(P, out):
    base = measure(P)
    rng = np.random.default_rng(1)
    th0 = base["a"]["theta"]
    tests = [("Points kept (%)", f, lambda f=f: P[rng.random(len(P)) < f / 100]) for f in (50, 25, 10, 5, 2)]
    tests += [("Added noise (mm)", n, lambda n=n: P + rng.normal(0, n, P.shape)) for n in (3, 5, 10, 20)]

    def strip(w):
        c = rng.uniform(0, 360)
        return P[np.abs((th0 - c + 180) % 360 - 180) > w / 2]
    tests += [("Missing wall strip (deg)", w, lambda w=w: strip(w)) for w in (15, 30, 60, 90)]

    rows = []
    for name, level, make in tests:
        r = [measure(make()) for _ in range(REPEATS)]
        lean = [np.hypot(*(x["t"] - base["t"])) * base["H"] for x in r]       # shift of the lean vector
        rows.append(dict(test=name, level=level, lean_shift_median_mm=np.median(lean), lean_shift_max_mm=np.max(lean),
                         radius_shift_max_mm=np.max([abs(x["R"] - base["R"]) for x in r]),
                         radius_shift_max_mill_fit_mm=np.max([abs(x["R_reused"] - base["R_reused"]) for x in r]),
                         oor_shift_max_pct=np.max([abs(x["oor"] - base["oor"]) for x in r])))
        print(rows[-1], flush=True)
    write(out / "robustness_real.csv", rows)

    names = list(dict.fromkeys(r["test"] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for axi, n in zip(ax, names):
        rr = [r for r in rows if r["test"] == n]
        x = [r["level"] for r in rr]
        axi.plot(x, [r["lean_shift_max_mm"] for r in rr], "o-", color="#eb6834", lw=2, ms=8, mec="white", label="Worst of 10")
        axi.plot(x, [r["lean_shift_median_mm"] for r in rr], "s-", color="#2a78d6", lw=2, ms=8, mec="white", label="Median of 10")
        axi.set(xlabel=n, ylabel="Change in lean vs full data (mm)", ylim=(0, None))
        axi.grid(color="0.9")
        axi.spines[["top", "right"]].set_visible(False)
    ax[0].invert_xaxis()
    ax[0].legend(frameon=False)
    fig.suptitle(f"Real tank: stability of the {np.hypot(*base['t']) * base['H']:.0f} mm lean result")
    sf.save(fig, out, "11_robustness_real")


BASE = dict(tilt=4.0, oval=30.0, grow=0.0, dent=-60.0, noise=3.0, uneven=0.5, hole=0.0)
SWEEPS = dict(tilt=(0, 2, 4, 8, 16), oval=(0, 15, 30, 60, 120), grow=(0, 15, 30, 60, 120), dent=(0, -30, -60, -120, -240),
              noise=(1, 3, 5, 10, 20), uneven=(0, 0.25, 0.5, 0.75, 0.9), hole=(0, 30, 60, 90, 120))
LABEL = dict(tilt="True tilt (mm/m)", oval="Ovality, same at all heights (mm)", grow="Lobes growing with height (mm at top)",
             dent="Dent depth (mm)", noise="Noise (mm)",
             uneven="Share of points in the dense side", hole="Missing wall strip (deg)")
R0, H0 = 11880.0, 11900.0


def tank(p, seed, n=400_000):
    """Known tank: tilt towards 40 deg, n = 2 ovality, optional n = 2 + 3 lobes growing from zero at the base (the
    real tank's ovality grows with height), one Gaussian dent at mid-height, uneven density, 1 cm rounding."""
    rng = np.random.default_rng(seed)
    k = int(n * p["uneven"])
    th = np.r_[rng.vonmises(0, 2, k) % (2 * np.pi), rng.uniform(0, 2 * np.pi, n - k)]
    z = rng.uniform(0, H0, n)
    # Dent at mid-height: its first-harmonic pull on the slices is symmetric, so it doesn't tilt the fitted axis.
    arc = R0 * np.angle(np.exp(1j * (th - np.pi / 2)))
    dev = p["oval"] * np.cos(2 * th) + p["grow"] * z / H0 * (np.cos(2 * th - 1) + np.cos(3 * th + 2)) / 2 + p["dent"] * np.exp(-(arc ** 2 + (z - H0 / 2) ** 2) / (2 * 600.0 ** 2))
    r = R0 + dev + rng.normal(0, p["noise"], n)
    e = p["tilt"] * z / 1000
    d = np.radians(40)
    P = np.c_[32000 + e * np.cos(d) + r * np.cos(th), 2000 + e * np.sin(d) + r * np.sin(th), z]
    if p["hole"]:
        P = P[np.abs(np.degrees(np.angle(np.exp(1j * (th - np.radians(200)))))) > p["hole"] / 2]
    return np.round(P, -1)


def synthetic(out):
    rows = []
    for factor, levels in SWEEPS.items():
        for lv in levels:
            p = dict(BASE, **{factor: lv})
            true = p["tilt"] * np.array([np.cos(np.radians(40)), np.sin(np.radians(40))])
            e = []
            for seed in range(SEEDS):
                m = measure(tank(p, seed))
                a = m["a"]
                near = (np.abs(a["theta"] - 90) < 0.5) & (np.abs(a["h"] - H0 / 2) < 100)
                lobes = p["grow"] / 2 * (np.cos(np.pi - 1) + np.cos(1.5 * np.pi + 2)) / 2     # at 90 deg, mid-height
                dent = np.median(a["d"][near]) + p["oval"] - lobes if near.any() else np.nan   # ovality is -oval at 90 deg
                e.append((np.hypot(*(m["t"] - true)) * H0 / 1000, np.hypot(*(m["t_reused"] - true)) * H0 / 1000,
                          m["R"] - R0, m["R_reused"] - R0, m["oor"] - 100 * 4 * p["oval"] / (2 * R0), dent - p["dent"]))
            e = np.array(e)
            rows.append(dict(factor=factor, level=lv, lean_err_mm=np.median(e[:, 0]), lean_err_mill_fit_mm=np.median(e[:, 1]),
                             radius_err_mm=np.median(e[:, 2]), radius_err_mill_fit_mm=np.median(e[:, 3]),
                             oor_err_pct=np.nan if f_grow(factor) else np.median(e[:, 4]), dent_err_mm=np.nanmedian(e[:, 5])))
            print(rows[-1], flush=True)
    write(out / "accuracy_synthetic.csv", rows)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    ax[1, 3].set_visible(False)
    for axi, f in zip(ax.flat, SWEEPS):
        rr = [r for r in rows if r["factor"] == f]
        x = [abs(r["level"]) for r in rr]
        axi.plot(x, [r["lean_err_mill_fit_mm"] for r in rr], "s-", color="#eb6834", lw=2, ms=8, mec="white", label="Mill fit as reused")
        axi.plot(x, [r["lean_err_mm"] for r in rr], "o-", color="#2a78d6", lw=2, ms=8, mec="white", label="Area centroid (used)")
        axi.set(xlabel=LABEL[f], ylabel="Lean error (mm)", ylim=(0, None))
        axi.grid(color="0.9")
        axi.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Synthetic tanks: lean error when one factor changes (others at baseline; median of 3 seeds)")
    sf.save(fig, out, "12_accuracy_synthetic")


def f_grow(factor):
    """Growing lobes change the ovality with height, so there's no single true value to compare with."""
    return factor == "grow"


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in rows)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "data.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "figures"
    plt.rcParams.update({"font.size": 9})
    synthetic(out)
    real_tank(load_point_cloud(src), out)
