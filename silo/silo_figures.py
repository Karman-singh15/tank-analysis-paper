"""Silo / tank wall geometry from a TLS point cloud -> paper figures.

Usage: python silo/silo_figures.py [points.txt] [out_dir]
Defaults: silo/data/data.txt -> silo/figures/
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reused import load_point_cloud, correct_yz_alignment, fit_shell_circle_trial9

SLICE_MM = 250.0       # height of each slice used for circle fits
RNG = np.random.default_rng(0)  # fixed seed: figures are reproducible


def analyse(P):
    """P: (n, 3) points in mm, silo axis roughly along Z.

    Returns per-slice fits, the fitted (tilted) axis, and per-point
    cylindrical coordinates (h, theta, rho) and deviation d = rho - R.
    """
    # Mill code expects the cylinder axis along X: (X, Y, Z) -> (Z, X, Y).
    pts = P[:, [2, 0, 1]].astype(float)

    shift = pts[:, 1:].mean(0)
    q = pts.copy()
    q[:, 1:] -= shift
    q2 = correct_yz_alignment(q)
    shift = shift + (q - q2)[0, 1:]
    q = q2

    edges = np.arange(q[:, 0].min(), q[:, 0].max() + SLICE_MM, SLICE_MM)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.count_nonzero((q[:, 0] >= a) & (q[:, 0] < b)) < 500:
            continue
        cy, cz, R = fit_shell_circle_trial9(q, a, b)
        rows.append((0.5 * (a + b), cy + shift[0], cz + shift[1], R))
    slices = np.array(rows)

    # Straight axis through the slice centres -> lean of the shell.
    kx = np.polyfit(slices[:, 0], slices[:, 1], 1)
    ky = np.polyfit(slices[:, 0], slices[:, 2], 1)

    h = pts[:, 0]
    dx = pts[:, 1] - np.polyval(kx, h)
    dy = pts[:, 2] - np.polyval(ky, h)
    # ponytail: radius measured in the horizontal plane, not perpendicular to
    # the tilted axis; error ~ R*tilt^2/2 (< 0.1 mm at 0.2 deg). Fine for tanks.
    rho = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dy, dx)) % 360.0
    R = float(np.median(slices[:, 3]))
    return dict(slices=slices, kx=kx, ky=ky, h=h, theta=theta, rho=rho, d=rho - R, R=R)


def out_of_roundness(theta, rho, R, step=5):
    """(Dmax - Dmin) / Dnom from median radius in `step`-degree bins."""
    idx = (theta // step).astype(int)
    n = 360 // step
    r = np.array([np.median(rho[idx == i]) if np.any(idx == i) else np.nan for i in range(n)])
    D = r[: n // 2] + r[n // 2:]
    return (np.nanmax(D) - np.nanmin(D)) / (2 * R)


def grid_mean(theta, h, v, t_step=1.0, h_step=100.0):
    t_edges = np.arange(0, 360 + t_step, t_step)
    h_edges = np.arange(h.min(), h.max() + h_step, h_step)
    s, _, _ = np.histogram2d(theta, h, bins=(t_edges, h_edges), weights=v)
    c, _, _ = np.histogram2d(theta, h, bins=(t_edges, h_edges))
    with np.errstate(invalid="ignore"):
        return (s / c).T, c.T, t_edges, h_edges


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(P, a, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    h0 = a["h"].min()
    H = (a["h"] - h0) / 1000.0            # height above lowest point, m
    lim = float(np.percentile(np.abs(a["d"]), 99))
    sub = RNG.choice(len(P), min(80_000, len(P)), replace=False)
    Pm = P[sub] / 1000.0

    # 1. Raw point cloud: plan and two elevations, coloured by the hidden axis.
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axi, (i, j, lab) in zip(ax, [(0, 1, "Plan (X-Y)"), (0, 2, "Elevation (X-Z)"), (1, 2, "Elevation (Y-Z)")]):
        k3 = 3 - i - j
        # Elevations: only the near half, so the far wall doesn't show through.
        o = np.ones(len(Pm), bool) if k3 == 2 else Pm[:, k3] < np.median(Pm[:, k3])
        sc = axi.scatter(Pm[o, i], Pm[o, j], c=Pm[o, k3], s=0.2, cmap="viridis", rasterized=True)
        axi.set(title=lab if k3 == 2 else lab + ", near half", xlabel="XYZ"[i] + " (m)",
                ylabel="XYZ"[j] + " (m)", aspect="equal")
        fig.colorbar(sc, ax=axi, label="XYZ"[k3] + " (m)", shrink=0.7)
    save(fig, out, "01_raw_views")

    # 2. Unrolled deviation map (main figure).
    m, _, te, he = grid_mean(a["theta"], a["h"], a["d"])
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(m, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, 360, 0, (he[-1] - h0) / 1000.0], interpolation="nearest")
    ax.set(xlabel="Azimuth θ (°, from +X towards +Y)", ylabel="Height (m)",
           title=f"Wall deviation from fitted cylinder (R = {a['R'] / 1000:.3f} m)", xticks=range(0, 361, 45))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Deviation (mm), + outward")
    save(fig, out, "02_deviation_map")

    # 3. Cross-sections at four heights, deviation exaggerated.
    k = 0.15 * a["R"] / lim               # max deviation drawn as 15% of R
    t = np.radians(np.arange(0, 360, 2) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    for axi, f in zip(ax.flat, [0.1, 0.35, 0.65, 0.9]):
        hc = h0 + f * np.ptp(a["h"])
        s = np.abs(a["h"] - hc) < SLICE_MM / 2
        idx = (a["theta"][s] // 2).astype(int)
        dm = np.array([np.median(a["d"][s][idx == i]) if np.any(idx == i) else np.nan for i in range(180)])
        R = a["R"] / 1000.0
        axi.plot(R * np.cos(t), R * np.sin(t), "k--", lw=0.8, label="Fitted circle")
        r = R + k * dm / 1000.0
        axi.plot(r * np.cos(t), r * np.sin(t), "C3", lw=1.2, label=f"Measured (×{k:.0f})")
        axi.set(title=f"h = {(hc - h0) / 1000:.1f} m", aspect="equal", xlabel="m", ylabel="m")
    ax[0, 0].legend(loc="lower left", fontsize=7)
    save(fig, out, "03_cross_sections")

    # 8. Same cross-sections at true scale: the scanned points, no exaggeration.
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    for axi, f in zip(ax.flat, [0.1, 0.35, 0.65, 0.9]):
        hc = h0 + f * np.ptp(a["h"])
        s = np.abs(a["h"] - hc) < SLICE_MM / 2
        th, r = np.radians(a["theta"][s]), a["rho"][s] / 1000.0
        R = a["R"] / 1000.0
        axi.plot(R * np.cos(t), R * np.sin(t), "k--", lw=0.8, label="Fitted circle")
        axi.scatter(r * np.cos(th), r * np.sin(th), s=0.3, c="C3", rasterized=True, label="Scanned points")
        axi.set(title=f"h = {(hc - h0) / 1000:.1f} m", aspect="equal", xlabel="m", ylabel="m")
    ax[0, 0].legend(loc="lower left", fontsize=7, markerscale=10)
    save(fig, out, "08_cross_sections_true_scale")

    # 4. Vertical profiles: lean, radius, out-of-roundness.
    sl = a["slices"]
    hs = (sl[:, 0] - h0) / 1000.0
    off = np.hypot(sl[:, 1] - sl[0, 1], sl[:, 2] - sl[0, 2])
    lean = np.hypot(np.polyval(a["kx"], sl[:, 0]) - np.polyval(a["kx"], sl[0, 0]),
                    np.polyval(a["ky"], sl[:, 0]) - np.polyval(a["ky"], sl[0, 0]))
    ur = []
    for hc in sl[:, 0]:
        s = np.abs(a["h"] - hc) < SLICE_MM / 2
        ur.append(out_of_roundness(a["theta"][s], a["rho"][s], a["R"]))
    ur = np.array(ur)
    fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    ax[0].plot(off, hs, "o", ms=3, label="Slice centre")
    ax[0].plot(lean, hs, "k-", label="Fitted axis")
    ax[0].set(xlabel="Horizontal offset from bottom (mm)", ylabel="Height (m)", title="Verticality")
    ax[0].legend(fontsize=7)
    ax[1].plot(sl[:, 3] - a["R"], hs, "o-", ms=3)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set(xlabel="Slice radius − R (mm)", title="Radius")
    ax[2].plot(ur * 100, hs, "o-", ms=3)
    ax[2].set(xlabel="Out-of-roundness (Dmax−Dmin)/D (%)", title="Out-of-roundness")
    save(fig, out, "04_vertical_profiles")

    # 5. Deviation histogram.
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(a["d"], bins=200, range=(-2 * lim, 2 * lim), color="C0")
    for p in (5, 50, 95):
        ax.axvline(np.percentile(a["d"], p), color="k", lw=0.8, ls=":" if p != 50 else "-")
    ax.set(xlabel="Deviation (mm)", ylabel="Points", title="Distribution of wall deviation")
    save(fig, out, "05_deviation_histogram")

    # 6. 3D view coloured by deviation.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(Pm[:, 0], Pm[:, 1], Pm[:, 2], c=a["d"][sub], cmap="RdBu_r", vmin=-lim, vmax=lim,
                    s=0.3, rasterized=True)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_box_aspect(np.ptp(Pm, axis=0))
    fig.colorbar(sc, ax=ax, label="Deviation (mm)", shrink=0.6, pad=0.12)
    save(fig, out, "06_3d_deviation")

    # 7. Coverage: points per 1° x 0.1 m cell.
    _, c, _, he = grid_mean(a["theta"], a["h"], a["d"])
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(np.where(c > 0, c, np.nan), origin="lower", aspect="auto", cmap="magma", norm="log",
                   extent=[0, 360, 0, (he[-1] - h0) / 1000.0], interpolation="nearest")
    ax.set(xlabel="Azimuth θ (°)", ylabel="Height (m)", title="Scan coverage (empty = no data)",
           xticks=range(0, 361, 45))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Points per 1° × 0.1 m cell")
    save(fig, out, "07_coverage")

    # Tables for the paper.
    np.savetxt(out / "slices.csv",
               np.column_stack([hs, sl[:, 1], sl[:, 2], sl[:, 3], off, ur * 100]),
               delimiter=",", fmt="%.3f",
               header="height_m,centre_x_mm,centre_y_mm,radius_mm,offset_from_bottom_mm,out_of_roundness_pct",
               comments="")
    tilt = float(np.hypot(a["kx"][0], a["ky"][0]) * 1000)   # mm per m
    summary = {
        "points": int(len(P)),
        "radius_m": round(a["R"] / 1000, 4),
        "diameter_m": round(2 * a["R"] / 1000, 4),
        "height_m": round(float(np.ptp(a["h"])) / 1000, 3),
        "tilt_mm_per_m": round(tilt, 2),
        "lean_over_height_mm": round(tilt * float(np.ptp(a["h"])) / 1000, 1),
        "deviation_mm": {f"p{p}": round(float(np.percentile(a["d"], p)), 1) for p in (0.5, 5, 50, 95, 99.5)},
        "deviation_std_mm": round(float(a["d"].std()), 1),
        "max_out_of_roundness_pct": round(float(np.nanmax(ur)) * 100, 3),
        "coverage_pct_1deg_0.1m": round(float((c > 0).mean()) * 100, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "data.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "figures"
    P = load_point_cloud(src)
    print(json.dumps(make_figures(P, analyse(P), out), indent=2))
