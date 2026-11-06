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

from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from reused import load_point_cloud, correct_yz_alignment, fit_shell_circle_trial9

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tunnel"))
from tunnel_validation import wall_profile, despike

SLICE_MM = 250.0       # height of each slice used for circle fits
RNG = np.random.default_rng(0)  # fixed seed: figures are reproducible
MIN_SLICE_PTS = 500    # slices with fewer points are skipped
# Doubled wall: residuals here have two peaks ~25 mm apart (-20, +5), so two scans disagree; see scan_quality().
DOUBLED_AZ, DOUBLED_MIN_H, DOUBLED_SPLIT = (125, 145), 3.3, -7.5   # deg, m above lowest point, mm


def lsq_centre(yz, c0, R0):
    """Plain least-squares circle through all slice points (no balancing, no robust loss)."""
    f = lambda p: np.hypot(yz[:, 0] - p[0], yz[:, 1] - p[1]) - p[2]
    return least_squares(f, [c0[0], c0[1], R0]).x[:2]


def direction_radii(yz, c, step=2):
    """Median wall radius per `step`-degree direction about c; scan gaps bridged along the wall, not by a chord."""
    tt = np.radians(np.arange(0, 360, step) + step / 2)
    r = np.hypot(yz[:, 0] - c[0], yz[:, 1] - c[1])
    b = ((np.degrees(np.arctan2(yz[:, 1] - c[1], yz[:, 0] - c[0])) % 360) // step).astype(int)
    rr = np.array([np.median(r[b == i]) if np.any(b == i) else np.nan for i in range(len(tt))])
    ok = ~np.isnan(rr)
    return tt, np.interp(tt, tt[ok], rr[ok], period=2 * np.pi)


def tunnel_wall_radii(yz, c, step=2):
    """direction_radii with the tunnel's exact wall rule (filter, 1 deg bins, 5 deg despike); `step` is ignored."""
    tt = np.radians(np.arange(360) + 0.5)
    w = despike(wall_profile((yz[:, 0] - c[0]) / 1000.0, (yz[:, 1] - c[1]) / 1000.0)) * 1000.0
    ok = ~np.isnan(w)
    return tt, np.interp(tt, tt[ok], w[ok], period=2 * np.pi)


def area_centroid(yz, c0, step=2, it=3, radii=direction_radii):
    """Centroid of the polygon of median wall radius per `step`-degree bin, re-centred `it` times.

    Every direction counts once, however many points it has, so uneven scan density doesn't pull the centre.
    Returns the centre and the mean wall radius over all directions.
    """
    c = np.asarray(c0, float)
    for _ in range(it):
        tt, rr = radii(yz, c, step)
        x, y = c[0] + rr * np.cos(tt), c[1] + rr * np.sin(tt)
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        A = cr.sum() / 2
        c = np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])
    return c, float(rr.mean())
    # ponytail: gaps are filled by linear interpolation of radius; a dent hidden inside a gap is invisible.


def analyse(P, radii=direction_radii):
    """P: (n, 3) points in mm, silo axis roughly along Z.

    Returns per-slice fits, the fitted (tilted) axis, and per-point
    cylindrical coordinates (h, theta, rho) and deviation d = rho - R.

    slices columns: h, centre x, centre y, mean wall radius (area centroid; these define the axis and R),
    reused-fit centre x, y, plain-LSQ centre x, y, reused-fit radius. The area centroid is primary: on a
    semi-synthetic tank (this scan's deformation and point positions, known axis) its lean error was
    0.6 mm vs 11 mm (plain LSQ) and 22 mm (reused fit); see centre_study.py.
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
        if np.count_nonzero((q[:, 0] >= a) & (q[:, 0] < b)) < MIN_SLICE_PTS:
            continue

        m = (q[:, 0] >= a) & (q[:, 0] < b)
        cy, cz, R = fit_shell_circle_trial9(q, a, b)
        ly, lz = lsq_centre(q[m, 1:], (cy, cz), R)
        (ay, az), Rw = area_centroid(q[m, 1:], (cy, cz), radii=radii)
        rows.append((0.5 * (a + b), ay, az, Rw, cy, cz, ly, lz, R))
    slices = np.array(rows)
    slices[:, [1, 4, 6]] += shift[0]
    slices[:, [2, 5, 7]] += shift[1]

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


def harmonics(m, Hc, bands=((0, 3), (3, 6), (6, 9), (9, 12)), n_max=6):
    """Amplitude (mm) of each wall harmonic, from the mean deviation per direction in each height band (m)."""
    i, t = np.arange(360), np.radians(np.arange(360) + 0.5)
    res = {}
    for lo, hi in bands:
        r = np.nanmean(m[(Hc >= lo) & (Hc < hi)], 0)
        ok = ~np.isnan(r)
        r = np.interp(i, i[ok], r[ok], period=360)
        res[f"{lo}-{hi} m"] = {f"n{n}": round(float(2 * abs(np.mean(r * np.exp(-1j * n * t)))), 1) for n in range(1, n_max + 1)}
    return res


def first_harmonic_left(P, a):
    """n = 1 amplitude (mm) of each slice's equal-weight wall radii about each method's own centre.

    The area centroid zeroes this to first order, which is why it matches the test tanks' "no n = 1" truth.
    """
    sl = a["slices"]
    amp = {k: [] for k in ("area_centroid", "plain_lsq", "mill_fit")}
    for row in sl:
        yz = P[np.abs(a["h"] - row[0]) < SLICE_MM / 2, :2]
        for k, i in zip(amp, (1, 6, 4)):
            tt, rr = direction_radii(yz, row[i:i + 2])
            amp[k].append(2 * abs(np.mean(rr * np.exp(-1j * tt))))
    return {k: dict(median=round(float(np.median(v)), 2), max=round(float(np.max(v)), 2)) for k, v in amp.items()}


def scan_quality(P, a, m, c, he, out):
    """Can the scan itself be trusted? Within-cell spread map, doubled-wall patch, and what that patch does to the results."""
    h0 = a["h"].min()
    H = (a["h"] - h0) / 1000.0
    sq, _, _, _ = grid_mean(a["theta"], a["h"], a["d"] ** 2)
    sd = np.where(c >= 4, np.sqrt(np.maximum(sq - m ** 2, 0)), np.nan)
    # Residual from a smoothed wall, so the shape itself isn't counted as spread.
    smooth = median_filter(np.where(np.isnan(m), np.nanmedian(m), m), size=(11, 7), mode="wrap")
    res = a["d"] - smooth[np.clip(np.searchsorted(he, a["h"]) - 1, 0, len(smooth) - 1), a["theta"].astype(int) % 360]
    patch = (a["theta"] >= DOUBLED_AZ[0]) & (a["theta"] < DOUBLED_AZ[1]) & (H >= DOUBLED_MIN_H)
    near = (a["theta"] >= 150) & (a["theta"] < 180) & (H >= DOUBLED_MIN_H)

    def result(b):
        tilt = float(np.hypot(b["kx"][0], b["ky"][0]))
        return dict(tilt_mm_per_m=round(tilt * 1000, 2), lean_mm=round(tilt * float(np.ptp(b["h"])), 2), radius_m=round(b["R"] / 1000, 4))

    Hc = (he[:-1] - h0) / 1000.0
    in_patch = (Hc >= DOUBLED_MIN_H)[:, None] & (np.arange(360) >= DOUBLED_AZ[0]) & (np.arange(360) < DOUBLED_AZ[1])
    col_dev, col_n = np.nanmean(m, 0), c.sum(0)
    ok = ~np.isnan(col_dev) & (col_n > 0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4), width_ratios=(2, 1), layout="constrained")
    im = ax[0].imshow(sd, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=20,
                      extent=[0, 360, 0, (he[-1] - h0) / 1000.0], interpolation="nearest")
    ax[0].set(xlabel="Azimuth θ (°)", ylabel="Height (m)", title="Spread of points within a wall cell", xticks=range(0, 361, 45))
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label="Standard deviation per 1° × 0.1 m cell (mm)", extend="max")
    bins = np.arange(-50, 51, 5)
    ax[1].hist(res[near], bins=bins, density=True, color="#2a78d6", alpha=0.8, label=f"Neighbouring wall, 150–180° (sd {res[near].std():.0f} mm)")
    ax[1].hist(res[patch], bins=bins, density=True, histtype="step", lw=2, color="#eb6834",
               label=f"Doubled patch, {DOUBLED_AZ[0]}–{DOUBLED_AZ[1]}° (sd {res[patch].std():.0f} mm)")
    ax[1].set(xlabel="Point minus smoothed wall (mm)", ylabel="Density", title="Two surfaces in the patch")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, out, "13_scan_quality")

    b = analyse(P, tunnel_wall_radii)   # does the tank result depend on which wall rule is used?
    shift = float(np.hypot(b["kx"][0] - a["kx"][0], b["ky"][0] - a["ky"][0]) * np.ptp(a["h"]))

    return dict(
        tunnel_wall_rule=dict(result(b), lean_vector_shift_mm=round(shift, 2)),
        within_cell_sd_mm=dict(whole_wall_median=round(float(np.nanmedian(sd)), 1), doubled_patch_median=round(float(np.nanmedian(sd[in_patch])), 1)),
        density_vs_deviation_r_by_azimuth=round(float(np.corrcoef(np.log(col_n[ok]), col_dev[ok])[0, 1]), 2),
        doubled_patch=dict(azimuth_deg=DOUBLED_AZ, min_height_m=DOUBLED_MIN_H, points=int(patch.sum()),
                           points_pct=round(100 * float(patch.mean()), 1),
                           inner_surface_only=result(analyse(P[~(patch & (res >= DOUBLED_SPLIT))])),
                           outer_surface_only=result(analyse(P[~(patch & (res < DOUBLED_SPLIT))])),
                           removed=result(analyse(P[~patch]))),
    )


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

    # 5. Deviation histogram. Each wall cell counts once, so densely scanned areas don't dominate.
    cells = m[~np.isnan(m)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(cells, bins=100, range=(-2 * lim, 2 * lim), color="C0")
    for p in (5, 50, 95):
        ax.axvline(np.percentile(cells, p), color="k", lw=0.8, ls=":" if p != 50 else "-")
    ax.set(xlabel="Deviation (mm)", ylabel="Wall cells (1° × 0.1 m)", title="Distribution of wall deviation")
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

    # 9. Centre definition: the axis from each method, and how far the reused fit drifts from the centroid.
    H = np.ptp(a["h"]) / 1000.0
    methods = (("Area centroid (used)", 1, "#2a78d6", "o"), ("Plain least squares", 6, "#eb6834", "s"),
               ("Reused mill fit", 4, "#1baf7a", "^"))
    tilts = {}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for name, i, col, mk in methods:
        kx_, ky_ = np.polyfit(sl[:, 0], sl[:, i], 1), np.polyfit(sl[:, 0], sl[:, i + 1], 1)
        tilts[name] = (float(np.hypot(kx_[0], ky_[0]) * 1000), float(np.degrees(np.arctan2(ky_[0], kx_[0]))))
        for j, (axi, k_) in enumerate(zip(ax[:2], (kx_, ky_))):
            axi.plot(sl[:, i + j] - np.polyval(k_, sl[0, 0]), hs, mk + "-", color=col, ms=4, lw=1.2,
                     label=f"{name}: lean {tilts[name][0] * H:.0f} mm")
    ax[0].set(xlabel="Centre X offset from own axis at bottom (mm)", ylabel="Height (m)", title="Axis, X")
    ax[1].set(xlabel="Centre Y offset from own axis at bottom (mm)", title="Axis, Y")
    ax[0].legend(fontsize=7, frameon=False)
    gap = np.hypot(sl[:, 4] - sl[:, 1], sl[:, 5] - sl[:, 2])
    ax[2].scatter(ur * 100, gap, s=36, color="#1baf7a", edgecolor="white", linewidth=1)
    ax[2].set(xlabel="Out-of-roundness of slice (%)", ylabel="Reused fit − area centroid (mm)",
              title=f"Centre disagreement vs ovality (r = {np.corrcoef(ur, gap)[0, 1]:.2f})")
    save(fig, out, "09_centre_methods")

    # Tables for the paper.
    np.savetxt(out / "slices.csv",
               np.column_stack([hs, sl[:, 1], sl[:, 2], sl[:, 3], off, ur * 100, sl[:, 4], sl[:, 5], sl[:, 6], sl[:, 7], sl[:, 8]]),
               delimiter=",", fmt="%.3f",
               header="height_m,centre_x_mm,centre_y_mm,radius_mm,offset_from_bottom_mm,out_of_roundness_pct,"
                      "reused_centre_x_mm,reused_centre_y_mm,lsq_centre_x_mm,lsq_centre_y_mm,reused_radius_mm",
               comments="")
    tilt = float(np.hypot(a["kx"][0], a["ky"][0]) * 1000)   # mm per m
    pct = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 1) for p in (0.5, 5, 50, 95, 99.5)}
    summary = {
        "points": int(len(P)),
        "radius_m": round(a["R"] / 1000, 4),
        "radius_reused_fit_m": round(float(np.median(a["slices"][:, 8])) / 1000, 4),
        "diameter_m": round(2 * a["R"] / 1000, 4),
        "height_m": round(float(np.ptp(a["h"])) / 1000, 3),
        "tilt_mm_per_m": round(tilt, 2),
        "lean_over_height_mm": round(tilt * float(np.ptp(a["h"])) / 1000, 1),
        "centre_method": "area centroid",
        "lean_by_centre_method_mm": {k: dict(tilt_mm_per_m=round(v[0], 2), lean_mm=round(v[0] * H, 1),
                                             direction_deg=round(v[1])) for k, v in tilts.items()},
        "deviation_mm": pct(cells),                 # per 1 deg x 0.1 m wall cell
        "deviation_std_mm": round(float(cells.std()), 1),
        "deviation_per_point_mm": pct(a["d"]),      # for comparison only: densely scanned areas dominate
        "deviation_per_point_std_mm": round(float(a["d"].std()), 1),
        "radius_slope_mm_per_m": round(float(np.polyfit(hs, sl[:, 3], 1)[0]), 2),
        "harmonic_amplitude_mm": harmonics(m, (he[:-1] - h0) / 1000.0),
        "first_harmonic_left_mm": first_harmonic_left(P, a),
        "max_out_of_roundness_pct": round(float(np.nanmax(ur)) * 100, 3),
        "coverage_pct_1deg_0.1m": round(float((c > 0).mean()) * 100, 1),
        "scan_quality": scan_quality(P, a, m, c, he, out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "data.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "figures"
    P = load_point_cloud(src)
    print(json.dumps(make_figures(P, analyse(P), out), indent=2))
