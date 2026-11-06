"""Tunnel over/under-excavation per section, validated against Zhang et al. 2022 Table 1.

Run: python tunnel/tunnel_validation.py [points.npy] [central_axis.xlsx] [out_dir]
Defaults: tunnel/data/sections_full.npy (from extract_sections.py), tunnel/data/central_axis.xlsx -> tunnel/results/

Points are x y z [intensity] in metres. Deviation d = rho - R about the published
design axis, as in the paper; the reused mill fit gives each section's actual centre.
"""
import sys, json, warnings
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "silo"))
from reused import fit_shell_circle_trial9  # REUSED from integ.py via silo/reused.py

warnings.filterwarnings("ignore")

SECTIONS_M = np.arange(0, 170, 5)   # 34 sections, as in Table 1
HALF_THICK = 0.25                   # m, section slab = D +/- 0.25 m
ARC_BELOW = 50                      # deg below the springline still evaluated; bottom 80 deg (floor/track bed) excluded
SHELL_BAND = 0.05                   # m, wall = points within 5 cm of the outermost return in each 1 deg bin
DESPIKE_DEG = 5                     # deg, angular median filter: drops 1-2 deg spikes (stray points in the floor gutters)
# The text never states R. Digitised from the "Designed" curve of the paper's Figs. 2-6 (upper arch,
# 3.202-3.209 m across the five figures, +/-2 cm from line width); independent of Table 1.
DESIGN_R = 3.205                    # m

# Zhang, Zhang, Xu (2022) Data in Brief 41:107993, Table 1. None = "-" (not reported, i.e. no under-excavation).
PAPER = dict(
    w1=[61.8,100,100,100,100,76.7,60,100,91.7,100,100,100,93.3,95,73.3,38.33,45,85,90,88.3,75,85,81.7,65,93.3,70,76.7,60,72.5,95,100,93.3,67.5,95],
    w2=[1.3,None,None,None,None,4,37,None,3.3,None,None,None,None,None,5,20.7,20,None,1.3,None,0.7,None,None,25.3,0.7,1.3,None,27.2,11.8,None,None,None,7.3,None],
    M1=[22,25,39,53,44,46,26,27,30,50,64,69,54,56,55,25,22,21,22,37,38,39,33,43,40,30,34,46,46,44,58,64,70,46],
    M2=[12,None,None,None,None,1,7,None,8,None,None,None,None,None,3,7,4,None,10,None,3,None,None,22,6,2,None,11,8,None,None,None,6,None],
)


def axis_frame(points, axis_pts):
    """(s, u, v): chainage along the straight design axis, horizontal and vertical offset from it."""
    d = axis_pts[-1] - axis_pts[0]
    L = np.linalg.norm(d[:2])
    t = np.r_[d[:2] / L, 0]
    n = np.array([-t[1], t[0], 0])
    q = points[:, :3] - axis_pts[0]
    s = q @ t
    return s, q @ n, points[:, 2] - (axis_pts[0, 2] + d[2] * s / L)


def despike(w, k=DESPIKE_DEG):
    """Circular running median over k bins, NaN-aware; real overbreak spans many degrees, spikes don't."""
    return np.nanmedian(np.stack([np.roll(w, j, -1) for j in range(-(k // 2), k // 2 + 1)]), 0) if k > 1 else w


def wall_profile(u, v):
    """Raw wall radius per 1 deg bin (NaN where empty); 90 deg = crown, 270 deg = invert.

    ADAPTED from noise_removal.py _compute_radial_mask (radial percentile threshold), inverted:
    the mill code protects the inside, here the outer shell is kept and everything in front of it
    (pipes, brackets, people) is dropped.
    """
    rho = np.hypot(u, v)
    b = (np.degrees(np.arctan2(v, u)) % 360).astype(int)
    w = np.full(360, np.nan)
    for k in np.unique(b):
        r = rho[b == k]
        r = r[r > np.percentile(r, 98) - SHELL_BAND]
        if len(r) >= 3:
            w[k] = np.median(r)
    # ponytail: a bin where an object hides the wall completely reads as under-excavation; fine at 1 deg bins here.
    return w


def arc_stats(w, R, below=ARC_BELOW):
    """M1/M2 (cm) and w1/w2 (% of evaluated bins with data) over the arc -below .. 180+below deg."""
    d = (w[np.arange(-below, 181 + below) % 360] - R) * 100
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return dict(M1=np.nan, M2=np.nan, w1=np.nan, w2=np.nan)
    return dict(M1=d.max(), M2=max(0.0, -d.min()), w1=100 * np.mean(d > 0), w2=100 * np.mean(d < 0))


def mae(ours, paper):
    """Mean |ours - paper|; a "-" in the paper counts as 0 (none)."""
    p = np.array([0 if x is None else x for x in paper], float)
    return float(np.nanmean(np.abs(np.asarray(ours, float) - p)))


def design_radius(profiles, below=ARC_BELOW):
    """R that would best explain the paper's M1 column; cross-check on DESIGN_R only, not used for results."""
    arc = np.arange(-below, 181 + below) % 360
    return float(np.nanmedian(np.nanmax(profiles[:, arc], 1) - np.array(PAPER["M1"]) / 100))


def sensitivity(raw, param, values):
    """MAE (cm) of [M1, M2] vs the paper when one setting changes (R fixed at DESIGN_R)."""
    res = {}
    for x in values:
        k, below = (x, ARC_BELOW) if param == "despike" else (DESPIKE_DEG, x)
        P = np.array([despike(w, k) for w in raw])
        st = [arc_stats(w, DESIGN_R, below) for w in P]
        res[x] = [round(mae([s[m] for s in st], PAPER[m]), 2) for m in ("M1", "M2")]
    return res


def percentile_profile(u, v, q):
    """Wall radius per 1 deg bin as a plain percentile, no wall filter."""
    rho = np.hypot(u, v)
    b = (np.degrees(np.arctan2(v, u)) % 360).astype(int)
    w = np.full(360, np.nan)
    for k in np.unique(b):
        if np.sum(b == k) >= 3:
            w[k] = np.percentile(rho[b == k], q)
    return w


def filter_variants(s, u, v):
    """Does dropping the wall filter close the M2 gap? The source paper's own extraction isn't published, so try the
    plain alternatives: slab half-thickness x percentile x despike on/off, scored like the main result."""
    rep = np.array([x is not None for x in PAPER["M2"]])
    p1, p2 = np.array(PAPER["M1"], float), np.array([x or 0 for x in PAPER["M2"]], float)
    res = []
    for half in (0.25, 0.05):
        secs = [np.abs(s - D) < half for D in SECTIONS_M]
        for q in (50, 25, 10):
            raw = [percentile_profile(u[m], v[m], q) for m in secs]
            for k in (DESPIKE_DEG, 1):
                st = [arc_stats(despike(w, k), DESIGN_R) for w in raw]
                m1, m2 = np.array([x["M1"] for x in st]), np.array([x["M2"] for x in st])
                res.append(dict(slab_half_m=half, percentile=q, despike_deg=k, detected=int((m2[rep] >= 1).sum()),
                                false_alarms=int((m2[~rep] >= 1).sum()), M2_bias_cm=round(float(np.mean(m2[rep] - p2[rep])), 1),
                                M1_mae_cm=round(float(np.mean(np.abs(m1 - p1))), 1)))
    return res


def run(points, axis_pts, out):
    out.mkdir(parents=True, exist_ok=True)
    s, u, v = axis_frame(points, axis_pts)
    pts_mm = np.c_[s, u, v] * 1000.0     # mill convention: axis along X, mm
    rows, profiles = [], []
    for D in SECTIONS_M:
        m = np.abs(s - D) < HALF_THICK
        cy, cz, r_fit = fit_shell_circle_trial9(pts_mm, (D - HALF_THICK) * 1000, (D + HALF_THICK) * 1000)
        profiles.append(wall_profile(u[m], v[m]))   # raw; despiked below
        rows.append(dict(D_m=int(D), n_points=int(m.sum()), centre_u_mm=cy, centre_v_mm=cz, fit_radius_mm=r_fit))
    raw = np.array(profiles)
    profiles = np.array([despike(w) for w in raw])
    R = DESIGN_R

    for r, w, *paper in zip(rows, profiles, *PAPER.values()):
        r.update(arc_stats(w, R))
        r.update({f"paper_{k}": p for k, p in zip(PAPER, paper)})
    keys = list(rows[0])
    with open(out / "sections.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else (f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")

    col = lambda k: [r[k] for r in rows]
    offs = np.hypot(col("centre_u_mm"), col("centre_v_mm"))
    m2, rep2 = np.array(col("M2")), np.array([x is not None for x in PAPER["M2"]])
    p2 = np.array([x or 0 for x in PAPER["M2"]], float)
    summary = dict(
        points=int(len(points)), design_radius_m=R, implied_radius_from_M1_m=round(design_radius(profiles), 4),
        arc_below_springline_deg=ARC_BELOW, despike_deg=DESPIKE_DEG,
        mae_cm=dict(M1=round(mae(col("M1"), PAPER["M1"]), 2), M2=round(mae(col("M2"), PAPER["M2"]), 2)),
        mae_pct=dict(w1=round(mae(col("w1"), PAPER["w1"]), 1), w2=round(mae(col("w2"), PAPER["w2"]), 1)),
        corr_M1=round(float(np.corrcoef(col("M1"), PAPER["M1"])[0, 1]), 3),
        # The all-section M2 MAE is flattered by the 19 sections with no under-excavation; these aren't.
        M2_where_paper_reports_it=dict(sections=int(rep2.sum()), mae_cm=round(float(np.mean(np.abs(m2[rep2] - p2[rep2]))), 2),
                                       bias_cm=round(float(np.mean(m2[rep2] - p2[rep2])), 2),
                                       detected=int((m2[rep2] >= 1).sum()), false_alarms=int((m2[~rep2] >= 1).sum())),
        fitted_centre_vs_design_axis_mm=dict(median=round(float(np.median(offs))), max=round(float(offs.max()))),
        arc_sensitivity_mae_cm_M1_M2=sensitivity(raw, "arc", range(40, 61, 5)),
        despike_sensitivity_mae_cm_M1_M2=sensitivity(raw, "despike", (1, 3, 5, 7, 9)),
        M2_without_wall_filter=filter_variants(s, u, v),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Wall envelope vs design circle at the five sections the paper plots (its Figs. 2-6).
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), layout="constrained")
    th = np.linspace(0, 2 * np.pi, 361)
    for ax, D in zip(axs, (30, 75, 80, 115, 135)):
        k = int(D // 5)
        m = np.abs(s - D) < HALF_THICK
        ax.scatter(u[m], v[m], s=0.2, c="0.7", label="all points")
        ang = np.radians(np.arange(360) + 0.5)
        ax.plot(profiles[k] * np.cos(ang), profiles[k] * np.sin(ang), "k.", ms=2, label="wall (kept)")
        ax.plot(R * np.cos(th), R * np.sin(th), "r", lw=0.8, label=f"design R = {R:.3f} m")
        for c in np.radians([-ARC_BELOW, 180 + ARC_BELOW]):   # evaluated arc limits
            ax.plot([0, 3.8 * np.cos(c)], [0, 3.8 * np.sin(c)], "b--", lw=0.6)
        r = rows[k]
        ax.set(aspect="equal", title=f"D = {D} m\nM1 {r['M1']:.0f} (paper {PAPER['M1'][k]}) | "
               f"M2 {r['M2']:.0f} (paper {PAPER['M2'][k] or 0}) cm", xlabel="u (m)")
    axs[0].set_ylabel("v (m)")
    axs[0].legend(loc="lower left", fontsize=7, markerscale=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sections_vs_design.{ext}", dpi=300)
    plt.close(fig)
    validation_figure(rows, out)
    density_figure(s, u, v, rows, out)
    return summary


OURS, THEIRS = "#2a78d6", "#eb6834"   # categorical slots 1-2 (dataviz reference palette, CVD-validated)


def validation_figure(rows, out):
    """Ours vs Table 1 for all 34 sections: along the tunnel (top) and 1:1 scatter (bottom)."""
    D = np.array([r["D_m"] for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for j, (k, name) in enumerate((("M1", "Max over-excavation M1"), ("M2", "Max under-excavation M2"))):
        ours = np.array([r[k] for r in rows])
        paper = np.array([0 if x is None else x for x in PAPER[k]], float)
        ax = axs[0, j]
        ax.plot(D, paper, "o-", color=THEIRS, lw=2, ms=8, mec="white", mew=1.5, label="Zhang et al. 2022, Table 1")
        ax.plot(D, ours, "s-", color=OURS, lw=2, ms=8, mec="white", mew=1.5, label="This work")
        ax.set(xlabel="Chainage D (m)", ylabel=f"{k} (cm)", title=name)
        ax.legend(frameon=False)
        ax = axs[1, j]
        lim = max(ours.max(), paper.max()) * 1.08
        ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--", label="1:1")
        ax.scatter(paper, ours, s=64, color=OURS, edgecolor="white", linewidth=1.5, zorder=3)
        txt = f"MAE {np.mean(np.abs(ours - paper)):.1f} cm, all 34 sections"
        if k == "M1":
            txt += f"\nr = {np.corrcoef(ours, paper)[0, 1]:.3f}"
        else:
            rep = np.array([x is not None for x in PAPER["M2"]])
            txt += (f"\n{rep.sum()} sections with reported M2: bias {np.mean(ours[rep] - paper[rep]):+.1f} cm,"
                    f"\ndetected {(ours[rep] >= 1).sum()}/{rep.sum()}, false alarms {(ours[~rep] >= 1).sum()}/{(~rep).sum()}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9)
        ax.set(xlim=(-1, lim), ylim=(-1, lim), aspect="equal", xlabel=f"Paper {k} (cm)", ylabel=f"This work {k} (cm)")
    for ax in axs.flat:
        ax.grid(color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_vs_paper.{ext}", dpi=300)
    plt.close(fig)


def density_figure(s, u, v, rows, out):
    """C5: error vs point density, by thinning every section and across the sections' natural range."""
    rng = np.random.default_rng(0)
    p1 = np.array(PAPER["M1"], float)
    rep = np.array([x is not None for x in PAPER["M2"]])
    p2 = np.array([x or 0 for x in PAPER["M2"]], float)
    secs = [np.where(np.abs(s - D) < HALF_THICK)[0] for D in SECTIONS_M]
    table = []
    for keep in (100, 50, 20, 10, 5, 2, 1):
        res = []
        for _ in range(1 if keep == 100 else 5):
            st = []
            for i in secs:
                j = i[rng.random(len(i)) < keep / 100]
                st.append(arc_stats(despike(wall_profile(u[j], v[j])), DESIGN_R))
            m1, m2 = np.array([x["M1"] for x in st]), np.array([x["M2"] for x in st])
            res.append((np.nanmean(np.abs(m1 - p1)), np.nanmean(m2[rep] - p2[rep]), int(np.isnan(m1).sum())))
        table.append((keep, *np.mean(res, 0)))
        print(f"keep {keep}%: M1 MAE {table[-1][1]:.2f}, M2 bias {table[-1][2]:+.2f}, empty sections {table[-1][3]:.1f}", flush=True)
    T = np.array(table)
    np.savetxt(out / "density_robustness.csv", T, delimiter=",", fmt="%.2f", comments="",
               header="points_kept_pct,M1_mae_cm,M2_bias_reported_sections_cm,empty_sections")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax[0].plot(T[:, 0], T[:, 1], "o-", color=OURS, lw=2, ms=8, mec="white", label="M1 error (MAE)")
    ax[0].plot(T[:, 0], np.abs(T[:, 2]), "s-", color=THEIRS, lw=2, ms=8, mec="white", label="|M2 bias|, 15 reported sections")
    ax[0].set(xscale="log", xlabel="Points kept (%)", ylabel="cm", ylim=(0, None), title="Thinning every section")
    ax[0].invert_xaxis()
    ax[0].legend(frameon=False)
    n = np.array([r["n_points"] for r in rows], float)
    ax[1].scatter(n, np.abs(np.array([r["M1"] for r in rows]) - p1), s=64, color=OURS, edgecolor="white", linewidth=1.5)
    ax[1].set(xscale="log", xlabel="Points in the section", ylabel="|M1 error| (cm)", ylim=(0, None), title="Natural density range")
    for a_ in ax:
        a_.grid(color="0.9")
        a_.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"density_robustness.{ext}", dpi=300)
    plt.close(fig)


def unrolled_map(grid_file, out, R=DESIGN_R):
    """Deviation d = wall - R unrolled along the tunnel (chainage x angle), like the paper's Fig. 1."""
    g = np.load(grid_file)
    wall, edges = despike(g["wall"]), g["s_edges"]           # despike along angle, as for the sections
    ang = np.arange(-ARC_BELOW, 181 + ARC_BELOW)              # evaluated arc, springline-to-springline plus 50 deg
    d = (wall[:, ang % 360] - R) * 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("under_over", ["#0d366b", "#5598e7", "#f0efec", "#e66767", "#8f1d1d"])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(14, 4.8), layout="constrained")
    im = ax.pcolormesh(edges, np.r_[ang - 0.5, ang[-1] + 0.5], d.T, cmap=cmap, vmin=-50, vmax=50, shading="flat",
                       rasterized=True)
    ax.vlines(SECTIONS_M, ang[0] - 0.5, ang[-1] + 0.5, color="0.2", lw=0.4, alpha=0.4)
    ax.set_yticks([-50, 0, 90, 180, 230], ["-50", "0 (right\nspringline)", "90 (crown)", "180 (left\nspringline)", "230"])
    ax.set(xlabel="Chainage along the design axis (m)", ylabel="Angle (deg)", xlim=(-2, 172),
           title=f"Unrolled deviation from the design profile (R = {R} m); floor zone 230-310 deg not shown; "
                 "thin lines = the 34 Table 1 sections")
    cb = fig.colorbar(im, ax=ax, extend="both", pad=0.01)
    cb.set_label("d = wall radius - R (cm)\n< 0 under-excavation, > 0 over-excavation")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"unrolled_deviation_map.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    pts_file = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "sections_full.npy"
    axis_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "central_axis.xlsx"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "results"
    wb = openpyxl.load_workbook(axis_file, read_only=True)
    axis_pts = np.array([r[:3] for r in list(wb.active.iter_rows(values_only=True))[2:]], float)
    print(json.dumps(run(np.load(pts_file), axis_pts, out), indent=2))
    grid = pts_file.parent / "wall_grid.npz"
    if grid.exists():
        unrolled_map(grid, out)
