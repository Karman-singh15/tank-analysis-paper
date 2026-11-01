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
    return np.nanmedian(np.stack([np.roll(w, j) for j in range(-(k // 2), k // 2 + 1)]), 0) if k > 1 else w


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
    return summary


if __name__ == "__main__":
    pts_file = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "sections_full.npy"
    axis_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "central_axis.xlsx"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "results"
    wb = openpyxl.load_workbook(axis_file, read_only=True)
    axis_pts = np.array([r[:3] for r in list(wb.active.iter_rows(values_only=True))[2:]], float)
    print(json.dumps(run(np.load(pts_file), axis_pts, out), indent=2))
