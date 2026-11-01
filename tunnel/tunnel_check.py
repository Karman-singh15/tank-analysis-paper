"""Exploratory tunnel check: reused shell fit per section vs the paper's Table 1.

Run: python tunnel/tunnel_check.py [central_axis.xlsx] [sub20.npy]
Defaults: tunnel/data/central_axis.xlsx, tunnel/data/sub20.npy (every 20th point).
"""
import sys, warnings
from pathlib import Path

import numpy as np
import openpyxl

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "silo"))
from reused import fit_shell_circle_trial9  # REUSED from integ.py via silo/reused.py

warnings.filterwarnings("ignore")
axis_file = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "central_axis.xlsx"
pts_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "sub20.npy"

wb = openpyxl.load_workbook(axis_file, read_only=True)
A = np.array([r[:3] for r in list(wb.active.iter_rows(values_only=True))[2:]], float)
p = np.load(pts_file)
d = A[-1] - A[0]; L = np.linalg.norm(d[:2]); t = np.r_[d[:2] / L, 0]; n = np.array([-t[1], t[0], 0])
q = p[:, :3] - A[0]; s = q @ t; u = q @ n; v = p[:, 2] - (A[0, 2] + d[2] * s / L)
pts = np.c_[s, u, v] * 1000.0      # mill convention: axis along X, mm

# Zhang et al. 2022, Data in Brief 41:107993, Table 1: max over (M1) / under (M2) excavation, cm.
M1 = [22,25,39,53,44,46,26,27,30,50,64,69,54,56,55,25,22,21,22,37,38,39,33,43,40,30,34,46,46,44,58,64,70,46]
M2 = [12,None,None,None,None,1,7,None,8,None,None,None,None,None,3,7,4,None,10,None,3,None,None,22,6,2,None,11,8,None,None,None,6,None]

rows = []
for k, D in enumerate(range(0, 170, 5)):
    a, b = D * 1000 - 250, D * 1000 + 250
    m = (pts[:, 0] >= a) & (pts[:, 0] < b)
    cy, cz, R = fit_shell_circle_trial9(pts, a, b)
    sec = pts[m]
    rho = np.hypot(sec[:, 1], sec[:, 2]); phi = np.degrees(np.arctan2(sec[:, 2], sec[:, 1]))
    rows.append((D, m.sum(), cy, cz, R, rho, phi, M1[k], M2[k]))

print(" D   n     axis_off_u  axis_off_v  R_fit   [mm]")
for D, cnt, cy, cz, R, *_ in rows:
    print(f"{D:3d} {cnt:6d}  {cy:8.0f}  {cz:8.0f}  {R:7.0f}")
off = np.array([np.hypot(r[2], r[3]) for r in rows])
print("fitted centre vs design axis, median/max mm:", np.median(off).round(0), off.max().round(0))

# M1 = max rho - design R. The paper doesn't state R, so infer one R for all sections.
mx = np.array([np.percentile(r[5], 99.9) for r in rows])
Rd = np.median(mx - np.array(M1) * 10)
pred = (mx - Rd) / 10
print("implied design R (mm):", Rd.round(0))
print("M1 MAE (cm):", np.mean(np.abs(pred - np.array(M1))).round(1), " corr:", np.corrcoef(pred, M1)[0, 1].round(2))
# ponytail: min rho is polluted by equipment inside the tunnel, so M2 isn't compared yet (plan step C2).
