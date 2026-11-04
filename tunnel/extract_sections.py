"""Stream the full tunnel cloud out of the zip once: keep the 34 section slabs and grid the whole wall.

Run: python tunnel/extract_sections.py [point_cloud.zip] [central_axis.xlsx] [out_dir]
Defaults: tunnel/data/point_cloud.zip, tunnel/data/central_axis.xlsx -> tunnel/data/
Writes sections_full.npy (slab points) and wall_grid.npz (wall radius per 0.5 m x 1 deg, for the unrolled map).
Then: python tunnel/tunnel_validation.py
"""
import sys, zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from tunnel_validation import axis_frame, SECTIONS_M, HALF_THICK, SHELL_BAND

here = Path(__file__).resolve().parent
zip_file = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "point_cloud.zip"
axis_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "central_axis.xlsx"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "data"

S_STEP, S0, S1 = 0.5, -5.0, 185.0        # m, chainage bins for the map
R_STEP, R0, R1 = 0.01, 2.5, 4.2          # m, radius histogram per cell (1 cm)
NS, NR = int((S1 - S0) / S_STEP), int(round((R1 - R0) / R_STEP))

wb = openpyxl.load_workbook(axis_file, read_only=True)
axis_pts = np.array([r[:3] for r in list(wb.active.iter_rows(values_only=True))[2:]], float)

kept, total = [], 0
hist = np.zeros(NS * 360 * NR, np.int32)
with zipfile.ZipFile(zip_file) as z, z.open(z.namelist()[0]) as f:
    for chunk in pd.read_csv(f, sep=r"\s+", skiprows=1, header=None, usecols=[0, 1, 2, 3],
                             dtype=np.float64, chunksize=5_000_000):
        p = chunk.to_numpy()
        s, u, v = axis_frame(p, axis_pts)
        k = np.clip(np.rint(s / 5), 0, len(SECTIONS_M) - 1)     # nearest section index
        kept.append(p[np.abs(s - SECTIONS_M[k.astype(int)]) < HALF_THICK])
        si = ((s - S0) / S_STEP).astype(int)
        ai = (np.degrees(np.arctan2(v, u)) % 360).astype(int)
        ri = ((np.hypot(u, v) - R0) / R_STEP).astype(int)
        ok = (si >= 0) & (si < NS) & (ri >= 0) & (ri < NR)
        hist += np.bincount((si[ok] * 360 + ai[ok]) * NR + ri[ok], minlength=hist.size).astype(np.int32)
        total += len(p)
        print(f"{total / 1e6:6.1f} M read, {sum(map(len, kept)) / 1e6:5.2f} M kept", flush=True)

np.save(out / "sections_full.npy", np.concatenate(kept))

# Same rule as wall_profile(): median of the returns within SHELL_BAND of each cell's 98th percentile.
h = hist.reshape(NS, 360, NR)
cum = h.cumsum(-1)
n = cum[..., -1]
top = np.argmax(cum >= 0.98 * n[..., None], -1)
r_idx = np.arange(NR)
band = h * ((r_idx >= top[..., None] - round(SHELL_BAND / R_STEP)) & (r_idx <= top[..., None]))
bc = band.cumsum(-1)
med = np.argmax(bc >= 0.5 * bc[..., -1:], -1)
wall = np.where(n >= 3, R0 + (med + 0.5) * R_STEP, np.nan).astype(np.float32)
# ponytail: 1 cm histogram bins cap the map's radial resolution at ~1 cm; fine for a colour map.
np.savez(out / "wall_grid.npz", wall=wall, s_edges=S0 + S_STEP * np.arange(NS + 1))
print("saved", out / "sections_full.npy", out / "wall_grid.npz")
