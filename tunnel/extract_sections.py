"""Stream the full tunnel cloud out of the zip and keep only the 34 section slabs.

Run: python tunnel/extract_sections.py [point_cloud.zip] [central_axis.xlsx] [out.npy]
Defaults: tunnel/data/point_cloud.zip, tunnel/data/central_axis.xlsx -> tunnel/data/sections_full.npy
Then: python tunnel/tunnel_validation.py
"""
import sys, zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from tunnel_validation import axis_frame, SECTIONS_M, HALF_THICK

here = Path(__file__).resolve().parent
zip_file = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data" / "point_cloud.zip"
axis_file = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "data" / "central_axis.xlsx"
out = Path(sys.argv[3]) if len(sys.argv) > 3 else here / "data" / "sections_full.npy"

wb = openpyxl.load_workbook(axis_file, read_only=True)
axis_pts = np.array([r[:3] for r in list(wb.active.iter_rows(values_only=True))[2:]], float)

kept, total = [], 0
with zipfile.ZipFile(zip_file) as z, z.open(z.namelist()[0]) as f:
    for chunk in pd.read_csv(f, sep=r"\s+", skiprows=1, header=None, usecols=[0, 1, 2, 3],
                             dtype=np.float64, chunksize=5_000_000):
        p = chunk.to_numpy()
        s = axis_frame(p, axis_pts)[0]
        k = np.clip(np.rint(s / 5), 0, len(SECTIONS_M) - 1)     # nearest section index
        kept.append(p[np.abs(s - SECTIONS_M[k.astype(int)]) < HALF_THICK])
        total += len(p)
        print(f"{total / 1e6:6.1f} M read, {sum(map(len, kept)) / 1e6:5.2f} M kept", flush=True)

np.save(out, np.concatenate(kept))
print("saved", out)
