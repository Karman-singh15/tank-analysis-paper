"""Synthetic check: a tank with known radius, lean and dent must be recovered.

Run: python silo/test_fit.py
"""
import numpy as np

from silo_figures import analyse

R, H, TILT, DENT = 11880.0, 11900.0, 4.0, -60.0   # mm, mm, mm/m, mm


def synthetic_tank(n=300_000, seed=1):
    rng = np.random.default_rng(seed)
    # Uneven density like the real scan: ~8x more points near theta = 0.
    th = np.concatenate([rng.uniform(0, 2 * np.pi, n // 2), rng.vonmises(0, 2, n // 2) % (2 * np.pi)])
    z = rng.uniform(0, H, n)
    t_d, z_d = np.radians(90), 6000.0                       # dent at 90 deg, 6 m
    arc = R * (np.angle(np.exp(1j * (th - t_d))))
    r = R + DENT * np.exp(-(arc ** 2 + (z - z_d) ** 2) / (2 * 600.0 ** 2)) + rng.normal(0, 3, n)
    x = 32000 + r * np.cos(th) + TILT * z / 1000
    y = 2000 + r * np.sin(th)
    return np.round(np.column_stack([x, y, z]), -1)        # 1 cm rounding like data.txt


if __name__ == "__main__":
    a = analyse(synthetic_tank())
    tilt = np.hypot(a["kx"][0], a["ky"][0]) * 1000
    near = (np.abs(a["theta"] - 90) < 2) & (np.abs(a["h"] - 6000) < 300)
    dent = np.median(a["d"][near])
    print(f"radius bias {a['R'] - R:+.1f} mm | tilt {tilt:.2f} mm/m (true {TILT}) | dent {dent:.1f} mm (true {DENT})")
    assert abs(a["R"] - R) < 5, "radius off by > 5 mm"
    assert abs(tilt - TILT) < 0.5, "tilt off by > 0.5 mm/m"
    assert abs(dent - DENT) < 15, "dent depth off by > 15 mm"
    print("OK")
