"""Code reused verbatim from the mill-scanning project.

Every function below is copied unchanged from integ.py in the ball-mill repo
(~/dev/flsmith, not included here). The mill code assumes the cylinder axis lies along X; the silo
scripts permute columns so a vertical silo lies along X before calling these.
"""
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares


# REUSED from integ.py:19 (load_point_cloud) - unchanged.
def load_point_cloud(filepath):
    """Load point cloud from file and check units."""
    print(f"Loading: {Path(filepath).name}")
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in [".xyz", ".txt", ".csv"]:
        points = np.loadtxt(filepath)[:, :3]
    elif suffix == ".npy":
        points = np.load(filepath)
    elif suffix == ".ply":
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError("PLY file loaded but contains no points")
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    max_dim = np.ptp(points, axis=0).max()
    if max_dim < 100:
        print(f" Converting to mm (detected meters)...")
        points *= 1000.0

    print(f" Loaded {len(points):,} points")
    return points


# REUSED from integ.py:47 (correct_yz_alignment) - unchanged.
def correct_yz_alignment(points, n_iterations=3):
    """Iteratively correct YZ centering."""
    corrected = points.copy()
    
    for iteration in range(n_iterations):
        x_mid = np.median(corrected[:, 0])
        slice_mask = np.abs(corrected[:, 0] - x_mid) < 250.0
        slice_pts = corrected[slice_mask]

        if len(slice_pts) < 10: 
            break

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_thresh = np.percentile(r_all, 92.0)
        shell_mask = r_all >= r_thresh
        shell_pts_yz = slice_pts[shell_mask, 1:]
        r_shell = r_all[shell_mask]

        if len(r_shell) == 0: 
            break

        r_median = np.median(r_shell)
        r_std = np.std(r_shell)
        inlier = np.abs(r_shell - r_median) < 2.0 * r_std
        shell_pts_yz = azimuthally_balanced_shell_points(shell_pts_yz)


        if len(shell_pts_yz) < 100:
            break

        def residuals(params):
            cy, cz, R = params
            return np.sqrt((shell_pts_yz[:, 0] - cy) ** 2 + (shell_pts_yz[:, 1] - cz) ** 2) - R

        r_init = np.median(r_shell[inlier])
        result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=10.0)
        cy, cz, R_fit = result.x

        corrected[:, 1] -= cy
        corrected[:, 2] -= cz

        if np.sqrt(cy ** 2 + cz ** 2) < 3.0:
            break

    return corrected


# REUSED from integ.py:165 (azimuthally_balanced_shell_points) - unchanged.
def azimuthally_balanced_shell_points(shell_points):
    """
    Corrects biased shell sampling by enforcing angular balance.
    Automatically bypasses good scans.
    No hardcoded thresholds.
    """
    if len(shell_points) < 200:
        return shell_points

    y = shell_points[:, 0]
    z = shell_points[:, 1]

    theta = np.arctan2(z, y)
    radius = np.sqrt(y**2 + z**2)

    n_bins = max(24, int(np.sqrt(len(theta))))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    counts, _ = np.histogram(theta, bins=bins)
    mean_count = np.mean(counts[counts > 0])
    std_count = np.std(counts[counts > 0])

    if mean_count <= 0:
        return shell_points

    angular_cv = std_count / mean_count

    # GOOD SCAN → do nothing
    if angular_cv < np.percentile(counts[counts > 0], 75) / mean_count:
        return shell_points

    # PROBLEMATIC SCAN → balance angular contribution
    balanced = []

    for i in range(n_bins):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        if np.sum(mask) < 5:
            continue

        r_bin = radius[mask]
        pts_bin = shell_points[mask]

        r_peak = np.percentile(r_bin, 95)
        spread = np.std(r_bin)

        if spread <= 0:
            continue

        keep = np.abs(r_bin - r_peak) < spread
        if np.any(keep):
            balanced.append(pts_bin[keep])

    if not balanced:
        return shell_points

    return np.vstack(balanced)


# REUSED from integ.py:266 (fit_shell_circle_trial9) - unchanged.
def fit_shell_circle_trial9(points, x_min, x_max):
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    pts = points[in_cyl]
    y = pts[:, 1]
    z = pts[:, 2]
    r = np.sqrt(y*y + z*z)

    R_coarse = np.median(r)
    shell_min = 0.82 * R_coarse
    shell_max = 1.05 * R_coarse
    shell_mask = (r >= shell_min) & (r <= shell_max)
    shell_pts = pts[shell_mask]

    if len(shell_pts) < 100:
        return 0.0, 0.0, R_coarse

    shell_pts_yz = azimuthally_balanced_shell_points(shell_pts[:, 1:])

    def residuals(params):
        cy, cz, R = params
        return np.sqrt((shell_pts_yz[:, 0] - cy)**2 + (shell_pts_yz[:, 1] - cz)**2) - R

    result = least_squares(residuals, [0.0, 0.0, R_coarse], loss="soft_l1", f_scale=5.0)
    cy, cz, R = result.x
    return float(cy), float(cz), float(R)
