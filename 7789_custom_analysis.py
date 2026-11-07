#!/usr/bin/env python3
"""NGC 7789 experiment: initialize the existing models from a manual CMD polygon.

Run with no arguments to fit the same stars as the existing NGC 7789 results.
Use --draw-seed to replace the default polygon with a drawing in a plot window.
The polygon supplies initial parameters only; all rows participate in fitting.
Results go to outputs/NGC7789_custom-result_output, separate from the baseline.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path as PolygonPath
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
PMRA_COL = "pmra"
PMDEC_COL = "pmdec"
PARALLAX_COL = "parallax"
COV_REG = 1e-4

# Model parameters
MAX_ITER = 100
TOL_CENTER = 1e-4
TOL_LOGLIKE = 1e-5
MIN_MEMBERS = 10
MAX_PI = 0.08
MIN_PI = 0.001
MAX_PI_HARD = 0.35

# Quality thresholds
OVERLAP_PM_PLX_MIN = 0.3
OVERLAP_JOINT_PM_MIN = 0.5
OVERLAP_JOINT_PLX_MIN = 0.5
CORE_MEMBER_PROB_THRESHOLD = 0.8

# Gaia quality cuts
RUWE_MAX = 1.4
MIN_VISIBILITY_PERIODS = 8
MIN_PARALLAX_ERROR_RATIO = 3.0

# Standardized CMD extinction diagnostic zoom box.
# These limits are intentionally fixed, not auto-scaled, so raw vs extinction-
# corrected CMD products always show the exact same CMD window.
CMD_EXTINCTION_BOX_COLOR_MIN = 0.90
CMD_EXTINCTION_BOX_COLOR_MAX = 1.40
CMD_EXTINCTION_BOX_MAG_MIN = 16.5
CMD_EXTINCTION_BOX_MAG_MAX = 17.5

# Field stays broad: min "sigma" in each direction (mas/yr). Prevents field collapse.
FIELD_MIN_SIGMA_MAS = 5.0

# Effective uncertainty floor for projected sky coordinates (degrees).
# Prevents spatial dimensions from acting as zero-uncertainty channels.
SPATIAL_ERROR_FLOOR_DEG = 0.05



# Vertices are (raw BP-RP, raw G), in order around the central CMD concentration.
# Edit these points, or run --draw-seed to choose a different shape interactively.
MANUAL_CMD_VERTICES = np.array([
    [0.76, 13.25], [1.07, 13.25], [1.13, 14.70], [1.23, 15.70],
    [1.57, 18.50], [1.02, 18.50], [0.80, 16.10], [0.72, 14.65],
], dtype=float)


def manual_cmd_seed_mask(df, vertices=None):
    """Select a starting sample only, never a permanent membership mask."""
    vertices = MANUAL_CMD_VERTICES if vertices is None else np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("The manual CMD polygon needs at least three finite (color, G) vertices.")
    points = df[["bp_rp", "phot_g_mean_mag"]].to_numpy(dtype=float)
    mask = np.isfinite(points).all(axis=1) & PolygonPath(vertices).contains_points(points)
    if mask.sum() < MIN_MEMBERS:
        raise ValueError(f"Manual polygon contains only {mask.sum()} stars; at least {MIN_MEMBERS} are needed.")
    return mask


def require_manual_seed(seed_mask, n_rows):
    """Require an explicit manual sample, with no automatic fallback."""
    mask = np.asarray(seed_mask)
    if mask.shape != (n_rows,) or mask.dtype != np.dtype(bool) or mask.sum() < MIN_MEMBERS:
        raise ValueError("A boolean manual seed mask with at least ten selected stars is required.")
    return mask


def seed_location_scale(values):
    """Median and MAD scale of the manually selected sample, with a small floor."""
    values = np.asarray(values, dtype=float)
    center = np.median(values, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(values - center), axis=0), np.sqrt(COV_REG))
    return center, scale


def draw_manual_seed(df):
    """Let the user draw a polygon; return its vertices after the window closes."""
    from matplotlib.widgets import PolygonSelector
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df.bp_rp, df.phot_g_mean_mag, s=2, color="tab:blue", alpha=.2, rasterized=True)
    ax.set(xlabel="Raw BP - RP", ylabel="Raw G", xlim=(.2, 2.2), ylim=(18.5, 10),
           title="NGC 7789: click vertices, click the first point to close, then close the window")
    selection = {}
    def selected(vertices):
        selection["vertices"] = np.asarray(vertices, dtype=float)
        count = int(manual_cmd_seed_mask(df, vertices).sum())
        ax.set_title(f"Manual starting sample: {count} stars. Close window to fit.")
        fig.canvas.draw_idle()
    selector = PolygonSelector(ax, selected, props={"color": "tab:red", "linewidth": 2})
    plt.show()
    if "vertices" not in selection:
        raise ValueError("No completed polygon was drawn; no fit was run.")
    return selection["vertices"]


def save_manual_seed_preview(df, out_dir):
    """Save the exact starting selection and polygon for comparison with the fit."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df.bp_rp, df.phot_g_mean_mag, s=2, color="gray", alpha=.15, rasterized=True)
    selected = df.loc[df.manual_cmd_seed]
    ax.scatter(selected.bp_rp, selected.phot_g_mean_mag, s=3, color="tab:blue", alpha=.4, rasterized=True)
    vertices = np.vstack([MANUAL_CMD_VERTICES, MANUAL_CMD_VERTICES[0]])
    ax.plot(vertices[:, 0], vertices[:, 1], color="tab:red", linewidth=2)
    ax.set(xlabel="Raw BP - RP", ylabel="Raw G", xlim=(.2, 2.2), ylim=(18.5, 10),
           title=f"NGC 7789: manual initial sample ({len(selected)} of {len(df)} stars)")
    fig.tight_layout()
    fig.savefig(out_dir / "NGC7789_custom-result_manual_seed.png", dpi=160)
    plt.close(fig)


def load_astrometric_data(
    csv_path: Path,
    pmra_col: str = PMRA_COL,
    pmdec_col: str = PMDEC_COL,
    parallax_col: str = PARALLAX_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list, bool]:
    """
    Load astrometric data with quality cuts and optional error handling.
    
    Returns:
        pm_data: (n, 2) array of [pmra, pmdec]
        pmra_errors: (n,) array of pmra_error values (or None)
        pmdec_errors: (n,) array of pmdec_error values (or None)
        parallax_data: (n,) array of parallax values
        parallax_errors: (n,) array of parallax_error values (or None)
        df: cleaned DataFrame
        quality_flags: list of any quality issues encountered
        has_uncertainties: bool - whether uncertainty columns are available
    """
    df = pd.read_csv(csv_path, dtype={"source_id": "string"})
    quality_flags = []
    
    # Basic validity checks for measurements only
    required_cols = [pmra_col, pmdec_col, parallax_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required measurement columns: {missing_cols}")
    
    error_cols = ["pmra_error", "pmdec_error", "parallax_error"]
    has_error_cols = all(col in df.columns for col in error_cols)

    # Determine whether Gaia query is needed:
    # - any error column missing, OR
    # - any star has NaN/non-positive uncertainty in any required column.
    if has_error_cols:
        bad_error_mask_pre = (
            df["pmra_error"].isna()
            | df["pmdec_error"].isna()
            | df["parallax_error"].isna()
            | (df["pmra_error"] <= 0)
            | (df["pmdec_error"] <= 0)
            | (df["parallax_error"] <= 0)
        )
        needs_gaia_error_query = bool(bad_error_mask_pre.any())
    else:
        bad_error_mask_pre = np.ones(len(df), dtype=bool)
        needs_gaia_error_query = True

    # Apply Gaia quality cuts
    initial_n = len(df)
    
    # RUWE cut (if available)
    if 'ruwe' in df.columns:
        ruwe_mask = df['ruwe'] <= RUWE_MAX
        df = df[ruwe_mask]
        if len(df) < initial_n:
            quality_flags.append(f"RUWE_CUT: {initial_n - len(df)} stars removed")
    
    # Visibility periods cut (if available)
    if 'visibility_periods_used' in df.columns:
        vis_mask = df['visibility_periods_used'] >= MIN_VISIBILITY_PERIODS
        df = df[vis_mask]
        removed = initial_n - len(df)
        if removed > 0 and "RUWE_CUT" not in quality_flags:
            quality_flags.append(f"VISIBILITY_CUT: {removed} stars removed")
    
    # Enforce strict uncertainty policy: no missing/non-positive uncertainties allowed.
    bad_error_mask = (
        df["pmra_error"].isna()
        | df["pmdec_error"].isna()
        | df["parallax_error"].isna()
        | (df["pmra_error"] <= 0)
        | (df["pmdec_error"] <= 0)
        | (df["parallax_error"] <= 0)
    )
    if bad_error_mask.any():
        n_bad = int(bad_error_mask.sum())
        quality_flags.append(f"UNCERTAINTY_DROP: removed {n_bad} stars with unresolved missing/non-positive uncertainties")
        df = df[~bad_error_mask].copy()
    
    # Check for valid astrometric measurements (only measurements, not errors)
    valid_mask = (
        df[pmra_col].notna() & df[pmdec_col].notna() & 
        df[parallax_col].notna()
    )
    
    df = df[valid_mask].copy()
    if len(df) < initial_n:
        quality_flags.append(f"VALIDITY_CUT: {initial_n - len(df)} stars removed due to invalid measurements")
    
    # No fallback/no-imputation policy for uncertainties.
    
    # Extract data arrays
    pm_data = df[[pmra_col, pmdec_col]].values.astype(np.float64)
    parallax_data = df[parallax_col].values.astype(np.float64)
    
    # Extract error arrays (required by policy)
    pmra_errors = df['pmra_error'].values.astype(np.float64)
    pmdec_errors = df['pmdec_error'].values.astype(np.float64)
    parallax_errors = df['parallax_error'].values.astype(np.float64)
    has_uncertainties = True
    
    # Reset index to ensure contiguous indices matching probability arrays
    df = df.reset_index(drop=True)
    
    # Final validation
    if len(df) < MIN_MEMBERS:
        quality_flags.append(f"LOW_STAR_COUNT: only {len(df)} stars after quality cuts")
    
    return pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors, df, quality_flags, has_uncertainties


def load_pm_data(
    csv_path: Path,
    pmra_col: str = PMRA_COL,
    pmdec_col: str = PMDEC_COL,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load (pmra, pmdec) for stars with valid proper motions. No parallax"""
    df = pd.read_csv(csv_path)
    mask = df[pmra_col].notna() & df[pmdec_col].notna()
    pm = df.loc[mask, [pmra_col, pmdec_col]].values.astype(np.float64)
    sub = df.loc[mask].copy()
    # Reset index to ensure contiguous indices matching probability arrays
    sub = sub.reset_index(drop=True)
    return pm, sub











def distances_pm(pm: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Euclidean distance in PM space (mas/yr)"""
    return np.linalg.norm(pm - center, axis=1)


def weighted_center(pm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Center of mass in PM space with membership weights"""
    w = np.maximum(weights, 1e-12)
    return np.average(pm, axis=0, weights=w)


def weighted_covariance(pm: np.ndarray, center: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sample covariance (2x2) in PM space; regularized"""
    w = np.maximum(weights, 1e-12)
    w = w / w.sum()
    diff = pm - center
    cov = np.cov(diff.T, aweights=w)
    if cov.ndim == 0:
        cov = np.array([[cov, 0], [0, cov]])
    cov = np.atleast_2d(cov) + COV_REG * np.eye(2)
    return cov


def mvnpdf_log(pm: np.ndarray, center: np.ndarray, cov: np.ndarray, errors: np.ndarray = None) -> np.ndarray:
    """Log of 2D multivariate normal PDF with optional diagonal measurement errors"""
    diff = pm - center
    cov_model = np.atleast_2d(cov) + COV_REG * np.eye(2)
    
    # Add measurement errors if provided
    if errors is not None:
        # Use the proper implementation from mvnpdf_log_with_errors
        return mvnpdf_log_with_errors(pm, center, cov, errors[:, 0], errors[:, 1])
    else:
        cov_total = cov_model
    
    sign, logdet = np.linalg.slogdet(cov_total)
    if sign <= 0:
        return np.full(len(pm), -1e10)
    try:
        L = np.linalg.cholesky(cov_total)
        Linv = np.linalg.inv(L)
        quad = np.sum((diff @ Linv.T) ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.full(len(pm), -1e10)
    return -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)


def mvnpdf_log_with_errors(pm: np.ndarray, center: np.ndarray, cov: np.ndarray, pmra_errors: np.ndarray = None, pmdec_errors: np.ndarray = None) -> np.ndarray:
    """Log of 2D multivariate normal PDF with optional per-star diagonal measurement errors"""
    n = len(pm)
    log_likelihood = np.zeros(n)
    
    # Check if uncertainties are available
    has_errors = (pmra_errors is not None) and (pmdec_errors is not None)
    
    if has_errors:
        # Use measurement errors
        for i in range(n):
            diff = pm[i] - center
            cov_model = np.atleast_2d(cov) + COV_REG * np.eye(2)
            
            # By policy, uncertainties should be validated upstream and always valid here.
            if (
                not np.isfinite(pmra_errors[i])
                or not np.isfinite(pmdec_errors[i])
                or pmra_errors[i] <= 0
                or pmdec_errors[i] <= 0
            ):
                raise RuntimeError(
                    "Encountered invalid PM uncertainties after validation stage"
                )
            error_cov = np.diag([pmra_errors[i]**2, pmdec_errors[i]**2])
            cov_total = cov_model + error_cov
            
            sign, logdet = np.linalg.slogdet(cov_total)
            if sign <= 0:
                log_likelihood[i] = -1e10
                continue
                
            try:
                L = np.linalg.cholesky(cov_total)
                Linv = np.linalg.inv(L)
                quad = np.sum((diff @ Linv.T) ** 2)
                log_likelihood[i] = -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)
            except np.linalg.LinAlgError:
                log_likelihood[i] = -1e10
    else:
        # No uncertainties - use standard multivariate normal
        log_likelihood = mvnpdf_log(pm, center, cov)
    
    return log_likelihood



def p_member_two_component(
    pm: np.ndarray,
    center_cluster: np.ndarray,
    cov_cluster: np.ndarray,
    center_field: np.ndarray,
    cov_field: np.ndarray,
    pi_cluster: float,
) -> np.ndarray:
    """
    P(cluster | x) = π L_cluster(x) / (π L_cluster(x) + (1-π) L_field(x))
    Field is fixed; π is fixed. Only cluster parameters are learned
    """
    log_L_c = mvnpdf_log(pm, center_cluster, cov_cluster)
    log_L_f = mvnpdf_log(pm, center_field, cov_field)
    log_p_c = np.log(pi_cluster) + log_L_c
    log_p_f = np.log(1 - pi_cluster) + log_L_f
    max_log = np.maximum(log_p_c, log_p_f)
    p_c = np.exp(log_p_c - max_log)
    p_f = np.exp(log_p_f - max_log)
    return p_c / (p_c + p_f + 1e-300)


def fixed_field_component(pm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Define the field once: global mean, broad covariance. Never updated
    Ensures field stays unappealing so the cluster can be identified
    """
    center = np.mean(pm, axis=0)
    cov = np.cov(pm.T)
    cov = np.atleast_2d(cov) + COV_REG * np.eye(2)
    # Inflate so minimum eigenvalue >= (FIELD_MIN_SIGMA_MAS)^2
    eigvals, eigvecs = np.linalg.eigh(cov)
    min_eig = (FIELD_MIN_SIGMA_MAS ** 2)
    eigvals = np.maximum(eigvals, min_eig)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return center, cov





def mahalanobis_distances(pm: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Elliptical distance in PM space"""
    diff = pm - center
    cov_safe = np.atleast_2d(cov) + COV_REG * np.eye(2)
    L = np.linalg.cholesky(np.linalg.inv(cov_safe))
    white = diff @ L.T
    return np.linalg.norm(white, axis=1)


def fit_cluster_pm_only(
    pm: np.ndarray,
    pmra_errors: np.ndarray,
    pmdec_errors: np.ndarray,
    center_init: np.ndarray,
    sigma_init: float,
    center_field: np.ndarray,
    cov_field: np.ndarray,
    pi_cluster: float = None,
    max_iter: int = MAX_ITER,
    tol: float = TOL_CENTER,
    min_pi: float = MIN_PI,
    max_pi: float = MAX_PI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    PM-only mixture model with diagonal measurement errors and learned pi_cluster.
    Returns cluster_center, cluster_cov, P(member), Mahalanobis distances, learned pi_cluster
    """
    center_c = center_init.copy()
    cov_c = (sigma_init ** 2) * np.eye(2)
    
    # Initialize pi_cluster if not provided
    if pi_cluster is None:
        pi_cluster = 0.1

    for iteration in range(max_iter):
        # E-step: compute responsibilities with measurement errors
        log_L_c = mvnpdf_log_with_errors(pm, center_c, cov_c, pmra_errors, pmdec_errors)
        log_L_f = mvnpdf_log_with_errors(pm, center_field, cov_field, pmra_errors, pmdec_errors)
        
        log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
        log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
        
        max_log = np.maximum(log_p_c, log_p_f)
        p_c = np.exp(log_p_c - max_log)
        p_f = np.exp(log_p_f - max_log)
        prob = p_c / (p_c + p_f + 1e-300)
        
        # M-step: update parameters
        w = np.maximum(prob, 1e-12)
        center_c_new = weighted_center(pm, w)
        cov_c_new = weighted_covariance(pm, center_c_new, w)
        
        # Update pi_cluster (use raw prob, not w)
        pi_cluster_new = np.clip(np.mean(prob), min_pi, max_pi)
        
        # Check convergence
        drift = np.linalg.norm(center_c_new - center_c)
        center_c = center_c_new
        cov_c = cov_c_new
        pi_cluster = pi_cluster_new
        
        if drift < tol:
            break

    print(f"  PM EM iterations: {iteration + 1}; final center: {center_c}")

    # Final probabilities
    log_L_c = mvnpdf_log_with_errors(pm, center_c, cov_c, pmra_errors, pmdec_errors)
    log_L_f = mvnpdf_log_with_errors(pm, center_field, cov_field, pmra_errors, pmdec_errors)
    
    log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
    log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
    
    max_log = np.maximum(log_p_c, log_p_f)
    p_c = np.exp(log_p_c - max_log)
    p_f = np.exp(log_p_f - max_log)
    prob_final = p_c / (p_c + p_f + 1e-300)
    
    d_maha = mahalanobis_distances(pm, center_c, cov_c)
    return center_c, cov_c, prob_final, d_maha, pi_cluster


def membership_tier(prob: np.ndarray) -> np.ndarray:
    """Tiered labels: core / probable / possible / non-member"""
    tier = np.full(len(prob), "non-member", dtype=object)
    tier[prob >= CORE_MEMBER_PROB_THRESHOLD] = "core"
    tier[(prob >= 0.6) & (prob < CORE_MEMBER_PROB_THRESHOLD)] = "probable"
    tier[(prob >= 0.3) & (prob < 0.6)] = "possible"
    return tier




def fit_cluster_parallax_only(
    parallax: np.ndarray,
    parallax_errors: np.ndarray = None,
    center_init: float = None,
    sigma_init: float = None,
    pi_init: float = None,
    max_iter: int = MAX_ITER,
    tol_center: float = TOL_CENTER,
    min_pi: float = MIN_PI,
    max_pi: float = MAX_PI,
) -> tuple[float, float, np.ndarray, float]:
    """
    Independent parallax-only mixture model.
    1D Gaussian cluster + broad field, with individual measurement errors.
    
    Returns:
        center: parallax cluster center (mas)
        sigma: parallax cluster sigma (mas)
        prob: membership probabilities
        pi_cluster: learned cluster fraction
    """
    if center_init is None or sigma_init is None or pi_init is None:
        raise ValueError("Manual parallax center, scale, and initial fraction are required.")
    
    # Initial parameters
    center_c = float(center_init)
    sigma_c = float(sigma_init)
    pi_cluster = float(np.clip(pi_init, min_pi, max_pi))
    
    # Fixed broad field component
    field_center = np.median(parallax)
    field_sigma = np.std(parallax) * 2.0  # broad field
    field_sigma = max(field_sigma, 2.0)  # minimum field spread
    
    # Check if uncertainties are available
    has_errors = parallax_errors is not None
    
    n = len(parallax)
    loglike_old = -np.inf
    
    for iteration in range(max_iter):
        # E-step: compute responsibilities with measurement errors
        log_like_cluster = np.zeros(n)
        log_like_field = np.zeros(n)
        
        if has_errors:
            # Use measurement errors
            for i in range(n):
                if np.isfinite(parallax[i]) and np.isfinite(parallax_errors[i]):
                    # Cluster likelihood with measurement error
                    total_var_cluster = sigma_c**2 + parallax_errors[i]**2
                    log_like_cluster[i] = -0.5 * (np.log(2 * np.pi * total_var_cluster) + 
                                                  (parallax[i] - center_c)**2 / total_var_cluster)
                    
                    # Field likelihood with measurement error
                    total_var_field = field_sigma**2 + parallax_errors[i]**2
                    log_like_field[i] = -0.5 * (np.log(2 * np.pi * total_var_field) + 
                                               (parallax[i] - field_center)**2 / total_var_field)
                else:
                    log_like_cluster[i] = -1e10
                    log_like_field[i] = -1e10
        else:
            # No measurement errors - use standard Gaussian
            for i in range(n):
                if np.isfinite(parallax[i]):
                    # Cluster likelihood (standard Gaussian)
                    log_like_cluster[i] = -0.5 * (np.log(2 * np.pi * sigma_c**2) + 
                                                  (parallax[i] - center_c)**2 / sigma_c**2)
                    
                    # Field likelihood (standard Gaussian)
                    log_like_field[i] = -0.5 * (np.log(2 * np.pi * field_sigma**2) + 
                                               (parallax[i] - field_center)**2 / field_sigma**2)
                else:
                    log_like_cluster[i] = -1e10
                    log_like_field[i] = -1e10
        
        # Compute responsibilities
        log_p_cluster = np.log(max(pi_cluster, 1e-10)) + log_like_cluster
        log_p_field = np.log(max(1 - pi_cluster, 1e-10)) + log_like_field
        
        max_log = np.maximum(log_p_cluster, log_p_field)
        p_cluster = np.exp(log_p_cluster - max_log)
        p_field = np.exp(log_p_field - max_log)
        prob = p_cluster / (p_cluster + p_field + 1e-300)
        
        # M-step: update parameters
        w = np.maximum(prob, 1e-12)
        
        # Update cluster center
        center_c_new = np.average(parallax, weights=w)
        
        # Update cluster sigma (weighted standard deviation)
        variance_c = np.average((parallax - center_c_new)**2, weights=w)
        sigma_c_new = np.sqrt(max(variance_c, 0.01))  # minimum sigma
        
        # Update pi_cluster (use raw prob, not w)
        pi_cluster_new = np.clip(np.mean(prob), min_pi, max_pi)
        
        # Check convergence
        center_drift = abs(center_c_new - center_c)
        loglike_new = np.sum(
            np.log(
                pi_cluster * np.exp(log_like_cluster)
                + (1 - pi_cluster) * np.exp(log_like_field)
                + 1e-300
            )
        )
        
        if center_drift < tol_center and abs(loglike_new - loglike_old) < 1e-5:
            break
            
        # Update parameters
        center_c = center_c_new
        sigma_c = sigma_c_new
        pi_cluster = pi_cluster_new
        loglike_old = loglike_new
    
    # Final probabilities
    if has_errors:
        for i in range(n):
            if np.isfinite(parallax[i]) and np.isfinite(parallax_errors[i]):
                total_var_cluster = sigma_c**2 + parallax_errors[i]**2
                log_like_cluster[i] = -0.5 * (np.log(2 * np.pi * total_var_cluster) + 
                                              (parallax[i] - center_c)**2 / total_var_cluster)
                
                total_var_field = field_sigma**2 + parallax_errors[i]**2
                log_like_field[i] = -0.5 * (np.log(2 * np.pi * total_var_field) + 
                                           (parallax[i] - field_center)**2 / total_var_field)
            else:
                log_like_cluster[i] = -1e10
                log_like_field[i] = -1e10
    else:
        for i in range(n):
            if np.isfinite(parallax[i]):
                log_like_cluster[i] = -0.5 * (np.log(2 * np.pi * sigma_c**2) + 
                                              (parallax[i] - center_c)**2 / sigma_c**2)
                
                log_like_field[i] = -0.5 * (np.log(2 * np.pi * field_sigma**2) + 
                                           (parallax[i] - field_center)**2 / field_sigma**2)
            else:
                log_like_cluster[i] = -1e10
                log_like_field[i] = -1e10
    
    log_p_cluster = np.log(max(pi_cluster, 1e-10)) + log_like_cluster
    log_p_field = np.log(max(1 - pi_cluster, 1e-10)) + log_like_field
    
    max_log = np.maximum(log_p_cluster, log_p_field)
    p_cluster = np.exp(log_p_cluster - max_log)
    p_field = np.exp(log_p_field - max_log)
    prob_final = p_cluster / (p_cluster + p_field + 1e-300)
    
    return center_c, sigma_c, prob_final, pi_cluster


def add_projected_xy(df, ra0=None, dec0=None):
    """Return local sky offsets, wrapping RA across the 0/360-degree boundary."""
    ra = df["ra"].to_numpy(dtype=float)
    if ra0 is None:
        # Take a median in a continuous angular interval around the circular
        # mean. An ordinary RA median can put a boundary-crossing field at 180°.
        angle = np.deg2rad(ra)
        anchor = np.rad2deg(np.arctan2(np.nanmean(np.sin(angle)), np.nanmean(np.cos(angle))))
        ra0 = (anchor + np.nanmedian((ra - anchor + 180.0) % 360.0 - 180.0)) % 360.0
    if dec0 is None:
        dec0 = np.nanmedian(df["dec"])

    x = ((ra - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(dec0))
    y = df["dec"].values - dec0

    return x, y, ra0, dec0


def normalize_data_5d(data_5d):
    """Normalize 5D data to balanced scale and return transform parameters"""
    means = np.mean(data_5d, axis=0)
    stds = np.std(data_5d, axis=0)
    
    # Avoid division by zero
    stds = np.maximum(stds, 1e-8)
    
    data_normalized = (data_5d - means) / stds
    return data_normalized, means, stds





def compute_spatial_concentration(x: np.ndarray, y: np.ndarray, astro_prob: np.ndarray) -> float:
    """Compute spatial concentration from astrometric probabilities.

    concentration = field_r50 / core_r25, where
      - core_r25: 25th percentile radius of top-20% astro_prob stars
      - field_r50: 50th percentile radius of bottom-20% astro_prob stars
    """
    r = np.sqrt(x**2 + y**2)
    p = np.asarray(astro_prob)
    finite = np.isfinite(p) & np.isfinite(r)

    if finite.sum() < 50:
        return np.nan

    p = p[finite]
    rr = r[finite]

    high_cut = np.quantile(p, 0.80)
    low_cut = np.quantile(p, 0.20)
    high = p >= high_cut
    low = p <= low_cut

    if high.sum() < 20 or low.sum() < 20:
        return np.nan

    core_radius = np.percentile(rr[high], 25)
    field_radius = np.percentile(rr[low], 50)
    return float(field_radius / max(core_radius, 1e-6))


def concentration_to_spatial_weight(concentration: float, c_low: float = None, c_high: float = None) -> float:
    """Map concentration to bounded spatial weight.

    If calibrated bounds (c_low/c_high) are provided, use percentile-linear mapping.
    Otherwise, use the legacy smooth tanh mapping as fallback.
    """
    SPATIAL_WEIGHT_MIN = 0.25
    SPATIAL_WEIGHT_MAX = 0.75

    if not np.isfinite(concentration):
        return SPATIAL_WEIGHT_MIN

    if c_low is not None and c_high is not None and np.isfinite(c_low) and np.isfinite(c_high):
        z = (concentration - c_low) / max(c_high - c_low, 1e-6)
        z = float(np.clip(z, 0.0, 1.0))
        return SPATIAL_WEIGHT_MIN + z * (SPATIAL_WEIGHT_MAX - SPATIAL_WEIGHT_MIN)

    # Fallback mapping (used only when global calibration thresholds are unavailable)
    w = 0.25 + 0.5 * np.tanh((concentration - 1.5) / 2.0)
    return float(np.clip(w, SPATIAL_WEIGHT_MIN, SPATIAL_WEIGHT_MAX))


def adaptive_spatial_weight(
    x: np.ndarray,
    y: np.ndarray,
    astro_prob: np.ndarray,
    c_low: float = None,
    c_high: float = None,
) -> float:
    """
    Compute a bounded spatial weight based on how concentrated high-astrometric-probability
    stars are relative to low-astrometric-probability stars.

    Uses a quantile-based two-population contrast for robustness even when
    probabilities are compressed:
      high = top 20% of astro_prob
      low  = bottom 20% of astro_prob

    concentration = median_radius_low / median_radius_high

    Maps concentration to a spatial weight in [0.35, 0.75] via a smooth tanh curve:
      ~1.0  -> w ~0.35  (spatial not useful, barely weight it)
      ~3.5  -> w ~0.60  (moderate spatial concentration, moderate weight)
      ~7.0  -> w ~0.74  (strong concentration, near cap)

    The cap at 0.75 prevents spatial from dominating astrometry even for very
    compact clusters. The floor at 0.35 preserves a small spatial prior always.

    Falls back to 0.35 if there are too few finite values or either quantile
    population is too small.
    """
    concentration = compute_spatial_concentration(x, y, astro_prob)
    weight = concentration_to_spatial_weight(concentration, c_low=c_low, c_high=c_high)

    print(
        f"  Spatial weight debug: "
        f"concentration={concentration if np.isfinite(concentration) else np.nan:.3f}, "
        f"c_low={c_low if c_low is not None else np.nan:.3f}, "
        f"c_high={c_high if c_high is not None else np.nan:.3f}, "
        f"weight={weight:.3f}"
    )

    return weight





def fixed_field_component_nd(data: np.ndarray, pm_floor: float = None, plx_floor: float = None):
    """Create proper field component for 3D data with separate PM and parallax floors"""
    if pm_floor is None:
        pm_floor = FIELD_MIN_SIGMA_MAS
    if plx_floor is None:
        plx_floor = 0.3  # Much smaller floor for parallax
        
    center = np.mean(data, axis=0)
    cov = np.cov(data.T)
    cov = np.atleast_2d(cov) + COV_REG * np.eye(data.shape[1])

    # Apply floors in coordinate space, not eigenvalue space
    if data.shape[1] == 3:
        floors = np.array([pm_floor**2, pm_floor**2, plx_floor**2])
    else:
        floors = np.full(data.shape[1], pm_floor**2)

    for i in range(data.shape[1]):
        cov[i, i] = max(cov[i, i], floors[i])

    # Ensure numerical stability with eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, COV_REG)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return center, cov


def fixed_field_component_5d(data):
    """Create proper field component for 5D spatial+astrometric data"""
    center = np.mean(data, axis=0)
    cov = np.cov(data.T) + COV_REG * np.eye(5)

    # Different floors for different dimensions
    floors = np.array([
        5.0**2,    # pmra (mas/yr)
        5.0**2,    # pmdec (mas/yr)
        0.2**2,    # parallax (mas)
        0.2**2,    # x sky (degrees)
        0.2**2,    # y sky (degrees)
    ])

    for i in range(5):
        cov[i, i] = max(cov[i, i], floors[i])

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, COV_REG)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return center, cov


def fixed_field_component_5d_normalized(data):
    """Create proper field component for normalized 5D data"""
    center = np.mean(data, axis=0)
    cov = np.cov(data.T) + COV_REG * np.eye(5)

    floors = np.array([
        0.6**2,    # pmra (normalized)
        0.6**2,    # pmdec (normalized)
        0.3**2,    # parallax (normalized)
        0.3**2,    # x sky (normalized)
        0.3**2,    # y sky (normalized)
    ])

    for i in range(5):
        cov[i, i] = max(cov[i, i], floors[i])

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, COV_REG)
    return center, eigvecs @ np.diag(eigvals) @ eigvecs.T


def mvnpdf_log_nd(data, center, cov, errors=None):
    """Generic N-dimensional multivariate normal log PDF with optional uncertainties"""
    n, dim = data.shape
    diff = data - center
    cov_model = np.atleast_2d(cov) + COV_REG * np.eye(dim)
    
    # Check if uncertainties are available
    has_errors = errors is not None and errors.shape == (n, dim)
    
    if has_errors:
        # Add measurement errors to covariance for each star
        log_likelihood = np.zeros(n)
        for i in range(n):
            if np.any(np.isnan(errors[i])) or np.any(errors[i] <= 0):
                raise RuntimeError(
                    "Encountered invalid ND uncertainties after validation stage"
                )

            cov_total = cov_model + np.diag(errors[i]**2)
            sign, logdet = np.linalg.slogdet(cov_total)
            if sign <= 0:
                log_likelihood[i] = -1e10
                continue

            try:
                L = np.linalg.cholesky(cov_total)
                solved = np.linalg.solve(L, diff[i])
                quad = np.sum(solved**2)
                log_likelihood[i] = -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)
            except np.linalg.LinAlgError:
                log_likelihood[i] = -1e10
        return log_likelihood
    else:
        # No uncertainties - use standard calculation
        sign, logdet = np.linalg.slogdet(cov_model)
        if sign <= 0:
            return np.full(len(data), -1e10)

        try:
            L = np.linalg.cholesky(cov_model)
            solved = np.linalg.solve(L, diff.T)
            quad = np.sum(solved**2, axis=0)
        except np.linalg.LinAlgError:
            return np.full(len(data), -1e10)

        return -0.5 * (dim * np.log(2 * np.pi) + logdet + quad)



def mvnpdf_log_3d(
    data: np.ndarray, 
    center: np.ndarray, 
    cov: np.ndarray, 
    pmra_errors: np.ndarray = None, 
    pmdec_errors: np.ndarray = None, 
    parallax_errors: np.ndarray = None
) -> np.ndarray:
    """Log of 3D multivariate normal PDF with optional per-star diagonal measurement errors"""
    n = len(data)
    log_likelihood = np.zeros(n)
    
    # Check if uncertainties are available
    has_errors = (pmra_errors is not None) and (pmdec_errors is not None) and (parallax_errors is not None)
    
    if has_errors:
        # Use measurement errors
        for i in range(n):
            diff = data[i] - center
            cov_model = np.atleast_2d(cov) + COV_REG * np.eye(3)
            
            if (
                not np.isfinite(pmra_errors[i])
                or not np.isfinite(pmdec_errors[i])
                or not np.isfinite(parallax_errors[i])
                or pmra_errors[i] <= 0
                or pmdec_errors[i] <= 0
                or parallax_errors[i] <= 0
            ):
                raise RuntimeError(
                    "Encountered invalid 3D uncertainties after validation stage"
                )
            error_cov = np.diag([pmra_errors[i]**2, pmdec_errors[i]**2, parallax_errors[i]**2])
            cov_total = cov_model + error_cov
            
            sign, logdet = np.linalg.slogdet(cov_total)
            if sign <= 0:
                log_likelihood[i] = -1e10
                continue
                
            try:
                L = np.linalg.cholesky(cov_total)
                Linv = np.linalg.inv(L)
                quad = np.sum((diff @ Linv.T) ** 2)
                log_likelihood[i] = -0.5 * (3 * np.log(2 * np.pi) + logdet + quad)
            except np.linalg.LinAlgError:
                log_likelihood[i] = -1e10
    else:
        # No uncertainties - use standard 3D multivariate normal
        diff = data - center
        cov_model = np.atleast_2d(cov) + COV_REG * np.eye(3)

        sign, logdet = np.linalg.slogdet(cov_model)
        if sign <= 0:
            return np.full(n, -1e10)

        try:
            L = np.linalg.cholesky(cov_model)
            Linv = np.linalg.inv(L)
            quad = np.sum((diff @ Linv.T) ** 2, axis=1)
            return -0.5 * (3 * np.log(2 * np.pi) + logdet + quad)
        except np.linalg.LinAlgError:
            return np.full(n, -1e10)
    
    return log_likelihood


def fit_cluster_nd(
    data,
    center_init,
    cov_init,
    center_field,
    cov_field,
    errors=None,
    pi_cluster=None,
    max_iter=100,
    tol=1e-4,
    min_pi: float = MIN_PI,
    max_pi: float = MAX_PI,
):
    """Generic N-dimensional cluster fitting with EM algorithm"""
    if pi_cluster is None:
        pi_cluster = 0.1

    center_c = center_init.copy()
    cov_c = cov_init.copy()

    for iteration in range(max_iter):
        log_L_c = mvnpdf_log_nd(data, center_c, cov_c, errors)
        log_L_f = mvnpdf_log_nd(data, center_field, cov_field, errors)

        log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
        log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f

        max_log = np.maximum(log_p_c, log_p_f)
        p_c = np.exp(log_p_c - max_log)
        p_f = np.exp(log_p_f - max_log)
        prob = p_c / (p_c + p_f + 1e-300)

        w = np.maximum(prob, 1e-12)

        center_new = np.average(data, axis=0, weights=w)
        diff = data - center_new
        cov_new = np.average(diff[:, :, None] * diff[:, None, :], axis=0, weights=w)
        
        # Add covariance cap for 5D joint model to prevent over-expansion
        if data.shape[1] == 5:
            max_sigma = np.array([0.6, 0.6, 0.35, 0.35, 0.35])
            max_var = max_sigma**2
            
            # Remove cross-covariance to prevent tilted mega-net
            cov_new = np.diag(np.minimum(np.diag(cov_new), max_var))
        
        # Update cluster covariance with regularization
        cov_new += COV_REG * np.eye(data.shape[1])

        pi_new = np.clip(np.mean(prob), min_pi, max_pi)

        if np.linalg.norm(center_new - center_c) < tol:
            center_c, cov_c, pi_cluster = center_new, cov_new, pi_new
            break

        center_c, cov_c, pi_cluster = center_new, cov_new, pi_new

    print(f"  Joint EM iterations: {iteration + 1}; final normalized center: {center_c}")

    # Final probabilities
    log_L_c = mvnpdf_log_nd(data, center_c, cov_c, errors)
    log_L_f = mvnpdf_log_nd(data, center_field, cov_field, errors)

    log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
    log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f

    max_log = np.maximum(log_p_c, log_p_f)
    p_c = np.exp(log_p_c - max_log)
    p_f = np.exp(log_p_f - max_log)
    prob = p_c / (p_c + p_f + 1e-300)

    return center_c, cov_c, prob, pi_cluster














def estimate_adaptive_max_pi(pm_data: np.ndarray) -> float:
    """Estimate per-cluster max_pi from PM-space density contrast.

    Uses local PM density relative to its median so the estimate is truly
    cluster-dependent (not percentile-locked):
      local_density = 1 / d_k
      density_ratio = local_density / median(local_density)
      dense_fraction = mean(density_ratio > 3)
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        n = len(pm_data)
        if n < 50:
            return MAX_PI

        k = min(15, max(5, n - 1))
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(pm_data)
        dists, _ = nbrs.kneighbors(pm_data)
        dk = dists[:, -1]
        local_density = 1.0 / np.maximum(dk, 1e-6)

        median_density = np.median(local_density)
        density_ratio = local_density / max(median_density, 1e-12)
        dense_fraction = float(np.mean(density_ratio > 3.0))
        max_pi_adaptive = np.clip(dense_fraction * 1.5, 0.08, 0.35)
        return float(max_pi_adaptive)
    except Exception:
        return MAX_PI


def run_manual_seed_pm(pm_data, pmra_errors=None, pmdec_errors=None,
                       center_field=None, cov_field=None, min_pi=MIN_PI, max_pi=MAX_PI,
                       *, seed_mask):
    """Fit the whole PM field once, starting only from the manually selected stars."""
    seed_mask = require_manual_seed(seed_mask, len(pm_data))
    center_init, scale = seed_location_scale(pm_data[seed_mask])
    sigma_init = float(np.sqrt(np.mean(scale**2)))
    pi_init = float(np.clip(np.mean(seed_mask), min_pi, max_pi))
    center, cov, prob, _, pi_cluster = fit_cluster_pm_only(
        pm_data, pmra_errors, pmdec_errors, center_init, sigma_init,
        center_field, cov_field, pi_cluster=pi_init, min_pi=min_pi, max_pi=max_pi,
    )
    return {'status': 'SUCCESS', 'best': {'seed': 'manual_cmd', 'center': center,
            'cov': cov, 'prob': prob, 'pi_cluster': pi_cluster}}



def run_manual_seed_parallax(parallax_data, parallax_errors=None,
                             min_pi=MIN_PI, max_pi=MAX_PI, *, seed_mask):
    """Fit all parallaxes once, starting only from the manually selected stars."""
    seed_mask = require_manual_seed(seed_mask, len(parallax_data))
    center_init, sigma_init = seed_location_scale(parallax_data[seed_mask])
    center, sigma, prob, pi_cluster = fit_cluster_parallax_only(
        parallax_data, parallax_errors, float(center_init), float(sigma_init),
        pi_init=float(np.mean(seed_mask)), min_pi=min_pi, max_pi=max_pi,
    )
    return {'status': 'SUCCESS', 'best': {'seed': 'manual_cmd', 'center': center,
            'sigma': sigma, 'prob': prob, 'pi_cluster': pi_cluster}}



def run_manual_seed_joint(pm_data, parallax_data=None, df=None,
                          center_field=None, cov_field=None, pmra_errors=None,
                          pmdec_errors=None, parallax_errors=None, astro_prob=None,
                          c_low=None, c_high=None, spatial_weight_override=None,
                          min_pi=MIN_PI, max_pi=MAX_PI, *, seed_mask):
    """Fit all 5D rows once, using manual stars for the initial center and scales."""
    seed_mask = require_manual_seed(seed_mask, len(pm_data))
    x_sky, y_sky, _, _ = add_projected_xy(df)
    data_5d = np.column_stack([pm_data, parallax_data, x_sky, y_sky])
    normalized, means, stds = normalize_data_5d(data_5d)
    if spatial_weight_override is not None:
        spatial_weight = float(np.clip(spatial_weight_override, .25, .75))
    elif astro_prob is not None:
        spatial_weight = adaptive_spatial_weight(x_sky, y_sky, astro_prob, c_low=c_low, c_high=c_high)
    else:
        spatial_weight = .25
    print(f"  Adaptive spatial weight: {spatial_weight:.3f}")
    normalized[:, 3:] *= spatial_weight
    if center_field is None or cov_field is None:
        center_field, cov_field = fixed_field_component_5d_normalized(normalized)
    center_init, scale_init = seed_location_scale(normalized[seed_mask])
    cov_init = np.diag(scale_init**2)
    if all(e is not None for e in (pmra_errors, pmdec_errors, parallax_errors)):
        errors = np.column_stack([pmra_errors, pmdec_errors, parallax_errors,
                                 np.full(len(df), SPATIAL_ERROR_FLOOR_DEG),
                                 np.full(len(df), SPATIAL_ERROR_FLOOR_DEG)]) / stds
        errors[:, 3:] *= spatial_weight
    else:
        errors = None
    center_norm, cov_norm, prob, pi_cluster = fit_cluster_nd(
        normalized, center_init, cov_init, center_field, cov_field, errors,
        pi_cluster=float(np.clip(np.mean(seed_mask), min_pi, max_pi)),
        min_pi=min_pi, max_pi=max_pi,
    )
    scale = stds.copy()
    scale[3:] /= spatial_weight
    center = center_norm * scale + means
    cov = cov_norm * np.outer(scale, scale)
    return {'status': 'SUCCESS', 'best': {'seed': 'manual_cmd', 'center': center,
            'cov': cov, 'prob': prob, 'pi_cluster': pi_cluster}}



def compute_model_overlaps(pm_prob: np.ndarray, plx_prob: np.ndarray, joint_prob: np.ndarray) -> dict:
    """Compute overlap metrics between different model probability assignments"""
    overlaps = {}
    
    # Define core members (P >= 0.8)
    pm_core = pm_prob >= 0.8
    plx_core = plx_prob >= 0.8
    joint_core = joint_prob >= 0.8
    
    # PM-Parallax overlap
    if pm_core.any():
        overlaps['pm_plx_overlap'] = np.mean(plx_core[pm_core])
    else:
        overlaps['pm_plx_overlap'] = np.nan
    
    # Joint-PM overlap
    if joint_core.any():
        overlaps['joint_pm_overlap'] = np.mean(pm_core[joint_core])
    else:
        overlaps['joint_pm_overlap'] = np.nan
    
    # Joint-Parallax overlap
    if joint_core.any():
        overlaps['joint_plx_overlap'] = np.mean(plx_core[joint_core])
    else:
        overlaps['joint_plx_overlap'] = np.nan
    
    # Core member counts
    overlaps['pm_core_count'] = pm_core.sum()
    overlaps['plx_core_count'] = plx_core.sum()
    overlaps['joint_core_count'] = joint_core.sum()
    
    return overlaps


def evaluate_solution_quality(
    pm_result: dict,
    plx_result: dict,
    joint_result: dict,
    n_total: int
) -> dict:
    """Evaluate quality of all three model solutions and generate flags"""
    quality_flags = []
    
    # Check for failed models
    if pm_result['status'] == 'FAILED':
        quality_flags.append('PM_UNSTABLE')
    
    if plx_result['status'] == 'FAILED':
        quality_flags.append('PLX_UNSTABLE')
    
    if joint_result['status'] == 'FAILED':
        quality_flags.append('JOINT_UNSTABLE')
    
    # Get probabilities for successful models
    pm_prob = pm_result['best']['prob'] if pm_result['status'] == 'SUCCESS' else None
    plx_prob = plx_result['best']['prob'] if plx_result['status'] == 'SUCCESS' else None
    joint_prob = joint_result['best']['prob'] if joint_result['status'] == 'SUCCESS' else None
    
    # Check member counts
    if joint_prob is not None:
        n_joint_members = (joint_prob >= 0.8).sum()
        if n_joint_members < MIN_MEMBERS:
            quality_flags.append('LOW_JOINT_MEMBERS')
        
        member_fraction = n_joint_members / n_total
        if member_fraction > 0.35:
            quality_flags.append('HIGH_MEMBER_FRACTION')
    
    # Check for reasonable parameter values
    if joint_result['status'] == 'SUCCESS':
        joint_prob = joint_result['best']['prob']
        joint_center = joint_result['best']['center']
        
        # Check for over-admission of joint members
        n_joint_core = (joint_prob >= 0.8).sum()
        joint_core_fraction = n_joint_core / len(joint_prob)
        
        if joint_core_fraction > 0.35:
            quality_flags.append("JOINT_OVERADMITTED_MEMBERS")
            print(f"WARNING: Joint model over-admitted members ({n_joint_core}/{len(joint_prob)} = {joint_core_fraction:.2f}); do not trust joint_prob for this cluster.")
        
        # Check for reasonable parallax range
        if len(joint_center) >= 3:
            parallax_center = joint_center[2]
            if parallax_center < 0.1 or parallax_center > 10.0:
                quality_flags.append("JOINT_UNREASONABLE_PARALLAX")
        
        # Check for reasonable parallax spread (if covariance available)
        if 'cov' in joint_result['best']:
            joint_cov = joint_result['best']['cov']
            if len(joint_cov) >= 3:
                parallax_var = joint_cov[2, 2]
                parallax_std = np.sqrt(max(parallax_var, 0))
                if parallax_std > 2.0:  # Very spread out in parallax
                    quality_flags.append("JOINT_WIDE_PARALLAX")
    
    # Check model agreement
    if all(prob is not None for prob in [pm_prob, plx_prob, joint_prob]):
        overlaps = compute_model_overlaps(pm_prob, plx_prob, joint_prob)
        
        # PM-Parallax overlap
        if pm_prob is not None and plx_prob is not None:
            if overlaps['pm_plx_overlap'] < OVERLAP_PM_PLX_MIN:
                quality_flags.append('PM_PLX_DISAGREE')
        
        # Joint-PM disagreement
        if overlaps['joint_pm_overlap'] < OVERLAP_JOINT_PM_MIN:
            quality_flags.append('JOINT_PM_DISAGREE')
        
        # Joint-Parallax disagreement
        if overlaps['joint_plx_overlap'] < OVERLAP_JOINT_PLX_MIN:
            quality_flags.append('JOINT_PLX_DISAGREE')

        # Hard reject: spatial-dominated joint solution with zero astrometric support
        if joint_prob is not None:
            joint_core_fraction = float((joint_prob >= 0.8).mean())
            if (
                joint_core_fraction > 0.35
                and overlaps.get('joint_pm_overlap', np.nan) == 0
                and overlaps.get('joint_plx_overlap', np.nan) == 0
            ):
                quality_flags.append("JOINT_SPATIAL_DOMINATED_REJECT")
    
    # This is a single manual initialization, so between-seed stability is not assessed.

    # Overall assessment
    if len(quality_flags) == 0:
        overall_status = 'GOOD'
    elif len(quality_flags) <= 2:
        overall_status = 'ACCEPTABLE'
    elif len(quality_flags) <= 4:
        overall_status = 'QUESTIONABLE'
    else:
        overall_status = 'POOR'
        quality_flags.append('NO_CLEAR_SOLUTION')
    
    return {
        'overall_status': overall_status,
        'quality_flags': quality_flags,
        'n_total': n_total,
        'overlaps': compute_model_overlaps(pm_prob, plx_prob, joint_prob) if all(prob is not None for prob in [pm_prob, plx_prob, joint_prob]) else {}
    }


def validate_model_outputs(pm_result: dict, plx_result: dict, joint_result: dict) -> bool:
    """Validate that all model outputs are physically reasonable"""
    validation_flags = []
    
    for model_name, result in [('PM', pm_result), ('PARALLAX', plx_result), ('JOINT', joint_result)]:
        if result['status'] != 'SUCCESS':
            continue
        
        best = result['best']
        
        # Check probabilities are valid
        prob = best['prob']
        if not np.all((prob >= 0) & (prob <= 1)):
            validation_flags.append(f'{model_name}_INVALID_PROBABILITIES')
        
        # Check center is finite
        center = best['center']
        if not np.isfinite(center).all():
            validation_flags.append(f'{model_name}_INVALID_CENTER')
        
        # Check covariance is finite and positive definite (or sigma for parallax)
        if model_name == "PARALLAX":
            sigma = best.get("sigma", np.nan)
            if not np.isfinite(sigma) or sigma <= 0:
                validation_flags.append("PARALLAX_INVALID_SIGMA")
        else:
            cov = best['cov']
            if not np.isfinite(cov).all():
                validation_flags.append(f'{model_name}_INVALID_COVARIANCE')
            
            try:
                np.linalg.cholesky(cov + COV_REG * np.eye(cov.shape[0]))
            except np.linalg.LinAlgError:
                validation_flags.append(f'{model_name}_NON_POSITIVE_DEFINITE')
        
        # Check pi_cluster is reasonable
        pi_cluster = best['pi_cluster']
        if not (MIN_PI <= pi_cluster <= MAX_PI_HARD):
            validation_flags.append(f'{model_name}_INVALID_PI_CLUSTER')
    
    if validation_flags:
        print(f"Validation warnings: {validation_flags}")
        return False
    
    return True


def generate_three_model_cmd_plot(
    df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
    apply_extinction_correction: bool = False,
):
    """
    Generate three-panel CMD showing membership tiers for:
    PM-only, Parallax-only, and Final astrometry-gated membership.
    
    Args:
        df: DataFrame with photometry and membership data
        out_dir: Output directory for plots
        file_prefix: Prefix for output filenames
        apply_extinction_correction: If True, apply spatial extinction correction
    """
    if not HAS_MATPLOTLIB:
        print("  (Skipped three-model CMD: matplotlib not available)")
        return
    
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    extinction_suffix = " (EXTINCTION CORRECTED)" if mag_col == "phot_g_mean_mag_corrected" else ""
    
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped three-model CMD: missing photometry)")
        return

    x_label, y_label, _ = cmd_axis_labels(mag_col, color_col)
    
    # Check for valid photometry
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped three-model CMD: no valid photometry)")
        return
    
    # Create figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for membership tiers
    tier_colors = {
        "core": "#117733",
        "probable": "#2166AC",
        "possible": "#D55E00",
        "non-member": "#888888",
    }
    
    # Draw order: non-member first (bottom), then possible, probable, core (top)
    tier_order = ["non-member", "possible", "probable", "core"]
    
    # Panel 1: PM-only model
    ax = axes[0]
    if 'pm_only_tier' in df.columns:
        for tier_name in tier_order:
            mask = valid_photo & (df['pm_only_tier'] == tier_name)
            if mask.any():
                count = mask.sum()
                ax.scatter(
                    df.loc[mask, color_col],
                    df.loc[mask, mag_col],
                    c=tier_colors[tier_name],
                    s=6,
                    alpha=0.7,
                    label=f"{tier_name} (N={count})",
                )
        
        n_pm_core = (df['pm_only_prob'] >= 0.8).sum() if 'pm_only_prob' in df.columns else 0
        ax.set_title(f"PM-only Model{extinction_suffix}\nCore members: {n_pm_core}")
    else:
        ax.text(0.6, 0.6, "PM-only model\nfailed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("PM-only Model\nFAILED")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Parallax-only model
    ax = axes[1]
    if 'plx_only_tier' in df.columns:
        for tier_name in tier_order:
            mask = valid_photo & (df['plx_only_tier'] == tier_name)
            if mask.any():
                count = mask.sum()
                ax.scatter(
                    df.loc[mask, color_col],
                    df.loc[mask, mag_col],
                    c=tier_colors[tier_name],
                    s=6,
                    alpha=0.7,
                    label=f"{tier_name} (N={count})",
                )
        
        n_plx_core = (df['plx_only_prob'] >= 0.8).sum() if 'plx_only_prob' in df.columns else 0
        ax.set_title(f"Parallax-only Model{extinction_suffix}\nCore members: {n_plx_core}")
    else:
        ax.text(0.6, 0.6, "Parallax-only model\nfailed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Parallax-only Model\nFAILED")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Final astrometry-gated membership (science output)
    ax = axes[2]
    if 'final_tier' in df.columns:
        for tier_name in tier_order:
            mask = valid_photo & (df['final_tier'] == tier_name)
            if mask.any():
                count = mask.sum()
                ax.scatter(
                    df.loc[mask, color_col],
                    df.loc[mask, mag_col],
                    c=tier_colors[tier_name],
                    s=6,
                    alpha=0.7,
                    label=f"{tier_name} (N={count})",
                )
        
        n_final_core = (df['final_member_prob'] >= 0.8).sum() if 'final_member_prob' in df.columns else 0
        ax.set_title(f"Final Membership (Astrometry-Gated){extinction_suffix}\nCore members: {n_final_core}")
    else:
        ax.text(0.6, 0.6, "Final astrometry-gated\nmembership unavailable", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Final Membership\nUNAVAILABLE")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_three_model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_three_model_comparison.png")













def plot_cmd_colored_by_probability(
    df: pd.DataFrame,
    prob_values: np.ndarray,
    prob_label: str,
    out_dir: Path,
    filename: str,
    file_prefix: str,
    apply_extinction_correction: bool = False,
):
    """CMD (BP−RP vs G) colored by a continuous probability (blue scale)."""
    if not HAS_MATPLOTLIB:
        return
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    
    if mag_col not in df.columns or color_col not in df.columns:
        return
    ok = df[mag_col].notna() & df[color_col].notna()
    if not ok.any():
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df.loc[ok, color_col],
        df.loc[ok, mag_col],
        c=prob_values[ok],
        s=8,
        cmap="Blues",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD — colored by {prob_label}{title_suffix}")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(prob_label)
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_{filename}", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_{filename}")


def plot_cmd_core_only(
    df: pd.DataFrame,
    prob_values: np.ndarray,
    prob_label: str,
    out_dir: Path,
    filename: str,
    file_prefix: str,
    threshold: float = 0.8,
    apply_extinction_correction: bool = False,
):
    """CMD showing only stars with probability ≥ threshold (e.g., 80% core members)."""
    if not HAS_MATPLOTLIB:
        return
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    
    if mag_col not in df.columns or color_col not in df.columns:
        return
    
    # Only select high-probability stars
    ok = df[mag_col].notna() & df[color_col].notna() & (prob_values >= threshold)
    if not ok.any():
        print(f"  (Skipped {filename}: no stars with {prob_label} ≥ {threshold})")
        return
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        df.loc[ok, color_col],
        df.loc[ok, mag_col],
        s=10,
        color="darkblue",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD — Core members only ({prob_label} ≥ {threshold}){title_suffix}\nN = {ok.sum()}")
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_{filename}", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_{filename}")











def plot_cmd_core_only_comparison(
    df: pd.DataFrame,
    pm_prob: np.ndarray,
    parallax_prob: np.ndarray,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
    apply_extinction_correction: bool = False,
):
    """
    3-panel CMD showing ONLY the 80%+ core group for PM, parallax, and combined
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD core-only: missing photometry)")
        return
    
    # Check for valid photometry
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD core-only: no valid photometry)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    color_core = "royalblue"  # Medium blue for core group
    
    # --- Panel 1: PM probability core only ---
    ax = axes[0]
    mask_core = valid_photo & (pm_prob >= 0.8)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black", 
                  linewidths=0.6, label=f"Core (≥80%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by PM probability - Core only{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 2: Parallax probability core only ---
    ax = axes[1]
    mask_core = valid_photo & (parallax_prob >= 0.8)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.6, label=f"Core (≥80%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by Parallax probability - Core only{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 3: Bayesian posterior core only ---
    ax = axes[2]
    mask_core = valid_photo & (bayesian_posterior >= 0.8)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.6, label=f"Core (≥80%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by Combined probability - Core only{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_core_only_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_core_only_comparison.png")


def plot_cmd_comparison_cuts(
    df: pd.DataFrame,
    pm_prob: np.ndarray,
    parallax_prob: np.ndarray,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
    apply_extinction_correction: bool = False,
):
    """
    3-panel CMD comparing PM, parallax, and combined probabilities
    with two confidence levels: 60-80% (light) and 80%+ (medium blue)
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD comparison cuts: missing photometry)")
        return
    
    # Check for valid photometry
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD comparison cuts: no valid photometry)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for confidence levels
    color_60_80 = "lightblue"  # Light blue for 60-80%
    color_80_plus = "mediumblue"  # Medium blue for 80%+
    
    # --- Panel 1: PM probability ---
    ax = axes[0]
    for label, threshold_low, threshold_high, color in [
        ("60-80%", 0.6, 0.8, color_60_80),
        ("80%+", 0.8, 1.0, color_80_plus),
    ]:
        mask = valid_photo & (pm_prob >= threshold_low) & (pm_prob < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by PM probability{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 2: Parallax probability ---
    ax = axes[1]
    for label, threshold_low, threshold_high, color in [
        ("60-80%", 0.6, 0.8, color_60_80),
        ("80%+", 0.8, 1.0, color_80_plus),
    ]:
        mask = valid_photo & (parallax_prob >= threshold_low) & (parallax_prob < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by Parallax probability{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 3: Bayesian posterior ---
    ax = axes[2]
    for label, threshold_low, threshold_high, color in [
        ("60-80%", 0.6, 0.8, color_60_80),
        ("80%+", 0.8, 1.0, color_80_plus),
    ]:
        mask = valid_photo & (bayesian_posterior >= threshold_low) & (bayesian_posterior < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"CMD by Combined probability{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_comparison_cuts.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_comparison_cuts.png")





def plot_individual_probabilities(
    pm_prob: np.ndarray,
    parallax_prob: np.ndarray,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """Create individual plots for each probability type (full range + 50%+)."""
    if not HAS_MATPLOTLIB:
        return
    
    # --- Individual plots: Full range (0-100%) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PM probability
    ax = axes[0]
    ax.hist(pm_prob, bins=50, range=(0, 1), alpha=0.7, color="steelblue", 
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("PM membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability
    ax = axes[1]
    ax.hist(parallax_prob, bins=50, range=(0, 1), alpha=0.7, color="darkorange",
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Parallax membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior
    ax = axes[2]
    ax.hist(bayesian_posterior, bins=50, range=(0, 1), alpha=0.7, color="crimson",
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("P(cluster | PM + parallax)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Combined probability distribution")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_individual_probabilities_full.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_individual_probabilities_full.png")
    
    # --- Individual plots: 50%+ only ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PM probability (50%+ only)
    ax = axes[0]
    mask_pm_50 = pm_prob >= 0.5
    if mask_pm_50.any():
        ax.hist(pm_prob[mask_pm_50], bins=40, range=(0.5, 1.0), alpha=0.7, color="steelblue",
                edgecolor="black", linewidth=0.6)
    ax.set_xlabel("PM membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability (50%+ only)
    ax = axes[1]
    mask_parallax_50 = parallax_prob >= 0.5
    if mask_parallax_50.any():
        ax.hist(parallax_prob[mask_parallax_50], bins=40, range=(0.5, 1.0), alpha=0.7, color="darkorange",
                edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Parallax membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior (50%+ only)
    ax = axes[2]
    mask_bayesian_50 = bayesian_posterior >= 0.5
    if mask_bayesian_50.any():
        ax.hist(bayesian_posterior[mask_bayesian_50], bins=40, range=(0.5, 1.0), alpha=0.7, color="crimson",
                edgecolor="black", linewidth=0.6)
    ax.set_xlabel("P(cluster | PM + parallax)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Combined probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_individual_probabilities_50plus.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_individual_probabilities_50plus.png")











def robust_scatter(values: np.ndarray) -> float:
    """Robust 1-sigma scatter estimate from MAD."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def cmd_axis_labels(mag_col: str, color_col: str) -> tuple[str, str, str]:
    """Return CMD axis labels and title suffix matching raw/corrected photometry."""
    corrected_cmd = mag_col == "phot_g_mean_mag_corrected" and color_col == "bp_rp_corrected"
    if corrected_cmd:
        return "BP − RP corrected", "G corrected (mag)", " (empirical extinction corrected)"
    return "BP − RP", "G (mag)", ""


def cmd_extinction_zoom_limits() -> tuple[float, float, float, float]:
    """Return the fixed CMD diagnostic box as (color_min, color_max, mag_min, mag_max)."""
    return (
        CMD_EXTINCTION_BOX_COLOR_MIN,
        CMD_EXTINCTION_BOX_COLOR_MAX,
        CMD_EXTINCTION_BOX_MAG_MIN,
        CMD_EXTINCTION_BOX_MAG_MAX,
    )


def add_cmd_extinction_zoom_box(ax) -> None:
    """Draw the standardized extinction-analysis CMD zoom rectangle on an axis."""
    color_min, color_max, mag_min, mag_max = cmd_extinction_zoom_limits()
    rect = Rectangle(
        (color_min, mag_min),
        color_max - color_min,
        mag_max - mag_min,
        fill=False,
        edgecolor="crimson",
        linewidth=1.8,
        linestyle="--",
        alpha=0.95,
        zorder=10,
    )
    ax.add_patch(rect)
    ax.text(
        color_min,
        mag_min,
        " fixed extinction\n zoom box",
        color="crimson",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
        zorder=11,
    )





def select_cmd_photometry(
    df: pd.DataFrame,
    apply_extinction_correction: bool = False,
) -> tuple[str, str]:
    """
    Select CMD photometry columns for plotting without recomputing corrections.

    Extinction-corrected CMD plots are allowed only when the one-time precompute
    step has already added valid corrected columns to the result table.  Plotting
    must not call compute_spatial_extinction_correction() itself, because a failed
    precompute should fall back to raw photometry once rather than being retried by
    every CMD plot.
    """
    corrected_cols = ["phot_g_mean_mag_corrected", "bp_rp_corrected"]
    if apply_extinction_correction and all(col in df.columns for col in corrected_cols):
        valid_corrected = (
            df["phot_g_mean_mag_corrected"].notna()
            & df["bp_rp_corrected"].notna()
        )
        # Legacy outputs can contain raw copies labeled as corrected. Do not
        # advertise a corrected render variant when no correction was applied.
        changed = (
            (df["phot_g_mean_mag_corrected"] - df["phot_g_mean_mag"]).abs() > 1e-10
        ) | ((df["bp_rp_corrected"] - df["bp_rp"]).abs() > 1e-10)
        if (valid_corrected & changed).any():
            return "phot_g_mean_mag_corrected", "bp_rp_corrected"

    return "phot_g_mean_mag", "bp_rp"


def has_precomputed_extinction_photometry(df: pd.DataFrame) -> bool:
    """Return True only when valid corrected CMD columns are present in the table."""
    return select_cmd_photometry(df, apply_extinction_correction=True) == (
        "phot_g_mean_mag_corrected",
        "bp_rp_corrected",
    )


def fallback_photometry_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Safely return raw CMD arrays, using NaNs when columns are unavailable."""
    color = df["bp_rp"].values.astype(float) if "bp_rp" in df.columns else np.full(len(df), np.nan)
    mag = df["phot_g_mean_mag"].values.astype(float) if "phot_g_mean_mag" in df.columns else np.full(len(df), np.nan)
    return color, mag


def compute_spatial_extinction_correction(
    df: pd.DataFrame,
    prob_col: str = "final_member_prob",
    prob_threshold: float = CORE_MEMBER_PROB_THRESHOLD,
    n_bins: int = 5,
    smoothing_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute independent 2D spatial extinction corrections for both magnitude and color.
    
    Uses only high-probability core members (default: final_member_prob >= 0.8)
    to map magnitude and color residuals across the field, then creates separate
    smooth correction surfaces for each dimension.  The resulting correction
    surfaces are still evaluated for every star so corrected CMD columns remain
    available for the full table, but non-core stars never define the extinction
    surface.
    
    Args:
        df: DataFrame with photometry and coordinates
        prob_col: Column name for membership probabilities used to define the
            core reference group
        prob_threshold: Minimum probability to use as reference stars; defaults
            to 0.8 for the 80%+ core group
        n_bins: Number of spatial bins for initial correction map
        smoothing_sigma: Gaussian smoothing sigma (in bins) for correction surface
    
    Returns:
        corrected_color: Array of corrected BP-RP colors
        corrected_mag: Array of corrected G magnitudes  
        correction_info: Dict with correction metrics and parameters
    """
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    
    fallback_color, fallback_mag = fallback_photometry_arrays(df)

    required = ["ra", "dec", "phot_g_mean_mag", "bp_rp"]
    if any(col not in df.columns for col in required):
        print("  (Spatial extinction correction: missing required columns)")
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "missing_columns"
        }

    # If the pipeline already prepared corrected photometry, reuse it instead of
    # recomputing a potentially slightly different smoothed correction surface in
    # every plotting function.  This keeps all CMD outputs and the saved CSV
    # internally consistent for a given run.
    if "bp_rp_corrected" in df.columns and "phot_g_mean_mag_corrected" in df.columns:
        corrected_color = df["bp_rp_corrected"].values.astype(float)
        corrected_mag = df["phot_g_mean_mag_corrected"].values.astype(float)
        if has_precomputed_extinction_photometry(df):
            return corrected_color, corrected_mag, {
                "success": True,
                "reason": "precomputed_columns",
                "prob_col": prob_col,
                "prob_threshold": prob_threshold,
                "n_reference_stars": int(df.get("extinction_reference_star", pd.Series(False, index=df.index)).sum())
                if "extinction_reference_star" in df.columns else np.nan,
                "median_correction_color": float(np.nanmedian(df["bp_rp"] - corrected_color)),
                "median_correction_mag": float(np.nanmedian(df["phot_g_mean_mag"] - corrected_mag)),
                "max_correction_color": float(np.nanmax(np.abs(df["bp_rp"] - corrected_color))),
                "max_correction_mag": float(np.nanmax(np.abs(df["phot_g_mean_mag"] - corrected_mag))),
            }
    
    g = df["phot_g_mean_mag"].values.astype(float)
    color = df["bp_rp"].values.astype(float)
    x, y, _, _ = add_projected_xy(df)
    
    # Use only the high-probability core group as the reference set for
    # calculating the empirical extinction/differential-reddening surface.  Do
    # not fall back to all stars here: all stars should receive the correction,
    # but field stars should not define it.
    if prob_col not in df.columns:
        print(f"  (Spatial extinction correction: missing reference probability column '{prob_col}')")
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "missing_probability_column",
            "prob_col": prob_col,
            "prob_threshold": prob_threshold,
        }

    prob_values = df[prob_col].values.astype(float)
    finite_prob = np.isfinite(prob_values)
    if not finite_prob.any():
        print(f"  (Spatial extinction correction: no finite values in '{prob_col}')")
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "invalid_probability_column",
            "prob_col": prob_col,
            "prob_threshold": prob_threshold,
        }

    membership_mask = finite_prob & (prob_values >= prob_threshold)
    n_core_candidates = int(membership_mask.sum())
    if n_core_candidates == 0:
        print(
            f"  (Spatial extinction correction: no core reference stars with "
            f"{prob_col} >= {prob_threshold:.2f})"
        )
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "no_core_reference_stars",
            "prob_col": prob_col,
            "prob_threshold": prob_threshold,
            "n_core_candidates": n_core_candidates,
        }
    
    # Focus on main sequence stars for better reference
    # Use stars with G > 14 to avoid bright giants, G < 18 for good photometry
    ms_mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(color)
        & membership_mask
        & (g > 14.0) & (g < 18.0)
    )
    
    if ms_mask.sum() < 10:
        print(
            f"  (Spatial extinction correction: too few core reference stars after "
            f"photometry cuts; {ms_mask.sum()} usable from {n_core_candidates} "
            f"stars with {prob_col} >= {prob_threshold:.2f})"
        )
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "insufficient_reference_stars",
            "reference_mask": ms_mask,
            "n_reference_stars": int(ms_mask.sum()),
            "n_core_candidates": n_core_candidates,
            "prob_col": prob_col,
            "prob_threshold": prob_threshold,
        }
    
    xx = x[ms_mask]
    yy = y[ms_mask]
    cc = color[ms_mask]
    gg = g[ms_mask]
    
    # Compute independent residuals for both color and magnitude
    median_color = np.nanmedian(cc)
    median_mag = np.nanmedian(gg)
    delta_color = cc - median_color
    delta_mag = gg - median_mag
    
    # Create spatial bins for correction maps
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), n_bins + 1)
    
    # Bin both color and magnitude residuals independently
    binned_delta_color = np.full((n_bins, n_bins), np.nan)
    binned_delta_mag = np.full((n_bins, n_bins), np.nan)
    binned_count = np.zeros((n_bins, n_bins), dtype=int)
    
    for ix in range(n_bins):
        for iy in range(n_bins):
            in_cell = (
                (xx >= x_edges[ix]) & (xx <= x_edges[ix + 1])
                & (yy >= y_edges[iy]) & (yy <= y_edges[iy + 1])
            )
            if in_cell.sum() >= 2:
                binned_delta_color[iy, ix] = np.nanmedian(delta_color[in_cell])
                binned_delta_mag[iy, ix] = np.nanmedian(delta_mag[in_cell])
                binned_count[iy, ix] = int(in_cell.sum())
    
    # Check spatial coverage
    finite_mask = np.isfinite(binned_delta_color) & np.isfinite(binned_delta_mag)
    if finite_mask.sum() < 4:
        print("  (Spatial extinction correction: insufficient spatial coverage)")
        return fallback_color, fallback_mag, {
            "success": False,
            "reason": "insufficient_spatial_coverage",
            "reference_mask": ms_mask,
            "n_reference_stars": int(ms_mask.sum()),
            "n_core_candidates": n_core_candidates,
            "prob_col": prob_col,
            "prob_threshold": prob_threshold,
        }
    
    # Replace NaN with median for smoothing
    delta_color_filled = binned_delta_color.copy()
    delta_mag_filled = binned_delta_mag.copy()
    delta_color_filled[~np.isfinite(binned_delta_color)] = np.nanmedian(binned_delta_color[finite_mask])
    delta_mag_filled[~np.isfinite(binned_delta_mag)] = np.nanmedian(binned_delta_mag[finite_mask])
    
    # Apply Gaussian smoothing to both maps independently
    delta_color_smoothed = gaussian_filter(delta_color_filled, sigma=smoothing_sigma)
    delta_mag_smoothed = gaussian_filter(delta_mag_filled, sigma=smoothing_sigma)
    
    # Create interpolators for both correction fields
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    interpolator_color = RegularGridInterpolator(
        (y_centers, x_centers),
        delta_color_smoothed,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    interpolator_mag = RegularGridInterpolator(
        (y_centers, x_centers),
        delta_mag_smoothed,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    # The grid is defined at bin CENTERS, but stars extend to bin EDGES.
    # Hold the nearest boundary value outside the center grid instead of
    # silently assigning zero to the outer half of every edge bin. Before the
    # RA wrap fix, NGC 7789's huge artificial gap put every star outside this
    # center grid, so both arrays were identically zero despite success=True.
    points = np.column_stack([
        np.clip(y, y_centers[0], y_centers[-1]),
        np.clip(x, x_centers[0], x_centers[-1]),
    ])
    valid_position = np.isfinite(x) & np.isfinite(y)
    correction_color = np.zeros(len(df), dtype=float)
    correction_mag = np.zeros(len(df), dtype=float)
    correction_color[valid_position] = interpolator_color(points[valid_position])
    correction_mag[valid_position] = interpolator_mag(points[valid_position])
    
    corrected_color = color - correction_color
    corrected_mag = g - correction_mag
    
    # Compute improvement metrics
    original_color_scatter = robust_scatter(delta_color)
    original_mag_scatter = robust_scatter(delta_mag)
    corrected_color_residuals = corrected_color[ms_mask] - np.nanmedian(corrected_color[ms_mask])
    corrected_mag_residuals = corrected_mag[ms_mask] - np.nanmedian(corrected_mag[ms_mask])
    corrected_color_scatter = robust_scatter(corrected_color_residuals)
    corrected_mag_scatter = robust_scatter(corrected_mag_residuals)
    
    color_improvement = original_color_scatter - corrected_color_scatter
    mag_improvement = original_mag_scatter - corrected_mag_scatter
    color_improvement_fraction = color_improvement / original_color_scatter if original_color_scatter > 0 else 0
    mag_improvement_fraction = mag_improvement / original_mag_scatter if original_mag_scatter > 0 else 0

    correction_info = {
        "success": True,
        "reference_selection": f"{prob_col} >= {prob_threshold:.2f}",
        "prob_col": prob_col,
        "prob_threshold": prob_threshold,
        "n_core_candidates": n_core_candidates,
        "n_bins": n_bins,
        "smoothing_sigma": smoothing_sigma,
        "n_reference_stars": int(ms_mask.sum()),
        "reference_mask": ms_mask,
        "original_color_scatter": original_color_scatter,
        "corrected_color_scatter": corrected_color_scatter,
        "original_mag_scatter": original_mag_scatter,
        "corrected_mag_scatter": corrected_mag_scatter,
        "color_scatter_improvement": color_improvement,
        "mag_scatter_improvement": mag_improvement,
        "color_improvement_fraction": color_improvement_fraction,
        "mag_improvement_fraction": mag_improvement_fraction,
        "median_correction_color": float(np.nanmedian(correction_color)),
        "median_correction_mag": float(np.nanmedian(correction_mag)),
        "max_correction_color": float(np.nanmax(np.abs(correction_color))),
        "max_correction_mag": float(np.nanmax(np.abs(correction_mag))),
        "n_stars_corrected": int(((np.abs(correction_color) > 1e-10)
                                  | (np.abs(correction_mag) > 1e-10)).sum()),
    }

    if correction_info["n_stars_corrected"] == 0:
        correction_info.update(success=False, reason="no_spatial_correction")
        print("  (Spatial extinction correction: no nonzero correction available)")
        return fallback_color, fallback_mag, correction_info

    # Guard against using a smoothing surface that makes both CMD dimensions worse.
    # Small negative fractions can happen from noise; require a meaningful worsening
    # before rejecting the correction.
    worsens_color = np.isfinite(color_improvement_fraction) and color_improvement_fraction < -0.02
    worsens_mag = np.isfinite(mag_improvement_fraction) and mag_improvement_fraction < -0.02
    if worsens_color and worsens_mag:
        correction_info.update({
            "success": False,
            "reason": "scatter_worsened",
        })
        print("  (Spatial extinction correction rejected: both color and magnitude scatter worsened)")
        print(f"    Color: scatter {original_color_scatter:.4f} → {corrected_color_scatter:.4f} ({color_improvement_fraction*100:.1f}%)")
        print(f"    Mag:   scatter {original_mag_scatter:.4f} → {corrected_mag_scatter:.4f} ({mag_improvement_fraction*100:.1f}%)")
        return fallback_color, fallback_mag, correction_info
    
    print(f"  Spatial extinction correction applied:")
    print(f"    Color: scatter {original_color_scatter:.4f} → {corrected_color_scatter:.4f} ({color_improvement_fraction*100:.1f}%)")
    print(f"    Mag:   scatter {original_mag_scatter:.4f} → {corrected_mag_scatter:.4f} ({mag_improvement_fraction*100:.1f}%)")
    print(f"    Median corrections: color={correction_info['median_correction_color']:.4f}, mag={correction_info['median_correction_mag']:.4f}")
    
    return corrected_color, corrected_mag, correction_info


def add_extinction_corrected_photometry(
    df: pd.DataFrame,
    prob_col: str = "final_member_prob",
    prob_threshold: float = CORE_MEMBER_PROB_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """
    Add empirical differential-reddening/photometric-flattening columns once.

    The correction is intentionally saved as additional columns rather than
    replacing Gaia observed photometry.  It is an empirical spatial CMD correction
    for plotting/diagnostics, not an absolute extinction law correction using
    external E(B-V), A_G, or Gaia passband coefficients.

    Only the 80%+ core membership group (by default final_member_prob >= 0.8)
    is used to define the correction surface.  The resulting correction is then
    evaluated for every row so full-table CMD plots remain available.
    """
    corrected_color, corrected_mag, correction_info = compute_spatial_extinction_correction(
        df, prob_col=prob_col, prob_threshold=prob_threshold
    )
    if not correction_info.get("success", False):
        return df, correction_info

    out = df.copy()
    out["bp_rp_corrected"] = corrected_color
    out["phot_g_mean_mag_corrected"] = corrected_mag
    out["bp_rp_extinction_delta"] = out["bp_rp"].astype(float) - out["bp_rp_corrected"].astype(float)
    out["phot_g_extinction_delta"] = out["phot_g_mean_mag"].astype(float) - out["phot_g_mean_mag_corrected"].astype(float)

    reference_mask = correction_info.get("reference_mask")
    if reference_mask is not None and len(reference_mask) == len(out):
        out["extinction_reference_star"] = np.asarray(reference_mask, dtype=bool)
    elif prob_col in out.columns:
        out["extinction_reference_star"] = out[prob_col].astype(float) >= prob_threshold
    else:
        out["extinction_reference_star"] = False

    correction_info["n_reference_stars"] = int(out["extinction_reference_star"].sum())
    correction_info["reference_selection"] = f"{prob_col} >= {prob_threshold:.2f} plus CMD reference cuts"
    correction_info["prob_col"] = prob_col
    correction_info["prob_threshold"] = prob_threshold
    correction_info.pop("reference_mask", None)

    return out, correction_info


def plot_differential_reddening_color_gradient_diagnostic(
    df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
    prob_col: str = "final_member_prob",
):
    """
    QC diagnostic: map BP-RP color across the sky for stars with precise-enough
    photometry (default G in 17--18). This does not alter membership selection.
    """
    empty_metrics = {
        "color_gradient_delta": np.nan,
        "color_gradient_residual_sigma": np.nan,
    }
    if not HAS_MATPLOTLIB:
        return empty_metrics

    required = ["ra", "dec", "phot_g_mean_mag", "bp_rp"]
    if any(col not in df.columns for col in required):
        print("  (Skipped color-gradient diagnostic: missing ra/dec/G/BP-RP columns)")
        return empty_metrics

    g = df["phot_g_mean_mag"].values.astype(float)
    color = df["bp_rp"].values.astype(float)
    x, y, _, _ = add_projected_xy(df)

    # Use high-confidence final members when available; otherwise fall back to all stars.
    if prob_col in df.columns and not df[prob_col].isna().all():
        membership_mask = df[prob_col].values.astype(float) >= 0.8
        membership_label = f"{prob_col} ≥ 0.8"
    else:
        membership_mask = np.ones(len(df), dtype=bool)
        membership_label = "all valid stars"

    precise = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(color)
        & (g > 17.0) & (g < 18.0) & membership_mask
    )

    if precise.sum() < 8:
        # Relax only the magnitude window, not the membership criterion, so sparse
        # clusters still produce a QC plot while remaining focused on bright stars.
        precise = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(color)
            & (g > 16.5) & (g < 18.5) & membership_mask
        )
        mag_label = "16.5 < G < 18.5 (fallback)"
    else:
        mag_label = "17 < G < 18"

    if precise.sum() < 5:
        print("  (Skipped color-gradient diagnostic: too few stars in precise-photometry sample)")
        return empty_metrics

    xx = x[precise]
    yy = y[precise]
    cc = color[precise]
    delta_color = cc - np.nanmedian(cc)
    resid_scatter = robust_scatter(delta_color)

    # Bin the field spatially and show the median color residual in each cell.
    n_bins = 5 if len(cc) >= 40 else 4
    x_edges = np.linspace(np.nanmin(xx), np.nanmax(xx), n_bins + 1)
    y_edges = np.linspace(np.nanmin(yy), np.nanmax(yy), n_bins + 1)
    binned_delta = np.full((n_bins, n_bins), np.nan)
    binned_count = np.zeros((n_bins, n_bins), dtype=int)
    for ix in range(n_bins):
        for iy in range(n_bins):
            in_cell = (
                (xx >= x_edges[ix]) & (xx <= x_edges[ix + 1])
                & (yy >= y_edges[iy]) & (yy <= y_edges[iy + 1])
            )
            if in_cell.sum() >= 2:
                binned_delta[iy, ix] = np.nanmedian(delta_color[in_cell])
                binned_count[iy, ix] = int(in_cell.sum())

    finite_bins = np.isfinite(binned_delta)
    color_gradient_delta = (
        float(np.nanmax(binned_delta[finite_bins]) - np.nanmin(binned_delta[finite_bins]))
        if finite_bins.any() else np.nan
    )

    # Simple left/right split for the advisor's eyeball question.
    left = xx < np.nanmedian(xx)
    right = ~left
    left_med = float(np.nanmedian(delta_color[left])) if left.any() else np.nan
    right_med = float(np.nanmedian(delta_color[right])) if right.any() else np.nan

    vmax = max(0.03, float(np.nanpercentile(np.abs(delta_color), 95)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    ax = axes[0]
    sc = ax.scatter(
        xx, yy, c=delta_color, s=28, cmap="coolwarm", alpha=0.9,
        edgecolors="none", vmin=-vmax, vmax=vmax
    )
    ax.set_xlabel("Projected X = ΔRA cos(Dec) [deg]")
    ax.set_ylabel("Projected Y = ΔDec [deg]")
    ax.set_title(f"Color residuals on sky\n{mag_label}, {membership_label}, N={len(cc)}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Δ(BP − RP) from slice median")

    ax = axes[1]
    im = ax.imshow(
        binned_delta,
        origin="lower",
        extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()],
        cmap="coolwarm",
        aspect="auto", vmin=-vmax, vmax=vmax,
    )
    for ix in range(n_bins):
        for iy in range(n_bins):
            if binned_count[iy, ix] > 0:
                ax.text(
                    0.5 * (x_edges[ix] + x_edges[ix + 1]),
                    0.5 * (y_edges[iy] + y_edges[iy + 1]),
                    str(binned_count[iy, ix]),
                    ha="center", va="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.5, edgecolor="none"),
                )
    ax.set_xlabel("Projected X [deg]")
    ax.set_ylabel("Projected Y [deg]")
    ax.set_title(
        "Binned median color residual\n"
        f"cell range≈{color_gradient_delta:.3f} mag; σ_rob={resid_scatter:.3f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("median Δ(BP − RP)")

    ax = axes[2]
    bins = np.linspace(-vmax, vmax, 18)
    if left.any():
        ax.hist(delta_color[left], bins=bins, alpha=0.65, color="#4477AA", edgecolor="black",
                label=f"left/X-low median={left_med:+.3f}")
    if right.any():
        ax.hist(delta_color[right], bins=bins, alpha=0.65, color="#CC6677", edgecolor="black",
                label=f"right/X-high median={right_med:+.3f}")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Δ(BP − RP) from slice median")
    ax.set_ylabel("Number of stars")
    ax.set_title("Left/right color-residual comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_qc_color_gradient_diagnostic.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_qc_color_gradient_diagnostic.png")
    return {
        "color_gradient_delta": color_gradient_delta,
        "color_gradient_residual_sigma": resid_scatter,
    }


def plot_blue_straggler_parallax_validation(
    df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
    prob_col: str = "final_member_prob",
):
    """
    QC diagnostic comparing parallaxes for ridge-selected Blue Straggler-like
    candidates and a post-selection Main Sequence ridge-strip comparison sample.

    The BSS candidate geometry is derived from the full reliable, high-confidence
    member CMD ridge.  The MS comparison sample is selected only after the
    turnoff/BSS candidates are defined, as a ridge-centered strip below the
    turnoff.  This does not alter membership selection.
    """
    empty_metrics = {
        "N_BSS": 0,
        "N_MS": 0,
        "median_parallax_BSS": np.nan,
        "median_parallax_MS": np.nan,
        "delta_median_parallax": np.nan,
        "robust_sigma_BSS": np.nan,
        "robust_sigma_MS": np.nan,
    }
    if not HAS_MATPLOTLIB:
        return empty_metrics

    required = ["phot_g_mean_mag", "bp_rp", PARALLAX_COL]
    if any(col not in df.columns for col in required):
        print("  (Skipped blue-straggler parallax validation: missing CMD/parallax columns)")
        return empty_metrics

    g = df["phot_g_mean_mag"].values.astype(float)
    color = df["bp_rp"].values.astype(float)
    plx = df[PARALLAX_COL].values.astype(float)
    valid_cmd = np.isfinite(g) & np.isfinite(color)
    valid_plx = np.isfinite(plx)
    valid = valid_cmd & valid_plx

    if prob_col in df.columns and not df[prob_col].isna().all():
        member_mask = df[prob_col].values.astype(float) >= 0.8
        member_label = f"{prob_col} ≥ 0.8"
    else:
        member_mask = np.ones(len(df), dtype=bool)
        member_label = "all valid stars"

    # Trace the ridge from the full reliable high-confidence member sequence,
    # not from a bright/MS comparison band.  The 17<G<18 color-gradient slice is
    # unrelated to this QA and must not constrain BSS/MS ridge tracing.
    ridge_sample = valid_cmd & member_mask
    if ridge_sample.sum() < 15:
        print("  (Skipped blue-straggler parallax validation: too few member stars for ridge tracing)")
        return empty_metrics

    gm = g[ridge_sample]
    cm = color[ridge_sample]

    def smooth_series(values: np.ndarray) -> np.ndarray:
        """Small dependency-free triangular smoothing for ridge colors."""
        if len(values) < 3:
            return values
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        if len(values) < len(kernel):
            kernel = np.ones(3)
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        padded = np.pad(values, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def trace_cmd_ridge(g_values: np.ndarray, c_values: np.ndarray):
        """
        Trace the main-sequence ridge as median BP-RP in sliding G bins.

        Bins are intentionally magnitude-based over the available high-confidence
        CMD range; no pre-limited bright/MS comparison band is used.  The turnoff
        is later derived from ridge morphology.
        """
        finite = np.isfinite(g_values) & np.isfinite(c_values)
        if finite.sum() < 15:
            return None
        g_values = g_values[finite]
        c_values = c_values[finite]

        g_lo, g_hi = float(np.nanmin(g_values)), float(np.nanmax(g_values))
        if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_hi <= g_lo:
            return None

        n_bins = int(np.clip(len(g_values) // 8, 8, 32))
        edges = np.linspace(g_lo, g_hi, n_bins + 1)
        ridge_g = []
        ridge_c = []
        min_bin_count = 3 if len(g_values) < 80 else 5
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (g_values >= lo) & (g_values < hi)
            if hi == edges[-1]:
                in_bin = (g_values >= lo) & (g_values <= hi)
            if in_bin.sum() < min_bin_count:
                continue
            # Use the blue/central ridge rather than the red binary/evolved tail.
            c_bin = c_values[in_bin]
            ridge_g.append(0.5 * (lo + hi))
            ridge_c.append(float(np.nanpercentile(c_bin, 45)))

        if len(ridge_g) < 5:
            return None

        ridge_g = np.asarray(ridge_g, dtype=float)
        ridge_c = smooth_series(np.asarray(ridge_c, dtype=float))
        order = np.argsort(ridge_g)
        return ridge_g[order], ridge_c[order]

    ridge = trace_cmd_ridge(gm, cm)
    if ridge is None:
        print("  (Skipped blue-straggler parallax validation: could not trace CMD ridge)")
        return empty_metrics

    ridge_g, ridge_c = ridge

    def estimate_turnoff_from_lower_ms(rg: np.ndarray, rc: np.ndarray) -> int:
        """
        Estimate turnoff as the brightest ridge point still consistent with the
        lower/main-sequence trend, not as the bluest upper-ridge point.

        The traced ridge is ordered from bright to faint (increasing G).  We use
        the faint/lower portion to define the monotonic lower-MS color--magnitude
        trend, then walk from the faint end toward brighter magnitudes until the
        ridge departs from that trend in slope or residual.  This avoids using a
        subgiant/turnoff hook blue extremum as the anchor in old open clusters.
        """
        n = len(rg)
        if n < 5:
            return int(np.argmin(rg))

        slope = np.gradient(rc, rg)
        lower_anchor = rg >= np.nanpercentile(rg, 55)
        if lower_anchor.sum() < 4:
            lower_anchor = rg >= np.nanmedian(rg)
        if lower_anchor.sum() < 4:
            lower_anchor = np.ones(n, dtype=bool)

        lower_slope = float(np.nanmedian(slope[lower_anchor]))
        if not np.isfinite(lower_slope):
            lower_slope = 0.0

        # Robust lower-MS trend and residual scale.
        try:
            fit_coeff = np.polyfit(rg[lower_anchor], rc[lower_anchor], deg=1)
            trend_c = np.polyval(fit_coeff, rg)
            lower_resid = rc[lower_anchor] - np.polyval(fit_coeff, rg[lower_anchor])
            resid_sigma = robust_scatter(lower_resid)
        except Exception:
            trend_c = np.interp(rg, rg[lower_anchor], rc[lower_anchor])
            resid_sigma = robust_scatter(rc[lower_anchor] - np.nanmedian(rc[lower_anchor]))

        if not np.isfinite(resid_sigma) or resid_sigma <= 0:
            resid_sigma = 0.04

        # Consistency with the lower MS: the local ridge should keep the lower-MS
        # redder-when-fainter slope and remain close to the lower-MS extrapolation.
        slope_floor = max(0.01, 0.25 * lower_slope) if lower_slope > 0 else -0.02
        resid_tol = max(0.08, 3.0 * resid_sigma)
        consistent = (slope >= slope_floor) & (np.abs(rc - trend_c) <= resid_tol)

        # The anchor portion defines the lower MS by construction.
        consistent[lower_anchor] = True

        # Walk from faint to bright and choose the brightest point in the
        # contiguous lower-MS sequence.  Allow one isolated failed bin so noisy
        # ridge bins do not prematurely terminate the sequence.
        first_consistent_idx = n - 1
        misses = 0
        for idx in range(n - 1, -1, -1):
            if consistent[idx]:
                first_consistent_idx = idx
                misses = 0
            else:
                misses += 1
                if misses > 1:
                    break

        return int(first_consistent_idx)

    # Turnoff proxy: brightest point where the full traced ridge is still
    # consistent with the monotonic lower/main-sequence trend.  This explicitly
    # avoids choosing the bluest point on the upper ridge/subgiant hook.
    turnoff_idx = estimate_turnoff_from_lower_ms(ridge_g, ridge_c)
    turnoff_mag = float(ridge_g[turnoff_idx])
    turnoff_color = float(ridge_c[turnoff_idx])

    # For stars brighter than the first ridge bin, compare against the turnoff
    # color rather than rejecting them for being outside the interpolation range.
    # Fainter than the traced ridge remains invalid for ridge-strip comparisons.
    ridge_color_at_star = np.interp(g, ridge_g, ridge_c, left=turnoff_color, right=np.nan)
    color_resid = color - ridge_color_at_star
    ridge_valid = np.isfinite(ridge_color_at_star)

    # Estimate ridge width from stars below the turnoff; use conservative floors
    # so sparse clusters still get usable geometric regions.
    ms_width_sample = (
        ridge_sample & ridge_valid
        & (g >= turnoff_mag + 0.35)
        & (g <= min(turnoff_mag + 2.4, float(np.nanmax(ridge_g))))
    )
    if ms_width_sample.sum() >= 8:
        ridge_sigma = robust_scatter(color_resid[ms_width_sample])
    else:
        ridge_sigma = robust_scatter(color_resid[ridge_sample & ridge_valid])
    if not np.isfinite(ridge_sigma) or ridge_sigma <= 0:
        ridge_sigma = 0.08
    ridge_sigma = float(np.clip(ridge_sigma, 0.06, 0.18))

    # Blue Straggler-like QA sample: existing high-confidence members that are
    # brighter than the morphology-derived turnoff and blueward of the local
    # turnoff/main-sequence geometry.  No rectangular classifier/box is used.
    bss_red_boundary = np.minimum(ridge_color_at_star, turnoff_color) - 0.35 * ridge_sigma

    bs_mask = (
        valid & member_mask & ridge_valid
        & (g < turnoff_mag)
        & (color < bss_red_boundary)
    )

    # Only after defining BSS candidates, select the MS comparison stars as a
    # ridge-centered strip below the turnoff across the traced reliable ridge.
    # There is deliberately no hard 17<G<18, 18.5-mag, or preset bright-band cap;
    # any faint/reliability limit should enter through the available parallax data
    # rather than clipping the CMD comparison geometry plotted here.
    ms_g_min = turnoff_mag + 0.35
    ms_g_max = float(np.nanmax(ridge_g))
    ms_blue_offset = -1.25 * ridge_sigma
    ms_red_offset = 1.75 * ridge_sigma

    ms_mask = (
        valid & member_mask & ridge_valid
        & (g >= ms_g_min) & (g <= ms_g_max)
        & (color_resid >= ms_blue_offset) & (color_resid <= ms_red_offset)
    )

    if bs_mask.sum() == 0 or ms_mask.sum() < 3:
        print("  (Skipped blue-straggler parallax validation: insufficient BS/MS comparison stars)")
        return empty_metrics

    bs_plx = plx[bs_mask]
    ms_plx = plx[ms_mask]

    def stats_text(values):
        return (
            f"N={len(values)}\n"
            f"median={np.median(values):.3f} mas\n"
            f"std={np.std(values, ddof=1) if len(values) > 1 else np.nan:.3f} mas\n"
            f"robust σ={robust_scatter(values):.3f} mas"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    ax = axes[0]
    ax.scatter(color[valid_cmd], g[valid_cmd], s=5, c="#D0D0D0", alpha=0.25, label="all valid CMD stars")
    ax.scatter(color[valid_cmd & member_mask], g[valid_cmd & member_mask], s=8, c="#8F8F8F", alpha=0.45,
               label=member_label)
    ax.scatter(color[ms_mask], g[ms_mask], s=30, c="#117733", alpha=0.9, edgecolors="black", linewidths=0.4,
               label=f"MS ridge-strip comparison (N={ms_mask.sum()})")
    ax.scatter(color[bs_mask], g[bs_mask], s=36, c="#4477AA", alpha=0.95, edgecolors="black", linewidths=0.5,
               label=f"Blue Straggler candidates (N={bs_mask.sum()})")
    ax.plot(ridge_c, ridge_g, color="black", lw=2.0, label="traced CMD ridge")
    ax.scatter([turnoff_color], [turnoff_mag], marker="*", s=170, c="#CC6677",
               edgecolors="black", linewidths=0.6, zorder=6, label="ridge turnoff proxy")

    ms_line_g = np.linspace(ms_g_min, ms_g_max, 120)
    ms_line_c = np.interp(ms_line_g, ridge_g, ridge_c)
    ax.plot(ms_line_c + ms_blue_offset, ms_line_g, color="#117733", lw=2, ls="--",
            label="MS ridge strip")
    ax.plot(ms_line_c + ms_red_offset, ms_line_g, color="#117733", lw=2, ls="--")
    ax.fill_betweenx(ms_line_g, ms_line_c + ms_blue_offset, ms_line_c + ms_red_offset,
                     color="#117733", alpha=0.10)
    ax.axhline(turnoff_mag, color="#CC6677", lw=1.2, ls=":", alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title(
        "Ridge-derived turnoff-relative CMD validation regions\n"
        f"turnoff≈({turnoff_color:.2f}, G={turnoff_mag:.2f}), ridge σ≈{ridge_sigma:.2f}"
    )
    ax.legend(loc="best", framealpha=0.85, fontsize=8)
    ax.grid(True, alpha=0.3)

    all_plx = np.concatenate([bs_plx, ms_plx])
    lo, hi = np.nanpercentile(all_plx, [1, 99])
    pad = max(0.02, 0.15 * (hi - lo))
    bins = np.linspace(lo - pad, hi + pad, 20)

    for ax, values, title, color_name in [
        (axes[1], bs_plx, "Blue Straggler candidate parallaxes", "#4477AA"),
        (axes[2], ms_plx, "MS ridge-strip comparison parallaxes", "#117733"),
    ]:
        ax.hist(values, bins=bins, color=color_name, alpha=0.7, edgecolor="black", density=False)
        ax.axvline(np.median(values), color="crimson", ls="--", lw=1.8, label="median")
        if len(values) >= 3:
            try:
                kde = gaussian_kde(values)
                grid = np.linspace(bins.min(), bins.max(), 200)
                bin_width = bins[1] - bins[0]
                ax.plot(grid, kde(grid) * len(values) * bin_width, color="black", lw=1.5, label="KDE")
            except Exception:
                pass
        ax.text(0.03, 0.97, stats_text(values), ha="left", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=9)
        ax.set_xlabel("Parallax (mas)")
        ax.set_ylabel("Number of sources")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    median_diff = np.median(bs_plx) - np.median(ms_plx)
    fig.suptitle(
        f"{file_prefix} QC: Blue Straggler vs Main Sequence parallax validation "
        f"(median Δ={median_diff:+.3f} mas)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_qc_blue_straggler_parallax_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {file_prefix}_qc_blue_straggler_parallax_validation.png")
    return {
        "N_BSS": int(bs_mask.sum()),
        "N_MS": int(ms_mask.sum()),
        "median_parallax_BSS": float(np.median(bs_plx)),
        "median_parallax_MS": float(np.median(ms_plx)),
        "delta_median_parallax": float(median_diff),
        "robust_sigma_BSS": robust_scatter(bs_plx),
        "robust_sigma_MS": robust_scatter(ms_plx),
    }


def perform_quadrature_analysis(df, out_dir, file_prefix):
    """Plot the proper-motion distribution only; no seed is selected here."""
    pm_total = np.hypot(df.pmra, df.pmdec)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(pm_total[np.isfinite(pm_total)], bins=5000, color='steelblue', alpha=.7, edgecolor='black')
    ax.set(xlim=(0, 100), xlabel='Total Proper Motion [mas/yr]', ylabel='Number of Stars',
           title='Proper Motion Quadrature Distribution')
    ax.grid(alpha=.3)
    fig.savefig(out_dir / f'{file_prefix}_quadrature_analysis.png', dpi=150)
    plt.close(fig)






def run_three_model_analysis(
    pm_data: np.ndarray,
    pmra_errors: np.ndarray = None,
    pmdec_errors: np.ndarray = None,
    parallax_data: np.ndarray = None,
    parallax_errors: np.ndarray = None,
    df: pd.DataFrame = None,
    out_dir: Path = None,
    file_prefix: str = None,
    c_low: float = None,
    c_high: float = None,
    spatial_weight_override: float = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run complete three-model analysis: PM-only, parallax-only, and joint 3D.
    
    Returns:
        result_df: DataFrame with all model probabilities and metadata
        quality_summary: Quality assessment and flags
    """
    print("\n=== THREE-MODEL MEMBERSHIP ANALYSIS ===")
    seed_mask = df["manual_cmd_seed"].to_numpy(dtype=bool)
    seed_values = np.column_stack([pm_data, parallax_data])[seed_mask]
    print(f"Manual CMD starting stars: {seed_mask.sum()} / {len(df)}")
    print(f"Initial median (pmra, pmdec, parallax): {np.median(seed_values, axis=0)}")
    
    # Fixed field component (same for all models)
    center_field_pm, cov_field_pm = fixed_field_component(pm_data)

    # Adaptive per-cluster pi ceiling to avoid hard saturation at global MAX_PI
    max_pi_adaptive = estimate_adaptive_max_pi(pm_data)
    print(f"Adaptive max_pi: {max_pi_adaptive:.3f}")
    
    # Create proper 3D field component for joint model
    joint_data = np.column_stack([pm_data, parallax_data])
    center_field_3d, cov_field_3d = fixed_field_component_nd(joint_data)
    
    # Run PM-only model
    print("Running PM-only model...")
    pm_result = run_manual_seed_pm(
        pm_data, pmra_errors, pmdec_errors, center_field_pm, cov_field_pm,
        seed_mask=seed_mask, min_pi=MIN_PI, max_pi=max_pi_adaptive,
    )
    
    # Run parallax-only model
    print("Running parallax-only model...")
    plx_result = run_manual_seed_parallax(
        parallax_data, parallax_errors,
        seed_mask=seed_mask, min_pi=MIN_PI, max_pi=max_pi_adaptive,
    )
    
    # Build astrometric support probability (PM-primary, parallax-penalty).
    # PM drives membership; parallax only downweights strong disagreements.
    # This keeps spatial from creating core membership by itself while avoiding
    # over-suppression from conservative parallax-only fits.
    astro_prob = None
    if pm_result['status'] == 'SUCCESS' and plx_result['status'] == 'SUCCESS':
        pm_prob = np.clip(pm_result['best']['prob'], 0, 1)
        plx_prob = np.clip(plx_result['best']['prob'], 0, 1)
        astro_prob = pm_prob.copy()
        bad_plx = plx_prob < 0.2
        astro_prob[bad_plx] *= 0.5

    # Run 5D spatial+astrometric model
    print("Running Joint 5D Spatial+Astrometric Model...")
    joint_result = run_manual_seed_joint(
        pm_data, parallax_data, df,
        pmra_errors=pmra_errors, pmdec_errors=pmdec_errors, parallax_errors=parallax_errors,
        center_field=None, cov_field=None,
        astro_prob=astro_prob,
        c_low=c_low,
        c_high=c_high,
        spatial_weight_override=spatial_weight_override,
        seed_mask=seed_mask,
        min_pi=MIN_PI,
        max_pi=max_pi_adaptive,
    )

    # Save calibration diagnostics
    if astro_prob is not None:
        quality_summary_concentration = compute_spatial_concentration(
            *add_projected_xy(df)[:2], astro_prob
        )
    else:
        quality_summary_concentration = np.nan
    
    # Validate outputs
    if not validate_model_outputs(pm_result, plx_result, joint_result):
        print("WARNING: Model output validation failed!")
    
    # Quality assessment
    quality_summary = evaluate_solution_quality(pm_result, plx_result, joint_result, len(df))
    quality_summary['seed_stability_assessed'] = False
    quality_summary['max_pi_adaptive'] = max_pi_adaptive
    quality_summary['spatial_concentration'] = quality_summary_concentration
    quality_summary['spatial_c_low'] = c_low
    quality_summary['spatial_c_high'] = c_high
    
    # Prepare results DataFrame
    result_df = df.copy()
    
    # Add PM-only results
    if pm_result['status'] == 'SUCCESS':
        result_df['pm_only_prob'] = pm_result['best']['prob']
        result_df['pm_only_tier'] = membership_tier(pm_result['best']['prob'])
        result_df['pm_only_pi'] = pm_result['best']['pi_cluster']
        result_df['pm_only_center_pmra'] = pm_result['best']['center'][0]
        result_df['pm_only_center_pmdec'] = pm_result['best']['center'][1]
    else:
        result_df['pm_only_prob'] = np.nan
        result_df['pm_only_tier'] = 'non-member'
        result_df['pm_only_pi'] = np.nan
        result_df['pm_only_center_pmra'] = np.nan
        result_df['pm_only_center_pmdec'] = np.nan
    
    # Add parallax-only results
    if plx_result['status'] == 'SUCCESS':
        result_df['plx_only_prob'] = plx_result['best']['prob']
        result_df['plx_only_tier'] = membership_tier(plx_result['best']['prob'])
        result_df['plx_only_pi'] = plx_result['best']['pi_cluster']
        result_df['plx_only_center'] = plx_result['best']['center']
        result_df['plx_only_sigma'] = plx_result['best']['sigma']
    else:
        result_df['plx_only_prob'] = np.nan
        result_df['plx_only_tier'] = 'non-member'
        result_df['plx_only_pi'] = np.nan
        result_df['plx_only_center'] = np.nan
        result_df['plx_only_sigma'] = np.nan
    
    # Add joint 5D results
    if joint_result['status'] == 'SUCCESS':
        result_df['joint_5d_prob'] = joint_result['best']['prob']
        result_df['joint_prob'] = joint_result['best']['prob']  # backward-compatible alias
        result_df['joint_tier'] = membership_tier(joint_result['best']['prob'])
        result_df['joint_pi'] = joint_result['best']['pi_cluster']
        result_df['joint_center_pmra'] = joint_result['best']['center'][0]
        result_df['joint_center_pmdec'] = joint_result['best']['center'][1]
        result_df['joint_center_parallax'] = joint_result['best']['center'][2]
    else:
        result_df['joint_5d_prob'] = np.nan
        result_df['joint_prob'] = np.nan
        result_df['joint_tier'] = 'non-member'
        result_df['joint_pi'] = np.nan
        result_df['joint_center_pmra'] = np.nan
        result_df['joint_center_pmdec'] = np.nan
        result_df['joint_center_parallax'] = np.nan

    # Add explicit astrometric 3D probability output
    if astro_prob is not None:
        result_df['astro_3d_prob'] = astro_prob
    else:
        result_df['astro_3d_prob'] = np.nan
    
    # Add quality flags
    result_df['quality_flags'] = ', '.join(quality_summary['quality_flags'])

    # Final astrometry-gated membership probability.
    # Joint 5D may refine membership, but cannot rescue stars with poor astrometric support.
    result_df['final_prob'] = result_df['joint_5d_prob'].copy()
    bad_joint = (
        (result_df['joint_5d_prob'] >= 0.8)
        & (result_df['astro_3d_prob'] < 0.5)
    )
    result_df.loc[bad_joint, 'final_prob'] = np.minimum(
        result_df.loc[bad_joint, 'final_prob'],
        result_df.loc[bad_joint, 'astro_3d_prob']
    )

    # Backward-compatible alias + final tiering for science/member lists
    result_df['final_member_prob'] = result_df['final_prob']
    result_df['final_tier'] = membership_tier(result_df['final_prob'].fillna(0).values)
    
    # Print summary
    print(f"\nModel Results Summary:")
    print(f"  PM-only: {pm_result['status']}")
    if pm_result['status'] == 'SUCCESS':
        n_pm_core = (pm_result['best']['prob'] >= 0.8).sum()
        print(f"    Core members: {n_pm_core}")
        print(f"    π_cluster: {pm_result['best']['pi_cluster']:.3f}")
    
    print(f"  Parallax-only: {plx_result['status']}")
    if plx_result['status'] == 'SUCCESS':
        n_plx_core = (plx_result['best']['prob'] >= 0.8).sum()
        print(f"    Core members: {n_plx_core}")
        print(f"    π_cluster: {plx_result['best']['pi_cluster']:.3f}")
    
    print(f"  Joint 5D: {joint_result['status']}")
    if joint_result['status'] == 'SUCCESS':
        n_joint_core = (joint_result['best']['prob'] >= 0.8).sum()
        print(f"    Core members: {n_joint_core}")
        print(f"    π_cluster: {joint_result['best']['pi_cluster']:.3f}")
        print(f"    Center (pmra, pmdec, parallax): {joint_result['best']['center']}")

    if 'final_prob' in result_df.columns:
        n_final_core = int((result_df['final_prob'] >= 0.8).sum())
        print(f"  Final (astrometry-gated):")
        print(f"    Core members: {n_final_core}")
    
    print(f"\nQuality Assessment: {quality_summary['overall_status']}")
    if quality_summary['quality_flags']:
        print(f"  Flags: {', '.join(quality_summary['quality_flags'])}")
    
    if quality_summary['overlaps']:
        overlaps = quality_summary['overlaps']
        print(f"  Model Overlaps:")
        print(f"    PM-Parallax: {overlaps.get('pm_plx_overlap', 'N/A'):.2f}")
        print(f"    Joint-PM: {overlaps.get('joint_pm_overlap', 'N/A'):.2f}")
        print(f"    Joint-Parallax: {overlaps.get('joint_plx_overlap', 'N/A'):.2f}")
    
    return result_df, quality_summary




def make_plot_output_dirs(out_dir: Path, include_extinction: bool = False) -> dict:
    """Create and return standardized plot render-target directories."""
    plot_root = out_dir / "plots"
    plot_dirs = {
        "png": plot_root / "png",
        "html": plot_root / "html",
    }
    if include_extinction:
        plot_dirs["png_with_extinction"] = plot_root / "png_with_extinction"
        plot_dirs["html_with_extinction"] = plot_root / "html_with_extinction"

    for path in plot_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return plot_dirs


def plotly_hover_columns(df: pd.DataFrame) -> list[str]:
    """Return useful per-star columns to include in interactive hover tooltips."""
    candidates = [
        "source_id", "ra", "dec", "pmra", "pmdec", PARALLAX_COL,
        "pm_only_prob", "plx_only_prob", "joint_prob", "final_member_prob",
        "pm_only_tier", "plx_only_tier", "joint_tier", "final_tier",
        "phot_g_mean_mag", "bp_rp",
        "phot_g_mean_mag_corrected", "bp_rp_corrected",
    ]
    return [col for col in candidates if col in df.columns]


def save_interactive_cmd_probability(
    df: pd.DataFrame,
    prob_values: np.ndarray,
    prob_label: str,
    out_dir: Path,
    filename: str,
    file_prefix: str,
    apply_extinction_correction: bool = False,
    threshold: float = None,
):
    """Save a zoomable Plotly CMD colored by probability."""
    if not HAS_PLOTLY:
        return

    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    if mag_col not in df.columns or color_col not in df.columns:
        return

    plot_df = df.copy()
    plot_df[prob_label] = prob_values
    ok = plot_df[mag_col].notna() & plot_df[color_col].notna() & np.isfinite(prob_values)
    if threshold is not None:
        ok &= prob_values >= threshold
    plot_df = plot_df.loc[ok].copy()
    if plot_df.empty:
        return

    fig = px.scatter(
        plot_df,
        x=color_col,
        y=mag_col,
        color=prob_label,
        color_continuous_scale="Blues",
        range_color=[0, 1],
        hover_data=plotly_hover_columns(plot_df),
        title=f"CMD — {prob_label}{title_suffix}" + (f" (≥ {threshold})" if threshold is not None else ""),
        labels={color_col: x_label, mag_col: y_label},
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=720)
    output_name = Path(filename).with_suffix(".html").name
    fig.write_html(out_dir / f"{file_prefix}_{output_name}", include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_{output_name}")








def plotly_output_path(out_dir: Path, file_prefix: str, stem: str) -> Path:
    """Standardize interactive HTML output filenames."""
    return out_dir / f"{file_prefix}_{stem}.html"


def plotly_tier_style() -> tuple[list[str], dict[str, str]]:
    """Shared tier draw order and colors for Plotly CMD plots."""
    tier_order = ["non-member", "possible", "probable", "core"]
    tier_colors = {
        "core": "#117733",
        "probable": "#2166AC",
        "possible": "#D55E00",
        "non-member": "#888888",
    }
    return tier_order, tier_colors


def add_cmd_tier_traces(
    fig,
    df: pd.DataFrame,
    tier_values,
    mag_col: str,
    color_col: str,
    row: int,
    col: int,
    showlegend: bool = True,
):
    """Add real data-backed tier-colored CMD traces to a Plotly subplot."""
    tier_order, tier_colors = plotly_tier_style()
    plot_df = df.copy()
    plot_df["_plot_tier"] = np.asarray(tier_values, dtype=object)
    valid = plot_df[mag_col].notna() & plot_df[color_col].notna()
    hover_cols = plotly_hover_columns(plot_df)
    customdata = plot_df[hover_cols].astype(str).values if hover_cols else None
    hovertemplate = "BP-RP=%{x}<br>G=%{y}<br>tier=%{meta}<extra></extra>"

    for tier_name in tier_order:
        mask = valid & (plot_df["_plot_tier"] == tier_name)
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=plot_df.loc[mask, color_col],
                y=plot_df.loc[mask, mag_col],
                mode="markers",
                name=f"{tier_name} (N={int(mask.sum())})",
                marker=dict(size=5, color=tier_colors[tier_name], opacity=0.75),
                customdata=customdata[mask.values] if customdata is not None else None,
                meta=tier_name,
                hovertemplate=hovertemplate,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )


def save_interactive_three_model_cmd(df: pd.DataFrame, out_dir: Path, file_prefix: str, apply_extinction_correction: bool = False):
    """Interactive three-panel PM/parallax/final CMD comparison."""
    if not HAS_PLOTLY:
        return
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    if mag_col not in df.columns or color_col not in df.columns:
        return
    fig = make_subplots(rows=1, cols=3, subplot_titles=("PM-only", "Parallax-only", "Final membership"))
    for i, tier_col in enumerate(["pm_only_tier", "plx_only_tier", "final_tier"], start=1):
        if tier_col in df.columns:
            add_cmd_tier_traces(fig, df, df[tier_col].values, mag_col, color_col, 1, i, showlegend=(i == 1))
        fig.update_xaxes(title_text=x_label, row=1, col=i)
        fig.update_yaxes(title_text=y_label, autorange="reversed", row=1, col=i)
    fig.update_layout(title=f"CMD three-model comparison{title_suffix}", template="plotly_white", height=720)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "cmd_three_model_comparison"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_cmd_three_model_comparison.html")


def save_interactive_cmd_core_only_comparison(df, pm_prob, plx_prob, joint_prob, out_dir, file_prefix, apply_extinction_correction=False):
    """Interactive three-panel core-only CMD comparison."""
    if not HAS_PLOTLY:
        return
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    if mag_col not in df.columns or color_col not in df.columns:
        return
    fig = make_subplots(rows=1, cols=3, subplot_titles=("PM core", "Parallax core", "Combined core"))
    for i, (prob, label) in enumerate([(pm_prob, "PM"), (plx_prob, "Parallax"), (joint_prob, "Combined")], start=1):
        ok = df[mag_col].notna() & df[color_col].notna() & (prob >= 0.8)
        if ok.any():
            fig.add_trace(go.Scattergl(x=df.loc[ok, color_col], y=df.loc[ok, mag_col], mode="markers",
                                       name=f"{label} core N={int(ok.sum())}", marker=dict(size=6, color="royalblue", opacity=0.8)), row=1, col=i)
        fig.update_xaxes(title_text=x_label, row=1, col=i)
        fig.update_yaxes(title_text=y_label, autorange="reversed", row=1, col=i)
    fig.update_layout(title=f"CMD core-only comparison{title_suffix}", template="plotly_white", height=720)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "cmd_core_only_comparison"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_cmd_core_only_comparison.html")


def save_interactive_cmd_comparison_cuts(df, pm_prob, plx_prob, joint_prob, out_dir, file_prefix, apply_extinction_correction=False):
    """Interactive three-panel 60-80/80+ CMD comparison."""
    if not HAS_PLOTLY:
        return
    mag_col, color_col = select_cmd_photometry(df, apply_extinction_correction)
    x_label, y_label, title_suffix = cmd_axis_labels(mag_col, color_col)
    if mag_col not in df.columns or color_col not in df.columns:
        return
    fig = make_subplots(rows=1, cols=3, subplot_titles=("PM", "Parallax", "Combined"))
    for i, (prob, label) in enumerate([(pm_prob, "PM"), (plx_prob, "Parallax"), (joint_prob, "Combined")], start=1):
        for lo, hi, color, name in [(0.6, 0.8, "lightblue", "60-80%"), (0.8, 1.01, "mediumblue", "80%+")]:
            ok = df[mag_col].notna() & df[color_col].notna() & (prob >= lo) & (prob < hi)
            if ok.any():
                fig.add_trace(go.Scattergl(x=df.loc[ok, color_col], y=df.loc[ok, mag_col], mode="markers",
                                           name=f"{label} {name} N={int(ok.sum())}", marker=dict(size=5, color=color, opacity=0.75)), row=1, col=i)
        fig.update_xaxes(title_text=x_label, row=1, col=i)
        fig.update_yaxes(title_text=y_label, autorange="reversed", row=1, col=i)
    fig.update_layout(title=f"CMD probability cuts comparison{title_suffix}", template="plotly_white", height=720)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "cmd_comparison_cuts"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_cmd_comparison_cuts.html")


def save_interactive_probability_histograms(pm_prob, plx_prob, joint_prob, out_dir, file_prefix):
    """Interactive versions of the two individual probability-distribution PNGs."""
    if not HAS_PLOTLY:
        return
    fig = make_subplots(rows=1, cols=3, subplot_titles=("PM probability", "Parallax probability", "Combined probability"))
    for i, (values, color) in enumerate([(pm_prob, "steelblue"), (plx_prob, "darkorange"), (joint_prob, "crimson")], start=1):
        fig.add_trace(go.Histogram(x=np.asarray(values, dtype=float), nbinsx=50, marker_color=color), row=1, col=i)
        fig.update_xaxes(title_text="Probability", row=1, col=i)
        fig.update_yaxes(title_text="Number of sources", row=1, col=i)
    fig.update_layout(title="Individual probability distributions", template="plotly_white", height=600, showlegend=False)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "individual_probabilities_full"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_individual_probabilities_full.html")

    fig = make_subplots(rows=1, cols=3, subplot_titles=("PM probability 50%+", "Parallax probability 50%+", "Combined probability 50%+"))
    for i, (values, color) in enumerate([(pm_prob, "steelblue"), (plx_prob, "darkorange"), (joint_prob, "crimson")], start=1):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals) & (vals >= 0.5)]
        fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color=color), row=1, col=i)
        fig.update_xaxes(title_text="Probability", range=[0.5, 1], row=1, col=i)
        fig.update_yaxes(title_text="Number of sources", row=1, col=i)
    fig.update_layout(title="Individual probability distributions (50%+)", template="plotly_white", height=600, showlegend=False)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "individual_probabilities_50plus"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_individual_probabilities_50plus.html")





def save_interactive_quadrature(df: pd.DataFrame, out_dir: Path, file_prefix: str):
    """Interactive quadrature histogram from the same proper-motion data."""
    if not HAS_PLOTLY or "pmra" not in df.columns or "pmdec" not in df.columns:
        return
    pm_total = np.sqrt(df["pmra"].astype(float) ** 2 + df["pmdec"].astype(float) ** 2)
    fig = go.Figure(go.Histogram(x=pm_total[np.isfinite(pm_total)], nbinsx=500, marker_color="steelblue"))
    fig.update_layout(title="Proper Motion Quadrature Distribution", xaxis_title="Total Proper Motion [mas/yr]",
                      yaxis_title="Number of Stars", template="plotly_white", height=650)
    fig.update_xaxes(range=[0, 100])
    fig.write_html(plotly_output_path(out_dir, file_prefix, "quadrature_analysis"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_quadrature_analysis.html")


def save_interactive_qc_color_gradient(df: pd.DataFrame, out_dir: Path, file_prefix: str):
    """Interactive color-gradient QC scatter from the same source columns."""
    if not HAS_PLOTLY or any(col not in df.columns for col in ["ra", "dec", "phot_g_mean_mag", "bp_rp"]):
        return
    x, y, _, _ = add_projected_xy(df)
    plot_df = df.copy()
    plot_df["projected_x"] = x
    plot_df["projected_y"] = y
    plot_df["bp_rp_residual"] = plot_df["bp_rp"].astype(float) - np.nanmedian(plot_df["bp_rp"].astype(float))
    ok = np.isfinite(plot_df["projected_x"]) & np.isfinite(plot_df["projected_y"]) & np.isfinite(plot_df["bp_rp_residual"])
    plot_df = plot_df.loc[ok].copy()
    if plot_df.empty:
        return
    fig = px.scatter(plot_df, x="projected_x", y="projected_y", color="bp_rp_residual", color_continuous_scale="RdBu_r",
                     hover_data=plotly_hover_columns(plot_df), title="QC color-gradient diagnostic",
                     labels={"projected_x": "Projected X [deg]", "projected_y": "Projected Y [deg]", "bp_rp_residual": "BP-RP residual"})
    fig.update_layout(template="plotly_white", height=700)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "qc_color_gradient_diagnostic"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_qc_color_gradient_diagnostic.html")


def save_interactive_blue_straggler_qc(df: pd.DataFrame, out_dir: Path, file_prefix: str):
    """Interactive QC CMD/parallax overview using real CMD and parallax columns."""
    if not HAS_PLOTLY or any(col not in df.columns for col in ["phot_g_mean_mag", "bp_rp", PARALLAX_COL]):
        return
    plot_df = df.copy()
    ok = plot_df["phot_g_mean_mag"].notna() & plot_df["bp_rp"].notna() & plot_df[PARALLAX_COL].notna()
    plot_df = plot_df.loc[ok].copy()
    if plot_df.empty:
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CMD", "Parallax distribution"))
    color_col = "final_member_prob" if "final_member_prob" in plot_df.columns else None
    marker = dict(size=5, opacity=0.7, color=plot_df[color_col], colorscale="Blues", cmin=0, cmax=1, showscale=True) if color_col else dict(size=5, opacity=0.7)
    fig.add_trace(go.Scattergl(x=plot_df["bp_rp"], y=plot_df["phot_g_mean_mag"], mode="markers", name="CMD", marker=marker), row=1, col=1)
    fig.add_trace(go.Histogram(x=plot_df[PARALLAX_COL], nbinsx=50, name="Parallax", marker_color="steelblue"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="G (mag)", row=1, col=1)
    fig.update_xaxes(title_text="BP-RP", row=1, col=1)
    fig.update_xaxes(title_text="Parallax (mas)", row=1, col=2)
    fig.update_yaxes(title_text="Number of sources", row=1, col=2)
    fig.update_layout(title="QC blue-straggler parallax validation overview", template="plotly_white", height=700)
    fig.write_html(plotly_output_path(out_dir, file_prefix, "qc_blue_straggler_parallax_validation"), include_plotlyjs="cdn")
    print(f"  Saved: {file_prefix}_qc_blue_straggler_parallax_validation.html")





def generate_interactive_plot_suite(result_df, out_dir, file_prefix, apply_extinction_correction=False):
    """Generate only the requested HTML plots in the selected render folder."""
    if not HAS_PLOTLY:
        return
    has_pm = "pm_only_prob" in result_df and result_df["pm_only_prob"].notna().any()
    has_plx = "plx_only_prob" in result_df and result_df["plx_only_prob"].notna().any()
    has_joint = "joint_prob" in result_df and result_df["joint_prob"].notna().any()
    save_interactive_three_model_cmd(result_df, out_dir, file_prefix, apply_extinction_correction)
    if has_pm:
        save_interactive_cmd_probability(result_df, result_df["pm_only_prob"].values,
            "PM-only probability", out_dir, "cmd_pm_only.html", file_prefix, apply_extinction_correction)
    if has_pm and has_plx and has_joint:
        pm, plx, joint = (result_df[col].values for col in ["pm_only_prob", "plx_only_prob", "joint_prob"])
        save_interactive_cmd_probability(result_df, joint, "P(cluster | PM + parallax)",
            out_dir, "cmd_core_bayesian_only.html", file_prefix, apply_extinction_correction, threshold=0.8)
        save_interactive_cmd_core_only_comparison(result_df, pm, plx, joint, out_dir, file_prefix, apply_extinction_correction)
        save_interactive_cmd_comparison_cuts(result_df, pm, plx, joint, out_dir, file_prefix, apply_extinction_correction)
        save_interactive_probability_histograms(pm, plx, joint, out_dir, file_prefix)
    save_interactive_quadrature(result_df, out_dir, file_prefix)
    save_interactive_qc_color_gradient(result_df, out_dir, file_prefix)
    save_interactive_blue_straggler_qc(result_df, out_dir, file_prefix)


def generate_all_plots(result_df, out_dir, file_prefix, plot_dirs: dict = None):
    """Generate the requested plots, preserving format and extinction folders."""
    has_extinction = has_precomputed_extinction_photometry(result_df)
    if plot_dirs is None:
        plot_dirs = make_plot_output_dirs(out_dir, include_extinction=has_extinction)
    has_pm = "pm_only_prob" in result_df and result_df["pm_only_prob"].notna().any()
    has_plx = "plx_only_prob" in result_df and result_df["plx_only_prob"].notna().any()
    has_joint = "joint_prob" in result_df and result_df["joint_prob"].notna().any()
    variants = [(False, "png", "html")]
    if has_extinction:
        variants.append((True, "png_with_extinction", "html_with_extinction"))
    for corrected, png_key, html_key in variants:
        png_dir = plot_dirs.get(png_key)
        if HAS_MATPLOTLIB and png_dir is not None:
            generate_three_model_cmd_plot(result_df, png_dir, file_prefix, corrected)
            if has_pm:
                plot_cmd_colored_by_probability(result_df, result_df["pm_only_prob"].values,
                    "PM-only probability", png_dir, "cmd_pm_only.png", file_prefix, corrected)
            if has_pm and has_plx and has_joint:
                pm, plx, joint = (result_df[col].values for col in ["pm_only_prob", "plx_only_prob", "joint_prob"])
                plot_cmd_core_only(result_df, joint, "P(cluster | PM + parallax)",
                    png_dir, "cmd_core_bayesian_only.png", file_prefix, threshold=0.8,
                    apply_extinction_correction=corrected)
                plot_cmd_core_only_comparison(result_df, pm, plx, joint, png_dir, file_prefix, corrected)
                plot_cmd_comparison_cuts(result_df, pm, plx, joint, png_dir, file_prefix, corrected)
                plot_individual_probabilities(pm, plx, joint, png_dir, file_prefix)
            plot_differential_reddening_color_gradient_diagnostic(result_df, png_dir, file_prefix)
            plot_blue_straggler_parallax_validation(result_df, png_dir, file_prefix)
            # The raw quadrature PNG is already produced as a descriptive histogram.
            if corrected:
                perform_quadrature_analysis(result_df, png_dir, file_prefix)
        html_dir = plot_dirs.get(html_key)
        if HAS_PLOTLY and html_dir is not None:
            generate_interactive_plot_suite(result_df, html_dir, file_prefix, corrected)


def process_single_csv(csv_path: Path, outputs_dir: Path, c_low: float = None, c_high: float = None) -> None:
    """Process a single CSV file with three-model analysis and save results"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {csv_path.name}")
    print(f"{'='*60}")
    
    # Create subfolder for this specific file
    file_prefix = "NGC7789_custom-result"
    file_output_dir = outputs_dir / f"{file_prefix}_output"
    file_output_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = make_plot_output_dirs(file_output_dir, include_extinction=False)
    print(f"Output subdirectory: {file_output_dir}")
    
    # Load astrometric data with quality cuts
    print(f"Loading astrometric data from: {csv_path}")
    try:
        pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors, df, quality_flags, has_uncertainties = load_astrometric_data(csv_path)
    except ValueError as e:
        print(f"ERROR loading data: {e}")
        return
    
    if len(df) < MIN_MEMBERS:
        print(f"WARNING: Too few stars ({len(df)}) after quality cuts in {csv_path.name}. Skipping.")
        return
    
    print(f"Loaded {len(df)} stars after quality cuts")
    if quality_flags:
        print(f"Quality flags: {', '.join(quality_flags)}")
    
    stale = [c for c in df if c.startswith(("pm_only_", "plx_only_", "joint_", "astro_3d_", "final_"))
             or c in {"quality_flags", "bp_rp_corrected", "phot_g_mean_mag_corrected",
                      "bp_rp_extinction_delta", "phot_g_extinction_delta", "extinction_reference_star", "manual_cmd_seed"}]
    df = df.drop(columns=stale)
    df["manual_cmd_seed"] = manual_cmd_seed_mask(df)
    save_manual_seed_preview(df, plot_dirs["png"])

    # Reference histogram only; initialization comes exclusively from the manual polygon.
    perform_quadrature_analysis(df, plot_dirs["png"], file_prefix)
    
    # Step 2: Three-model analysis
    result_df, quality_summary = run_three_model_analysis(
        pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors,
        df, file_output_dir, file_prefix,
        c_low=c_low, c_high=c_high,
    )

    # Empirical differential-reddening correction for CMD diagnostics is always
    # attempted once.  The correction surface is fit only from the 80%+ final
    # core-membership group, then evaluated for every row.  If successful,
    # corrected columns are saved to the result CSV and rendered as an additional
    # plot suite under plots/*_with_extinction.
    extinction_info = {"success": False, "reason": "not_attempted"}
    print("\n=== PREPARING EMPIRICAL DIFFERENTIAL-REDDENING CMD COLUMNS ===")
    print(
        f"  Reference group for extinction surface: final_member_prob >= "
        f"{CORE_MEMBER_PROB_THRESHOLD:.2f} (core members only)"
    )
    result_df, extinction_info = add_extinction_corrected_photometry(
        result_df,
        prob_col="final_member_prob",
        prob_threshold=CORE_MEMBER_PROB_THRESHOLD,
    )
    if extinction_info.get("success", False):
        print("  Saved corrected CMD columns to result table:")
        print("    bp_rp_corrected, phot_g_mean_mag_corrected")
        print("    bp_rp_extinction_delta, phot_g_extinction_delta")
        print(f"  Extinction reference stars used: {extinction_info.get('n_reference_stars', 0)}")
        plot_dirs = make_plot_output_dirs(file_output_dir, include_extinction=True)
    else:
        print(f"  Extinction correction unavailable: {extinction_info.get('reason', 'unknown')}")
    
    # Output validation
    assert np.all((result_df['pm_only_prob'] >= 0) & (result_df['pm_only_prob'] <= 1)) | np.isnan(result_df['pm_only_prob']).all(), "PM-only probabilities out of range"
    assert np.all((result_df['plx_only_prob'] >= 0) & (result_df['plx_only_prob'] <= 1)) | np.isnan(result_df['plx_only_prob']).all(), "Parallax-only probabilities out of range"
    assert np.all((result_df['joint_prob'] >= 0) & (result_df['joint_prob'] <= 1)) | np.isnan(result_df['joint_prob']).all(), "Joint probabilities out of range"
    
    # Validate centers are finite
    if 'pm_only_center_pmra' in result_df.columns:
        assert np.isfinite(result_df['pm_only_center_pmra'].iloc[0]) or np.isnan(result_df['pm_only_center_pmra'].iloc[0])
        assert np.isfinite(result_df['pm_only_center_pmdec'].iloc[0]) or np.isnan(result_df['pm_only_center_pmdec'].iloc[0])
    
    if 'joint_center_pmra' in result_df.columns:
        assert np.isfinite(result_df['joint_center_pmra'].iloc[0]) or np.isnan(result_df['joint_center_pmra'].iloc[0])
        assert np.isfinite(result_df['joint_center_pmdec'].iloc[0]) or np.isnan(result_df['joint_center_pmdec'].iloc[0])
        assert np.isfinite(result_df['joint_center_parallax'].iloc[0]) or np.isnan(result_df['joint_center_parallax'].iloc[0])
    
    # Save main results
    output_csv = file_output_dir / f"{file_prefix}_cluster_membership_results.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Save quality summary
    quality_summary_df = pd.DataFrame([{
        'file_prefix': file_prefix,
        'initialization': 'manual_cmd_polygon',
        'manual_seed_count': int(df.manual_cmd_seed.sum()),
        'manual_seed_vertices': repr(MANUAL_CMD_VERTICES.tolist()),
        'seed_stability_assessed': False,
        'overall_status': quality_summary['overall_status'],
        'quality_flags': ', '.join(quality_summary['quality_flags']),
        'n_total': quality_summary['n_total'],
        'max_pi_adaptive': quality_summary.get('max_pi_adaptive', np.nan),
        'spatial_concentration': quality_summary.get('spatial_concentration', np.nan),
        'spatial_c_low': quality_summary.get('spatial_c_low', np.nan),
        'spatial_c_high': quality_summary.get('spatial_c_high', np.nan),
        'extinction_correction_attempted': True,
        'extinction_correction_success': bool(extinction_info.get('success', False)),
        'extinction_correction_reason': extinction_info.get('reason', ''),
        'extinction_reference_selection': extinction_info.get('reference_selection', ''),
        'extinction_prob_col': extinction_info.get('prob_col', ''),
        'extinction_prob_threshold': extinction_info.get('prob_threshold', np.nan),
        'extinction_n_core_candidates': extinction_info.get('n_core_candidates', np.nan),
        'extinction_n_reference_stars': extinction_info.get('n_reference_stars', np.nan),
        'extinction_median_delta_bp_rp': extinction_info.get('median_correction_color', np.nan),
        'extinction_median_delta_g': extinction_info.get('median_correction_mag', np.nan),
        'extinction_max_abs_delta_bp_rp': extinction_info.get('max_correction_color', np.nan),
        'extinction_max_abs_delta_g': extinction_info.get('max_correction_mag', np.nan),
        **quality_summary['overlaps']
    }])
    quality_csv = file_output_dir / f"{file_prefix}_quality_summary.csv"
    quality_summary_df.to_csv(quality_csv, index=False)
    print(f"Quality summary saved to: {quality_csv}")
    
    # Step 3: Generate diagnostic plots
    if HAS_MATPLOTLIB:
        print("\n=== GENERATING DIAGNOSTIC PLOTS ===")
        generate_all_plots(result_df, file_output_dir, file_prefix, plot_dirs=plot_dirs)
    
    print(f"\n=== COMPLETED: {csv_path.name} ===")
    print(f"Final quality status: {quality_summary['overall_status']}")
    if quality_summary['quality_flags']:
        print(f"Warning flags: {', '.join(quality_summary['quality_flags'])}")


def main():
    global MANUAL_CMD_VERTICES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/NGC7789-result_output/NGC7789-result_cluster_membership_results.csv",
                        help="NGC 7789 input CSV; defaults to the existing result table (old memberships are discarded)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs",
                        help="Parent output directory; uses the NGC7789_custom-result_output subfolder")
    parser.add_argument("--draw-seed", action="store_true", help="Draw a new seed polygon on the raw CMD before fitting")
    args = parser.parse_args()
    if not args.input_csv.is_file() or "ngc7789" not in args.input_csv.name.lower():
        parser.error("Provide an existing NGC7789 CSV, not a directory or another cluster.")
    target = args.out_dir / "NGC7789_custom-result_output"
    if args.input_csv.resolve().parent == target.resolve():
        parser.error("The output folder must differ from the input folder.")
    if args.draw_seed:
        *_, frame, _, _ = load_astrometric_data(args.input_csv)
        MANUAL_CMD_VERTICES = draw_manual_seed(frame)
    process_single_csv(args.input_csv, args.out_dir)


if __name__ == "__main__":
    main()
