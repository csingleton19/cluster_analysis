#!/usr/bin/env python3
"""
Combined Open Cluster Analysis Tool

This script combines the functionality of:
1. simple_pm_quadrature.py - Find initial guess from quadrature peak analysis
2. quadrature_pm_membership.py - Full membership analysis using that guess

Process:
1. Prompt user for input directory containing CSV files
2. Process all CSV files in the directory
3. For each file: perform quadrature analysis + full membership analysis
4. Output results and visualizations to individual subfolders in outputs/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
PMRA_COL = "pmra"
PMDEC_COL = "pmdec"
PARALLAX_COL = "parallax"
COV_REG = 1e-4

# Field stays broad: min "sigma" in each direction (mas/yr). Prevents field collapse.
FIELD_MIN_SIGMA_MAS = 12.0
# Prior: most stars are field. Fixed; not learned.
PI_CLUSTER_FIXED = 0.15


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


def find_quadrature_peak(pm_total, bins=5000):
    """Find the peak value in quadrature distribution"""
    # Filter out NaN values
    pm_total_valid = pm_total[np.isfinite(pm_total)]
    hist, bin_edges = np.histogram(pm_total_valid, bins=bins)
    peak_bin_idx = np.argmax(hist)
    peak_value = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2
    return peak_value


def find_closest_star_to_peak(df, peak_value):
    """Find the star closest to the quadrature peak"""
    pm_total = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    # Filter out NaN values
    valid_mask = np.isfinite(pm_total)
    distances = np.abs(pm_total - peak_value)
    # Set invalid distances to infinity so they won't be chosen
    distances[~valid_mask] = np.inf
    closest_star_idx = np.argmin(distances)
    initial_guess = df.iloc[closest_star_idx][['pmra', 'pmdec']].values
    # Convert numpy.float64 to regular Python floats
    initial_guess = np.array([float(initial_guess[0]), float(initial_guess[1])])
    return initial_guess, closest_star_idx


def find_cluster_center(pm: np.ndarray) -> np.ndarray:
    """
    Initial cluster center = median of the tightest ~15% of points around global median
    Avoids KDE (which can be heavy or unstable on large fields). Robust and sufficient
    for the fixed-field mixture
    """
    median_pt = np.median(pm, axis=0)
    d = np.linalg.norm(pm - median_pt, axis=1)
    k = max(1, int(0.15 * len(pm)))
    idx = np.argpartition(d, k)[:k]
    return np.median(pm[idx], axis=0)


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


def mvnpdf_log(pm: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log of 2D multivariate normal PDF: Multivariate Normal Probability Density Function"""
    diff = pm - center
    cov = np.atleast_2d(cov) + COV_REG * np.eye(2)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.full(len(pm), -1e10)
    try:
        L = np.linalg.cholesky(cov)
        Linv = np.linalg.inv(L)
        quad = np.sum((diff @ Linv.T) ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.full(len(pm), -1e10)
    return -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)


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


def estimate_sigma_from_core(pm: np.ndarray, center: np.ndarray, frac: float = 0.15) -> float:
    """Robust scale for initial cluster covariance (isotropic)"""
    d = distances_pm(pm, center)
    k = max(1, int(frac * len(d)))
    nearest = np.partition(d, k)[:k]
    return np.median(nearest) * 1.5


def mahalanobis_distances(pm: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Elliptical distance in PM space"""
    diff = pm - center
    cov_safe = np.atleast_2d(cov) + COV_REG * np.eye(2)
    L = np.linalg.cholesky(np.linalg.inv(cov_safe))
    white = diff @ L.T
    return np.linalg.norm(white, axis=1)


def fit_cluster_pm_only(
    pm: np.ndarray,
    center_init: np.ndarray,
    sigma_init: float,
    center_field: np.ndarray,
    cov_field: np.ndarray,
    pi_cluster: float = PI_CLUSTER_FIXED,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate only the cluster (center, cov). Field and π are fixed
    Returns cluster_center, cluster_cov, P(member), Mahalanobis distances
    """
    center_c = center_init.copy()
    cov_c = (sigma_init ** 2) * np.eye(2)

    for _ in range(max_iter):
        prob = p_member_two_component(
            pm, center_c, cov_c, center_field, cov_field, pi_cluster
        )
        w = np.maximum(prob, 1e-12)
        center_c_new = weighted_center(pm, w)
        cov_c_new = weighted_covariance(pm, center_c_new, w)
        drift = np.linalg.norm(center_c_new - center_c)
        center_c = center_c_new
        cov_c = cov_c_new
        if drift < tol:
            break

    prob_final = p_member_two_component(
        pm, center_c, cov_c, center_field, cov_field, pi_cluster
    )
    d_maha = mahalanobis_distances(pm, center_c, cov_c)
    return center_c, cov_c, prob_final, d_maha


def membership_tier(prob: np.ndarray) -> np.ndarray:
    """Tiered labels: core / probable / candidate / non-member"""
    tier = np.full(len(prob), "non-member", dtype=object)
    tier[prob >= 0.9] = "core"
    tier[(prob >= 0.5) & (prob < 0.9)] = "probable"
    tier[(prob >= 0.2) & (prob < 0.5)] = "candidate"
    return tier


def parallax_scatter_step2(df: pd.DataFrame, prob: np.ndarray, out_dir: Path, file_prefix: str):
    """
    STEP 2: Parallax scatter plots for different membership probability bins.
    Creates scatter-based visualizations instead of histograms.
    """
    if PARALLAX_COL not in df.columns:
        return

    plx = df[PARALLAX_COL].values
    valid = np.isfinite(plx)

    if not valid.any():
        print("  (Skipped Step 2: no valid parallax values)")
        return

    # --- Plot 1: Parallax vs PM probability, colored continuously by PM prob ---
    fig, ax = plt.subplots(figsize=(9, 6))
    order = np.argsort(prob[valid])  # plot low-prob first so high-prob on top
    sc = ax.scatter(
        plx[valid][order], prob[valid][order],
        c=prob[valid][order], s=6, cmap="viridis", alpha=0.6, vmin=0, vmax=1,
    )
    ax.set_xlabel("Parallax (mas)")
    ax.set_ylabel("PM membership probability")
    ax.set_title("Parallax vs PM probability")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("PM probability")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_parallax_vs_pm_prob_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_parallax_vs_pm_prob_scatter.png")

    # --- Plot 2: Parallax scatter by tier (subplots, jittered strip) ---
    tier_defs = [
        ("Core (P ≥ 0.9)", prob >= 0.9, "#117733"),
        ("Probable (0.5–0.9)", (prob >= 0.5) & (prob < 0.9), "#332288"),
        ("Candidate (0.2–0.5)", (prob >= 0.2) & (prob < 0.5), "#DDCC77"),
        ("Field (P < 0.2)", prob < 0.2, "#888888"),
    ]
    fig, axes = plt.subplots(len(tier_defs), 1, figsize=(9, 2.8 * len(tier_defs)))
    for i, (label, mask, color) in enumerate(tier_defs):
        ax = axes[i]
        m = mask & valid
        n_stars = m.sum()
        if n_stars > 0:
            plx_tier = plx[m]
            jitter = np.random.default_rng(42).normal(0, 0.15, size=n_stars)
            ax.scatter(plx_tier, jitter, s=4, alpha=0.5, color=color, edgecolors="none")
            ax.axvline(np.median(plx_tier), color="red", ls="--", lw=1.5,
                       label=f"median = {np.median(plx_tier):.3f} mas")
        ax.set_xlabel("Parallax (mas)")
        ax.set_yticks([])
        ax.set_title(f"{label} (N = {n_stars})")
        ax.legend(loc="upper right")
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_parallax_strip_by_tier.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_parallax_strip_by_tier.png")

    # --- Plot 3: All tiers overlaid on one parallax scatter ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, mask, color in reversed(tier_defs):  # field first (bottom layer)
        m = mask & valid
        if m.sum() > 0:
            ax.scatter(plx[m], prob[m], s=6, alpha=0.5, color=color,
                       edgecolors="none", label=f"{label} (N={m.sum()})")
    ax.set_xlabel("Parallax (mas)")
    ax.set_ylabel("PM probability")
    ax.set_title("Parallax vs PM probability — colored by tier")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_parallax_vs_pm_prob_by_tier.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_parallax_vs_pm_prob_by_tier.png")


def compute_parallax_probability(df: pd.DataFrame, prob_pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute parallax membership probability using the advisor's simpler approach:
    take the Core stars' parallax median as "ground truth" and compute probabilities
    based on deviation from that value.

    Approach:
    1. Use Core stars (PM P ≥ 0.9) to define the cluster parallax reference
    2. Compute how many sigma each star is from that reference
    3. Convert sigma deviation to probability (closer = higher probability)
    4. Use PM probability as prior for Bayesian combination

    Returns
    -------
    parallax_prob_simple : np.ndarray
        P(cluster | parallax) based on deviation from Core median (0–1).
    bayesian_posterior : np.ndarray
        P(cluster | PM + parallax) using PM prob as prior (0–1).
    parallax_median : float
        Core parallax median (mas) - the "ground truth" reference.
    parallax_sigma : float
        Core parallax sigma (mas) - the natural spread.
    """
    n = len(prob_pm)
    if PARALLAX_COL not in df.columns:
        return np.ones(n), prob_pm.copy(), np.nan, np.nan

    plx = df[PARALLAX_COL].values
    valid = np.isfinite(plx)

    core_mask = (prob_pm >= 0.9) & valid
    if core_mask.sum() < 20:
        return np.ones(n), prob_pm.copy(), np.nan, np.nan

    # --- Core parallax reference (advisor's "ground truth") ---
    plx_core = plx[core_mask]
    parallax_median = float(np.median(plx_core))
    parallax_sigma = float(np.std(plx_core, ddof=1))
    parallax_sigma = max(parallax_sigma, 0.05)  # floor to avoid singularity

    # --- Simple parallax probability based on deviation ---
    parallax_prob_simple = np.ones(n)
    
    if valid.any():
        # Compute z-scores: how many sigma from Core median?
        z_scores = np.abs((plx[valid] - parallax_median) / parallax_sigma)
        
        # Convert to probability: closer to median = higher probability
        # Use exponential decay: P = exp(-z²/2) which gives P=1 at z=0, P~0.6 at z=1, P~0.14 at z=2
        parallax_prob_simple[valid] = np.exp(-0.5 * z_scores**2)
        
        # Optional: Cap at reasonable maximum for very close matches
        parallax_prob_simple[valid] = np.minimum(parallax_prob_simple[valid], 0.99)

    # --- Bayesian posterior using PM probability as prior ---
    bayesian_posterior = np.zeros(n)
    if valid.any():
        pm_prior = np.clip(prob_pm[valid], 1e-6, 1.0 - 1e-6)
        
        # Simple Bayesian update: posterior ∝ prior × likelihood
        # where likelihood is our parallax probability
        unnormalized_posterior = pm_prior * parallax_prob_simple[valid]
        
        # Normalize (assuming only two options: cluster vs not cluster)
        bayesian_posterior[valid] = unnormalized_posterior / (unnormalized_posterior + (1 - pm_prior) * (1 - parallax_prob_simple[valid]) + 1e-300)
    
    # For invalid parallax, just return PM probability
    bayesian_posterior[~valid] = prob_pm[~valid]

    return parallax_prob_simple, bayesian_posterior, parallax_median, parallax_sigma


def compute_combined_probability(pm_prob: np.ndarray, parallax_prob: np.ndarray, method: str = "rms") -> np.ndarray:
    """
    Combine PM and parallax probabilities.
    
    Methods:
    --------
    "rms" : sqrt((PM^2 + Parallax^2) / 2)
    "geometric_mean" : sqrt(PM * Parallax)
    """
    if method == "rms":
        return np.sqrt((pm_prob**2 + parallax_prob**2) / 2)
    elif method == "geometric_mean":
        return np.sqrt(pm_prob * parallax_prob)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_cmd_core_only_comparison(
    df: pd.DataFrame,
    pm_prob: np.ndarray,
    parallax_prob: np.ndarray,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """
    3-panel CMD showing ONLY the 90%+ core group for PM, parallax, and combined
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
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
    mask_core = valid_photo & (pm_prob >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black", 
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by PM probability - Core only")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 2: Parallax probability core only ---
    ax = axes[1]
    mask_core = valid_photo & (parallax_prob >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Parallax probability - Core only")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 3: Bayesian posterior core only ---
    ax = axes[2]
    mask_core = valid_photo & (bayesian_posterior >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
        ax.legend()
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Combined probability - Core only")
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
):
    """
    3-panel CMD comparing PM, parallax, and combined probabilities
    with two confidence levels: 80-90% (light) and 90%+ (medium blue)
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD comparison: missing photometry)")
        return
    
    # Check for valid photometry
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD comparison: no valid photometry)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for the two cuts
    color_core = "royalblue"      # 90%+ (Core) - medium blue
    color_probable = "lightcoral" # 80-90% (Probable) - light red/pink
    
    # --- Panel 1: PM probability cuts ---
    ax = axes[0]
    
    # Core: 90%+
    mask_core = valid_photo & (pm_prob >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black", 
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
    
    # Probable: 80-90%
    mask_probable = valid_photo & (pm_prob >= 0.8) & (pm_prob < 0.9)
    if mask_probable.any():
        ax.scatter(df.loc[mask_probable, color_col], df.loc[mask_probable, mag_col],
                  s=10, color=color_probable, alpha=0.7, edgecolors="black",
                  linewidths=0.5, label=f"Probable (80-90%): N={mask_probable.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by PM probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 2: Parallax probability cuts ---
    ax = axes[1]
    
    # Core: 90%+
    mask_core = valid_photo & (parallax_prob >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
    
    # Probable: 80-90%
    mask_probable = valid_photo & (parallax_prob >= 0.8) & (parallax_prob < 0.9)
    if mask_probable.any():
        ax.scatter(df.loc[mask_probable, color_col], df.loc[mask_probable, mag_col],
                  s=10, color=color_probable, alpha=0.7, edgecolors="black",
                  linewidths=0.5, label=f"Probable (80-90%): N={mask_probable.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Parallax probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 3: Bayesian posterior cuts ---
    ax = axes[2]
    
    # Core: 90%+
    mask_core = valid_photo & (bayesian_posterior >= 0.9)
    if mask_core.any():
        ax.scatter(df.loc[mask_core, color_col], df.loc[mask_core, mag_col],
                  s=12, color=color_core, alpha=0.8, edgecolors="black",
                  linewidths=0.5, label=f"Core (≥90%): N={mask_core.sum()}")
    
    # Probable: 80-90%
    mask_probable = valid_photo & (bayesian_posterior >= 0.8) & (bayesian_posterior < 0.9)
    if mask_probable.any():
        ax.scatter(df.loc[mask_probable, color_col], df.loc[mask_probable, mag_col],
                  s=10, color=color_probable, alpha=0.7, edgecolors="black",
                  linewidths=0.5, label=f"Probable (80-90%): N={mask_probable.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Combined probability (PM + Parallax)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_comparison_cuts.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_comparison_cuts.png")


def plot_phase2_summary(
    df: pd.DataFrame,
    pm_prob: np.ndarray,
    parallax_prob: np.ndarray,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """
    Minimal Phase 2 plots:
      1) Histogram overlay of PM_prob, Parallax_prob, Combined (Bayesian posterior)
      2) CMD showing only 90%+ Bayesian posterior members
      3) CMD comparison of cuts across probability types
    """
    if not HAS_MATPLOTLIB:
        return

    # --- Plot 1: Histogram overlay of all three probabilities (full range) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # PM probability
    ax.hist(pm_prob, bins=50, range=(0, 1), alpha=0.5, color="steelblue", 
            label="PM probability", edgecolor="black", linewidth=0.5)
    
    # Parallax probability (standalone)
    ax.hist(parallax_prob, bins=50, range=(0, 1), alpha=0.5, color="darkorange",
            label="Parallax probability", edgecolor="black", linewidth=0.5)
    
    # Bayesian posterior (true combined)
    ax.hist(bayesian_posterior, bins=50, range=(0, 1), alpha=0.5, color="crimson",
            label="P(cluster | PM + parallax)", edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Probability of membership")
    ax.set_ylabel("Number of sources")
    ax.set_title("Distribution of membership probabilities (full range)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_probability_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_probability_distributions.png")

    # --- Plot 1b: Histogram overlay of 50%+ probabilities (zoomed) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # PM probability (50%+ only)
    mask_pm_50 = pm_prob >= 0.5
    if mask_pm_50.any():
        ax.hist(pm_prob[mask_pm_50], bins=40, range=(0.5, 1.0), alpha=0.5, color="steelblue", 
                label="PM probability", edgecolor="black", linewidth=0.5)
    
    # Parallax probability (50%+ only)
    mask_parallax_50 = parallax_prob >= 0.5
    if mask_parallax_50.any():
        ax.hist(parallax_prob[mask_parallax_50], bins=40, range=(0.5, 1.0), alpha=0.5, color="darkorange",
                label="Parallax probability", edgecolor="black", linewidth=0.5)
    
    # Bayesian posterior (50%+ only)
    mask_bayesian_50 = bayesian_posterior >= 0.5
    if mask_bayesian_50.any():
        ax.hist(bayesian_posterior[mask_bayesian_50], bins=40, range=(0.5, 1.0), alpha=0.5, color="crimson",
                label="P(cluster | PM + parallax)", edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Probability of membership")
    ax.set_ylabel("Number of sources")
    ax.set_title("Distribution of membership probabilities (50%+ only)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_probability_distributions_50plus.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_probability_distributions_50plus.png")

    # --- Plot 2: CMD of 90%+ Bayesian posterior members ---
    plot_cmd_core_only(
        df, bayesian_posterior, "P(cluster | PM + parallax)", 
        out_dir, "cmd_core_bayesian_only.png", file_prefix, threshold=0.9
    )
    
    # --- Plot 3: Individual probability distributions ---
    plot_individual_probabilities(pm_prob, parallax_prob, bayesian_posterior, out_dir, file_prefix)
    
    # --- Plot 4: CMD comparison of cuts ---
    plot_cmd_core_only_comparison(df, pm_prob, parallax_prob, bayesian_posterior, out_dir, file_prefix)
    plot_cmd_comparison_cuts(df, pm_prob, parallax_prob, bayesian_posterior, out_dir, file_prefix)


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
    ax.hist(pm_prob, bins=50, range=(0, 1), color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("PM probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability
    ax = axes[1]
    ax.hist(parallax_prob, bins=50, range=(0, 1), color="darkorange", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Parallax probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior
    ax = axes[2]
    ax.hist(bayesian_posterior, bins=50, range=(0, 1), color="crimson", alpha=0.7, edgecolor="black")
    ax.set_xlabel("P(cluster | PM + parallax)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Bayesian posterior distribution")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_individual_probabilities_full.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_individual_probabilities_full.png")
    
    # --- Individual plots: 50%+ only ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PM probability (50%+)
    ax = axes[0]
    mask_pm_50 = pm_prob >= 0.5
    if mask_pm_50.any():
        ax.hist(pm_prob[mask_pm_50], bins=40, range=(0.5, 1.0), color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("PM probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability (50%+)
    ax = axes[1]
    mask_parallax_50 = parallax_prob >= 0.5
    if mask_parallax_50.any():
        ax.hist(parallax_prob[mask_parallax_50], bins=40, range=(0.5, 1.0), color="darkorange", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Parallax probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior (50%+)
    ax = axes[2]
    mask_bayesian_50 = bayesian_posterior >= 0.5
    if mask_bayesian_50.any():
        ax.hist(bayesian_posterior[mask_bayesian_50], bins=40, range=(0.5, 1.0), color="crimson", alpha=0.7, edgecolor="black")
    ax.set_xlabel("P(cluster | PM + parallax)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Bayesian posterior distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_individual_probabilities_50plus.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_individual_probabilities_50plus.png")


def plot_cmd_colored_by_probability(
    df: pd.DataFrame,
    prob_values: np.ndarray,
    prob_label: str,
    out_dir: Path,
    filename: str,
    file_prefix: str,
):
    """CMD (BP−RP vs G) colored by a continuous probability (plasma)."""
    if not HAS_MATPLOTLIB:
        return
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
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
        cmap="plasma",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title(f"CMD — colored by {prob_label}")
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
    threshold: float = 0.9,
):
    """CMD showing only stars with probability ≥ threshold (e.g., 90% core members)."""
    if not HAS_MATPLOTLIB:
        return
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
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
        linewidths=0.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title(f"CMD — Core members only ({prob_label} ≥ {threshold})\nN = {ok.sum()}")
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_{filename}", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_{filename}")


def plot_cmd_pm_only_variant(
    df: pd.DataFrame,
    pm_prob: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """
    CMD sanity check - Proper Motion only variant.
    Shows all stars colored by PM membership tier (core, probable, candidate, non-member).
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD PM-only: missing photometry)")
        return
    
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD PM-only: no valid photometry)")
        return
    
    # Create membership tiers from PM probability
    tier = np.full(len(pm_prob), "non-member", dtype=object)
    tier[pm_prob >= 0.9] = "core"
    tier[(pm_prob >= 0.5) & (pm_prob < 0.9)] = "probable"
    tier[(pm_prob >= 0.2) & (pm_prob < 0.5)] = "candidate"
    
    # Draw order: non-member first (bottom), then candidate, probable, core (top)
    tier_order = ["non-member", "candidate", "probable", "core"]
    tier_colors = {
        "core": "#117733",
        "probable": "#332288", 
        "candidate": "#DDCC77",
        "non-member": "#888888",
    }
    
    fig, ax = plt.subplots(figsize=(7, 6))
    for tier_name in tier_order:
        mask = valid_photo & (tier == tier_name)
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
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD Sanity Check - Proper Motion Only")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_sanity_check_pm_only.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_sanity_check_pm_only.png")


def plot_cmd_parallax_only_variant(
    df: pd.DataFrame,
    parallax_prob: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """
    CMD sanity check - Parallax only variant.
    Shows all stars colored by parallax membership tier (core, probable, candidate, non-member).
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD parallax-only: missing photometry)")
        return
    
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD parallax-only: no valid photometry)")
        return
    
    # Create membership tiers from parallax probability
    tier = np.full(len(parallax_prob), "non-member", dtype=object)
    tier[parallax_prob >= 0.9] = "core"
    tier[(parallax_prob >= 0.5) & (parallax_prob < 0.9)] = "probable"
    tier[(parallax_prob >= 0.2) & (parallax_prob < 0.5)] = "candidate"
    
    # Draw order: non-member first (bottom), then candidate, probable, core (top)
    tier_order = ["non-member", "candidate", "probable", "core"]
    tier_colors = {
        "core": "#117733",
        "probable": "#332288",
        "candidate": "#DDCC77", 
        "non-member": "#888888",
    }
    
    fig, ax = plt.subplots(figsize=(7, 6))
    for tier_name in tier_order:
        mask = valid_photo & (tier == tier_name)
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
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD Sanity Check - Parallax Only")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_sanity_check_parallax_only.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_sanity_check_parallax_only.png")


def plot_cmd_combined_variant(
    df: pd.DataFrame,
    bayesian_posterior: np.ndarray,
    out_dir: Path,
    file_prefix: str,
):
    """
    CMD sanity check - Combined PM-Parallax variant.
    Shows all stars colored by combined (Bayesian posterior) membership tier.
    """
    if not HAS_MATPLOTLIB:
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped CMD combined: missing photometry)")
        return
    
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD combined: no valid photometry)")
        return
    
    # Create membership tiers from combined (Bayesian posterior) probability
    tier = np.full(len(bayesian_posterior), "non-member", dtype=object)
    tier[bayesian_posterior >= 0.9] = "core"
    tier[(bayesian_posterior >= 0.5) & (bayesian_posterior < 0.9)] = "probable"
    tier[(bayesian_posterior >= 0.2) & (bayesian_posterior < 0.5)] = "candidate"
    
    # Draw order: non-member first (bottom), then candidate, probable, core (top)
    tier_order = ["non-member", "candidate", "probable", "core"]
    tier_colors = {
        "core": "#117733",
        "probable": "#332288",
        "candidate": "#DDCC77",
        "non-member": "#888888",
    }
    
    fig, ax = plt.subplots(figsize=(7, 6))
    for tier_name in tier_order:
        mask = valid_photo & (tier == tier_name)
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
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD Sanity Check - Combined PM + Parallax")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_cmd_sanity_check_combined.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_cmd_sanity_check_combined.png")


def perform_quadrature_analysis(df: pd.DataFrame, out_dir: Path, file_prefix: str) -> np.ndarray:
    """
    Perform quadrature analysis to find optimal initial guess.
    
    Returns:
        initial_guess: Array with [pmra, pmdec] coordinates
    """
    print("=== QUADRATURE ANALYSIS ===")
    
    # Calculate quadrature: sqrt(pmra^2 + pmdec^2)
    pm_total = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    
    # Find quadrature peak and closest star for initial guess
    peak_value = find_quadrature_peak(pm_total)
    initial_guess, star_idx = find_closest_star_to_peak(df, peak_value)
    
    print(f"Quadrature peak: {peak_value:.3f} mas/yr")
    print(f"Closest star index: {star_idx}")
    print(f"Initial guess (pmra, pmdec): {initial_guess}")
    print(f"Mean PM: {np.mean(pm_total):.3f} mas/yr")
    print(f"Median PM: {np.median(pm_total):.3f} mas/yr")
    
    # Create quadrature histogram plot
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 6))
        plt.hist(pm_total, bins=5000, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlim(0, 100)
        plt.xlabel('Total Proper Motion [mas/yr]')
        plt.ylabel('Number of Stars')
        plt.title('Proper Motion Quadrature Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save quadrature plot with file prefix
        plt.savefig(out_dir / f"{file_prefix}_quadrature_analysis.png", dpi=150)
        plt.close()
        print(f"Saved quadrature plot to: {out_dir / f'{file_prefix}_quadrature_analysis.png'}")
    
    return initial_guess


def perform_membership_analysis(pm: np.ndarray, df: pd.DataFrame, initial_guess: np.ndarray, out_dir: Path, file_prefix: str) -> pd.DataFrame:
    """
    Perform full membership analysis using quadrature-derived initial guess.
    """
    print("\n=== MEMBERSHIP ANALYSIS ===")
    
    # Fixed field (never updated)
    center_field, cov_field = fixed_field_component(pm)

    # Use quadrature-derived initial guess
    center0 = initial_guess
    sigma0 = estimate_sigma_from_core(pm, center0)
    
    print(f"Using quadrature-derived initial guess: {center0}")
    print(f"Initial sigma: {sigma0:.3f}")

    # Iterate only cluster; field and π fixed
    center, cov, prob, d_maha = fit_cluster_pm_only(
        pm, center0, sigma0, center_field, cov_field, pi_cluster=PI_CLUSTER_FIXED
    )

    df = df.copy()
    df["pm_membership_prob"] = prob
    df["pm_dist_mahalanobis"] = d_maha
    df["membership_tier"] = membership_tier(prob)

    # Summary statistics
    n_core = (prob >= 0.9).sum()
    n_probable = ((prob >= 0.5) & (prob < 0.9)).sum()
    n_candidate = ((prob >= 0.2) & (prob < 0.5)).sum()
    n_non = (prob < 0.2).sum()
    
    print("M67 / NGC 2682 — PM-only membership (2D Gaussian cluster + fixed field)")
    print("  Cluster center (pmra, pmdec) [mas/yr]:", center)
    print("  Tiered counts:")
    print("    core       (P ≥ 0.9):", n_core)
    print("    probable   (0.5–0.9):", n_probable)
    print("    candidate   (0.2–0.5):", n_candidate)
    print("    non-member (P < 0.2):", n_non)

    # Create visualizations
    if HAS_MATPLOTLIB:
        # PM space visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(pm[:, 0], pm[:, 1], c=prob, s=8, cmap="viridis", alpha=0.7)
        ax.plot(center[0], center[1], "r*", ms=14, label="cluster center")
        theta = np.linspace(0, 2 * np.pi, 100)
        L = np.linalg.cholesky(np.atleast_2d(cov) + COV_REG * np.eye(2))
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        ellipse = center + (circle @ L.T)
        ax.plot(ellipse[:, 0], ellipse[:, 1], "r-", lw=1.5, alpha=0.8, label="1σ ellipse")
        ax.set_xlabel("μ_RA (pmra) [mas/yr]")
        ax.set_ylabel("μ_Dec (pmdec) [mas/yr]")
        ax.set_title("Proper-motion space: P(member)")
        ax.legend()
        ax.set_aspect("equal")

        ax = axes[1]
        ax.hist(prob, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(0.2, color="gray", ls=":", alpha=0.7)
        ax.axvline(0.5, color="red", ls="--", label="0.5 (candidate)")
        ax.axvline(0.9, color="darkgreen", ls="--", label="0.9 (core)")
        ax.set_xlabel("P(member)")
        ax.set_ylabel("N stars")
        ax.set_title("Membership probability")
        ax.legend()

        plt.tight_layout()
        plt.savefig(out_dir / f"{file_prefix}_membership_analysis.png", dpi=150)
        plt.close()
        print(f"  Saved membership analysis plot: {file_prefix}_membership_analysis.png")

        # CMD sanity check: color by membership tier
        mag_col = "phot_g_mean_mag"
        color_col = "bp_rp"
        if mag_col in df.columns and color_col in df.columns:
            ok = df[mag_col].notna() & df[color_col].notna()
            if ok.any():
                # Draw order: non-member first (bottom), then candidate, probable, core (top)
                tier_order = ["non-member", "candidate", "probable", "core"]
                tier_colors = {
                    "core": "#117733",
                    "probable": "#332288",
                    "candidate": "#DDCC77",
                    "non-member": "#888888",
                }
                fig, ax = plt.subplots(figsize=(7, 6))
                for tier in tier_order:
                    mask = ok & (df["membership_tier"] == tier)
                    if mask.any():
                        ax.scatter(
                            df.loc[mask, color_col],
                            df.loc[mask, mag_col],
                            c=tier_colors[tier],
                            s=6,
                            alpha=0.7,
                            label=tier,
                        )
                ax.invert_yaxis()
                ax.set_xlabel("BP − RP")
                ax.set_ylabel("G (mag)")
                ax.set_title("CMD sanity check — colored by membership tier")
                handles, labels = ax.get_legend_handles_labels()
                order = ["core", "probable", "candidate", "non-member"]
                handles = [h for L in order for h, l in zip(handles, labels) if l == L]
                labels = [L for L in order if L in labels]
                ax.legend(handles, labels, loc="upper right", framealpha=0.9)
                ax.set_aspect("auto")
                plt.tight_layout()
                plt.savefig(out_dir / f"{file_prefix}_cmd_sanity_check.png", dpi=150)
                plt.close()
                print(f"  Saved CMD sanity check: {file_prefix}_cmd_sanity_check.png")
                
                # Generate CMD sanity check variants
                plot_cmd_pm_only_variant(df, prob, out_dir, file_prefix)
            else:
                print("  (Skipped CMD: no valid phot_g_mean_mag / bp_rp)")
        else:
            print("  (Skipped CMD: missing phot_g_mean_mag or bp_rp)")
    else:
        print("  (Install matplotlib to generate membership_analysis.png)")

    return df


def process_single_csv(csv_path: Path, outputs_dir: Path):
    """Process a single CSV file and save results to its own subfolder"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {csv_path.name}")
    print(f"{'='*60}")
    
    # Create subfolder for this specific file
    file_prefix = csv_path.stem  # filename without .csv
    file_output_dir = outputs_dir / f"{file_prefix}_output"
    file_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output subdirectory: {file_output_dir}")
    
    # Load data
    print(f"Loading data from: {csv_path}")
    pm, df = load_pm_data(csv_path)
    if len(pm) < 10:
        print(f"WARNING: Too few stars with valid proper motions in {csv_path.name}. Skipping.")
        return
    
    print(f"Loaded {len(pm)} stars with valid proper motions")
    
    # Step 1: Quadrature analysis to find initial guess
    initial_guess = perform_quadrature_analysis(df, file_output_dir, file_prefix)
    
    # Step 2: Full membership analysis using that guess
    result_df = perform_membership_analysis(pm, df, initial_guess, file_output_dir, file_prefix)
    
    # Save results
    output_csv = file_output_dir / f"{file_prefix}_cluster_membership_results.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Step 3: Parallax analysis if available
    if HAS_MATPLOTLIB and PARALLAX_COL in result_df.columns:
        print("\n=== PARALLAX ANALYSIS ===")
        
        # Compute parallax probabilities and Bayesian posterior
        parallax_prob, bayesian_post, plx_median, plx_sigma = compute_parallax_probability(result_df, result_df["pm_membership_prob"])
        if not np.isnan(plx_median):
            print(f"Core parallax model: median = {plx_median:.4f} mas, sigma = {plx_sigma:.4f} mas")
            
            # Add to dataframe
            result_df["parallax_probability"] = parallax_prob
            result_df["bayesian_posterior"] = bayesian_post
            
            # Combined probabilities (two ad-hoc methods for comparison)
            combined_rms = compute_combined_probability(result_df["pm_membership_prob"], parallax_prob, method="rms")
            combined_geom = compute_combined_probability(result_df["pm_membership_prob"], parallax_prob, method="geometric_mean")
            result_df["combined_probability_rms"] = combined_rms
            result_df["combined_probability_geometric"] = combined_geom
            
            # Phase 2 plots: parallax scatter + full summary
            parallax_scatter_step2(result_df, result_df["pm_membership_prob"], file_output_dir, file_prefix)
            plot_phase2_summary(result_df, result_df["pm_membership_prob"], parallax_prob, bayesian_post, file_output_dir, file_prefix)
            
            # Additional visualizations
            plot_cmd_colored_by_probability(result_df, result_df["pm_membership_prob"], "PM probability", file_output_dir, "cmd_colored_by_pm.png", file_prefix)
            plot_cmd_colored_by_probability(result_df, bayesian_post, "P(cluster | PM + parallax)", file_output_dir, "cmd_colored_by_combined.png", file_prefix)
            
            # Generate CMD sanity check variants for parallax and combined
            plot_cmd_parallax_only_variant(result_df, parallax_prob, file_output_dir, file_prefix)
            plot_cmd_combined_variant(result_df, bayesian_post, file_output_dir, file_prefix)
            
            # Additional CMD variants - unfiltered continuous probability plots
            plot_cmd_colored_by_probability(result_df, result_df["pm_membership_prob"], "PM probability", file_output_dir, "cmd_unfiltered_pm.png", file_prefix)
            plot_cmd_colored_by_probability(result_df, parallax_prob, "Parallax probability", file_output_dir, "cmd_unfiltered_parallax.png", file_prefix)
            plot_cmd_colored_by_probability(result_df, bayesian_post, "P(cluster | PM + parallax)", file_output_dir, "cmd_unfiltered_combined.png", file_prefix)
        
        # Re-save CSV with Phase 2 columns included
        result_df.to_csv(output_csv, index=False)
        print(f"Updated CSV with parallax & combined probabilities: {output_csv}")
    else:
        if not HAS_MATPLOTLIB:
            print("\n(Skipped parallax analysis: matplotlib not available)")
        elif PARALLAX_COL not in result_df.columns:
            print("\n(Skipped parallax analysis: no parallax column)")
    
    print(f"\n=== COMPLETED: {csv_path.name} ===")


def main():
    """Main analysis pipeline - processes multiple CSV files"""
    parser = argparse.ArgumentParser(description="Combined Open Cluster Analysis Tool - Batch Processing")
    parser.add_argument("input_dir", nargs="?", 
                       help="Directory containing CSV files to process")
    parser.add_argument("-o", "--out-dir", default=None, 
                       help="Output directory for plots and results (default: ./outputs)")
    
    args = parser.parse_args()
    
    # Prompt for input directory if not provided
    if args.input_dir is None:
        print("Please enter the path to the directory containing CSV files:")
        input_dir = input().strip()
        if not input_dir:
            print("No input directory provided. Exiting.")
            sys.exit(1)
    else:
        input_dir = args.input_dir
    
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Directory '{input_path}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Set up main outputs directory
    if args.out_dir is None:
        outputs_dir = Path(__file__).parent / "outputs"
    else:
        outputs_dir = Path(args.out_dir)
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Main outputs directory: {outputs_dir}")
    
    # Find all CSV files in the input directory
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in directory: {input_path}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            process_single_csv(csv_file, outputs_dir)
        except Exception as e:
            print(f"ERROR processing {csv_file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(csv_files)} files")
    print(f"Results saved in: {outputs_dir}")
    print("Each file has its own subfolder with prefix '_output'")


if __name__ == "__main__":
    main()
