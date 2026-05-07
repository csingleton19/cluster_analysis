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
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

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

# Quality thresholds
OVERLAP_PM_PLX_MIN = 0.3
OVERLAP_JOINT_PM_MIN = 0.5
OVERLAP_JOINT_PLX_MIN = 0.5

# Gaia quality cuts
RUWE_MAX = 1.4
MIN_VISIBILITY_PERIODS = 8
MIN_PARALLAX_ERROR_RATIO = 3.0

# Field stays broad: min "sigma" in each direction (mas/yr). Prevents field collapse.
FIELD_MIN_SIGMA_MAS = 5.0


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
    df = pd.read_csv(csv_path)
    quality_flags = []
    
    # Basic validity checks for measurements only
    required_cols = [pmra_col, pmdec_col, parallax_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required measurement columns: {missing_cols}")
    
    # Check for uncertainty columns
    has_pmra_error = 'pmra_error' in df.columns
    has_pmdec_error = 'pmdec_error' in df.columns
    has_parallax_error = 'parallax_error' in df.columns
    has_uncertainties = has_pmra_error and has_pmdec_error and has_parallax_error
    
    if not has_uncertainties:
        quality_flags.append("NO_UNCERTAINTIES: running without measurement errors")
    
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
    
    # Parallax error ratio cut (if available)
    if 'parallax_over_error' in df.columns:
        plx_ratio_mask = df['parallax_over_error'] >= MIN_PARALLAX_ERROR_RATIO
        df = df[plx_ratio_mask]
        removed = initial_n - len(df)
        if removed > 0 and "RUWE_CUT" not in quality_flags and "VISIBILITY_CUT" not in quality_flags:
            quality_flags.append(f"PARALLAX_RATIO_CUT: {removed} stars removed")
    
    # Check for valid astrometric measurements (only measurements, not errors)
    valid_mask = (
        df[pmra_col].notna() & df[pmdec_col].notna() & 
        df[parallax_col].notna()
    )
    
    df = df[valid_mask].copy()
    if len(df) < initial_n:
        quality_flags.append(f"VALIDITY_CUT: {initial_n - len(df)} stars removed due to invalid measurements")
    
    # If uncertainties exist, check for positive errors
    if has_uncertainties:
        positive_error_mask = (
            (df['pmra_error'] > 0) & 
            (df['pmdec_error'] > 0) & 
            (df['parallax_error'] > 0)
        )
        
        invalid_errors = (~positive_error_mask).sum()
        if invalid_errors > 0:
            quality_flags.append(f"POSITIVE_ERROR_CUT: {invalid_errors} stars removed due to non-positive errors")
            df = df[positive_error_mask].copy()
    
    # Extract data arrays
    pm_data = df[[pmra_col, pmdec_col]].values.astype(np.float64)
    parallax_data = df[parallax_col].values.astype(np.float64)
    
    # Extract error arrays if available, otherwise set to None
    if has_uncertainties:
        pmra_errors = df['pmra_error'].values.astype(np.float64)
        pmdec_errors = df['pmdec_error'].values.astype(np.float64)
        parallax_errors = df['parallax_error'].values.astype(np.float64)
    else:
        pmra_errors = None
        pmdec_errors = None
        parallax_errors = None
    
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
            
            # Add measurement errors for this star
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
    """Tiered labels: core / probable / candidate / non-member"""
    tier = np.full(len(prob), "non-member", dtype=object)
    tier[prob >= 0.9] = "core"
    tier[(prob >= 0.5) & (prob < 0.9)] = "probable"
    tier[(prob >= 0.2) & (prob < 0.5)] = "candidate"
    return tier




def fit_cluster_parallax_only(
    parallax: np.ndarray,
    parallax_errors: np.ndarray = None,
    center_init: float = None,
    sigma_init: float = None,
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
    # Set default initial values if not provided
    if center_init is None:
        center_init = np.median(parallax)
    if sigma_init is None:
        sigma_init = np.std(parallax) * 0.5
    
    # Initial parameters
    center_c = float(center_init)
    sigma_c = float(sigma_init)
    pi_cluster = 0.1  # initial guess
    
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
    """Add projected spatial coordinates relative to cluster center"""
    if ra0 is None:
        ra0 = np.median(df["ra"])
    if dec0 is None:
        dec0 = np.median(df["dec"])

    x = (df["ra"].values - ra0) * np.cos(np.deg2rad(dec0))
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


def estimate_sigma_spatial_core(x_sky, y_sky, frac=0.15):
    """Estimate spatial scale using core-based approach like PM model"""
    d = np.sqrt(x_sky**2 + y_sky**2)
    k = max(1, int(frac * len(d)))
    core = np.partition(d, k)[:k]
    sigma_spatial = np.median(core) * 1.5
    return sigma_spatial


def generate_spatially_aware_seed(x_sky, y_sky):
    """Find densest region in sky for better seed initialization"""
    from sklearn.neighbors import KernelDensity
    
    xy = np.column_stack([x_sky, y_sky])
    kde = KernelDensity(bandwidth=0.1).fit(xy)
    log_dens = kde.score_samples(xy)
    idx = np.argmax(log_dens)
    
    return x_sky[idx], y_sky[idx]


def fixed_field_component_nd(data: np.ndarray, pm_floor: float = None, plx_floor: float = None):
    """Create proper field component for 3D data with separate PM and parallax floors"""
    if pm_floor is None:
        pm_floor = FIELD_MIN_SIGMA_MAS
    if plx_floor is None:
        plx_floor = 0.2  # Much smaller floor for parallax
        
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
        0.5**2,    # pmra (normalized)
        0.5**2,    # pmdec (normalized)
        0.3**2,    # parallax (normalized)
        0.3**2,    # x sky (normalized)
        0.3**2,    # y sky (normalized)
    ])

    for i in range(5):
        cov[i, i] = max(cov[i, i], floors[i])

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, COV_REG)
    return center, eigvecs @ np.diag(eigvals) @ eigvecs.T


def mvnpdf_log_nd(data, center, cov):
    """Generic N-dimensional multivariate normal log PDF (no uncertainties)"""
    dim = data.shape[1]
    diff = data - center
    cov = np.atleast_2d(cov) + COV_REG * np.eye(dim)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.full(len(data), -1e10)

    try:
        L = np.linalg.cholesky(cov)
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
            
            # Add measurement errors for this star
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
    pi_cluster=None,
    max_iter=100,
    tol=1e-4,
):
    """Generic N-dimensional cluster fitting with EM algorithm"""
    if pi_cluster is None:
        pi_cluster = 0.1

    center_c = center_init.copy()
    cov_c = cov_init.copy()

    for _ in range(max_iter):
        log_L_c = mvnpdf_log_nd(data, center_c, cov_c)
        log_L_f = mvnpdf_log_nd(data, center_field, cov_field)

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
        cov_new += COV_REG * np.eye(data.shape[1])

        pi_new = np.clip(np.mean(prob), MIN_PI, MAX_PI)

        if np.linalg.norm(center_new - center_c) < tol:
            center_c, cov_c, pi_cluster = center_new, cov_new, pi_new
            break

        center_c, cov_c, pi_cluster = center_new, cov_new, pi_new

    # Final probabilities
    log_L_c = mvnpdf_log_nd(data, center_c, cov_c)
    log_L_f = mvnpdf_log_nd(data, center_field, cov_field)

    log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
    log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f

    max_log = np.maximum(log_p_c, log_p_f)
    p_c = np.exp(log_p_c - max_log)
    p_f = np.exp(log_p_f - max_log)
    prob = p_c / (p_c + p_f + 1e-300)

    return center_c, cov_c, prob, pi_cluster


def fit_cluster_3d_joint(
    pm_data: np.ndarray,
    pmra_errors: np.ndarray,
    pmdec_errors: np.ndarray,
    parallax_data: np.ndarray,
    parallax_errors: np.ndarray,
    center_init: np.ndarray,
    sigma_init: float,
    center_field: np.ndarray,
    cov_field: np.ndarray,
    pi_cluster: float = None,
    max_iter: int = MAX_ITER,
    tol: float = TOL_CENTER,
    min_pi: float = MIN_PI,
    max_pi: float = MAX_PI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Joint 3D PM+parallax mixture model with diagonal measurement errors and learned pi_cluster.
    Returns cluster_center, cluster_cov, P(member), learned pi_cluster
    """
    # Combine data into 3D array
    data = np.column_stack([pm_data, parallax_data])
    
    center_c = center_init.copy()
    
    # Initialize cluster covariance with proper dimensional scaling
    sigma_pm = sigma_init
    sigma_plx = max(0.05, 0.5 * np.std(parallax_data))
    cov_c = np.diag([
        sigma_pm**2,
        sigma_pm**2,
        sigma_plx**2,
    ])
    
    # Initialize pi_cluster if not provided
    if pi_cluster is None:
        pi_cluster = 0.1

    for iteration in range(max_iter):
        # E-step: compute responsibilities with measurement errors
        log_L_c = mvnpdf_log_3d(data, center_c, cov_c, pmra_errors, pmdec_errors, parallax_errors)
        log_L_f = mvnpdf_log_3d(data, center_field, cov_field, pmra_errors, pmdec_errors, parallax_errors)
        
        log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
        log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
        
        max_log = np.maximum(log_p_c, log_p_f)
        p_c = np.exp(log_p_c - max_log)
        p_f = np.exp(log_p_f - max_log)
        prob = p_c / (p_c + p_f + 1e-300)
        
        # M-step: update parameters
        w = np.maximum(prob, 1e-12)
        
        # Update cluster center (weighted mean)
        center_c_new = np.average(data, axis=0, weights=w)
        
        # Update cluster covariance (weighted covariance)
        diff = data - center_c_new
        cov_c_new = np.atleast_2d(np.average(diff[:, :, None] * diff[:, None, :], axis=0, weights=w))
        cov_c_new = np.atleast_2d(cov_c_new) + COV_REG * np.eye(3)
        
        # Update pi_cluster (use raw prob, not w)
        pi_cluster_new = np.clip(np.mean(prob), min_pi, max_pi)
        
        # Check convergence
        drift = np.linalg.norm(center_c_new - center_c)
        center_c = center_c_new
        cov_c = cov_c_new
        pi_cluster = pi_cluster_new
        
        if drift < tol:
            break

    # Final probabilities
    log_L_c = mvnpdf_log_3d(data, center_c, cov_c, pmra_errors, pmdec_errors, parallax_errors)
    log_L_f = mvnpdf_log_3d(data, center_field, cov_field, pmra_errors, pmdec_errors, parallax_errors)
    
    log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
    log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
    
    max_log = np.maximum(log_p_c, log_p_f)
    p_c = np.exp(log_p_c - max_log)
    p_f = np.exp(log_p_f - max_log)
    prob_final = p_c / (p_c + p_f + 1e-300)
    
    return center_c, cov_c, prob_final, pi_cluster


def generate_pm_seeds(pm_data: np.ndarray) -> list:
    """Generate multiple initialization seeds for PM model"""
    seeds = []
    
    # Seed 1: Median seed
    median_seed = np.median(pm_data, axis=0)
    seeds.append(("median", median_seed))
    
    # Seed 2: Quadrature seed (existing method)
    pm_total = np.sqrt(pm_data[:, 0]**2 + pm_data[:, 1]**2)
    peak_value = find_quadrature_peak(pm_total)
    _, star_idx = find_closest_star_to_peak(pd.DataFrame({PMRA_COL: pm_data[:, 0], PMDEC_COL: pm_data[:, 1]}), peak_value)
    quadrature_seed = pm_data[star_idx]
    seeds.append(("quadrature", quadrature_seed))
    
    # Seed 3: Density seed (highest local density)
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=1.0).fit(pm_data)
    log_dens = kde.score_samples(pm_data)
    density_idx = np.argmax(log_dens)
    density_seed = pm_data[density_idx]
    seeds.append(("density", density_seed))
    
    return seeds


def generate_parallax_seeds(parallax_data: np.ndarray) -> list:
    """Generate multiple initialization seeds for parallax model"""
    seeds = []
    
    # Seed 1: Median seed
    median_seed = np.median(parallax_data)
    seeds.append(("median", median_seed))
    
    # Seed 2: Mode seed (highest density)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(parallax_data)
    x_grid = np.linspace(np.min(parallax_data), np.max(parallax_data), 100)
    density = kde(x_grid)
    mode_seed = x_grid[np.argmax(density)]
    seeds.append(("mode", mode_seed))
    
    # Seed 3: Central percentile seed
    central_seed = np.percentile(parallax_data, 50)
    seeds.append(("percentile", central_seed))
    
    return seeds


def generate_joint_seeds(pm_data: np.ndarray, parallax_data: np.ndarray) -> list:
    """Generate multiple initialization seeds for joint 3D model"""
    seeds = []
    
    # Seed 1: Combined median
    pm_median = np.median(pm_data, axis=0)
    plx_median = np.median(parallax_data)
    joint_median = np.array([pm_median[0], pm_median[1], plx_median])
    seeds.append(("median", joint_median))
    
    # Seed 2: PM quadrature + parallax median
    pm_total = np.sqrt(pm_data[:, 0]**2 + pm_data[:, 1]**2)
    peak_value = find_quadrature_peak(pm_total)
    _, star_idx = find_closest_star_to_peak(pd.DataFrame({PMRA_COL: pm_data[:, 0], PMDEC_COL: pm_data[:, 1]}), peak_value)
    pm_quadrature = pm_data[star_idx]
    joint_quadrature = np.array([pm_quadrature[0], pm_quadrature[1], plx_median])
    seeds.append(("quadrature", joint_quadrature))
    
    # Seed 3: Density-based
    data_3d = np.column_stack([pm_data, parallax_data])
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=2.0).fit(data_3d)
    log_dens = kde.score_samples(data_3d)
    density_idx = np.argmax(log_dens)
    density_seed = data_3d[density_idx]
    seeds.append(("density", density_seed))
    
    return seeds


def run_multi_seed_pm(
    pm_data: np.ndarray,
    pmra_errors: np.ndarray = None,
    pmdec_errors: np.ndarray = None,
    center_field: np.ndarray = None,
    cov_field: np.ndarray = None,
) -> dict:
    """Run PM-only model with multiple seeds"""
    seeds = generate_pm_seeds(pm_data)
    results = []
    
    for seed_name, seed_center in seeds:
        try:
            sigma_init = estimate_sigma_from_core(pm_data, seed_center)
            center, cov, prob, d_maha, pi_cluster = fit_cluster_pm_only(
                pm_data, pmra_errors, pmdec_errors, seed_center, sigma_init,
                center_field, cov_field
            )
            
            # Calculate log-likelihood
            log_L_c = mvnpdf_log_with_errors(pm_data, center, cov, pmra_errors, pmdec_errors)
            log_L_f = mvnpdf_log_with_errors(pm_data, center_field, cov_field, pmra_errors, pmdec_errors)
            log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
            log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
            max_log = np.maximum(log_p_c, log_p_f)
            log_likelihood = np.sum(np.log(np.exp(log_p_c - max_log) + np.exp(log_p_f - max_log) + 1e-300) + max_log)
            
            results.append({
                'seed': seed_name,
                'center': center,
                'cov': cov,
                'prob': prob,
                'pi_cluster': pi_cluster,
                'log_likelihood': log_likelihood,
                'converged': True
            })
        except Exception as e:
            results.append({
                'seed': seed_name,
                'center': np.array([np.nan, np.nan]),
                'cov': np.eye(2) * np.nan,
                'prob': np.full(len(pm_data), np.nan),
                'pi_cluster': np.nan,
                'log_likelihood': -np.inf,
                'converged': False,
                'error': str(e)
            })
    
    # Select best result by log-likelihood among converged solutions
    converged_results = [r for r in results if r['converged']]
    if not converged_results:
        return {'status': 'FAILED', 'results': results}
    
    best_result = max(converged_results, key=lambda x: x['log_likelihood'])
    best_result['all_results'] = results
    
    return {'status': 'SUCCESS', 'best': best_result}


def run_multi_seed_parallax(
    parallax_data: np.ndarray,
    parallax_errors: np.ndarray = None,
) -> dict:
    """Run parallax-only model with multiple seeds"""
    seeds = generate_parallax_seeds(parallax_data)
    results = []
    
    for seed_name, seed_center in seeds:
        try:
            sigma_init = np.std(parallax_data) * 0.5  # Initial sigma estimate
            center, sigma, prob, pi_cluster = fit_cluster_parallax_only(
                parallax_data, parallax_errors, seed_center, sigma_init
            )
            
            # Calculate log-likelihood
            log_likelihood = 0
            if parallax_errors is not None:
                for i in range(len(parallax_data)):
                    if np.isfinite(parallax_data[i]) and np.isfinite(parallax_errors[i]):
                        total_var = sigma**2 + parallax_errors[i]**2
                        log_likelihood += -0.5 * (np.log(2 * np.pi * total_var) + 
                                                (parallax_data[i] - center)**2 / total_var)
            else:
                # No uncertainties - use standard Gaussian log-likelihood
                for i in range(len(parallax_data)):
                    if np.isfinite(parallax_data[i]):
                        log_likelihood += -0.5 * (np.log(2 * np.pi * sigma**2) + 
                                                (parallax_data[i] - center)**2 / sigma**2)
            
            results.append({
                'seed': seed_name,
                'center': center,
                'sigma': sigma,
                'prob': prob,
                'pi_cluster': pi_cluster,
                'log_likelihood': log_likelihood,
                'converged': True
            })
        except Exception as e:
            results.append({
                'seed': seed_name,
                'center': np.nan,
                'sigma': np.nan,
                'prob': np.full(len(parallax_data), np.nan),
                'pi_cluster': np.nan,
                'log_likelihood': -np.inf,
                'converged': False,
                'error': str(e)
            })
    
    # Select best result
    converged_results = [r for r in results if r['converged']]
    if not converged_results:
        return {'status': 'FAILED', 'results': results}
    
    best_result = max(converged_results, key=lambda x: x['log_likelihood'])
    best_result['all_results'] = results
    
    return {'status': 'SUCCESS', 'best': best_result}


def run_multi_seed_joint(
    pm_data: np.ndarray,
    parallax_data: np.ndarray = None,
    df: pd.DataFrame = None,
    center_field: np.ndarray = None,
    cov_field: np.ndarray = None,
) -> dict:
    """Run 5D spatial+astrometric model with multiple seeds"""
    # Add spatial coordinates
    x_sky, y_sky, _, _ = add_projected_xy(df)
    
    # Build 5D data matrix
    data_5d = np.column_stack([
        pm_data[:, 0],  # pmra
        pm_data[:, 1],  # pmdec
        parallax_data,  # parallax
        x_sky,          # x sky
        y_sky,          # y sky
    ])
    
    # Normalize dimensions to balanced scale
    data_5d_norm, data_means, data_stds = normalize_data_5d(data_5d)
    
    # Create 5D field component if not provided (use normalized data)
    if center_field is None or cov_field is None:
        center_field, cov_field = fixed_field_component_5d_normalized(data_5d_norm)
    
    # Generate spatially aware seed
    x_seed, y_seed = generate_spatially_aware_seed(x_sky, y_sky)
    
    # Generate 5D seeds with spatial awareness
    seeds = generate_joint_seeds(pm_data, parallax_data)
    results = []
    
    for seed_name, seed_center in seeds:
        try:
            # Initialize 5D cluster parameters with core-based spatial scaling
            sigma_pm = estimate_sigma_from_core(pm_data, seed_center[:2])
            sigma_plx = max(0.05, 0.5 * np.std(parallax_data))
            sigma_spatial = estimate_sigma_spatial_core(x_sky, y_sky)
            
            # Convert 3D seed to 5D seed with spatially aware position
            seed_center_5d_raw = np.array([
                seed_center[0],  # pmra
                seed_center[1],  # pmdec
                seed_center[2],  # parallax
                x_seed,          # x sky (densest region)
                y_seed,          # y sky (densest region)
            ])
            
            # Normalize seed center using same transform
            seed_center_5d = (seed_center_5d_raw - data_means) / data_stds
            
            # Initialize 5D cluster covariance with proper scaling
            sigma_norm = np.array([sigma_pm, sigma_pm, sigma_plx, sigma_spatial, sigma_spatial]) / data_stds
            cov_init = np.diag(sigma_norm**2)
            
            # Run 5D model with normalized data
            center_norm, cov_norm, prob, pi_cluster = fit_cluster_nd(
                data_5d_norm, seed_center_5d, cov_init, center_field, cov_field
            )
            
            # Convert results back to original scale
            center = center_norm * data_stds + data_means
            cov = cov_norm * np.outer(data_stds, data_stds)
            
            # Calculate log-likelihood using normalized space (consistent coordinate system)
            log_L_c = mvnpdf_log_nd(data_5d_norm, center_norm, cov_norm)
            log_L_f = mvnpdf_log_nd(data_5d_norm, center_field, cov_field)
            log_p_c = np.log(max(pi_cluster, 1e-10)) + log_L_c
            log_p_f = np.log(max(1 - pi_cluster, 1e-10)) + log_L_f
            max_log = np.maximum(log_p_c, log_p_f)
            log_likelihood = np.sum(np.log(np.exp(log_p_c - max_log) + np.exp(log_p_f - max_log) + 1e-300) + max_log)
            
            results.append({
                'seed': seed_name,
                'center': center,
                'cov': cov,
                'prob': prob,
                'pi_cluster': pi_cluster,
                'log_likelihood': log_likelihood,
                'converged': True
            })
        except Exception as e:
            results.append({
                'seed': seed_name,
                'center': np.array([np.nan, np.nan, np.nan]),
                'cov': np.eye(3) * np.nan,
                'prob': np.full(len(pm_data), np.nan),
                'pi_cluster': np.nan,
                'log_likelihood': -np.inf,
                'converged': False,
                'error': str(e)
            })
    
    # Select best result
    converged_results = [r for r in results if r['converged']]
    if not converged_results:
        return {'status': 'FAILED', 'results': results}
    
    best_result = max(converged_results, key=lambda x: x['log_likelihood'])
    best_result['all_results'] = results
    
    return {'status': 'SUCCESS', 'best': best_result}


def compute_model_overlaps(pm_prob: np.ndarray, plx_prob: np.ndarray, joint_prob: np.ndarray) -> dict:
    """Compute overlap metrics between different model probability assignments"""
    overlaps = {}
    
    # Define core members (P >= 0.9)
    pm_core = pm_prob >= 0.9
    plx_core = plx_prob >= 0.9
    joint_core = joint_prob >= 0.9
    
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
        n_joint_members = (joint_prob >= 0.9).sum()
        if n_joint_members < MIN_MEMBERS:
            quality_flags.append('LOW_JOINT_MEMBERS')
        
        member_fraction = n_joint_members / n_total
        if member_fraction > 0.25:
            quality_flags.append('HIGH_MEMBER_FRACTION')
    
    # Check parameter reasonableness for successful models
    if joint_result['status'] == 'SUCCESS':
        joint_center = joint_result['best']['center']
        # Check for reasonable parallax values (0.1-10 mas typical for clusters)
        if joint_center[2] < 0.1 or joint_center[2] > 10:
            quality_flags.append('UNREASONABLE_PARALLAX')
        
        # Check parallax spread
        joint_cov = joint_result['best']['cov']
        parallax_sigma = np.sqrt(joint_cov[2, 2])
        if parallax_sigma > 2.0:  # Very broad parallax distribution
            quality_flags.append('BROAD_PARALLAX')
    
    # Check model agreement
    if all(prob is not None for prob in [pm_prob, plx_prob, joint_prob]):
        overlaps = compute_model_overlaps(pm_prob, plx_prob, joint_prob)
        
        # PM-Parallax disagreement
        if overlaps['pm_plx_overlap'] < OVERLAP_PM_PLX_MIN:
            quality_flags.append('PM_PLX_DISAGREE')
        
        # Joint-PM disagreement
        if overlaps['joint_pm_overlap'] < OVERLAP_JOINT_PM_MIN:
            quality_flags.append('JOINT_PM_DISAGREE')
        
        # Joint-Parallax disagreement
        if overlaps['joint_plx_overlap'] < OVERLAP_JOINT_PLX_MIN:
            quality_flags.append('JOINT_PLX_DISAGREE')
    
    # Check seed stability
    def check_seed_stability(result_dict):
        if result_dict['status'] != 'SUCCESS':
            return False
        
        all_results = result_dict['best']['all_results']
        converged_results = [r for r in all_results if r['converged']]
        
        if len(converged_results) < 2:
            return False
        
        # Check center similarity between best seeds
        best_center = result_dict['best']['center']
        for result in converged_results:
            if np.array_equal(result['center'], best_center):
                continue
            center_diff = np.linalg.norm(result['center'] - best_center)
            if center_diff > 1.0:  # Arbitrary threshold for center difference
                return False
        
        return True
    
    if not check_seed_stability(pm_result):
        quality_flags.append('PM_UNSTABLE_SEEDS')
    
    if not check_seed_stability(plx_result):
        quality_flags.append('PLX_UNSTABLE_SEEDS')
    
    if not check_seed_stability(joint_result):
        quality_flags.append('JOINT_UNSTABLE_SEEDS')
    
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
        if not (MIN_PI <= pi_cluster <= MAX_PI):
            validation_flags.append(f'{model_name}_INVALID_PI_CLUSTER')
    
    if validation_flags:
        print(f"Validation warnings: {validation_flags}")
        return False
    
    return True


def generate_three_model_cmd_plot(
    df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
):
    """
    Generate three-panel CMD showing membership tiers for all three models:
    PM-only, Parallax-only, and Joint 3D.
    """
    if not HAS_MATPLOTLIB:
        print("  (Skipped three-model CMD: matplotlib not available)")
        return
    
    mag_col = "phot_g_mean_mag"
    color_col = "bp_rp"
    
    if mag_col not in df.columns or color_col not in df.columns:
        print("  (Skipped three-model CMD: missing photometry)")
        return
    
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
        "probable": "#332288",
        "candidate": "#DDCC77",
        "non-member": "#888888",
    }
    
    # Draw order: non-member first (bottom), then candidate, probable, core (top)
    tier_order = ["non-member", "candidate", "probable", "core"]
    
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
        
        n_pm_core = (df['pm_only_prob'] >= 0.9).sum() if 'pm_only_prob' in df.columns else 0
        ax.set_title(f"PM-only Model\nCore members: {n_pm_core}")
    else:
        ax.text(0.5, 0.5, "PM-only model\nfailed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("PM-only Model\nFAILED")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
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
        
        n_plx_core = (df['plx_only_prob'] >= 0.9).sum() if 'plx_only_prob' in df.columns else 0
        ax.set_title(f"Parallax-only Model\nCore members: {n_plx_core}")
    else:
        ax.text(0.5, 0.5, "Parallax-only model\nfailed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Parallax-only Model\nFAILED")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Joint 3D model
    ax = axes[2]
    if 'joint_tier' in df.columns:
        for tier_name in tier_order:
            mask = valid_photo & (df['joint_tier'] == tier_name)
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
        
        n_joint_core = (df['joint_prob'] >= 0.9).sum() if 'joint_prob' in df.columns else 0
        ax.set_title(f"Joint 3D Model\nCore members: {n_joint_core}")
    else:
        ax.text(0.5, 0.5, "Joint 3D model\nfailed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Joint 3D Model\nFAILED")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP - RP")
    ax.set_ylabel("G (mag)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
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


def plot_parallax_vs_pm_prob_scatter(df: pd.DataFrame, prob: np.ndarray, out_dir: Path, file_prefix: str):
    """Parallax vs PM probability scatter plot"""
    if not HAS_MATPLOTLIB:
        return
    
    plx = df[PARALLAX_COL].values
    valid = np.isfinite(plx)

    if not valid.any():
        print("  (Skipped parallax vs PM scatter: no valid parallax values)")
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


def plot_parallax_strip_by_tier(df: pd.DataFrame, prob: np.ndarray, out_dir: Path, file_prefix: str):
    """Parallax strip plot by membership tier"""
    if not HAS_MATPLOTLIB:
        return
    
    plx = df[PARALLAX_COL].values
    valid = np.isfinite(plx)

    if not valid.any():
        print("  (Skipped parallax strip by tier: no valid parallax values)")
        return

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


def plot_parallax_vs_pm_prob_by_tier(df: pd.DataFrame, prob: np.ndarray, out_dir: Path, file_prefix: str):
    """Parallax vs PM probability plot colored by tier"""
    if not HAS_MATPLOTLIB:
        return
    
    plx = df[PARALLAX_COL].values
    valid = np.isfinite(plx)

    if not valid.any():
        print("  (Skipped parallax vs PM by tier: no valid parallax values)")
        return

    # --- Plot 3: All tiers overlaid on one parallax scatter ---
    tier_defs = [
        ("Core (P ≥ 0.9)", prob >= 0.9, "#117733"),
        ("Probable (0.5–0.9)", (prob >= 0.5) & (prob < 0.9), "#332288"),
        ("Candidate (0.2–0.5)", (prob >= 0.2) & (prob < 0.5), "#DDCC77"),
        ("Field (P < 0.2)", prob < 0.2, "#888888"),
    ]
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
        print("  (Skipped CMD comparison cuts: missing photometry)")
        return
    
    # Check for valid photometry
    valid_photo = df[mag_col].notna() & df[color_col].notna()
    if not valid_photo.any():
        print("  (Skipped CMD comparison cuts: no valid photometry)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for confidence levels
    color_80_90 = "lightblue"  # Light blue for 80-90%
    color_90_plus = "mediumblue"  # Medium blue for 90%+
    
    # --- Panel 1: PM probability ---
    ax = axes[0]
    for label, threshold_low, threshold_high, color in [
        ("80-90%", 0.8, 0.9, color_80_90),
        ("90%+", 0.9, 1.0, color_90_plus),
    ]:
        mask = valid_photo & (pm_prob >= threshold_low) & (pm_prob < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by PM probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 2: Parallax probability ---
    ax = axes[1]
    for label, threshold_low, threshold_high, color in [
        ("80-90%", 0.8, 0.9, color_80_90),
        ("90%+", 0.9, 1.0, color_90_plus),
    ]:
        mask = valid_photo & (parallax_prob >= threshold_low) & (parallax_prob < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Parallax probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    
    # --- Panel 3: Bayesian posterior ---
    ax = axes[2]
    for label, threshold_low, threshold_high, color in [
        ("80-90%", 0.8, 0.9, color_80_90),
        ("90%+", 0.9, 1.0, color_90_plus),
    ]:
        mask = valid_photo & (bayesian_posterior >= threshold_low) & (bayesian_posterior < threshold_high)
        if mask.any():
            ax.scatter(df.loc[mask, color_col], df.loc[mask, mag_col],
                      s=8, color=color, alpha=0.7, edgecolors="black",
                      linewidths=0.3, label=f"{label}: N={mask.sum()}")
    
    ax.invert_yaxis()
    ax.set_xlabel("BP − RP")
    ax.set_ylabel("G (mag)")
    ax.set_title("CMD by Combined probability")
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
    ax.hist(pm_prob, bins=50, range=(0, 1), alpha=0.7, color="steelblue", 
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("PM membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability
    ax = axes[1]
    ax.hist(parallax_prob, bins=50, range=(0, 1), alpha=0.7, color="darkorange",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Parallax membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior
    ax = axes[2]
    ax.hist(bayesian_posterior, bins=50, range=(0, 1), alpha=0.7, color="crimson",
            edgecolor="black", linewidth=0.5)
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
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("PM membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("PM probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Parallax probability (50%+ only)
    ax = axes[1]
    mask_parallax_50 = parallax_prob >= 0.5
    if mask_parallax_50.any():
        ax.hist(parallax_prob[mask_parallax_50], bins=40, range=(0.5, 1.0), alpha=0.7, color="darkorange",
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Parallax membership probability")
    ax.set_ylabel("Number of sources")
    ax.set_title("Parallax probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    # Bayesian posterior (50%+ only)
    ax = axes[2]
    mask_bayesian_50 = bayesian_posterior >= 0.5
    if mask_bayesian_50.any():
        ax.hist(bayesian_posterior[mask_bayesian_50], bins=40, range=(0.5, 1.0), alpha=0.7, color="crimson",
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("P(cluster | PM + parallax)")
    ax.set_ylabel("Number of sources")
    ax.set_title("Combined probability distribution (50%+ only)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_individual_probabilities_50plus.png", dpi=150)
    plt.close()
    print(f"  Saved: {file_prefix}_individual_probabilities_50plus.png")


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
    CMD sanity check - Combined variant.
    Shows all stars colored by combined Bayesian posterior membership tier (core, probable, candidate, non-member).
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
    
    # Create membership tiers from combined probability
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


def run_spatial_astrometric_model(
    pm_data: np.ndarray,
    parallax_data: np.ndarray,
    df: pd.DataFrame = None,
    out_dir: Path = None,
    file_prefix: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run 5D spatial+astrometric membership analysis.
    
    Returns:
        result_df: DataFrame with model probabilities and metadata
        quality_summary: Quality assessment and flags
    """
    print("\n=== 5D SPATIAL + ASTROMETRIC MEMBERSHIP ANALYSIS ===")
    
    # Add projected spatial coordinates
    x_sky, y_sky, ra0, dec0 = add_projected_xy(df)
    
    # Build 5D data matrix: [pmra, pmdec, parallax, x, y]
    joint_data = np.column_stack([
        pm_data[:, 0],  # pmra
        pm_data[:, 1],  # pmdec
        parallax_data,  # parallax
        x_sky,          # x sky
        y_sky,          # y sky
    ])
    
    # Create 5D field component
    center_field_5d, cov_field_5d = fixed_field_component_5d(joint_data)
    
    # Initialize cluster parameters
    sigma_pm = estimate_sigma_from_core(pm_data, np.median(pm_data, axis=0))
    sigma_plx = max(0.05, 0.5 * np.std(parallax_data))
    sigma_x = max(0.02, 0.25 * np.std(x_sky))
    sigma_y = max(0.02, 0.25 * np.std(y_sky))
    
    # Initial cluster center (use medians)
    center_init = np.array([
        np.median(pm_data[:, 0]),  # pmra
        np.median(pm_data[:, 1]),  # pmdec
        np.median(parallax_data),  # parallax
        np.median(x_sky),          # x sky
        np.median(y_sky),          # y sky
    ])
    
    # Initial cluster covariance
    cov_init = np.diag([
        sigma_pm**2,
        sigma_pm**2,
        sigma_plx**2,
        sigma_x**2,
        sigma_y**2,
    ])
    
    # Run 5D model
    print("Running 5D spatial+astrometric model...")
    center_c, cov_c, prob, pi_cluster = fit_cluster_nd(
        joint_data, center_init, cov_init, center_field_5d, cov_field_5d
    )
    
    # Create results DataFrame
    result_df = df.copy()
    result_df['spatial_astrometric_prob'] = prob
    result_df['spatial_astrometric_tier'] = pd.cut(
        prob, bins=[0, 0.1, 0.5, 0.9, 1.0], 
        labels=['FIELD', 'POSSIBLE', 'LIKELY', 'CORE']
    )
    
    # Quality assessment
    quality_summary = {
        'overall_status': 'GOOD' if pi_cluster < MAX_PI else 'QUESTIONABLE',
        'quality_flags': [],
        'n_total': len(df),
        'pi_cluster': pi_cluster,
        'cluster_center': center_c.tolist(),
        'spatial_center': (ra0, dec0)
    }
    
    return result_df, quality_summary


def run_three_model_analysis(
    pm_data: np.ndarray,
    pmra_errors: np.ndarray = None,
    pmdec_errors: np.ndarray = None,
    parallax_data: np.ndarray = None,
    parallax_errors: np.ndarray = None,
    df: pd.DataFrame = None,
    out_dir: Path = None,
    file_prefix: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run complete three-model analysis: PM-only, parallax-only, and joint 3D.
    
    Returns:
        result_df: DataFrame with all model probabilities and metadata
        quality_summary: Quality assessment and flags
    """
    print("\n=== THREE-MODEL MEMBERSHIP ANALYSIS ===")
    
    # Fixed field component (same for all models)
    center_field_pm, cov_field_pm = fixed_field_component(pm_data)
    
    # Create proper 3D field component for joint model
    joint_data = np.column_stack([pm_data, parallax_data])
    center_field_3d, cov_field_3d = fixed_field_component_nd(joint_data)
    
    # Run PM-only model
    print("Running PM-only model...")
    pm_result = run_multi_seed_pm(pm_data, pmra_errors, pmdec_errors, center_field_pm, cov_field_pm)
    
    # Run parallax-only model
    print("Running parallax-only model...")
    plx_result = run_multi_seed_parallax(parallax_data, parallax_errors)
    
    # Run 5D spatial+astrometric model
    print("Running Joint 5D Spatial+Astrometric Model...")
    joint_result = run_multi_seed_joint(
        pm_data, parallax_data, df
    )
    
    # Validate outputs
    if not validate_model_outputs(pm_result, plx_result, joint_result):
        print("WARNING: Model output validation failed!")
    
    # Quality assessment
    quality_summary = evaluate_solution_quality(pm_result, plx_result, joint_result, len(df))
    
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
    
    # Add joint 3D results
    if joint_result['status'] == 'SUCCESS':
        result_df['joint_prob'] = joint_result['best']['prob']
        result_df['joint_tier'] = membership_tier(joint_result['best']['prob'])
        result_df['joint_pi'] = joint_result['best']['pi_cluster']
        result_df['joint_center_pmra'] = joint_result['best']['center'][0]
        result_df['joint_center_pmdec'] = joint_result['best']['center'][1]
        result_df['joint_center_parallax'] = joint_result['best']['center'][2]
    else:
        result_df['joint_prob'] = np.nan
        result_df['joint_tier'] = 'non-member'
        result_df['joint_pi'] = np.nan
        result_df['joint_center_pmra'] = np.nan
        result_df['joint_center_pmdec'] = np.nan
        result_df['joint_center_parallax'] = np.nan
    
    # Add quality flags
    result_df['quality_flags'] = ', '.join(quality_summary['quality_flags'])
    
    # Print summary
    print(f"\nModel Results Summary:")
    print(f"  PM-only: {pm_result['status']}")
    if pm_result['status'] == 'SUCCESS':
        n_pm_core = (pm_result['best']['prob'] >= 0.9).sum()
        print(f"    Core members: {n_pm_core}")
        print(f"    π_cluster: {pm_result['best']['pi_cluster']:.3f}")
    
    print(f"  Parallax-only: {plx_result['status']}")
    if plx_result['status'] == 'SUCCESS':
        n_plx_core = (plx_result['best']['prob'] >= 0.9).sum()
        print(f"    Core members: {n_plx_core}")
        print(f"    π_cluster: {plx_result['best']['pi_cluster']:.3f}")
    
    print(f"  Joint 3D: {joint_result['status']}")
    if joint_result['status'] == 'SUCCESS':
        n_joint_core = (joint_result['best']['prob'] >= 0.9).sum()
        print(f"    Core members: {n_joint_core}")
        print(f"    π_cluster: {joint_result['best']['pi_cluster']:.3f}")
        print(f"    Center (pmra, pmdec, parallax): {joint_result['best']['center']}")
    
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




def generate_all_plots(result_df, out_dir, file_prefix):
    """Dedicated plotting orchestrator - generates all available plots"""
    if not HAS_MATPLOTLIB:
        return

    has_pm = "pm_only_prob" in result_df.columns and not np.isnan(result_df["pm_only_prob"]).all()
    has_plx = "plx_only_prob" in result_df.columns and not np.isnan(result_df["plx_only_prob"]).all()
    has_joint = "joint_prob" in result_df.columns and not np.isnan(result_df["joint_prob"]).all()

    # Always generate model comparison CMD if possible
    generate_three_model_cmd_plot(result_df, out_dir, file_prefix)

    if has_pm:
        pm_prob = result_df["pm_only_prob"].values

        plot_cmd_colored_by_probability(
            result_df, pm_prob, "PM-only probability",
            out_dir, "cmd_pm_only.png", file_prefix
        )

        plot_cmd_core_only(
            result_df, pm_prob, "PM-only probability",
            out_dir, "cmd_pm_core_only.png", file_prefix, threshold=0.9
        )

        plot_cmd_pm_only_variant(result_df, pm_prob, out_dir, file_prefix)

        plot_parallax_vs_pm_prob_scatter(result_df, pm_prob, out_dir, file_prefix)
        plot_parallax_strip_by_tier(result_df, pm_prob, out_dir, file_prefix)
        plot_parallax_vs_pm_prob_by_tier(result_df, pm_prob, out_dir, file_prefix)

    if has_plx:
        plx_prob = result_df["plx_only_prob"].values

        plot_cmd_colored_by_probability(
            result_df, plx_prob, "Parallax-only probability",
            out_dir, "cmd_plx_only.png", file_prefix
        )

        plot_cmd_parallax_only_variant(result_df, plx_prob, out_dir, file_prefix)

    if has_joint:
        joint_prob = result_df["joint_prob"].values

        plot_cmd_colored_by_probability(
            result_df, joint_prob, "Joint 5D spatial+astrometric probability",
            out_dir, "cmd_joint_5d.png", file_prefix
        )

        plot_cmd_core_only(
            result_df, joint_prob, "Joint 5D probability",
            out_dir, "cmd_joint_core_only.png", file_prefix, threshold=0.9
        )

        plot_cmd_combined_variant(result_df, joint_prob, out_dir, file_prefix)

    if has_pm and has_plx and has_joint:
        pm_prob = result_df["pm_only_prob"].values
        plx_prob = result_df["plx_only_prob"].values
        joint_prob = result_df["joint_prob"].values

        plot_cmd_core_only_comparison(
            result_df, pm_prob, plx_prob, joint_prob, out_dir, file_prefix
        )

        plot_cmd_comparison_cuts(
            result_df, pm_prob, plx_prob, joint_prob, out_dir, file_prefix
        )

        plot_phase2_summary(
            result_df, pm_prob, plx_prob, joint_prob, out_dir, file_prefix
        )

        plot_individual_probabilities(
            pm_prob, plx_prob, joint_prob, out_dir, file_prefix
        )


def process_single_csv(csv_path: Path, outputs_dir: Path) -> None:
    """Process a single CSV file with three-model analysis and save results"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {csv_path.name}")
    print(f"{'='*60}")
    
    # Create subfolder for this specific file
    file_prefix = csv_path.stem  # filename without .csv
    file_output_dir = outputs_dir / f"{file_prefix}_output"
    file_output_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Step 1: Quadrature analysis (for reference only)
    initial_guess = perform_quadrature_analysis(df, file_output_dir, file_prefix)
    
    # Step 2: Three-model analysis
    result_df, quality_summary = run_three_model_analysis(
        pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors,
        df, file_output_dir, file_prefix
    )
    
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
        'overall_status': quality_summary['overall_status'],
        'quality_flags': ', '.join(quality_summary['quality_flags']),
        'n_total': quality_summary['n_total'],
        **quality_summary['overlaps']
    }])
    quality_csv = file_output_dir / f"{file_prefix}_quality_summary.csv"
    quality_summary_df.to_csv(quality_csv, index=False)
    print(f"Quality summary saved to: {quality_csv}")
    
    # Step 3: Generate diagnostic plots
    if HAS_MATPLOTLIB:
        print("\n=== GENERATING DIAGNOSTIC PLOTS ===")
        generate_all_plots(result_df, file_output_dir, file_prefix)
    
    print(f"\n=== COMPLETED: {csv_path.name} ===")
    print(f"Final quality status: {quality_summary['overall_status']}")
    if quality_summary['quality_flags']:
        print(f"Warning flags: {', '.join(quality_summary['quality_flags'])}")


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
