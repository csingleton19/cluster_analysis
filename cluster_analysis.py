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
MAX_PI = 0.2
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
FIELD_MIN_SIGMA_MAS = 12.0


def load_astrometric_data(
    csv_path: Path,
    pmra_col: str = PMRA_COL,
    pmdec_col: str = PMDEC_COL,
    parallax_col: str = PARALLAX_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list]:
    """
    Load complete astrometric data with quality cuts and error handling.
    
    Returns:
        pm_data: (n, 2) array of [pmra, pmdec]
        pmra_errors: (n,) array of pmra_error values
        pmdec_errors: (n,) array of pmdec_error values
        parallax_data: (n,) array of parallax values
        parallax_errors: (n,) array of parallax_error values
        df: cleaned DataFrame
        quality_flags: list of any quality issues encountered
    """
    df = pd.read_csv(csv_path)
    quality_flags = []
    
    # Basic validity checks
    required_cols = [pmra_col, pmdec_col, parallax_col, 'pmra_error', 'pmdec_error', 'parallax_error']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
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
    
    # Check for valid astrometric measurements
    valid_mask = (
        df[pmra_col].notna() & df[pmdec_col].notna() & 
        df[parallax_col].notna() &
        df['pmra_error'].notna() & df['pmdec_error'].notna() & 
        df['parallax_error'].notna()
    )
    
    df = df[valid_mask].copy()
    if len(df) < initial_n:
        quality_flags.append(f"VALIDITY_CUT: {initial_n - len(df)} stars removed due to invalid measurements")
    
    # Check for positive errors
    positive_error_mask = (
        (df['pmra_error'] > 0) & 
        (df['pmdec_error'] > 0) & 
        (df['parallax_error'] > 0)
    )
    
    invalid_errors = (~positive_error_mask).sum()
    if invalid_errors > 0:
        quality_flags.append(f"POSITIVE_ERROR_CUT: {invalid_errors} stars removed due to non-positive errors")
        df = df[positive_error_mask].copy()
    
    # Handle missing/invalid errors with fallbacks
    fallback_applied = False
    
    # Check for zero or extremely small errors
    tiny_error_threshold = 1e-6
    tiny_error_mask = (
        (df['pmra_error'] < tiny_error_threshold) | 
        (df['pmdec_error'] < tiny_error_threshold) | 
        (df['parallax_error'] < tiny_error_threshold)
    )
    
    if tiny_error_mask.any():
        # Apply fallback: use median errors
        median_pmra_error = np.median(df['pmra_error'])
        median_pmdec_error = np.median(df['pmdec_error'])
        median_parallax_error = np.median(df['parallax_error'])
        
        df.loc[tiny_error_mask, 'pmra_error'] = median_pmra_error
        df.loc[tiny_error_mask, 'pmdec_error'] = median_pmdec_error
        df.loc[tiny_error_mask, 'parallax_error'] = median_parallax_error
        
        quality_flags.append("MISSING_ERROR_FALLBACK: median errors applied for tiny values")
        fallback_applied = True
    
    # Extract data arrays
    pm_data = df[[pmra_col, pmdec_col]].values.astype(np.float64)
    pmra_errors = df['pmra_error'].values.astype(np.float64)
    pmdec_errors = df['pmdec_error'].values.astype(np.float64)
    parallax_data = df[parallax_col].values.astype(np.float64)
    parallax_errors = df['parallax_error'].values.astype(np.float64)
    
    # Reset index to ensure contiguous indices matching probability arrays
    df = df.reset_index(drop=True)
    
    # Final validation
    if len(df) < MIN_MEMBERS:
        quality_flags.append(f"LOW_STAR_COUNT: only {len(df)} stars after quality cuts")
    
    return pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors, df, quality_flags


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
        # errors should be shape (n, 2) for pmra_error, pmdec_error
        error_cov = np.array([errors[i, 0]**2, errors[i, 1]**2] for i in range(len(errors)))
        cov_total = cov_model + np.diag(error_cov.T)  # This is wrong - need per-star covariance
        # For now, just use model covariance
        cov_total = cov_model
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


def mvnpdf_log_with_errors(pm: np.ndarray, center: np.ndarray, cov: np.ndarray, pmra_errors: np.ndarray, pmdec_errors: np.ndarray) -> np.ndarray:
    """Log of 2D multivariate normal PDF with per-star diagonal measurement errors"""
    n = len(pm)
    log_likelihood = np.zeros(n)
    
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
        
        # Update pi_cluster
        pi_cluster_new = np.clip(np.mean(w), min_pi, max_pi)
        
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
    parallax_errors: np.ndarray,
    center_init: float,
    sigma_init: float,
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
    # Initial parameters
    center_c = float(center_init)
    sigma_c = float(sigma_init)
    pi_cluster = 0.1  # initial guess
    
    # Fixed broad field component
    field_center = np.median(parallax)
    field_sigma = np.std(parallax) * 2.0  # broad field
    field_sigma = max(field_sigma, 2.0)  # minimum field spread
    
    n = len(parallax)
    loglike_old = -np.inf
    
    for iteration in range(max_iter):
        # E-step: compute responsibilities with measurement errors
        log_like_cluster = np.zeros(n)
        log_like_field = np.zeros(n)
        
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
        
        # Update pi_cluster
        pi_cluster_new = np.clip(np.mean(w), min_pi, max_pi)
        
        # Check convergence
        center_drift = abs(center_c_new - center_c)
        loglike_new = np.sum(np.log(w * np.exp(log_like_cluster) + (1-w) * np.exp(log_like_field) + 1e-300))
        
        if center_drift < tol_center and abs(loglike_new - loglike_old) < 1e-5:
            break
            
        # Update parameters
        center_c = center_c_new
        sigma_c = sigma_c_new
        pi_cluster = pi_cluster_new
        loglike_old = loglike_new
    
    # Final probabilities
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
    
    log_p_cluster = np.log(max(pi_cluster, 1e-10)) + log_like_cluster
    log_p_field = np.log(max(1 - pi_cluster, 1e-10)) + log_like_field
    
    max_log = np.maximum(log_p_cluster, log_p_field)
    p_cluster = np.exp(log_p_cluster - max_log)
    p_field = np.exp(log_p_field - max_log)
    prob_final = p_cluster / (p_cluster + p_field + 1e-300)
    
    return center_c, sigma_c, prob_final, pi_cluster


def mvnpdf_log_3d(
    data: np.ndarray, 
    center: np.ndarray, 
    cov: np.ndarray, 
    pmra_errors: np.ndarray, 
    pmdec_errors: np.ndarray, 
    parallax_errors: np.ndarray
) -> np.ndarray:
    """Log of 3D multivariate normal PDF with per-star diagonal measurement errors"""
    n = len(data)
    log_likelihood = np.zeros(n)
    
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
    
    return log_likelihood


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
    cov_c = np.eye(3) * (sigma_init ** 2)
    
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
        cov_c_new = np.cov(diff.T, aweights=w)
        if cov_c_new.ndim == 0:
            cov_c_new = np.eye(3) * cov_c_new
        cov_c_new = np.atleast_2d(cov_c_new) + COV_REG * np.eye(3)
        
        # Update pi_cluster
        pi_cluster_new = np.clip(np.mean(w), min_pi, max_pi)
        
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
    pmra_errors: np.ndarray,
    pmdec_errors: np.ndarray,
    center_field: np.ndarray,
    cov_field: np.ndarray,
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
    parallax_errors: np.ndarray,
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
            for i in range(len(parallax_data)):
                if np.isfinite(parallax_data[i]) and np.isfinite(parallax_errors[i]):
                    total_var = sigma**2 + parallax_errors[i]**2
                    log_likelihood += -0.5 * (np.log(2 * np.pi * total_var) + 
                                            (parallax_data[i] - center)**2 / total_var)
            
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
    pmra_errors: np.ndarray,
    pmdec_errors: np.ndarray,
    parallax_data: np.ndarray,
    parallax_errors: np.ndarray,
    center_field: np.ndarray,
    cov_field: np.ndarray,
) -> dict:
    """Run joint 3D model with multiple seeds"""
    seeds = generate_joint_seeds(pm_data, parallax_data)
    results = []
    
    for seed_name, seed_center in seeds:
        try:
            sigma_init = estimate_sigma_from_core(pm_data, seed_center[:2])  # Use PM part for sigma
            center, cov, prob, pi_cluster = fit_cluster_3d_joint(
                pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors,
                seed_center, sigma_init, center_field, cov_field
            )
            
            # Calculate log-likelihood
            data_3d = np.column_stack([pm_data, parallax_data])
            log_L_c = mvnpdf_log_3d(data_3d, center, cov, pmra_errors, pmdec_errors, parallax_errors)
            log_L_f = mvnpdf_log_3d(data_3d, center_field, cov_field, pmra_errors, pmdec_errors, parallax_errors)
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
        
        # Check covariance is finite and positive definite
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


def run_three_model_analysis(
    pm_data: np.ndarray,
    pmra_errors: np.ndarray,
    pmdec_errors: np.ndarray,
    parallax_data: np.ndarray,
    parallax_errors: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    file_prefix: str,
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
    center_field_3d = np.array([center_field_pm[0], center_field_pm[1], np.median(parallax_data)])
    cov_field_3d = np.eye(3) * (np.std(pm_data)**2 + np.std(parallax_data)**2) * 4.0  # Broad field
    
    # Run PM-only model
    print("Running PM-only model...")
    pm_result = run_multi_seed_pm(pm_data, pmra_errors, pmdec_errors, center_field_pm, cov_field_pm)
    
    # Run parallax-only model
    print("Running parallax-only model...")
    plx_result = run_multi_seed_parallax(parallax_data, parallax_errors)
    
    # Run joint 3D model
    print("Running joint 3D model...")
    joint_result = run_multi_seed_joint(
        pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors,
        center_field_3d, cov_field_3d
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




def process_single_csv(csv_path: Path, outputs_dir: Path):
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
        pm_data, pmra_errors, pmdec_errors, parallax_data, parallax_errors, df, quality_flags = load_astrometric_data(csv_path)
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
        
        # Three-model CMD diagnostic plot
        generate_three_model_cmd_plot(result_df, file_output_dir, file_prefix)
        
        # Legacy plots for comparison (if desired)
        if 'pm_only_prob' in result_df.columns and not np.isnan(result_df['pm_only_prob']).all():
            plot_cmd_colored_by_probability(result_df, result_df['pm_only_prob'], "PM-only probability", file_output_dir, "cmd_pm_only.png", file_prefix)
        
        if 'joint_prob' in result_df.columns and not np.isnan(result_df['joint_prob']).all():
            plot_cmd_colored_by_probability(result_df, result_df['joint_prob'], "Joint 3D probability", file_output_dir, "cmd_joint_3d.png", file_prefix)
    
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
