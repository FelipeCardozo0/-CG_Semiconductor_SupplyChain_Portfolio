#!/usr/bin/env python3
"""
Numerical Diagnostics for Iterative Solvers
============================================

This script evaluates numerical solution quality independently of portfolio 
performance by computing solver-specific diagnostics.

Metrics Computed:
    1. Relative Residual Norm: ||b - Ax|| / ||b||
    2. A-norm Error: ||x - x*||_A = sqrt((x-x*)^T A (x-x*))
       where x* is a high-precision reference solution
    3. Loss of Conjugacy: measures deviation from A-orthogonality of search directions
    4. Wall-clock Runtime: per-solve timing

All metrics are reported as distributions over rebalancing dates, not single-point
summaries. No portfolio returns, backtesting, or market performance measures are
included.

Author: Numerical Diagnostics for Portfolio Optimization Solvers
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import os
import time
import warnings
from scipy.linalg import solve as direct_solve
from scipy.linalg import cholesky

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Solver parameters (identical across all solvers)
TOLERANCE = 1e-10
MAX_ITERATIONS = 1000
RISK_AVERSION = 1.0

# Reference solution tolerance (much tighter for comparison baseline)
REFERENCE_TOLERANCE = 1e-15

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# =============================================================================
# DATA CLASSES FOR DIAGNOSTICS
# =============================================================================

@dataclass
class SolverDiagnostics:
    """Container for detailed numerical diagnostics of a single solve."""
    # Basic info
    rebal_date: str
    solver_name: str
    
    # Convergence metrics
    converged: bool
    iterations: int
    wall_clock_time: float
    
    # Residual metrics
    initial_residual_norm: float
    final_residual_norm: float
    relative_residual: float
    residual_history: List[float]
    
    # Error metrics (relative to reference solution)
    a_norm_error: float  # ||x - x*||_A
    two_norm_error: float  # ||x - x*||_2
    relative_solution_error: float  # ||x - x*||_2 / ||x*||_2
    
    # Problem conditioning
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    
    # CG-specific: conjugacy loss (only for CG/PCG methods)
    conjugacy_loss: Optional[List[float]] = None
    max_conjugacy_loss: Optional[float] = None
    mean_conjugacy_loss: Optional[float] = None


@dataclass
class SolverDiagnosticsCollection:
    """Collection of diagnostics across all rebalancing dates for one solver."""
    solver_name: str
    diagnostics: List[SolverDiagnostics] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        records = []
        for d in self.diagnostics:
            records.append({
                'rebal_date': d.rebal_date,
                'solver': d.solver_name,
                'converged': d.converged,
                'iterations': d.iterations,
                'wall_clock_time_ms': d.wall_clock_time * 1000,
                'initial_residual_norm': d.initial_residual_norm,
                'final_residual_norm': d.final_residual_norm,
                'relative_residual': d.relative_residual,
                'a_norm_error': d.a_norm_error,
                'two_norm_error': d.two_norm_error,
                'relative_solution_error': d.relative_solution_error,
                'condition_number': d.condition_number,
                'min_eigenvalue': d.min_eigenvalue,
                'max_eigenvalue': d.max_eigenvalue,
                'max_conjugacy_loss': d.max_conjugacy_loss,
                'mean_conjugacy_loss': d.mean_conjugacy_loss,
            })
        return pd.DataFrame(records)
    
    def get_distribution_stats(self, metric: str) -> Dict[str, float]:
        """Compute distribution statistics for a given metric."""
        df = self.to_dataframe()
        values = df[metric].dropna()
        return {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'q25': values.quantile(0.25),
            'median': values.median(),
            'q75': values.quantile(0.75),
            'max': values.max(),
        }


# =============================================================================
# ENHANCED SOLVERS WITH DIAGNOSTICS
# =============================================================================

def compute_reference_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute high-precision reference solution using direct solver.
    
    Uses scipy's direct solve which employs LAPACK routines for
    maximum numerical precision.
    """
    return direct_solve(A, b, assume_a='pos')


def compute_a_norm_error(x: np.ndarray, x_ref: np.ndarray, A: np.ndarray) -> float:
    """
    Compute A-norm error: ||x - x*||_A = sqrt((x-x*)^T A (x-x*))
    
    The A-norm is the natural norm for the quadratic minimization problem.
    """
    diff = x - x_ref
    return np.sqrt(np.abs(diff @ A @ diff))


def gradient_descent_with_diagnostics(A: np.ndarray, 
                                       b: np.ndarray,
                                       x0: np.ndarray,
                                       x_ref: np.ndarray,
                                       tol: float = TOLERANCE,
                                       max_iter: int = MAX_ITERATIONS) -> Tuple[np.ndarray, dict]:
    """
    Gradient Descent with detailed diagnostics tracking.
    """
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    
    initial_residual_norm = np.linalg.norm(r)
    residual_history = [initial_residual_norm]
    
    start_time = time.perf_counter()
    
    converged = False
    iteration = 0
    
    while iteration < max_iter:
        residual_norm = np.linalg.norm(r)
        
        if residual_norm <= tol:
            converged = True
            break
        
        # Optimal step size
        Ar = A @ r
        rTr = np.dot(r, r)
        rTAr = np.dot(r, Ar)
        
        if abs(rTAr) < 1e-15:
            break
        
        alpha = rTr / rTAr
        
        x = x + alpha * r
        r = r - alpha * Ar
        
        residual_history.append(np.linalg.norm(r))
        iteration += 1
    
    elapsed_time = time.perf_counter() - start_time
    
    final_residual_norm = np.linalg.norm(b - A @ x)
    
    diagnostics = {
        'converged': converged,
        'iterations': iteration,
        'wall_clock_time': elapsed_time,
        'initial_residual_norm': initial_residual_norm,
        'final_residual_norm': final_residual_norm,
        'relative_residual': final_residual_norm / np.linalg.norm(b),
        'residual_history': residual_history,
        'a_norm_error': compute_a_norm_error(x, x_ref, A),
        'two_norm_error': np.linalg.norm(x - x_ref),
        'relative_solution_error': np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref),
        'conjugacy_loss': None,  # N/A for GD
    }
    
    return x, diagnostics


def conjugate_gradient_with_diagnostics(A: np.ndarray,
                                         b: np.ndarray,
                                         x0: np.ndarray,
                                         x_ref: np.ndarray,
                                         tol: float = TOLERANCE,
                                         max_iter: int = MAX_ITERATIONS,
                                         preconditioner: Optional[Callable] = None,
                                         solver_name: str = "CG") -> Tuple[np.ndarray, dict]:
    """
    Conjugate Gradient (optionally preconditioned) with detailed diagnostics.
    
    Tracks loss of conjugacy: for true CG, d_i^T A d_j = 0 for i != j.
    We measure deviation from this property.
    """
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    
    if preconditioner is not None:
        z = preconditioner(r)
    else:
        z = r.copy()
    
    d = z.copy()
    
    initial_residual_norm = np.linalg.norm(r)
    residual_history = [initial_residual_norm]
    
    # Track search directions for conjugacy loss computation
    search_directions = [d.copy()]
    conjugacy_losses = []
    
    rTz = np.dot(r, z)
    
    start_time = time.perf_counter()
    
    converged = False
    iteration = 0
    
    while iteration < max_iter:
        residual_norm = np.linalg.norm(r)
        
        if residual_norm <= tol:
            converged = True
            break
        
        Ad = A @ d
        dTAd = np.dot(d, Ad)
        
        if abs(dTAd) < 1e-15:
            break
        
        alpha = rTz / dTAd
        
        x = x + alpha * d
        r = r - alpha * Ad
        
        residual_history.append(np.linalg.norm(r))
        
        if preconditioner is not None:
            z = preconditioner(r)
        else:
            z = r.copy()
        
        rTz_new = np.dot(r, z)
        beta = rTz_new / rTz
        
        d = z + beta * d
        rTz = rTz_new
        
        # Track search direction for conjugacy analysis
        search_directions.append(d.copy())
        
        # Compute conjugacy loss: |d_new^T A d_old| / (||d_new||_A ||d_old||_A)
        if len(search_directions) >= 2:
            d_new = search_directions[-1]
            d_old = search_directions[-2]
            
            Ad_new = A @ d_new
            Ad_old = A @ d_old
            
            norm_A_new = np.sqrt(abs(np.dot(d_new, Ad_new)))
            norm_A_old = np.sqrt(abs(np.dot(d_old, Ad_old)))
            
            if norm_A_new > 1e-15 and norm_A_old > 1e-15:
                conjugacy = abs(np.dot(d_new, Ad_old)) / (norm_A_new * norm_A_old)
                conjugacy_losses.append(conjugacy)
        
        iteration += 1
    
    elapsed_time = time.perf_counter() - start_time
    
    final_residual_norm = np.linalg.norm(b - A @ x)
    
    diagnostics = {
        'converged': converged,
        'iterations': iteration,
        'wall_clock_time': elapsed_time,
        'initial_residual_norm': initial_residual_norm,
        'final_residual_norm': final_residual_norm,
        'relative_residual': final_residual_norm / np.linalg.norm(b),
        'residual_history': residual_history,
        'a_norm_error': compute_a_norm_error(x, x_ref, A),
        'two_norm_error': np.linalg.norm(x - x_ref),
        'relative_solution_error': np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref),
        'conjugacy_loss': conjugacy_losses if conjugacy_losses else None,
    }
    
    return x, diagnostics


# =============================================================================
# PRECONDITIONERS
# =============================================================================

def create_jacobi_preconditioner(A: np.ndarray) -> Callable:
    """Jacobi (diagonal) preconditioner."""
    diag_inv = 1.0 / np.diag(A)
    return lambda r: diag_inv * r


def create_ssor_preconditioner(A: np.ndarray, omega: float = 1.2) -> Callable:
    """SSOR preconditioner."""
    n = A.shape[0]
    D = np.diag(A)
    L = np.tril(A, -1)
    D_omega = D / omega
    
    def apply(r):
        # Forward solve
        y = np.zeros(n)
        for i in range(n):
            y[i] = (r[i] - np.dot(L[i, :i], y[:i])) / D_omega[i]
        
        # Scale
        z = D * y
        
        # Backward solve
        x = np.zeros(n)
        U = np.triu(A, 1)
        for i in range(n-1, -1, -1):
            x[i] = (z[i] - np.dot(U[i, i+1:], x[i+1:])) / D_omega[i]
        
        return omega * (2 - omega) * x
    
    return apply


def create_ichol_preconditioner(A: np.ndarray) -> Callable:
    """Incomplete Cholesky preconditioner."""
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for j in range(n):
        sum_sq = np.dot(L[j, :j], L[j, :j])
        L[j, j] = np.sqrt(max(A[j, j] - sum_sq, 1e-10))
        
        for i in range(j+1, n):
            sum_prod = np.dot(L[i, :j], L[j, :j])
            L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    def apply(r):
        # Forward solve L y = r
        y = np.zeros(n)
        for i in range(n):
            y[i] = (r[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        
        # Backward solve L^T x = y
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
        
        return x
    
    return apply


# =============================================================================
# MAIN DIAGNOSTICS RUNNER
# =============================================================================

def load_problem_data(data_dir: str, date_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load problem data for a specific rebalancing date."""
    rolling_dir = os.path.join(data_dir, 'rolling_parameters')
    mu = np.load(os.path.join(rolling_dir, f'mu_{date_str}.npy'))
    cov = np.load(os.path.join(rolling_dir, f'cov_{date_str}.npy'))
    
    # RHS of linear system: b = mu / lambda
    b = mu / RISK_AVERSION
    
    return cov, b, mu


def run_diagnostics_for_date(A: np.ndarray, 
                              b: np.ndarray,
                              date_str: str) -> Dict[str, SolverDiagnostics]:
    """Run all solvers with full diagnostics for a single date."""
    n = len(b)
    
    # Compute problem conditioning
    eigenvalues = np.linalg.eigvalsh(A)
    min_eig = eigenvalues.min()
    max_eig = eigenvalues.max()
    cond_num = max_eig / min_eig
    
    # Compute high-precision reference solution
    x_ref = compute_reference_solution(A, b)
    
    # Initial guess (same for all solvers)
    x0 = np.ones(n) / n
    
    results = {}
    
    # 1. Gradient Descent
    x_gd, diag_gd = gradient_descent_with_diagnostics(A, b, x0, x_ref)
    results['GD'] = SolverDiagnostics(
        rebal_date=date_str,
        solver_name='GD',
        condition_number=cond_num,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        **diag_gd
    )
    
    # 2. Conjugate Gradient
    x_cg, diag_cg = conjugate_gradient_with_diagnostics(A, b, x0, x_ref, solver_name='CG')
    results['CG'] = SolverDiagnostics(
        rebal_date=date_str,
        solver_name='CG',
        condition_number=cond_num,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        max_conjugacy_loss=max(diag_cg['conjugacy_loss']) if diag_cg['conjugacy_loss'] else None,
        mean_conjugacy_loss=np.mean(diag_cg['conjugacy_loss']) if diag_cg['conjugacy_loss'] else None,
        **{k: v for k, v in diag_cg.items() if k != 'conjugacy_loss'}
    )
    
    # 3. PCG with Jacobi
    precond_jacobi = create_jacobi_preconditioner(A)
    x_pcg_j, diag_pcg_j = conjugate_gradient_with_diagnostics(
        A, b, x0, x_ref, preconditioner=precond_jacobi, solver_name='PCG_Jacobi'
    )
    results['PCG_Jacobi'] = SolverDiagnostics(
        rebal_date=date_str,
        solver_name='PCG_Jacobi',
        condition_number=cond_num,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        max_conjugacy_loss=max(diag_pcg_j['conjugacy_loss']) if diag_pcg_j['conjugacy_loss'] else None,
        mean_conjugacy_loss=np.mean(diag_pcg_j['conjugacy_loss']) if diag_pcg_j['conjugacy_loss'] else None,
        **{k: v for k, v in diag_pcg_j.items() if k != 'conjugacy_loss'}
    )
    
    # 4. PCG with SSOR
    precond_ssor = create_ssor_preconditioner(A, omega=1.2)
    x_pcg_s, diag_pcg_s = conjugate_gradient_with_diagnostics(
        A, b, x0, x_ref, preconditioner=precond_ssor, solver_name='PCG_SSOR'
    )
    results['PCG_SSOR'] = SolverDiagnostics(
        rebal_date=date_str,
        solver_name='PCG_SSOR',
        condition_number=cond_num,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        max_conjugacy_loss=max(diag_pcg_s['conjugacy_loss']) if diag_pcg_s['conjugacy_loss'] else None,
        mean_conjugacy_loss=np.mean(diag_pcg_s['conjugacy_loss']) if diag_pcg_s['conjugacy_loss'] else None,
        **{k: v for k, v in diag_pcg_s.items() if k != 'conjugacy_loss'}
    )
    
    # 5. PCG with Incomplete Cholesky
    precond_ichol = create_ichol_preconditioner(A)
    x_pcg_ic, diag_pcg_ic = conjugate_gradient_with_diagnostics(
        A, b, x0, x_ref, preconditioner=precond_ichol, solver_name='PCG_IChol'
    )
    results['PCG_IChol'] = SolverDiagnostics(
        rebal_date=date_str,
        solver_name='PCG_IChol',
        condition_number=cond_num,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        max_conjugacy_loss=max(diag_pcg_ic['conjugacy_loss']) if diag_pcg_ic['conjugacy_loss'] else None,
        mean_conjugacy_loss=np.mean(diag_pcg_ic['conjugacy_loss']) if diag_pcg_ic['conjugacy_loss'] else None,
        **{k: v for k, v in diag_pcg_ic.items() if k != 'conjugacy_loss'}
    )
    
    return results


def run_full_diagnostics(data_dir: str, verbose: bool = True) -> Dict[str, SolverDiagnosticsCollection]:
    """Run diagnostics across all rebalancing dates."""
    
    # Load rebalancing schedule
    schedule = pd.read_csv(os.path.join(data_dir, 'rebalancing_schedule.csv'))
    
    if verbose:
        print("="*80)
        print("NUMERICAL DIAGNOSTICS FOR ITERATIVE SOLVERS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Tolerance: {TOLERANCE:.2e}")
        print(f"  Max Iterations: {MAX_ITERATIONS}")
        print(f"  Number of rebalancing dates: {len(schedule)}")
        print(f"  Reference solution: Direct solver (LAPACK)")
    
    # Initialize collections
    solvers = ['GD', 'CG', 'PCG_Jacobi', 'PCG_SSOR', 'PCG_IChol']
    collections = {name: SolverDiagnosticsCollection(solver_name=name) for name in solvers}
    
    if verbose:
        print("\n" + "-"*80)
        print("Running diagnostics for each rebalancing date...")
        print("-"*80)
    
    for idx, row in schedule.iterrows():
        date_str = pd.to_datetime(row['rebal_date']).strftime('%Y%m%d')
        
        # Load problem
        A, b, mu = load_problem_data(data_dir, date_str)
        
        # Run all solvers
        results = run_diagnostics_for_date(A, b, date_str)
        
        # Store results
        for solver_name, diag in results.items():
            collections[solver_name].diagnostics.append(diag)
        
        if verbose and idx % 6 == 0:
            cond = results['GD'].condition_number
            print(f"\n  [{idx+1}/{len(schedule)}] Date: {date_str}, κ(Σ) = {cond:.1f}")
            for solver in solvers:
                d = results[solver]
                status = "✓" if d.converged else "✗"
                print(f"    {solver:12s}: {d.iterations:4d} iters, "
                      f"||r||/||b|| = {d.relative_residual:.2e}, "
                      f"||x-x*||_A = {d.a_norm_error:.2e} [{status}]")
    
    return collections


def compute_distribution_tables(collections: Dict[str, SolverDiagnosticsCollection]) -> Dict[str, pd.DataFrame]:
    """Compute distribution statistics tables for all metrics."""
    
    metrics = [
        'iterations',
        'wall_clock_time_ms',
        'relative_residual',
        'a_norm_error',
        'two_norm_error',
        'relative_solution_error',
        'max_conjugacy_loss',
        'mean_conjugacy_loss',
    ]
    
    tables = {}
    
    for metric in metrics:
        rows = []
        for solver_name, collection in collections.items():
            stats = collection.get_distribution_stats(metric)
            stats['solver'] = solver_name
            rows.append(stats)
        
        df = pd.DataFrame(rows)
        df = df[['solver', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max']]
        tables[metric] = df
    
    return tables


def save_diagnostics(collections: Dict[str, SolverDiagnosticsCollection],
                     tables: Dict[str, pd.DataFrame],
                     output_dir: str):
    """Save all diagnostics to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw diagnostics per solver
    all_diagnostics = []
    for solver_name, collection in collections.items():
        df = collection.to_dataframe()
        all_diagnostics.append(df)
        
        # Save individual solver diagnostics
        df.to_csv(os.path.join(output_dir, f'diagnostics_{solver_name}.csv'), index=False)
    
    # Combined diagnostics
    combined = pd.concat(all_diagnostics, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, 'all_diagnostics.csv'), index=False)
    
    # Save distribution tables
    for metric, table in tables.items():
        table.to_csv(os.path.join(output_dir, f'distribution_{metric}.csv'), index=False)
    
    # Save summary table
    summary_rows = []
    for solver_name, collection in collections.items():
        df = collection.to_dataframe()
        summary_rows.append({
            'solver': solver_name,
            'convergence_rate': df['converged'].mean(),
            'mean_iterations': df['iterations'].mean(),
            'std_iterations': df['iterations'].std(),
            'mean_time_ms': df['wall_clock_time_ms'].mean(),
            'mean_relative_residual': df['relative_residual'].mean(),
            'max_relative_residual': df['relative_residual'].max(),
            'mean_a_norm_error': df['a_norm_error'].mean(),
            'max_a_norm_error': df['a_norm_error'].max(),
            'mean_relative_solution_error': df['relative_solution_error'].mean(),
            'max_conjugacy_loss': df['max_conjugacy_loss'].max() if 'max_conjugacy_loss' in df else None,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    return combined, summary_df


def print_distribution_report(tables: Dict[str, pd.DataFrame], 
                               collections: Dict[str, SolverDiagnosticsCollection]):
    """Print formatted distribution report."""
    
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS OVER ALL REBALANCING DATES")
    print("="*80)
    
    # Iterations distribution
    print("\n" + "-"*80)
    print("ITERATIONS DISTRIBUTION")
    print("-"*80)
    print(tables['iterations'].to_string(index=False, float_format='%.2f'))
    
    # Runtime distribution
    print("\n" + "-"*80)
    print("WALL-CLOCK TIME (ms) DISTRIBUTION")
    print("-"*80)
    print(tables['wall_clock_time_ms'].to_string(index=False, float_format='%.4f'))
    
    # Relative residual distribution
    print("\n" + "-"*80)
    print("RELATIVE RESIDUAL ||b-Ax||/||b|| DISTRIBUTION")
    print("-"*80)
    df = tables['relative_residual'].copy()
    for col in df.columns:
        if col != 'solver':
            df[col] = df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
    print(df.to_string(index=False))
    
    # A-norm error distribution
    print("\n" + "-"*80)
    print("A-NORM ERROR ||x-x*||_A DISTRIBUTION")
    print("-"*80)
    df = tables['a_norm_error'].copy()
    for col in df.columns:
        if col != 'solver':
            df[col] = df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
    print(df.to_string(index=False))
    
    # Solution error distribution
    print("\n" + "-"*80)
    print("RELATIVE SOLUTION ERROR ||x-x*||/||x*|| DISTRIBUTION")
    print("-"*80)
    df = tables['relative_solution_error'].copy()
    for col in df.columns:
        if col != 'solver':
            df[col] = df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
    print(df.to_string(index=False))
    
    # Conjugacy loss (CG methods only)
    print("\n" + "-"*80)
    print("CONJUGACY LOSS (CG-BASED METHODS ONLY)")
    print("-"*80)
    print("Max conjugacy loss per solve (deviation from A-orthogonality):")
    df = tables['max_conjugacy_loss'].copy()
    for col in df.columns:
        if col != 'solver':
            df[col] = df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
    print(df.to_string(index=False))
    
    # Conditioning analysis
    print("\n" + "-"*80)
    print("PROBLEM CONDITIONING ACROSS DATES")
    print("-"*80)
    
    # Get conditioning from any solver (same for all)
    df = collections['GD'].to_dataframe()
    print(f"Condition Number κ(Σ):")
    print(f"  Mean:   {df['condition_number'].mean():.2f}")
    print(f"  Std:    {df['condition_number'].std():.2f}")
    print(f"  Min:    {df['condition_number'].min():.2f}")
    print(f"  Max:    {df['condition_number'].max():.2f}")
    print(f"\nMinimum Eigenvalue λ_min:")
    print(f"  Mean:   {df['min_eigenvalue'].mean():.2e}")
    print(f"  Min:    {df['min_eigenvalue'].min():.2e}")
    print(f"  Max:    {df['min_eigenvalue'].max():.2e}")


def print_convergence_analysis(collections: Dict[str, SolverDiagnosticsCollection]):
    """Print convergence behavior analysis."""
    
    print("\n" + "="*80)
    print("CONVERGENCE BEHAVIOR ANALYSIS")
    print("="*80)
    
    print("\n" + "-"*80)
    print("CONVERGENCE RATE BY SOLVER")
    print("-"*80)
    print(f"{'Solver':<15} {'Converged':>10} {'Total':>8} {'Rate':>10}")
    print("-"*45)
    
    for solver_name, collection in collections.items():
        df = collection.to_dataframe()
        converged = df['converged'].sum()
        total = len(df)
        rate = converged / total * 100
        print(f"{solver_name:<15} {converged:>10} {total:>8} {rate:>9.1f}%")
    
    print("\n" + "-"*80)
    print("NUMERICAL STABILITY ASSESSMENT")
    print("-"*80)
    
    for solver_name, collection in collections.items():
        df = collection.to_dataframe()
        
        # Check for any numerical issues
        has_nan = df.isna().any().any()
        max_residual = df['relative_residual'].max()
        max_error = df['relative_solution_error'].max()
        
        status = "✓ STABLE" if max_residual < 1e-3 and max_error < 1e-3 else "⚠ CHECK"
        
        print(f"\n{solver_name}:")
        print(f"  Max relative residual:       {max_residual:.2e}")
        print(f"  Max relative solution error: {max_error:.2e}")
        print(f"  Contains NaN values:         {has_nan}")
        print(f"  Status:                      {status}")


def main():
    """Main execution function."""
    
    # Run diagnostics
    collections = run_full_diagnostics(DATA_DIR, verbose=True)
    
    # Compute distribution tables
    tables = compute_distribution_tables(collections)
    
    # Save results
    output_dir = os.path.join(DATA_DIR, 'numerical_diagnostics')
    combined, summary = save_diagnostics(collections, tables, output_dir)
    
    # Print reports
    print_distribution_report(tables, collections)
    print_convergence_analysis(collections)
    
    print("\n" + "="*80)
    print(f"Diagnostics saved to: {output_dir}")
    print("="*80)
    
    return collections, tables


if __name__ == '__main__':
    collections, tables = main()


