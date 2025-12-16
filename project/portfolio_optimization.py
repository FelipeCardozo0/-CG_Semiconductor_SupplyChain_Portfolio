#!/usr/bin/env python3
"""
Portfolio Optimization via Iterative Linear Solvers
===================================================

This script solves the mean-variance portfolio optimization problem at each 
rebalancing date using iterative methods.

Problem Formulation:
    max_w  w^T μ - (λ/2) w^T Σ w
    
    First-order optimality: Σ w = (1/λ) μ
    
    We solve: Σ w = μ̃  where μ̃ = μ/λ (or equivalently scale after solving)

Solvers Compared:
    1. Gradient Descent (GD) - baseline iterative method
    2. Conjugate Gradient (CG) - Krylov subspace method
    3. Preconditioned CG (PCG) - accelerated via preconditioning

Portfolio Constraints:
    - Budget constraint: sum(w) = 1 (normalized post-solution)
    - Long-only: w >= 0 (projected post-solution, not embedded)

Objective:
    Evaluate numerical convergence, stability, and computational efficiency
    under realistic covariance conditioning.

Author: Portfolio Optimization Experiments
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import time
import warnings

# Import solvers
from solvers import (
    GradientDescentSolver,
    ConjugateGradientSolver,
    PreconditionedCGSolver,
)
from solvers.base import SolverResult, verify_spd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Solver parameters (identical across all solvers for fair comparison)
TOLERANCE = 1e-10
MAX_ITERATIONS = 1000

# Risk aversion parameter (λ in the mean-variance objective)
# Higher λ = more risk-averse
RISK_AVERSION = 1.0

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioSolution:
    """Container for portfolio optimization results at a single rebalancing date."""
    rebal_date: pd.Timestamp
    raw_weights: np.ndarray
    normalized_weights: np.ndarray
    long_only_weights: np.ndarray
    solver_result: SolverResult
    
    # Problem characteristics
    condition_number: float = 0.0
    min_eigenvalue: float = 0.0
    
    def __post_init__(self):
        self.budget_constraint_satisfied = np.isclose(np.sum(self.normalized_weights), 1.0)
        self.long_only_satisfied = np.all(self.long_only_weights >= -1e-10)


@dataclass
class SolverComparison:
    """Container for comparing solver performance across rebalancing dates."""
    solver_name: str
    solutions: List[PortfolioSolution] = field(default_factory=list)
    
    @property
    def n_dates(self) -> int:
        return len(self.solutions)
    
    @property
    def convergence_rate(self) -> float:
        """Fraction of problems that converged."""
        if self.n_dates == 0:
            return 0.0
        return sum(1 for s in self.solutions if s.solver_result.converged) / self.n_dates
    
    @property
    def mean_iterations(self) -> float:
        """Mean iterations across all rebalancing dates."""
        if self.n_dates == 0:
            return 0.0
        return np.mean([s.solver_result.iterations for s in self.solutions])
    
    @property
    def total_time(self) -> float:
        """Total solve time across all dates."""
        return sum(s.solver_result.elapsed_time for s in self.solutions)
    
    @property
    def mean_time(self) -> float:
        """Mean solve time per problem."""
        if self.n_dates == 0:
            return 0.0
        return self.total_time / self.n_dates
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        for sol in self.solutions:
            records.append({
                'rebal_date': sol.rebal_date,
                'solver': self.solver_name,
                'converged': sol.solver_result.converged,
                'iterations': sol.solver_result.iterations,
                'elapsed_time': sol.solver_result.elapsed_time,
                'final_residual': sol.solver_result.final_residual_norm,
                'relative_residual': sol.solver_result.relative_residual,
                'condition_number': sol.condition_number,
                'min_eigenvalue': sol.min_eigenvalue,
            })
        return pd.DataFrame(records)


# =============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# =============================================================================

def load_rebalancing_data(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load rebalancing schedule and ticker information.
    
    Returns
    -------
    tuple
        (rebalancing_schedule DataFrame, list of tickers)
    """
    schedule_path = os.path.join(data_dir, 'rebalancing_schedule.csv')
    schedule = pd.read_csv(schedule_path)
    schedule['rebal_date'] = pd.to_datetime(schedule['rebal_date'])
    
    tickers = np.load(os.path.join(data_dir, 'tickers.npy'))
    
    return schedule, list(tickers)


def load_rolling_parameters(data_dir: str, 
                            date_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load μ and Σ for a specific rebalancing date.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    date_str : str
        Date string in YYYYMMDD format
        
    Returns
    -------
    tuple
        (expected_returns μ, covariance_matrix Σ)
    """
    rolling_dir = os.path.join(data_dir, 'rolling_parameters')
    mu = np.load(os.path.join(rolling_dir, f'mu_{date_str}.npy'))
    cov = np.load(os.path.join(rolling_dir, f'cov_{date_str}.npy'))
    return mu, cov


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights to satisfy budget constraint: sum(w) = 1.
    
    Parameters
    ----------
    w : np.ndarray
        Raw portfolio weights
        
    Returns
    -------
    np.ndarray
        Normalized weights summing to 1
    """
    total = np.sum(w)
    if abs(total) < 1e-15:
        # Fallback to equal weights if sum is zero
        return np.ones_like(w) / len(w)
    return w / total


def project_long_only(w: np.ndarray) -> np.ndarray:
    """
    Project weights to long-only constraint: w >= 0, sum(w) = 1.
    
    Uses simple projection: clamp negatives to zero, then renormalize.
    
    Parameters
    ----------
    w : np.ndarray
        Portfolio weights (possibly negative)
        
    Returns
    -------
    np.ndarray
        Long-only weights (non-negative, sum to 1)
    """
    w_proj = np.maximum(w, 0)
    total = np.sum(w_proj)
    if total < 1e-15:
        # All weights were negative, fallback to equal weights
        return np.ones_like(w) / len(w)
    return w_proj / total


def solve_portfolio_optimization(mu: np.ndarray,
                                  cov: np.ndarray,
                                  solver,
                                  risk_aversion: float = RISK_AVERSION) -> Tuple[np.ndarray, SolverResult]:
    """
    Solve the mean-variance portfolio optimization problem.
    
    Problem: Σ w = μ / λ
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector (n,)
    cov : np.ndarray
        Covariance matrix (n x n)
    solver : IterativeSolver
        Iterative solver instance
    risk_aversion : float
        Risk aversion parameter λ
        
    Returns
    -------
    tuple
        (raw_weights, solver_result)
    """
    # Right-hand side: μ / λ
    b = mu / risk_aversion
    
    # Initial guess: equal weights
    n = len(mu)
    x0 = np.ones(n) / n
    
    # Solve: Σ w = b
    result = solver.solve(cov, b, x0)
    
    return result.solution, result


def run_optimization_experiment(data_dir: str,
                                 tolerance: float = TOLERANCE,
                                 max_iterations: int = MAX_ITERATIONS,
                                 risk_aversion: float = RISK_AVERSION,
                                 verbose: bool = True) -> Dict[str, SolverComparison]:
    """
    Run portfolio optimization across all rebalancing dates with all solvers.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    tolerance : float
        Convergence tolerance for all solvers
    max_iterations : int
        Maximum iterations for all solvers
    risk_aversion : float
        Risk aversion parameter λ
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Dictionary mapping solver names to SolverComparison objects
    """
    # Load rebalancing schedule
    schedule, tickers = load_rebalancing_data(data_dir)
    n_dates = len(schedule)
    n_assets = len(tickers)
    
    if verbose:
        print("="*70)
        print("PORTFOLIO OPTIMIZATION EXPERIMENT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Number of rebalancing dates: {n_dates}")
        print(f"  Number of assets: {n_assets}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Risk aversion (λ): {risk_aversion}")
    
    # Initialize solvers with identical parameters
    solvers = {
        'GD': GradientDescentSolver(
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=False
        ),
        'CG': ConjugateGradientSolver(
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=False
        ),
        'PCG_Jacobi': PreconditionedCGSolver(
            preconditioner='jacobi',
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=False
        ),
        'PCG_SSOR': PreconditionedCGSolver(
            preconditioner='ssor',
            precond_params={'omega': 1.2},
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=False
        ),
        'PCG_IChol': PreconditionedCGSolver(
            preconditioner='ichol',
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=False
        ),
    }
    
    # Initialize results containers
    results = {name: SolverComparison(solver_name=name) for name in solvers}
    
    if verbose:
        print(f"\nSolvers: {list(solvers.keys())}")
        print("\n" + "-"*70)
        print("Running optimization at each rebalancing date...")
        print("-"*70)
    
    # Process each rebalancing date
    for idx, row in schedule.iterrows():
        rebal_date = row['rebal_date']
        date_str = rebal_date.strftime('%Y%m%d')
        
        # Load parameters for this date
        mu, cov = load_rolling_parameters(data_dir, date_str)
        
        # Verify SPD
        is_spd, min_eig = verify_spd(cov)
        eigenvalues = np.linalg.eigvalsh(cov)
        cond_num = eigenvalues.max() / eigenvalues.min()
        
        if verbose and idx % 6 == 0:
            print(f"\n  [{idx+1}/{n_dates}] {rebal_date.strftime('%Y-%m-%d')}: "
                  f"κ(Σ) = {cond_num:.1f}, λ_min = {min_eig:.2e}")
        
        # Solve with each solver
        for solver_name, solver in solvers.items():
            raw_weights, solver_result = solve_portfolio_optimization(
                mu, cov, solver, risk_aversion
            )
            
            # Normalize weights
            normalized_weights = normalize_weights(raw_weights)
            
            # Project to long-only
            long_only_weights = project_long_only(normalized_weights)
            
            # Create solution object
            solution = PortfolioSolution(
                rebal_date=rebal_date,
                raw_weights=raw_weights,
                normalized_weights=normalized_weights,
                long_only_weights=long_only_weights,
                solver_result=solver_result,
                condition_number=cond_num,
                min_eigenvalue=min_eig,
            )
            
            results[solver_name].solutions.append(solution)
        
        if verbose and idx % 6 == 0:
            # Print solver comparison for this date
            for solver_name in solvers:
                sol = results[solver_name].solutions[-1]
                status = "✓" if sol.solver_result.converged else "✗"
                print(f"    {solver_name:12s}: {sol.solver_result.iterations:4d} iters, "
                      f"||r|| = {sol.solver_result.final_residual_norm:.2e}, "
                      f"t = {sol.solver_result.elapsed_time*1000:.2f} ms [{status}]")
    
    return results


def analyze_results(results: Dict[str, SolverComparison],
                    tickers: List[str],
                    output_dir: str) -> pd.DataFrame:
    """
    Analyze and save optimization results.
    
    Parameters
    ----------
    results : dict
        Dictionary of solver comparisons
    tickers : list
        Asset ticker symbols
    output_dir : str
        Directory to save results
        
    Returns
    -------
    pd.DataFrame
        Summary statistics DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results into single DataFrame
    all_results = pd.concat([
        comp.to_dataframe() for comp in results.values()
    ], ignore_index=True)
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'solver_results.csv')
    all_results.to_csv(results_path, index=False)
    
    # Compute summary statistics
    summary_records = []
    for solver_name, comp in results.items():
        df = comp.to_dataframe()
        summary_records.append({
            'solver': solver_name,
            'convergence_rate': comp.convergence_rate,
            'mean_iterations': comp.mean_iterations,
            'std_iterations': df['iterations'].std(),
            'min_iterations': df['iterations'].min(),
            'max_iterations': df['iterations'].max(),
            'mean_time_ms': comp.mean_time * 1000,
            'total_time_s': comp.total_time,
            'mean_residual': df['final_residual'].mean(),
            'max_residual': df['final_residual'].max(),
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, 'solver_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save portfolio weights for each solver
    weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    for solver_name, comp in results.items():
        # Normalized weights
        weights_df = pd.DataFrame(
            [sol.normalized_weights for sol in comp.solutions],
            columns=tickers,
            index=[sol.rebal_date for sol in comp.solutions]
        )
        weights_df.index.name = 'rebal_date'
        weights_df.to_csv(os.path.join(weights_dir, f'weights_{solver_name}.csv'))
        
        # Long-only weights
        lo_weights_df = pd.DataFrame(
            [sol.long_only_weights for sol in comp.solutions],
            columns=tickers,
            index=[sol.rebal_date for sol in comp.solutions]
        )
        lo_weights_df.index.name = 'rebal_date'
        lo_weights_df.to_csv(os.path.join(weights_dir, f'weights_long_only_{solver_name}.csv'))
    
    # Save convergence histories
    convergence_dir = os.path.join(output_dir, 'convergence')
    os.makedirs(convergence_dir, exist_ok=True)
    
    for solver_name, comp in results.items():
        histories = {}
        for sol in comp.solutions:
            date_str = sol.rebal_date.strftime('%Y%m%d')
            histories[date_str] = sol.solver_result.residual_history
        
        # Save as separate file per solver
        max_len = max(len(h) for h in histories.values())
        padded_histories = {
            k: v + [np.nan] * (max_len - len(v)) 
            for k, v in histories.items()
        }
        conv_df = pd.DataFrame(padded_histories)
        conv_df.index.name = 'iteration'
        conv_df.to_csv(os.path.join(convergence_dir, f'convergence_{solver_name}.csv'))
    
    return summary_df


def print_summary(summary_df: pd.DataFrame, results: Dict[str, SolverComparison]):
    """Print formatted summary of results."""
    print("\n" + "="*70)
    print("SOLVER COMPARISON SUMMARY")
    print("="*70)
    
    print("\n" + "-"*70)
    print("CONVERGENCE & ITERATION STATISTICS")
    print("-"*70)
    print(f"{'Solver':<15} {'Conv.%':>8} {'Mean Iter':>10} {'Std':>8} {'Min':>6} {'Max':>6}")
    print("-"*70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['solver']:<15} {row['convergence_rate']*100:>7.1f}% "
              f"{row['mean_iterations']:>10.1f} {row['std_iterations']:>8.1f} "
              f"{row['min_iterations']:>6.0f} {row['max_iterations']:>6.0f}")
    
    print("\n" + "-"*70)
    print("TIMING & RESIDUAL STATISTICS")
    print("-"*70)
    print(f"{'Solver':<15} {'Mean Time (ms)':>14} {'Total Time (s)':>15} {'Mean ||r||':>12} {'Max ||r||':>12}")
    print("-"*70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['solver']:<15} {row['mean_time_ms']:>14.3f} "
              f"{row['total_time_s']:>15.4f} "
              f"{row['mean_residual']:>12.2e} {row['max_residual']:>12.2e}")
    
    print("\n" + "-"*70)
    print("RELATIVE PERFORMANCE (vs Gradient Descent)")
    print("-"*70)
    
    gd_row = summary_df[summary_df['solver'] == 'GD'].iloc[0]
    gd_iters = gd_row['mean_iterations']
    gd_time = gd_row['mean_time_ms']
    
    print(f"{'Solver':<15} {'Iter Speedup':>14} {'Time Speedup':>14}")
    print("-"*70)
    
    for _, row in summary_df.iterrows():
        iter_speedup = gd_iters / row['mean_iterations'] if row['mean_iterations'] > 0 else 0
        time_speedup = gd_time / row['mean_time_ms'] if row['mean_time_ms'] > 0 else 0
        print(f"{row['solver']:<15} {iter_speedup:>14.2f}x {time_speedup:>14.2f}x")


def main():
    """Main execution function."""
    print("="*70)
    print("MEAN-VARIANCE PORTFOLIO OPTIMIZATION")
    print("Iterative Solver Comparison Experiment")
    print("="*70)
    
    # Run optimization experiment
    results = run_optimization_experiment(
        DATA_DIR,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        risk_aversion=RISK_AVERSION,
        verbose=True
    )
    
    # Load tickers for output
    tickers = list(np.load(os.path.join(DATA_DIR, 'tickers.npy')))
    
    # Analyze and save results
    output_dir = os.path.join(DATA_DIR, 'optimization_results')
    summary_df = analyze_results(results, tickers, output_dir)
    
    # Print summary
    print_summary(summary_df, results)
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return results, summary_df


if __name__ == '__main__':
    results, summary = main()


