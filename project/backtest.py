#!/usr/bin/env python3
"""
Out-of-Sample Portfolio Backtest
================================

This script conducts a strictly out-of-sample backtest over the test period
using precomputed portfolio weights from the rolling optimization stage.

Key Constraints:
    - No new estimation, tuning, or solver-specific adjustments
    - At each rebalancing date, apply precomputed weights and hold until next rebalance
    - Realized daily portfolio returns computed from asset returns only
    - No transaction costs, leverage, volatility targeting, or signal overlays
    - Performance reflects solely the numerical equivalence or divergence from solver choice

Performance Metrics:
    - Cumulative Return
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio (assuming 0% risk-free rate)
    - Maximum Drawdown
    - Portfolio Turnover

Purpose:
    Verify that numerically distinct solvers yield economically indistinguishable
    portfolios when applied correctly under identical information sets.

Author: Portfolio Backtest for Solver Comparison
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Benchmark ticker (for context, not alpha attribution)
BENCHMARK_TICKER = 'SPY'

# Annualization factor
TRADING_DAYS_PER_YEAR = 252

# Risk-free rate (assumed 0 for simplicity)
RISK_FREE_RATE = 0.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for portfolio performance metrics."""
    solver_name: str
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_turnover: float
    avg_turnover_per_rebal: float
    n_rebalances: int
    
    def to_dict(self) -> dict:
        return {
            'Solver': self.solver_name,
            'Cumulative Return (%)': self.cumulative_return * 100,
            'Annualized Return (%)': self.annualized_return * 100,
            'Annualized Volatility (%)': self.annualized_volatility * 100,
            'Sharpe Ratio': self.sharpe_ratio,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Total Turnover': self.total_turnover,
            'Avg Turnover/Rebal': self.avg_turnover_per_rebal,
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_returns(data_dir: str) -> pd.DataFrame:
    """Load test period asset returns."""
    returns = pd.read_csv(
        os.path.join(data_dir, 'test_returns.csv'),
        index_col='Date',
        parse_dates=True
    )
    return returns


def load_rebalancing_schedule(data_dir: str) -> pd.DataFrame:
    """Load rebalancing schedule."""
    schedule = pd.read_csv(os.path.join(data_dir, 'rebalancing_schedule.csv'))
    schedule['rebal_date'] = pd.to_datetime(schedule['rebal_date'])
    return schedule


def load_portfolio_weights(data_dir: str, solver: str, long_only: bool = False) -> pd.DataFrame:
    """
    Load precomputed portfolio weights for a solver.
    
    Parameters
    ----------
    data_dir : str
        Data directory path
    solver : str
        Solver name (GD, CG, PCG_Jacobi, PCG_SSOR, PCG_IChol)
    long_only : bool
        If True, load long-only projected weights
    """
    weights_dir = os.path.join(data_dir, 'optimization_results', 'weights')
    
    if long_only:
        filename = f'weights_long_only_{solver}.csv'
    else:
        filename = f'weights_{solver}.csv'
    
    weights = pd.read_csv(
        os.path.join(weights_dir, filename),
        index_col='rebal_date',
        parse_dates=True
    )
    return weights


def fetch_benchmark_returns(start_date: str, end_date: str, ticker: str = BENCHMARK_TICKER) -> pd.Series:
    """
    Fetch benchmark returns for comparison.
    
    Parameters
    ----------
    start_date, end_date : str
        Date range
    ticker : str
        Benchmark ticker symbol
    """
    print(f"Fetching benchmark data ({ticker})...")
    
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    if len(data) == 0:
        print(f"  Warning: Could not fetch {ticker} data")
        return None
    
    # Handle both single ticker (Series) and multi-ticker (DataFrame with MultiIndex) cases
    if 'Close' in data.columns:
        close_prices = data['Close']
    else:
        close_prices = data
    
    # Ensure it's a Series
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    
    # Compute log returns
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    # Ensure it's a proper 1D Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    returns = pd.Series(returns.values, index=returns.index, name=ticker)
    
    return returns


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def compute_portfolio_returns(weights: pd.DataFrame,
                               asset_returns: pd.DataFrame,
                               rebal_dates: List[pd.Timestamp]) -> Tuple[pd.Series, List[float]]:
    """
    Compute realized daily portfolio returns using precomputed weights.
    
    At each rebalancing date, apply the precomputed weights and hold
    until the next rebalance.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Precomputed portfolio weights (indexed by rebal_date)
    asset_returns : pd.DataFrame
        Daily asset returns
    rebal_dates : list
        List of rebalancing dates
        
    Returns
    -------
    tuple
        (portfolio_returns Series, turnover_list)
    """
    portfolio_returns = []
    turnover_list = []
    
    dates = asset_returns.index
    prev_weights = None
    
    for i, date in enumerate(dates):
        # Find the most recent rebalancing date on or before current date
        applicable_rebal_dates = [d for d in rebal_dates if d <= date]
        
        if not applicable_rebal_dates:
            # Before first rebalance, use first weights
            current_rebal_date = rebal_dates[0]
        else:
            current_rebal_date = max(applicable_rebal_dates)
        
        # Get weights for this rebalancing period
        current_weights = weights.loc[current_rebal_date].values
        
        # Compute daily portfolio return: w^T * r
        daily_return = np.dot(current_weights, asset_returns.loc[date].values)
        portfolio_returns.append(daily_return)
        
        # Track turnover at rebalancing dates
        if date in rebal_dates:
            if prev_weights is not None:
                turnover = np.sum(np.abs(current_weights - prev_weights))
                turnover_list.append(turnover)
            prev_weights = current_weights.copy()
    
    portfolio_returns = pd.Series(portfolio_returns, index=dates, name='portfolio_return')
    
    return portfolio_returns, turnover_list


def compute_performance_metrics(returns: pd.Series,
                                  turnover_list: List[float],
                                  solver_name: str) -> PerformanceMetrics:
    """
    Compute performance metrics from portfolio returns.
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    turnover_list : list
        Turnover at each rebalancing
    solver_name : str
        Name of the solver
    """
    # Ensure returns is a 1D Series with numeric values
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    if hasattr(returns, 'values'):
        returns = pd.Series(returns.values.flatten(), index=returns.index).astype(float)
    else:
        returns = pd.Series(returns).astype(float)
    
    # Cumulative return
    cumulative_return = float(np.exp(returns.sum()) - 1)
    
    # Annualized return (from log returns)
    n_days = len(returns)
    annualized_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    
    # Annualized volatility
    annualized_volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    # Sharpe ratio
    if annualized_volatility > 0:
        sharpe_ratio = float((annualized_return - RISK_FREE_RATE) / annualized_volatility)
    else:
        sharpe_ratio = 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())
    
    # Turnover
    total_turnover = sum(turnover_list) if turnover_list else 0.0
    n_rebalances = len(turnover_list)
    avg_turnover = total_turnover / n_rebalances if n_rebalances > 0 else 0.0
    
    return PerformanceMetrics(
        solver_name=solver_name,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        total_turnover=total_turnover,
        avg_turnover_per_rebal=avg_turnover,
        n_rebalances=n_rebalances,
    )


def run_backtest(data_dir: str,
                  solvers: List[str],
                  long_only: bool = True,
                  include_benchmark: bool = True,
                  verbose: bool = True) -> Tuple[Dict[str, pd.Series], Dict[str, PerformanceMetrics]]:
    """
    Run backtest for all solvers.
    
    Parameters
    ----------
    data_dir : str
        Data directory path
    solvers : list
        List of solver names
    long_only : bool
        Use long-only weights
    include_benchmark : bool
        Include benchmark comparison
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (returns_dict, metrics_dict)
    """
    if verbose:
        print("="*80)
        print("OUT-OF-SAMPLE PORTFOLIO BACKTEST")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Solvers: {solvers}")
        print(f"  Long-only weights: {long_only}")
        print(f"  Benchmark: {BENCHMARK_TICKER if include_benchmark else 'None'}")
    
    # Load data
    asset_returns = load_test_returns(data_dir)
    schedule = load_rebalancing_schedule(data_dir)
    rebal_dates = schedule['rebal_date'].tolist()
    
    if verbose:
        print(f"\nTest Period:")
        print(f"  Start: {asset_returns.index[0].strftime('%Y-%m-%d')}")
        print(f"  End: {asset_returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Trading Days: {len(asset_returns)}")
        print(f"  Rebalancing Dates: {len(rebal_dates)}")
    
    returns_dict = {}
    metrics_dict = {}
    
    # Run backtest for each solver
    if verbose:
        print("\n" + "-"*80)
        print("Running backtests...")
        print("-"*80)
    
    for solver in solvers:
        if verbose:
            print(f"\n  Processing {solver}...")
        
        # Load weights
        weights = load_portfolio_weights(data_dir, solver, long_only=long_only)
        
        # Compute portfolio returns
        portfolio_returns, turnover_list = compute_portfolio_returns(
            weights, asset_returns, rebal_dates
        )
        
        # Compute metrics
        metrics = compute_performance_metrics(portfolio_returns, turnover_list, solver)
        
        returns_dict[solver] = portfolio_returns
        metrics_dict[solver] = metrics
        
        if verbose:
            print(f"    Cumulative Return: {metrics.cumulative_return*100:.2f}%")
            print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    
    # Fetch and process benchmark
    if include_benchmark:
        benchmark_returns = fetch_benchmark_returns(
            start_date=asset_returns.index[0].strftime('%Y-%m-%d'),
            end_date=asset_returns.index[-1].strftime('%Y-%m-%d'),
            ticker=BENCHMARK_TICKER
        )
        
        if benchmark_returns is not None:
            # Align benchmark with test returns
            benchmark_returns = benchmark_returns.reindex(asset_returns.index).dropna()
            
            # Fill any remaining gaps
            common_dates = asset_returns.index.intersection(benchmark_returns.index)
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            # Compute benchmark metrics
            benchmark_metrics = compute_performance_metrics(
                benchmark_aligned, [], BENCHMARK_TICKER
            )
            
            returns_dict[BENCHMARK_TICKER] = benchmark_aligned
            metrics_dict[BENCHMARK_TICKER] = benchmark_metrics
            
            if verbose:
                print(f"\n  Benchmark ({BENCHMARK_TICKER}):")
                print(f"    Cumulative Return: {benchmark_metrics.cumulative_return*100:.2f}%")
                print(f"    Sharpe Ratio: {benchmark_metrics.sharpe_ratio:.3f}")
    
    return returns_dict, metrics_dict


def compute_weight_divergence(data_dir: str, solvers: List[str], long_only: bool = True) -> pd.DataFrame:
    """
    Compute weight divergence between solvers to verify numerical equivalence.
    
    Measures the L1 distance between portfolio weights from different solvers.
    """
    weights_all = {}
    for solver in solvers:
        weights_all[solver] = load_portfolio_weights(data_dir, solver, long_only=long_only)
    
    # Compute pairwise divergence
    divergence_records = []
    
    for i, solver1 in enumerate(solvers):
        for solver2 in solvers[i+1:]:
            w1 = weights_all[solver1].values
            w2 = weights_all[solver2].values
            
            # L1 distance per rebalancing date
            l1_distances = np.sum(np.abs(w1 - w2), axis=1)
            
            divergence_records.append({
                'solver_pair': f'{solver1} vs {solver2}',
                'mean_l1_divergence': l1_distances.mean(),
                'max_l1_divergence': l1_distances.max(),
                'min_l1_divergence': l1_distances.min(),
            })
    
    return pd.DataFrame(divergence_records)


def compute_return_correlation(returns_dict: Dict[str, pd.Series], solvers: List[str]) -> pd.DataFrame:
    """Compute correlation matrix of portfolio returns across solvers."""
    returns_df = pd.DataFrame({s: returns_dict[s] for s in solvers if s in returns_dict})
    return returns_df.corr()


def save_results(returns_dict: Dict[str, pd.Series],
                  metrics_dict: Dict[str, PerformanceMetrics],
                  divergence_df: pd.DataFrame,
                  correlation_df: pd.DataFrame,
                  output_dir: str):
    """Save backtest results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save daily returns
    returns_df = pd.DataFrame(returns_dict)
    returns_df.index.name = 'Date'
    returns_df.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))
    
    # Save cumulative returns
    cumulative_df = (1 + returns_df).cumprod()
    cumulative_df.to_csv(os.path.join(output_dir, 'cumulative_returns.csv'))
    
    # Save performance metrics
    metrics_records = [m.to_dict() for m in metrics_dict.values()]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    # Save weight divergence
    divergence_df.to_csv(os.path.join(output_dir, 'weight_divergence.csv'), index=False)
    
    # Save return correlation
    correlation_df.to_csv(os.path.join(output_dir, 'return_correlation.csv'))
    
    return {
        'returns': returns_df,
        'cumulative': cumulative_df,
        'metrics': metrics_df,
        'divergence': divergence_df,
        'correlation': correlation_df,
    }


def print_results(metrics_dict: Dict[str, PerformanceMetrics],
                   divergence_df: pd.DataFrame,
                   correlation_df: pd.DataFrame,
                   solvers: List[str]):
    """Print formatted backtest results."""
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    
    # Performance comparison table
    print("\n" + "-"*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("-"*80)
    
    headers = ['Solver', 'Cum. Ret (%)', 'Ann. Ret (%)', 'Ann. Vol (%)', 
               'Sharpe', 'Max DD (%)', 'Turnover']
    
    print(f"{headers[0]:<15} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12} "
          f"{headers[4]:>8} {headers[5]:>12} {headers[6]:>10}")
    print("-"*80)
    
    for name, metrics in metrics_dict.items():
        print(f"{name:<15} {metrics.cumulative_return*100:>12.2f} "
              f"{metrics.annualized_return*100:>12.2f} "
              f"{metrics.annualized_volatility*100:>12.2f} "
              f"{metrics.sharpe_ratio:>8.3f} "
              f"{metrics.max_drawdown*100:>12.2f} "
              f"{metrics.total_turnover:>10.2f}")
    
    # Weight divergence analysis
    print("\n" + "-"*80)
    print("PORTFOLIO WEIGHT DIVERGENCE (L1 Distance)")
    print("-"*80)
    print("Purpose: Verify numerical equivalence across solvers")
    print()
    
    print(f"{'Solver Pair':<25} {'Mean L1':>12} {'Max L1':>12} {'Min L1':>12}")
    print("-"*65)
    
    for _, row in divergence_df.iterrows():
        print(f"{row['solver_pair']:<25} {row['mean_l1_divergence']:>12.6f} "
              f"{row['max_l1_divergence']:>12.6f} {row['min_l1_divergence']:>12.6f}")
    
    # Return correlation
    print("\n" + "-"*80)
    print("DAILY RETURN CORRELATION MATRIX")
    print("-"*80)
    print("Purpose: Verify portfolios track similarly despite numerical differences")
    print()
    
    # Format correlation matrix
    solver_corr = correlation_df.loc[solvers, solvers] if all(s in correlation_df.index for s in solvers) else correlation_df
    print(solver_corr.round(6).to_string())
    
    # Economic equivalence assessment
    print("\n" + "-"*80)
    print("ECONOMIC EQUIVALENCE ASSESSMENT")
    print("-"*80)
    
    solver_metrics = {k: v for k, v in metrics_dict.items() if k in solvers}
    
    if solver_metrics:
        returns = [m.cumulative_return for m in solver_metrics.values()]
        sharpes = [m.sharpe_ratio for m in solver_metrics.values()]
        
        return_spread = max(returns) - min(returns)
        sharpe_spread = max(sharpes) - min(sharpes)
        
        print(f"\nCumulative Return Spread: {return_spread*100:.4f}%")
        print(f"Sharpe Ratio Spread: {sharpe_spread:.6f}")
        
        # Check if economically equivalent (arbitrary threshold)
        if return_spread < 0.01 and sharpe_spread < 0.01:
            print("\n✓ CONCLUSION: Portfolios are ECONOMICALLY EQUIVALENT")
            print("  Small numerical differences do not materially impact performance.")
        else:
            print("\n⚠ CONCLUSION: Notable differences detected")
            print("  Investigate solver-specific numerical issues.")
    
    # Compare to benchmark
    if BENCHMARK_TICKER in metrics_dict:
        print("\n" + "-"*80)
        print(f"BENCHMARK COMPARISON ({BENCHMARK_TICKER})")
        print("-"*80)
        
        bm = metrics_dict[BENCHMARK_TICKER]
        print(f"\nBenchmark Performance:")
        print(f"  Cumulative Return: {bm.cumulative_return*100:.2f}%")
        print(f"  Sharpe Ratio: {bm.sharpe_ratio:.3f}")
        
        print(f"\nPortfolio vs Benchmark:")
        for name, metrics in solver_metrics.items():
            excess_return = metrics.cumulative_return - bm.cumulative_return
            print(f"  {name}: {excess_return*100:+.2f}% excess return")


def main():
    """Main execution function."""
    
    # Solver list
    solvers = ['GD', 'CG', 'PCG_Jacobi', 'PCG_SSOR', 'PCG_IChol']
    
    # Run backtest
    returns_dict, metrics_dict = run_backtest(
        DATA_DIR,
        solvers=solvers,
        long_only=True,
        include_benchmark=True,
        verbose=True
    )
    
    # Compute weight divergence
    print("\n" + "-"*80)
    print("Computing weight divergence...")
    divergence_df = compute_weight_divergence(DATA_DIR, solvers, long_only=True)
    
    # Compute return correlation
    correlation_df = compute_return_correlation(returns_dict, solvers)
    
    # Save results
    output_dir = os.path.join(DATA_DIR, 'backtest_results')
    saved = save_results(returns_dict, metrics_dict, divergence_df, correlation_df, output_dir)
    
    # Print results
    print_results(metrics_dict, divergence_df, correlation_df, solvers)
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    return returns_dict, metrics_dict, saved


if __name__ == '__main__':
    returns_dict, metrics_dict, saved = main()

