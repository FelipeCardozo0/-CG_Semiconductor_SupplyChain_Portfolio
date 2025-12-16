#!/usr/bin/env python3
"""
Train-Test Split for Portfolio Optimization
============================================

This script partitions the semiconductor equity return dataset into a strictly 
chronological train-test framework to eliminate look-ahead bias.

Framework:
- Training Window: January 1, 2018 - December 31, 2022
  Used exclusively for initial parameter estimation (expected returns, covariance)
  
- Test Window: January 1, 2023 - December 31, 2024
  Used solely for out-of-sample evaluation

Rolling Estimation:
- Portfolio weights are computed once per rebalancing date using information 
  available up to that date only
- Estimation windows are fixed-length and rolled forward in time
- No test-period data influences parameter estimation

Purpose:
- Assess numerical stability and convergence behavior
- Evaluate realized performance under realistic deployment conditions
- Avoid in-sample optimality bias

Author: Train-Test Split for Quadratic Portfolio Optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Train-Test Split Dates
TRAIN_START = '2018-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2024-12-31'

# Rolling Estimation Window Configurations (in trading days, approx.)
# 252 trading days ≈ 1 year
ESTIMATION_WINDOWS = {
    '1Y': 252,      # 1-year trailing window
    '2Y': 504,      # 2-year trailing window
    '3Y': 756,      # 3-year trailing window
    'expanding': None  # Expanding window (all available history)
}

# Default estimation window for primary analysis
DEFAULT_WINDOW = '2Y'

# Rebalancing frequency (in trading days)
# 21 ≈ monthly, 63 ≈ quarterly, 252 ≈ annually
REBALANCE_FREQUENCIES = {
    'monthly': 21,
    'quarterly': 63,
    'annually': 252
}

DEFAULT_REBALANCE = 'monthly'

# Ridge regularization parameter
RIDGE_LAMBDA = 1e-6

# Data directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_returns() -> pd.DataFrame:
    """
    Load the cleaned return data from CSV.
    
    Returns
    -------
    pd.DataFrame
        Daily log returns with DatetimeIndex
    """
    returns_path = os.path.join(DATA_DIR, 'semiconductor_returns.csv')
    returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
    return returns


def split_train_test(returns: pd.DataFrame,
                     train_start: str = TRAIN_START,
                     train_end: str = TRAIN_END,
                     test_start: str = TEST_START,
                     test_end: str = TEST_END) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split returns into training and testing sets based on dates.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Full sample of returns
    train_start, train_end : str
        Training period bounds
    test_start, test_end : str
        Testing period bounds
        
    Returns
    -------
    tuple
        (train_returns, test_returns)
    """
    train_mask = (returns.index >= train_start) & (returns.index <= train_end)
    test_mask = (returns.index >= test_start) & (returns.index <= test_end)
    
    train_returns = returns.loc[train_mask].copy()
    test_returns = returns.loc[test_mask].copy()
    
    return train_returns, test_returns


def compute_parameters(returns: pd.DataFrame, 
                       ridge_lambda: float = RIDGE_LAMBDA) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute expected returns (mean) and covariance matrix from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for estimation
    ridge_lambda : float
        Ridge regularization for covariance matrix
        
    Returns
    -------
    tuple
        (expected_returns, covariance_matrix)
    """
    mu = returns.mean()
    cov = returns.cov()
    
    # Add ridge regularization for numerical stability
    if ridge_lambda > 0:
        n = len(cov)
        cov = cov + pd.DataFrame(
            ridge_lambda * np.eye(n),
            index=cov.index,
            columns=cov.columns
        )
    
    return mu, cov


def get_rebalancing_dates(test_returns: pd.DataFrame,
                          frequency: str = DEFAULT_REBALANCE) -> List[pd.Timestamp]:
    """
    Generate rebalancing dates within the test period.
    
    Parameters
    ----------
    test_returns : pd.DataFrame
        Test period returns
    frequency : str
        Rebalancing frequency ('monthly', 'quarterly', 'annually')
        
    Returns
    -------
    list
        List of rebalancing dates
    """
    freq_days = REBALANCE_FREQUENCIES[frequency]
    test_dates = test_returns.index
    
    # Start with first test date
    rebal_dates = [test_dates[0]]
    
    # Add subsequent rebalancing dates
    current_idx = 0
    while current_idx + freq_days < len(test_dates):
        current_idx += freq_days
        rebal_dates.append(test_dates[current_idx])
    
    return rebal_dates


def compute_rolling_parameters(full_returns: pd.DataFrame,
                               rebal_dates: List[pd.Timestamp],
                               window_type: str = DEFAULT_WINDOW,
                               ridge_lambda: float = RIDGE_LAMBDA) -> Dict:
    """
    Compute parameters for each rebalancing date using only past information.
    
    Parameters
    ----------
    full_returns : pd.DataFrame
        Full return history (train + test)
    rebal_dates : list
        List of rebalancing dates
    window_type : str
        Type of estimation window ('1Y', '2Y', '3Y', 'expanding')
    ridge_lambda : float
        Ridge regularization parameter
        
    Returns
    -------
    dict
        Dictionary with parameters for each rebalancing date
    """
    window_size = ESTIMATION_WINDOWS[window_type]
    parameters = {}
    
    for rebal_date in rebal_dates:
        # Get all returns strictly before rebalancing date
        available_returns = full_returns.loc[full_returns.index < rebal_date]
        
        if window_size is not None and len(available_returns) > window_size:
            # Fixed window: use only last N observations
            estimation_returns = available_returns.iloc[-window_size:]
        else:
            # Expanding window: use all available history
            estimation_returns = available_returns
        
        # Compute parameters
        mu, cov = compute_parameters(estimation_returns, ridge_lambda)
        
        # Verify positive definiteness
        eigenvalues = np.linalg.eigvalsh(cov.values)
        min_eig = eigenvalues.min()
        is_pd = min_eig > 0
        
        parameters[rebal_date] = {
            'expected_returns': mu,
            'covariance_matrix': cov,
            'estimation_start': estimation_returns.index[0],
            'estimation_end': estimation_returns.index[-1],
            'n_observations': len(estimation_returns),
            'min_eigenvalue': min_eig,
            'is_positive_definite': is_pd,
            'condition_number': eigenvalues.max() / min_eig if min_eig > 0 else np.inf
        }
    
    return parameters


def compute_test_period_metrics(test_returns: pd.DataFrame,
                                rebal_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Compute holding period returns for each rebalancing interval.
    
    Parameters
    ----------
    test_returns : pd.DataFrame
        Test period returns
    rebal_dates : list
        Rebalancing dates
        
    Returns
    -------
    pd.DataFrame
        Holding period realized returns for each interval
    """
    holding_periods = []
    
    for i in range(len(rebal_dates)):
        start_date = rebal_dates[i]
        
        if i < len(rebal_dates) - 1:
            end_date = rebal_dates[i + 1]
            # Returns from start_date to day before next rebalance
            period_returns = test_returns.loc[
                (test_returns.index >= start_date) & 
                (test_returns.index < end_date)
            ]
        else:
            # Last period: from last rebal date to end of test
            period_returns = test_returns.loc[test_returns.index >= start_date]
        
        holding_periods.append({
            'rebal_date': start_date,
            'period_start': period_returns.index[0] if len(period_returns) > 0 else start_date,
            'period_end': period_returns.index[-1] if len(period_returns) > 0 else start_date,
            'n_days': len(period_returns),
            'cumulative_returns': period_returns.sum(),  # Log returns sum
            'period_volatility': period_returns.std() * np.sqrt(len(period_returns))
        })
    
    return holding_periods


def save_train_test_data(train_returns: pd.DataFrame,
                         test_returns: pd.DataFrame,
                         train_mu: pd.Series,
                         train_cov: pd.DataFrame,
                         rolling_params: Dict,
                         rebal_dates: List,
                         window_type: str,
                         rebal_freq: str,
                         output_dir: str) -> Dict[str, str]:
    """
    Save all train-test split data and parameters.
    
    Returns
    -------
    dict
        Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    files = {}
    
    # Save train returns
    train_path = os.path.join(output_dir, 'train_returns.csv')
    train_returns.to_csv(train_path)
    files['train_returns'] = train_path
    
    # Save test returns
    test_path = os.path.join(output_dir, 'test_returns.csv')
    test_returns.to_csv(test_path)
    files['test_returns'] = test_path
    
    # Save initial (full training period) parameters
    train_mu_path = os.path.join(output_dir, 'train_expected_returns.csv')
    train_mu.to_csv(train_mu_path, header=['expected_return'])
    files['train_mu'] = train_mu_path
    
    train_cov_path = os.path.join(output_dir, 'train_covariance_matrix.csv')
    train_cov.to_csv(train_cov_path)
    files['train_cov'] = train_cov_path
    
    # Save NumPy arrays for optimization
    np.save(os.path.join(output_dir, 'train_returns_matrix.npy'), train_returns.values)
    np.save(os.path.join(output_dir, 'test_returns_matrix.npy'), test_returns.values)
    np.save(os.path.join(output_dir, 'train_mu_vector.npy'), train_mu.values)
    np.save(os.path.join(output_dir, 'train_cov_matrix.npy'), train_cov.values)
    
    # Save rolling parameters for each rebalancing date
    rolling_dir = os.path.join(output_dir, 'rolling_parameters')
    os.makedirs(rolling_dir, exist_ok=True)
    
    rebal_summary = []
    for i, rebal_date in enumerate(rebal_dates):
        params = rolling_params[rebal_date]
        date_str = rebal_date.strftime('%Y%m%d')
        
        # Save expected returns
        mu_path = os.path.join(rolling_dir, f'mu_{date_str}.npy')
        np.save(mu_path, params['expected_returns'].values)
        
        # Save covariance matrix
        cov_path = os.path.join(rolling_dir, f'cov_{date_str}.npy')
        np.save(cov_path, params['covariance_matrix'].values)
        
        rebal_summary.append({
            'rebal_idx': i,
            'rebal_date': rebal_date,
            'estimation_start': params['estimation_start'],
            'estimation_end': params['estimation_end'],
            'n_observations': params['n_observations'],
            'min_eigenvalue': params['min_eigenvalue'],
            'is_positive_definite': params['is_positive_definite'],
            'condition_number': params['condition_number'],
            'mu_file': f'mu_{date_str}.npy',
            'cov_file': f'cov_{date_str}.npy'
        })
    
    # Save rebalancing summary
    rebal_df = pd.DataFrame(rebal_summary)
    rebal_path = os.path.join(output_dir, 'rebalancing_schedule.csv')
    rebal_df.to_csv(rebal_path, index=False)
    files['rebalancing_schedule'] = rebal_path
    
    # Save metadata
    metadata = {
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'test_start': TEST_START,
        'test_end': TEST_END,
        'train_observations': len(train_returns),
        'test_observations': len(test_returns),
        'n_assets': len(train_returns.columns),
        'estimation_window': window_type,
        'window_size_days': ESTIMATION_WINDOWS[window_type],
        'rebalancing_frequency': rebal_freq,
        'rebalancing_interval_days': REBALANCE_FREQUENCIES[rebal_freq],
        'n_rebalancing_dates': len(rebal_dates),
        'ridge_lambda': RIDGE_LAMBDA,
        'tickers': list(train_returns.columns),
        'data_prepared': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, 'split_metadata.csv')
    pd.Series(metadata).to_csv(metadata_path)
    files['metadata'] = metadata_path
    
    # Save tickers array
    np.save(os.path.join(output_dir, 'tickers.npy'), 
            np.array(train_returns.columns.tolist()))
    
    return files


def print_summary(train_returns: pd.DataFrame,
                  test_returns: pd.DataFrame,
                  train_mu: pd.Series,
                  train_cov: pd.DataFrame,
                  rolling_params: Dict,
                  rebal_dates: List,
                  window_type: str,
                  rebal_freq: str):
    """
    Print comprehensive summary of the train-test split.
    """
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT SUMMARY")
    print("="*70)
    
    print("\n" + "-"*70)
    print("DATA PARTITIONING")
    print("-"*70)
    print(f"Training Period: {train_returns.index[0].strftime('%Y-%m-%d')} to "
          f"{train_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Training Observations: {len(train_returns)} trading days")
    print(f"Testing Period: {test_returns.index[0].strftime('%Y-%m-%d')} to "
          f"{test_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Testing Observations: {len(test_returns)} trading days")
    print(f"Number of Assets: {len(train_returns.columns)}")
    
    print("\n" + "-"*70)
    print("ROLLING ESTIMATION CONFIGURATION")
    print("-"*70)
    print(f"Estimation Window Type: {window_type}")
    window_days = ESTIMATION_WINDOWS[window_type]
    if window_days:
        print(f"Window Size: {window_days} trading days (~{window_days/252:.1f} years)")
    else:
        print("Window Size: Expanding (all available history)")
    print(f"Rebalancing Frequency: {rebal_freq} ({REBALANCE_FREQUENCIES[rebal_freq]} days)")
    print(f"Number of Rebalancing Dates: {len(rebal_dates)}")
    
    print("\n" + "-"*70)
    print("TRAINING PERIOD PARAMETERS (Full Window)")
    print("-"*70)
    
    train_eigenvalues = np.linalg.eigvalsh(train_cov.values)
    print(f"Expected Return Range (annualized): "
          f"{train_mu.min()*252*100:.2f}% to {train_mu.max()*252*100:.2f}%")
    print(f"Covariance Min Eigenvalue: {train_eigenvalues.min():.2e}")
    print(f"Covariance Condition Number: {train_eigenvalues.max()/train_eigenvalues.min():.2f}")
    
    print("\n" + "-"*70)
    print("REBALANCING SCHEDULE")
    print("-"*70)
    print(f"{'Date':<12} {'Est. Start':<12} {'Est. End':<12} {'N Obs':>8} "
          f"{'Min Eig':>12} {'Cond #':>10}")
    print("-"*70)
    
    for i, rebal_date in enumerate(rebal_dates[:10]):  # Show first 10
        p = rolling_params[rebal_date]
        print(f"{rebal_date.strftime('%Y-%m-%d'):<12} "
              f"{p['estimation_start'].strftime('%Y-%m-%d'):<12} "
              f"{p['estimation_end'].strftime('%Y-%m-%d'):<12} "
              f"{p['n_observations']:>8} "
              f"{p['min_eigenvalue']:>12.2e} "
              f"{p['condition_number']:>10.2f}")
    
    if len(rebal_dates) > 10:
        print(f"... ({len(rebal_dates) - 10} more rebalancing dates)")
    
    print("\n" + "-"*70)
    print("TRAIN vs TEST PERIOD STATISTICS")
    print("-"*70)
    
    train_stats = pd.DataFrame({
        'Train Mean (%)': train_returns.mean() * 100,
        'Train Std (%)': train_returns.std() * 100,
        'Test Mean (%)': test_returns.mean() * 100,
        'Test Std (%)': test_returns.std() * 100,
    })
    train_stats['Mean Shift (%)'] = train_stats['Test Mean (%)'] - train_stats['Train Mean (%)']
    train_stats['Vol Shift (%)'] = train_stats['Test Std (%)'] - train_stats['Train Std (%)']
    
    print("\nDaily Return Statistics by Period:")
    print(train_stats.round(4).to_string())
    
    # Correlation stability
    train_corr = train_returns.corr()
    test_corr = test_returns.corr()
    corr_diff = (test_corr - train_corr).abs()
    
    print(f"\nCorrelation Stability:")
    print(f"  Mean Abs. Correlation Change: {corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean():.4f}")
    print(f"  Max Abs. Correlation Change: {corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].max():.4f}")


def main():
    """
    Main execution function for train-test split pipeline.
    """
    print("="*70)
    print("TRAIN-TEST SPLIT FOR PORTFOLIO OPTIMIZATION")
    print("Strictly Chronological Framework - No Look-Ahead Bias")
    print("="*70)
    
    # Configuration
    window_type = DEFAULT_WINDOW
    rebal_freq = DEFAULT_REBALANCE
    
    # Step 1: Load returns data
    print("\n[1/6] Loading return data...")
    full_returns = load_returns()
    print(f"Full sample: {len(full_returns)} observations, {len(full_returns.columns)} assets")
    
    # Step 2: Split into train/test
    print("\n[2/6] Splitting into train/test sets...")
    train_returns, test_returns = split_train_test(full_returns)
    print(f"Training set: {len(train_returns)} observations")
    print(f"Test set: {len(test_returns)} observations")
    
    # Verify no overlap
    train_end_date = train_returns.index[-1]
    test_start_date = test_returns.index[0]
    gap_days = (test_start_date - train_end_date).days
    print(f"Gap between train and test: {gap_days} calendar days")
    
    # Step 3: Compute training period parameters
    print("\n[3/6] Computing training period parameters...")
    train_mu, train_cov = compute_parameters(train_returns)
    
    train_eigenvalues = np.linalg.eigvalsh(train_cov.values)
    print(f"Training covariance min eigenvalue: {train_eigenvalues.min():.2e}")
    print(f"Training covariance is positive definite: {train_eigenvalues.min() > 0}")
    
    # Step 4: Generate rebalancing dates
    print("\n[4/6] Generating rebalancing schedule...")
    rebal_dates = get_rebalancing_dates(test_returns, rebal_freq)
    print(f"Generated {len(rebal_dates)} rebalancing dates ({rebal_freq})")
    
    # Step 5: Compute rolling parameters for each rebalancing date
    print("\n[5/6] Computing rolling parameters for each rebalancing date...")
    print(f"Estimation window: {window_type} ({ESTIMATION_WINDOWS[window_type]} days)")
    
    # Need full history for rolling estimation
    rolling_params = compute_rolling_parameters(
        full_returns, rebal_dates, window_type
    )
    
    # Check all matrices are positive definite
    all_pd = all(p['is_positive_definite'] for p in rolling_params.values())
    print(f"All covariance matrices positive definite: {all_pd}")
    
    # Step 6: Save all data
    print("\n[6/6] Saving train-test split data...")
    files = save_train_test_data(
        train_returns, test_returns,
        train_mu, train_cov,
        rolling_params, rebal_dates,
        window_type, rebal_freq,
        DATA_DIR
    )
    
    print("\nFiles saved:")
    for name, path in files.items():
        print(f"  {name}: {os.path.basename(path)}")
    
    # Print comprehensive summary
    print_summary(
        train_returns, test_returns,
        train_mu, train_cov,
        rolling_params, rebal_dates,
        window_type, rebal_freq
    )
    
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT COMPLETE")
    print("="*70)
    
    return {
        'train_returns': train_returns,
        'test_returns': test_returns,
        'train_mu': train_mu,
        'train_cov': train_cov,
        'rolling_parameters': rolling_params,
        'rebalancing_dates': rebal_dates,
        'files': files
    }


if __name__ == '__main__':
    data = main()


