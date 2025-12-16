#!/usr/bin/env python3
"""
Semiconductor Equity Data Preparation for Portfolio Optimization
================================================================

This script retrieves daily adjusted close price data for a fixed universe of 
large-cap U.S. and international semiconductor-related equities using yfinance.

Asset Universe (defined ex-ante, constant throughout sample):
- Foundries: TSM, UMC, GFS (where available)
- Equipment Manufacturers: ASML, AMAT, LRCX, KLAC, TER
- Fabless Designers: NVDA, AMD, QCOM, AVGO, MRVL, INTC, MU, TXN, ADI, NXPI, ON, MCHP

Period: January 1, 2018 - December 31, 2024

Data Processing:
- Prices converted to daily log returns
- Aligned on common trading days
- Cleaned by removing rows with missing values only
- No forward-looking filters, factor engineering, or return-based screening

Output:
- Expected returns: simple historical means
- Risk: sample covariance matrix (with optional ridge regularization)

Author: Data Preparation Script for Quadratic Portfolio Optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - FIXED UNIVERSE (EX-ANTE DEFINITION)
# =============================================================================

# Fixed universe of semiconductor-related equities
# Selected based on pre-2018 existence and large-cap status
# Categories: Foundries, Equipment Manufacturers, Fabless Designers

SEMICONDUCTOR_UNIVERSE = {
    # Foundries
    'TSM': 'Taiwan Semiconductor Manufacturing (Foundry)',
    'UMC': 'United Microelectronics Corp (Foundry)',
    
    # Equipment Manufacturers  
    'ASML': 'ASML Holding (Equipment)',
    'AMAT': 'Applied Materials (Equipment)',
    'LRCX': 'Lam Research (Equipment)',
    'KLAC': 'KLA Corporation (Equipment)',
    'TER': 'Teradyne (Equipment)',
    
    # Fabless Designers
    'NVDA': 'NVIDIA Corporation (Fabless)',
    'AMD': 'Advanced Micro Devices (Fabless)',
    'QCOM': 'Qualcomm (Fabless)',
    'AVGO': 'Broadcom Inc (Fabless)',
    'MRVL': 'Marvell Technology (Fabless)',
    
    # Integrated Device Manufacturers (IDM) - design + manufacturing
    'INTC': 'Intel Corporation (IDM)',
    'MU': 'Micron Technology (Memory)',
    'TXN': 'Texas Instruments (Analog)',
    'ADI': 'Analog Devices (Analog)',
    'NXPI': 'NXP Semiconductors (Automotive)',
    'ON': 'ON Semiconductor (Power)',
    'MCHP': 'Microchip Technology (MCU)',
}

# Sample period (fixed, no adjustments based on data availability)
START_DATE = '2018-01-01'
END_DATE = '2024-12-31'

# Ridge regularization parameter for covariance stabilization
RIDGE_LAMBDA = 1e-6

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices from Yahoo Finance.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close prices, indexed by date
    """
    print(f"Fetching price data for {len(tickers)} tickers...")
    print(f"Period: {start} to {end}")
    
    # Download all tickers at once for efficiency
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Use adjusted prices
        progress=True,
        threads=True
    )
    
    # Extract Close prices (auto_adjust=True means these are adjusted)
    if 'Close' in data.columns.get_level_values(0):
        prices = data['Close']
    else:
        # Single ticker case
        prices = data[['Close']]
        prices.columns = tickers
    
    # Ensure columns are in consistent order
    prices = prices[sorted(prices.columns)]
    
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to daily log returns.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices
        
    Returns
    -------
    pd.DataFrame
        DataFrame of log returns
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns


def clean_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Clean returns data by removing rows with any missing values.
    
    This is the only cleaning operation applied - no forward-looking 
    filters, factor engineering, or return-based screening.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of returns (may contain NaN)
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values
    """
    initial_rows = len(returns)
    
    # Drop rows with any missing values
    clean_returns = returns.dropna()
    
    final_rows = len(clean_returns)
    removed_rows = initial_rows - final_rows
    
    print(f"Removed {removed_rows} rows with missing values")
    print(f"Final dataset: {final_rows} trading days")
    
    return clean_returns


def compute_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """
    Compute expected returns as simple historical means.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of clean log returns
        
    Returns
    -------
    pd.Series
        Expected daily returns (historical mean)
    """
    return returns.mean()


def compute_covariance_matrix(returns: pd.DataFrame, ridge_lambda: float = 0.0) -> pd.DataFrame:
    """
    Compute sample covariance matrix with optional ridge regularization.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of clean log returns
    ridge_lambda : float
        Ridge regularization parameter for diagonal stabilization
        
    Returns
    -------
    pd.DataFrame
        Covariance matrix (optionally regularized)
    """
    # Sample covariance matrix
    cov_matrix = returns.cov()
    
    if ridge_lambda > 0:
        # Add small diagonal ridge for positive definiteness
        n_assets = len(cov_matrix)
        ridge = ridge_lambda * np.eye(n_assets)
        cov_matrix = cov_matrix + pd.DataFrame(
            ridge, 
            index=cov_matrix.index, 
            columns=cov_matrix.columns
        )
    
    return cov_matrix


def verify_positive_definiteness(cov_matrix: pd.DataFrame) -> bool:
    """
    Verify that the covariance matrix is positive definite.
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix
        
    Returns
    -------
    bool
        True if positive definite
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
    min_eigenvalue = eigenvalues.min()
    
    is_pd = min_eigenvalue > 0
    
    print(f"Minimum eigenvalue: {min_eigenvalue:.2e}")
    print(f"Positive definite: {is_pd}")
    
    return is_pd


def save_data(prices: pd.DataFrame, 
              returns: pd.DataFrame,
              expected_returns: pd.Series,
              cov_matrix: pd.DataFrame,
              output_dir: str) -> dict:
    """
    Save all data to CSV files.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    returns : pd.DataFrame
        Log returns (cleaned)
    expected_returns : pd.Series
        Historical mean returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    output_dir : str
        Output directory path
        
    Returns
    -------
    dict
        Dictionary of file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = {}
    
    # Save prices
    prices_path = os.path.join(output_dir, 'semiconductor_prices.csv')
    prices.to_csv(prices_path)
    files['prices'] = prices_path
    
    # Save returns
    returns_path = os.path.join(output_dir, 'semiconductor_returns.csv')
    returns.to_csv(returns_path)
    files['returns'] = returns_path
    
    # Save expected returns
    mu_path = os.path.join(output_dir, 'expected_returns.csv')
    expected_returns.to_csv(mu_path, header=['expected_return'])
    files['expected_returns'] = mu_path
    
    # Save covariance matrix
    cov_path = os.path.join(output_dir, 'covariance_matrix.csv')
    cov_matrix.to_csv(cov_path)
    files['covariance'] = cov_path
    
    # Save metadata
    metadata = {
        'universe': list(SEMICONDUCTOR_UNIVERSE.keys()),
        'descriptions': SEMICONDUCTOR_UNIVERSE,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'n_assets': len(returns.columns),
        'n_observations': len(returns),
        'ridge_lambda': RIDGE_LAMBDA,
        'data_prepared': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    pd.Series(metadata).to_csv(metadata_path)
    files['metadata'] = metadata_path
    
    # Save as NumPy arrays for optimization algorithms
    np.save(os.path.join(output_dir, 'returns_matrix.npy'), returns.values)
    np.save(os.path.join(output_dir, 'expected_returns_vector.npy'), expected_returns.values)
    np.save(os.path.join(output_dir, 'covariance_matrix.npy'), cov_matrix.values)
    np.save(os.path.join(output_dir, 'tickers.npy'), np.array(returns.columns.tolist()))
    
    return files


def print_summary(prices: pd.DataFrame, 
                  returns: pd.DataFrame,
                  expected_returns: pd.Series,
                  cov_matrix: pd.DataFrame):
    """
    Print summary statistics of the prepared data.
    """
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print(f"\nAsset Universe: {len(returns.columns)} securities")
    print(f"Sample Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading Days: {len(returns)}")
    
    print("\n" + "-"*70)
    print("TICKERS IN UNIVERSE:")
    print("-"*70)
    for ticker, desc in SEMICONDUCTOR_UNIVERSE.items():
        if ticker in returns.columns:
            print(f"  {ticker:6s} - {desc}")
    
    print("\n" + "-"*70)
    print("RETURN STATISTICS (Daily):")
    print("-"*70)
    
    stats = pd.DataFrame({
        'Mean (%)': expected_returns * 100,
        'Std (%)': returns.std() * 100,
        'Min (%)': returns.min() * 100,
        'Max (%)': returns.max() * 100,
        'Sharpe (ann.)': (expected_returns * 252) / (returns.std() * np.sqrt(252))
    })
    print(stats.round(4).to_string())
    
    print("\n" + "-"*70)
    print("ANNUALIZED STATISTICS:")
    print("-"*70)
    print(f"Mean Return Range: {(expected_returns.min() * 252 * 100):.2f}% to {(expected_returns.max() * 252 * 100):.2f}%")
    print(f"Volatility Range: {(returns.std().min() * np.sqrt(252) * 100):.2f}% to {(returns.std().max() * np.sqrt(252) * 100):.2f}%")
    
    print("\n" + "-"*70)
    print("CORRELATION MATRIX:")
    print("-"*70)
    corr_matrix = returns.corr()
    print(f"Mean Pairwise Correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"Min Pairwise Correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")
    print(f"Max Pairwise Correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")


def main():
    """
    Main execution function for data preparation pipeline.
    """
    print("="*70)
    print("SEMICONDUCTOR EQUITY DATA PREPARATION")
    print("For Deterministic Quadratic Portfolio Optimization")
    print("="*70)
    
    # Get ticker list from fixed universe
    tickers = list(SEMICONDUCTOR_UNIVERSE.keys())
    
    # Step 1: Fetch price data
    print("\n[1/5] Fetching adjusted close prices...")
    prices = fetch_price_data(tickers, START_DATE, END_DATE)
    print(f"Downloaded data shape: {prices.shape}")
    
    # Check for any tickers that failed to download
    missing_tickers = set(tickers) - set(prices.columns)
    if missing_tickers:
        print(f"Warning: Could not fetch data for: {missing_tickers}")
    
    # Step 2: Compute log returns
    print("\n[2/5] Computing log returns...")
    returns = compute_log_returns(prices)
    print(f"Returns shape (before cleaning): {returns.shape}")
    
    # Step 3: Clean data (remove missing values only)
    print("\n[3/5] Cleaning data...")
    clean_returns = clean_data(returns)
    
    # Also clean prices to match returns dates
    clean_prices = prices.loc[clean_returns.index]
    
    # Step 4: Compute expected returns and covariance
    print("\n[4/5] Computing expected returns and covariance matrix...")
    expected_returns = compute_expected_returns(clean_returns)
    cov_matrix = compute_covariance_matrix(clean_returns, ridge_lambda=RIDGE_LAMBDA)
    
    # Verify positive definiteness
    verify_positive_definiteness(cov_matrix)
    
    # Step 5: Save all data
    print("\n[5/5] Saving data to disk...")
    files = save_data(clean_prices, clean_returns, expected_returns, cov_matrix, OUTPUT_DIR)
    
    print("\nFiles saved:")
    for name, path in files.items():
        print(f"  {name}: {os.path.basename(path)}")
    
    # Print summary
    print_summary(clean_prices, clean_returns, expected_returns, cov_matrix)
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    
    return {
        'prices': clean_prices,
        'returns': clean_returns,
        'expected_returns': expected_returns,
        'covariance_matrix': cov_matrix,
        'files': files
    }


if __name__ == '__main__':
    data = main()


