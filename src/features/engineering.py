"""
Feature Engineering for Time Series

This module provides functions for creating time-based features and lag features
for time series forecasting.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(
    df: pd.DataFrame,
    date_col: str = 'date',
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Create time-based features from datetime column.
    
    Features created:
    - year, month, day, day_of_week, day_of_year
    - hour, minute (if datetime has time component)
    - quarter, week_of_year
    - is_weekend, is_month_start, is_month_end
    - is_quarter_start, is_quarter_end
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of the datetime column
        drop_original: Whether to drop the original date column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Ensure datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Boolean features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    # Time features (if applicable)
    if df[date_col].dt.hour.sum() > 0:  # Has time component
        df['hour'] = df[date_col].dt.hour
        df['minute'] = df[date_col].dt.minute
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17)
        ).astype(int)
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    if drop_original:
        df = df.drop(columns=[date_col])
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = [1, 2, 3, 7, 14, 30],
    rolling_windows: Optional[List[int]] = [7, 14, 30]
) -> pd.DataFrame:
    """
    Create lag features and rolling statistics.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        lags: List of lag periods to create
        rolling_windows: List of window sizes for rolling statistics
        
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    
    # Create lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Create rolling statistics
    if rolling_windows:
        for window in rolling_windows:
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window).std()
            )
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window).min()
            )
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window).max()
            )
    
    # Create difference features
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_7'] = df[target_col].diff(7)
    
    # Create percentage change features
    df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
    df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(7)
    
    return df


def create_fourier_features(
    df: pd.DataFrame,
    date_col: str,
    period: int,
    order: int = 3
) -> pd.DataFrame:
    """
    Create Fourier features for capturing seasonality.
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of the datetime column
        period: Seasonal period (e.g., 365 for yearly, 7 for weekly)
        order: Number of Fourier terms to include
        
    Returns:
        DataFrame with added Fourier features
    """
    df = df.copy()
    
    # Ensure datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calculate time index
    t = np.arange(len(df))
    
    # Create Fourier terms
    for i in range(1, order + 1):
        df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    
    return df


def create_holiday_features(
    df: pd.DataFrame,
    date_col: str,
    country: str = 'US'
) -> pd.DataFrame:
    """
    Create holiday indicator features.
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of the datetime column
        country: Country code for holidays
        
    Returns:
        DataFrame with added holiday features
    """
    df = df.copy()
    
    try:
        import holidays
        
        # Get holidays for the country
        country_holidays = holidays.country_holidays(country)
        
        # Ensure datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create holiday indicator
        df['is_holiday'] = df[date_col].apply(
            lambda x: 1 if x in country_holidays else 0
        )
        
        # Days before/after holiday
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0
        
        for i in range(len(df)):
            date = df[date_col].iloc[i]
            
            # Find nearest holiday
            future_holidays = [h for h in country_holidays if h > date]
            past_holidays = [h for h in country_holidays if h < date]
            
            if future_holidays:
                df.loc[df.index[i], 'days_to_holiday'] = (
                    min(future_holidays) - date
                ).days
            
            if past_holidays:
                df.loc[df.index[i], 'days_from_holiday'] = (
                    date - max(past_holidays)
                ).days
        
    except ImportError:
        print("holidays package not installed. Skipping holiday features.")
    
    return df


def create_interaction_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create interaction features between specified columns.
    
    Args:
        df: DataFrame
        feature_cols: List of column names to create interactions from
        
    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()
    
    # Create pairwise interactions
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    print("Original DataFrame:")
    print(df.head())
    
    # Create time features
    df = create_time_features(df, 'date')
    print("\nAfter time features:")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    
    # Create lag features
    df = create_lag_features(df, 'value', lags=[1, 7], rolling_windows=[7])
    print("\nAfter lag features:")
    print(df.head(10))
    print(f"Columns: {df.columns.tolist()}")
    
    # Create Fourier features
    df = create_fourier_features(df, 'date', period=7, order=2)
    print("\nAfter Fourier features:")
    print(df.head())
    print(f"Total columns: {len(df.columns)}")
