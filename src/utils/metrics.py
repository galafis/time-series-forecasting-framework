"""
Evaluation Metrics for Time Series Forecasting

This module provides various metrics for evaluating forecast accuracy.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Dictionary with various metrics
    """
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    smape = symmetric_mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # Additional metrics
    mase = mean_absolute_scaled_error(actual, predicted)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'mase': mase
    }


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value
    """
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def symmetric_mean_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        sMAPE value
    """
    return np.mean(
        2.0 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))
    ) * 100


def mean_absolute_scaled_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        seasonality: Seasonal period for scaling
        
    Returns:
        MASE value
    """
    # Calculate MAE of forecast
    mae_forecast = np.mean(np.abs(actual - predicted))
    
    # Calculate MAE of naive forecast
    naive_forecast = actual[:-seasonality]
    naive_actual = actual[seasonality:]
    mae_naive = np.mean(np.abs(naive_actual - naive_forecast))
    
    # Avoid division by zero
    if mae_naive == 0:
        return np.inf
    
    return mae_forecast / mae_naive


def test_stationarity(data: np.ndarray) -> Dict[str, any]:
    """
    Test stationarity using Augmented Dickey-Fuller test.
    
    Args:
        data: Time series data
        
    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    # ADF test
    adf_result = adfuller(data)
    
    # KPSS test
    kpss_result = kpss(data)
    
    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_critical_values': kpss_result[3]
    }


if __name__ == "__main__":
    # Example usage
    actual = np.array([100, 105, 110, 108, 112, 115, 120])
    predicted = np.array([98, 107, 109, 110, 111, 116, 119])
    
    metrics = calculate_metrics(actual, predicted)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
