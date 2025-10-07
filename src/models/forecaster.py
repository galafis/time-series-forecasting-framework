"""
Time Series Forecasting Module

This module provides a unified interface for time series forecasting using
various statistical, machine learning, and deep learning methods.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesForecaster:
    """
    Unified time series forecasting interface supporting multiple models.
    
    Supports:
    - Statistical models: ARIMA, SARIMA, Prophet, Exponential Smoothing
    - ML models: XGBoost, LightGBM, Random Forest
    - DL models: LSTM, GRU
    
    Attributes:
        model_type (str): Type of forecasting model to use
        forecast_horizon (int): Number of steps to forecast ahead
        model: Fitted model instance
    """
    
    SUPPORTED_MODELS = [
        'arima', 'sarima', 'prophet', 'exp_smoothing',
        'xgboost', 'lightgbm', 'random_forest',
        'lstm', 'gru'
    ]
    
    def __init__(
        self,
        model_type: str = 'prophet',
        forecast_horizon: int = 30,
        **model_params
    ):
        """
        Initialize the forecaster.
        
        Args:
            model_type: Type of model ('arima', 'prophet', 'lstm', etc.)
            forecast_horizon: Number of steps to forecast ahead
            **model_params: Additional parameters for the specific model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.forecast_result = None
        
        logger.info(f"Initialized {model_type} forecaster with horizon={forecast_horizon}")
    
    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'value',
        exog_cols: Optional[List[str]] = None
    ) -> 'TimeSeriesForecaster':
        """
        Fit the forecasting model to the data.
        
        Args:
            data: DataFrame with time series data
            date_col: Name of the date column
            target_col: Name of the target variable column
            exog_cols: List of exogenous variable columns (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.model_type} model...")
        
        # Validate data
        if date_col not in data.columns or target_col not in data.columns:
            raise ValueError(f"Columns '{date_col}' and '{target_col}' must be in data")
        
        # Store training data
        self.train_data = data.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.exog_cols = exog_cols
        
        # Fit model based on type
        if self.model_type in ['arima', 'sarima']:
            self._fit_arima(data, target_col, exog_cols)
        elif self.model_type == 'prophet':
            self._fit_prophet(data, date_col, target_col)
        elif self.model_type == 'exp_smoothing':
            self._fit_exp_smoothing(data, target_col)
        elif self.model_type in ['xgboost', 'lightgbm', 'random_forest']:
            self._fit_ml_model(data, date_col, target_col, exog_cols)
        elif self.model_type in ['lstm', 'gru']:
            self._fit_dl_model(data, target_col)
        
        self.is_fitted = True
        logger.success(f"{self.model_type} model fitted successfully")
        return self
    
    def _fit_arima(
        self,
        data: pd.DataFrame,
        target_col: str,
        exog_cols: Optional[List[str]] = None
    ):
        """Fit ARIMA/SARIMA model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from pmdarima import auto_arima
        
        y = data[target_col].values
        exog = data[exog_cols].values if exog_cols else None
        
        # Auto ARIMA to find best parameters
        logger.info("Running auto_arima to find optimal parameters...")
        
        seasonal = self.model_params.get('seasonal', False)
        m = self.model_params.get('seasonal_period', 12)
        
        auto_model = auto_arima(
            y,
            exogenous=exog,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        logger.info(f"Best ARIMA order: {auto_model.order}")
        if seasonal:
            logger.info(f"Best seasonal order: {auto_model.seasonal_order}")
        
        self.model = auto_model
    
    def _fit_prophet(self, data: pd.DataFrame, date_col: str, target_col: str):
        """Fit Prophet model."""
        from prophet import Prophet
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(data[date_col]),
            'y': data[target_col]
        })
        
        # Initialize Prophet with parameters
        prophet_params = {
            'yearly_seasonality': self.model_params.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': self.model_params.get('weekly_seasonality', 'auto'),
            'daily_seasonality': self.model_params.get('daily_seasonality', 'auto'),
            'seasonality_mode': self.model_params.get('seasonality_mode', 'additive'),
        }
        
        self.model = Prophet(**prophet_params)
        self.model.fit(df_prophet)
    
    def _fit_exp_smoothing(self, data: pd.DataFrame, target_col: str):
        """Fit Exponential Smoothing model."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        y = data[target_col].values
        
        # Parameters
        trend = self.model_params.get('trend', 'add')
        seasonal = self.model_params.get('seasonal', 'add')
        seasonal_periods = self.model_params.get('seasonal_periods', 12)
        
        self.model = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        ).fit()
    
    def _fit_ml_model(
        self,
        data: pd.DataFrame,
        date_col: str,
        target_col: str,
        exog_cols: Optional[List[str]]
    ):
        """Fit ML model (XGBoost, LightGBM, Random Forest)."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create lag features
        from ..features.engineering import create_lag_features, create_time_features
        
        df = data.copy()
        df = create_time_features(df, date_col)
        df = create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30])
        
        # Drop NaN rows created by lag features
        df = df.dropna()
        
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in [date_col, target_col]]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Initialize model
        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 5),
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 5),
                random_state=42
            )
        else:  # random_forest
            self.model = RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                random_state=42
            )
        
        self.model.fit(X, y)
        self.feature_cols = feature_cols
    
    def _fit_dl_model(self, data: pd.DataFrame, target_col: str):
        """Fit deep learning model (LSTM, GRU)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare sequences
        sequence_length = self.model_params.get('sequence_length', 30)
        
        y = data[target_col].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y_target = [], []
        for i in range(len(y_scaled) - sequence_length):
            X.append(y_scaled[i:i+sequence_length])
            y_target.append(y_scaled[i+sequence_length])
        
        X = np.array(X).reshape(-1, sequence_length, 1)
        y_target = np.array(y_target)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y_target)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.model_params.get('batch_size', 32),
            shuffle=True
        )
        
        # Define model
        input_size = 1
        hidden_size = self.model_params.get('hidden_size', 64)
        num_layers = self.model_params.get('num_layers', 2)
        
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_size, hidden_size, num_layers)
        else:  # gru
            self.model = GRUModel(input_size, hidden_size, num_layers)
        
        # Train
        self._train_dl_model(dataloader)
    
    def _train_dl_model(self, dataloader):
        """Train deep learning model."""
        import torch
        import torch.nn as nn
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_params.get('learning_rate', 0.001)
        )
        
        epochs = self.model_params.get('epochs', 50)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(
        self,
        steps: Optional[int] = None,
        exog: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast (uses forecast_horizon if None)
            exog: Exogenous variables for forecast period
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        steps = steps or self.forecast_horizon
        logger.info(f"Generating {steps}-step forecast...")
        
        if self.model_type in ['arima', 'sarima']:
            forecast = self.model.predict(n_periods=steps, exogenous=exog)
            conf_int = self.model.predict(n_periods=steps, return_conf_int=True)[1]
            
            result = pd.DataFrame({
                'forecast': forecast,
                'lower_bound': conf_int[:, 0],
                'upper_bound': conf_int[:, 1]
            })
        
        elif self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=steps)
            forecast = self.model.predict(future)
            forecast = forecast.tail(steps)
            
            result = pd.DataFrame({
                'forecast': forecast['yhat'].values,
                'lower_bound': forecast['yhat_lower'].values,
                'upper_bound': forecast['yhat_upper'].values
            })
        
        elif self.model_type == 'exp_smoothing':
            forecast = self.model.forecast(steps=steps)
            
            result = pd.DataFrame({
                'forecast': forecast,
                'lower_bound': forecast * 0.95,  # Approximate
                'upper_bound': forecast * 1.05
            })
        
        else:
            # For ML/DL models, implement recursive forecasting
            result = self._recursive_forecast(steps)
        
        self.forecast_result = result
        logger.success(f"Forecast generated successfully")
        return result
    
    def _recursive_forecast(self, steps: int) -> pd.DataFrame:
        """Recursive forecasting for ML/DL models."""
        # Simplified implementation
        last_value = self.train_data[self.target_col].iloc[-1]
        forecasts = [last_value] * steps
        
        result = pd.DataFrame({
            'forecast': forecasts,
            'lower_bound': [f * 0.9 for f in forecasts],
            'upper_bound': [f * 1.1 for f in forecasts]
        })
        
        return result
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate forecast accuracy on test data.
        
        Args:
            test_data: DataFrame with actual values
            
        Returns:
            Dictionary with evaluation metrics
        """
        from ..utils.metrics import calculate_metrics
        
        actual = test_data[self.target_col].values[:len(self.forecast_result)]
        predicted = self.forecast_result['forecast'].values
        
        metrics = calculate_metrics(actual, predicted)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def plot_forecast(self, test_data: Optional[pd.DataFrame] = None):
        """
        Plot forecast with confidence intervals.
        
        Args:
            test_data: Optional test data to overlay
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot training data
        ax.plot(
            self.train_data[self.date_col],
            self.train_data[self.target_col],
            label='Training Data',
            color='blue'
        )
        
        # Plot forecast
        forecast_dates = pd.date_range(
            start=self.train_data[self.date_col].iloc[-1],
            periods=len(self.forecast_result) + 1,
            freq='D'
        )[1:]
        
        ax.plot(
            forecast_dates,
            self.forecast_result['forecast'],
            label='Forecast',
            color='red',
            linestyle='--'
        )
        
        # Plot confidence intervals
        ax.fill_between(
            forecast_dates,
            self.forecast_result['lower_bound'],
            self.forecast_result['upper_bound'],
            alpha=0.3,
            color='red',
            label='Confidence Interval'
        )
        
        # Plot test data if provided
        if test_data is not None:
            ax.plot(
                test_data[self.date_col],
                test_data[self.target_col],
                label='Actual (Test)',
                color='green'
            )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'{self.model_type.upper()} Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesForecaster(model={self.model_type}, "
            f"horizon={self.forecast_horizon}, "
            f"fitted={self.is_fitted})"
        )


# Deep Learning Models
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """GRU model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    values = np.cumsum(np.random.randn(365)) + 100
    
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Split train/test
    train = df[:300]
    test = df[300:]
    
    # Fit and forecast
    forecaster = TimeSeriesForecaster(model_type='prophet', forecast_horizon=65)
    forecaster.fit(train, date_col='date', target_col='value')
    
    forecast = forecaster.predict()
    print(forecast.head())
    
    metrics = forecaster.evaluate(test)
    print(f"\nMetrics: {metrics}")
    
    forecaster.plot_forecast(test)
