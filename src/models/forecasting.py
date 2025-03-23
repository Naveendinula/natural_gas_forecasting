"""
Forecasting Models for Natural Gas Data

This module contains various time series forecasting models for natural gas data.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure logging
logging.basicConfig(
    filename='forecasting.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesModel:
    """Base class for time series models."""
    
    def __init__(self):
        """Initialize the base time series model."""
        self.model = None
        self.scaler = StandardScaler()
        
    def train_test_split(self, data, test_size=0.2):
        """Split the data into training and testing sets.
        
        Args:
            data (pd.DataFrame): Input dataframe with 'date' and 'value' columns.
            test_size (float, optional): Proportion of data to use for testing.
            
        Returns:
            tuple: (train_data, test_data)
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
            
        # Sort by date
        data = data.sort_values('date')
        
        # Determine split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split data
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def evaluate_model(self, true_values, predictions):
        """Evaluate the model performance.
        
        Args:
            true_values (array-like): Actual values.
            predictions (array-like): Predicted values.
            
        Returns:
            dict: Dictionary with evaluation metrics.
        """
        # Calculate metrics
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Model evaluation: {metrics}")
        
        return metrics
        
    def preprocess_data(self, data):
        """Preprocess data for modeling.
        
        Args:
            data (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Ensure data is sorted by date
        if 'date' in processed_data.columns:
            processed_data = processed_data.sort_values('date')
            
        return processed_data
        
    def fit(self, data):
        """Fit the model to the data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def predict(self, periods=12):
        """Generate predictions. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

class ARIMAModel(TimeSeriesModel):
    """ARIMA time series model."""
    
    def __init__(self, p=1, d=1, q=0):
        """Initialize the ARIMA model.
        
        Args:
            p (int, optional): AR order.
            d (int, optional): Differencing order.
            q (int, optional): MA order.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model = None
        self.history = None
        
    def fit(self, data, value_column='value'):
        """Fit the ARIMA model to the data.
        
        Args:
            data (pd.DataFrame): Input dataframe.
            value_column (str, optional): Column with values to forecast.
            
        Returns:
            ARIMAModel: Fitted model.
        """
        try:
            logger.info(f"Fitting ARIMA({self.p},{self.d},{self.q}) model")
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Store the time series values
            self.history = processed_data[value_column].values
            
            # Fit ARIMA model
            self.model = ARIMA(
                self.history, 
                order=(self.p, self.d, self.q)
            )
            
            self.fitted_model = self.model.fit()
            
            logger.info("ARIMA model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
            
    def predict(self, periods=12):
        """Generate forecasts with the fitted ARIMA model.
        
        Args:
            periods (int, optional): Number of periods to forecast.
            
        Returns:
            np.array: Predicted values.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet")
            
        try:
            logger.info(f"Generating {periods}-period forecast with ARIMA model")
            
            # Make forecast
            forecast = self.fitted_model.forecast(steps=periods)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise
            
    def get_forecast_df(self, start_date, periods=12, freq='MS'):
        """Get forecast as a DataFrame with dates.
        
        Args:
            start_date (str or datetime): First date of the forecast.
            periods (int, optional): Number of periods to forecast.
            freq (str, optional): Frequency of dates (MS=month start).
            
        Returns:
            pd.DataFrame: DataFrame with forecast and dates.
        """
        # Generate forecast values
        forecast_values = self.predict(periods=periods)
        
        # Generate date range
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast': forecast_values
        })
        
        return forecast_df

class SARIMAXModel(TimeSeriesModel):
    """SARIMAX time series model with exogenous variables."""
    
    def __init__(self, p=1, d=1, q=0, P=1, D=1, Q=0, s=12):
        """Initialize the SARIMAX model.
        
        Args:
            p (int, optional): AR order.
            d (int, optional): Differencing order.
            q (int, optional): MA order.
            P (int, optional): Seasonal AR order.
            D (int, optional): Seasonal differencing order.
            Q (int, optional): Seasonal MA order.
            s (int, optional): Seasonal period.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.model = None
        self.history = None
        self.exog_columns = []
        
    def fit(self, data, value_column='value', exog_columns=None):
        """Fit the SARIMAX model to the data.
        
        Args:
            data (pd.DataFrame): Input dataframe.
            value_column (str, optional): Column with values to forecast.
            exog_columns (list, optional): List of exogenous variable columns.
            
        Returns:
            SARIMAXModel: Fitted model.
        """
        try:
            order = (self.p, self.d, self.q)
            seasonal_order = (self.P, self.D, self.Q, self.s)
            
            logger.info(f"Fitting SARIMAX{order}x{seasonal_order} model")
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Store the time series values
            self.history = processed_data[value_column].values
            
            # Store exogenous variables if provided
            self.exog_columns = exog_columns if exog_columns else []
            exog = processed_data[self.exog_columns].values if self.exog_columns else None
            
            # Fit SARIMAX model
            self.model = SARIMAX(
                self.history,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order
            )
            
            self.fitted_model = self.model.fit(disp=False)
            
            logger.info("SARIMAX model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {e}")
            raise
            
    def predict(self, periods=12, exog_future=None):
        """Generate forecasts with the fitted SARIMAX model.
        
        Args:
            periods (int, optional): Number of periods to forecast.
            exog_future (np.array, optional): Future values of exogenous variables.
            
        Returns:
            np.array: Predicted values.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet")
            
        try:
            logger.info(f"Generating {periods}-period forecast with SARIMAX model")
            
            # Make forecast
            forecast = self.fitted_model.forecast(steps=periods, exog=exog_future)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating SARIMAX forecast: {e}")
            raise
            
    def get_forecast_df(self, start_date, periods=12, freq='MS', exog_future=None):
        """Get forecast as a DataFrame with dates.
        
        Args:
            start_date (str or datetime): First date of the forecast.
            periods (int, optional): Number of periods to forecast.
            freq (str, optional): Frequency of dates (MS=month start).
            exog_future (np.array, optional): Future values of exogenous variables.
            
        Returns:
            pd.DataFrame: DataFrame with forecast and dates.
        """
        # Generate forecast values
        forecast_values = self.predict(periods=periods, exog_future=exog_future)
        
        # Generate date range
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast': forecast_values
        })
        
        return forecast_df 