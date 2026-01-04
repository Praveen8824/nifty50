"""
ARIMA Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model for time series forecasting"""
    
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.is_trained = False
        self.last_values = None
    
    def check_stationarity(self, series):
        """Check if series is stationary"""
        try:
            result = adfuller(series.dropna())
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    def find_best_order(self, series, max_p=5, max_d=2, max_q=5):
        """Find best ARIMA order using AIC"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        series_clean = series.dropna()
        if len(series_clean) < 20:
            return best_order
        
        for p in range(1, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(1, max_q + 1):
                    try:
                        model = ARIMA(series_clean, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def train(self, series):
        """Train ARIMA model"""
        if series is None or len(series) < 10:
            return False
        
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return False
            
            # Auto-find best order if needed
            if self.order == (5, 1, 0):
                self.order = self.find_best_order(series_clean)
            
            self.model = ARIMA(series_clean, order=self.order)
            self.model = self.model.fit()
            self.last_values = series_clean.values[-10:]
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training ARIMA: {str(e)}")
            return False
    
    def predict(self, n_periods=1):
        """Make predictions"""
        if not self.is_trained:
            return None
        
        try:
            forecast = self.model.forecast(steps=n_periods)
            return forecast.values if hasattr(forecast, 'values') else [forecast]
        except Exception as e:
            print(f"Error predicting with ARIMA: {str(e)}")
            return None
    
    def predict_series(self, series):
        """Predict for a series of values using fitted values"""
        if not self.is_trained:
            return None
        
        try:
            # Use fitted values from the model for historical predictions
            # For future predictions, use forecast
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return None
            
            # Get fitted values (in-sample predictions)
            fitted_values = self.model.fittedvalues
            
            # Convert to numpy array if it's a Series
            if hasattr(fitted_values, 'values'):
                fitted_values = fitted_values.values
            elif hasattr(fitted_values, 'iloc'):
                fitted_values = fitted_values.values
            
            # Align with input series length
            if len(fitted_values) > len(series_clean):
                fitted_values = fitted_values[-len(series_clean):]
            elif len(fitted_values) < len(series_clean):
                # Pad with last fitted value
                if len(fitted_values) > 0:
                    padding = np.full(len(series_clean) - len(fitted_values), fitted_values[-1])
                    fitted_values = np.concatenate([padding, fitted_values])
                else:
                    return None
            
            return fitted_values
        except Exception as e:
            print(f"Error in ARIMA series prediction: {str(e)}")
            return None

