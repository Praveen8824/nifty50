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
    
    def __init__(self, order=None):
        self.order = order  # Will be auto-determined if None
        self.model = None
        self.is_trained = False
        self.last_values = None
        self.fitted_values = None
    
    def check_stationarity(self, series):
        """Check if series is stationary"""
        try:
            result = adfuller(series.dropna())
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    def find_best_order(self, series, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA order using AIC with improved search"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        series_clean = series.dropna()
        if len(series_clean) < 30:
            return (1, 1, 1)  # Default for small datasets
        
        # Limit search space for faster training
        # Try common ARIMA orders first
        common_orders = [
            (1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1),
            (2, 1, 1), (1, 1, 2), (3, 1, 2), (2, 1, 0)
        ]
        
        for order in common_orders:
            try:
                model = ARIMA(series_clean, order=order)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = order
            except:
                continue
        
        # If we have enough data, do a limited grid search
        if len(series_clean) > 50:
            for p in range(0, min(max_p + 1, 3)):
                for d in range(1, min(max_d + 1, 2)):  # Start from 1 for differencing
                    for q in range(0, min(max_q + 1, 3)):
                        if (p, d, q) in common_orders:
                            continue
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
        """Train ARIMA model with improved parameters"""
        if series is None or len(series) < 10:
            return False
        
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return False
            
            # Auto-find best order if not specified
            if self.order is None:
                self.order = self.find_best_order(series_clean)
            
            # Fit model with better parameters
            self.model = ARIMA(series_clean, order=self.order)
            try:
                self.model = self.model.fit(method='css-ml', trend='c')
            except:
                # Fallback to default fit
                self.model = self.model.fit()
            
            # Store fitted values for better predictions
            self.fitted_values = self.model.fittedvalues
            self.last_values = series_clean.values[-10:]
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training ARIMA: {str(e)}")
            # Fallback to simpler model
            try:
                self.order = (1, 1, 1)
                self.model = ARIMA(series_clean, order=self.order)
                self.model = self.model.fit()
                self.fitted_values = self.model.fittedvalues
                self.last_values = series_clean.values[-10:]
                self.is_trained = True
                return True
            except:
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
        """Predict for a series of values using fitted values with better alignment"""
        if not self.is_trained:
            return None
        
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return None
            
            # Get fitted values (in-sample predictions)
            fitted_values = self.fitted_values if self.fitted_values is not None else self.model.fittedvalues
            
            # Convert to numpy array
            if hasattr(fitted_values, 'values'):
                fitted_values = fitted_values.values
            elif isinstance(fitted_values, pd.Series):
                fitted_values = fitted_values.values
            
            # Ensure we have valid fitted values
            if len(fitted_values) == 0:
                # Fallback: use one-step ahead forecasts
                predictions = []
                for i in range(1, len(series_clean)):
                    try:
                        forecast = self.model.forecast(steps=1)
                        predictions.append(forecast[0] if hasattr(forecast, '__getitem__') else forecast)
                    except:
                        # Use last known value if forecast fails
                        predictions.append(series_clean.iloc[i-1] if i > 0 else series_clean.iloc[0])
                fitted_values = np.array(predictions)
            
            # Align with input series length - shift by 1 to match target
            target_len = len(series_clean) - 1
            
            if len(fitted_values) >= target_len:
                # Take the last target_len values
                fitted_values = fitted_values[-target_len:]
            elif len(fitted_values) < target_len:
                # Extend with forecasts
                needed = target_len - len(fitted_values)
                try:
                    forecast = self.model.forecast(steps=needed)
                    if hasattr(forecast, 'values'):
                        forecast = forecast.values
                    elif hasattr(forecast, '__len__') and len(forecast) > 0:
                        forecast = np.array(forecast) if not isinstance(forecast, np.ndarray) else forecast
                    else:
                        forecast = np.full(needed, fitted_values[-1] if len(fitted_values) > 0 else series_clean.iloc[-1])
                    
                    fitted_values = np.concatenate([fitted_values, forecast[:needed]])
                except:
                    # Pad with last fitted value
                    padding = np.full(needed, fitted_values[-1] if len(fitted_values) > 0 else series_clean.iloc[-1])
                    fitted_values = np.concatenate([fitted_values, padding])
            
            return fitted_values[:target_len]
        except Exception as e:
            print(f"Error in ARIMA series prediction: {str(e)}")
            return None

