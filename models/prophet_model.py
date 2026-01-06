"""
Prophet Model for Stock Price Prediction (Time Series Forecasting)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None


class ProphetModel:
    """Facebook Prophet for time series forecasting"""
    
    def __init__(self):
        self.is_available = PROPHET_AVAILABLE
        if not PROPHET_AVAILABLE:
            self.is_trained = False
            return
        
        self.model = None
        self.is_trained = False
        self.last_values = None
    
    def prepare_data(self, df, feature_columns=None):
        """Prepare data for Prophet (needs ds and y columns)"""
        if not PROPHET_AVAILABLE:
            return None, None
        
        df_clean = df.dropna()
        
        if df_clean.empty or len(df_clean) < 30:
            return None, None
        
        # Prophet requires 'ds' (date) and 'y' (target) columns
        if 'date' in df_clean.columns:
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df_clean['date']),
                'y': df_clean['close'].values
            })
        elif isinstance(df_clean.index, pd.DatetimeIndex):
            prophet_df = pd.DataFrame({
                'ds': df_clean.index,
                'y': df_clean['close'].values
            })
        elif df_clean.index.name == 'date':
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df_clean.index),
                'y': df_clean['close'].values
            })
        else:
            # Create date index if not available - use appropriate frequency
            # Try to infer frequency from time frame
            freq = 'D'  # Default to daily
            if len(df_clean) > 0:
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(df_clean), freq=freq),
                    'y': df_clean['close'].values
                })
            else:
                return None, None
        
        return prophet_df, df_clean['close'].values
    
    def train(self, prophet_df, y=None):
        """Train the Prophet model"""
        if not PROPHET_AVAILABLE:
            return False
        
        if prophet_df is None or len(prophet_df) < 30:
            return False
        
        try:
            # Create Prophet model with better parameters
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Fit the model
            self.model.fit(prophet_df)
            
            self.is_trained = True
            self.last_values = prophet_df['y'].values if y is None else y
            return True
        except Exception as e:
            print(f"Error training Prophet: {str(e)}")
            return False
    
    def predict(self, df):
        """Make predictions using Prophet"""
        if not PROPHET_AVAILABLE or not self.is_trained:
            return None
        
        try:
            # Create future dataframe for predictions
            periods = len(df) - 1  # Predict for all periods except first
            future = self.model.make_future_dataframe(periods=periods, freq='D')
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Extract predictions (skip first period to align with target)
            predictions = forecast['yhat'].values[1:len(df)]
            
            # Ensure predictions match data length
            if len(predictions) > len(df) - 1:
                predictions = predictions[:len(df) - 1]
            elif len(predictions) < len(df) - 1:
                # Pad with last prediction
                last_pred = predictions[-1] if len(predictions) > 0 else self.last_values[-1] if self.last_values is not None else 0
                predictions = np.append(predictions, [last_pred] * (len(df) - 1 - len(predictions)))
            
            return predictions
        except Exception as e:
            print(f"Error predicting with Prophet: {str(e)}")
            # Fallback: return shifted actual values
            if self.last_values is not None and len(self.last_values) > 1:
                return self.last_values[1:]
            return None

