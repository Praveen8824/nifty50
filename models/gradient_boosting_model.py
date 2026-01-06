"""
Gradient Boosting Regressor Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingModel:
    """Gradient Boosting Regressor for stock price prediction"""
    
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.is_trained = False
        self.feature_columns = None
    
    def prepare_data(self, df, feature_columns):
        """Prepare data for training"""
        df_clean = df.dropna()
        
        if df_clean.empty or len(df_clean) < 10:
            return None, None
        
        # Get available features
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        if len(available_features) == 0:
            return None, None
        
        # Prepare features and target
        X = df_clean[available_features].values
        y = df_clean['close'].values
        
        # Shift target by 1 to predict next period
        if len(X) > 1:
            X = X[:-1]
            y = y[1:]
        
        self.feature_columns = available_features
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        if X is None or len(X) == 0:
            return False
        
        try:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
                loss='squared_error',
                verbose=0
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Gradient Boosting: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or X is None:
            return None
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error predicting with Gradient Boosting: {str(e)}")
            return None

