"""
Support Vector Regressor (SVR) Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SVRModel:
    """Support Vector Regressor for stock price prediction"""
    
    def __init__(self, kernel='rbf', C=100.0, epsilon=0.01, gamma='scale'):
        # Increased C for better fit, reduced epsilon for tighter margin
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, max_iter=1000)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_data(self, df, feature_columns, target_column='close'):
        """Prepare data for training"""
        df_clean = df.dropna()
        
        if df_clean.empty:
            return None, None
        
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        # Create target (next period's close price)
        if len(y) > 1:
            y = y[1:]
            X = X[:-1]
        
        if len(X) == 0:
            return None, None
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        if X is None or len(X) == 0:
            return False
        
        try:
            # Scale features and target separately
            X_scaled = self.x_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
            
            self.model.fit(X_scaled, y_scaled)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training SVR: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or X is None:
            return None
        
        try:
            X_scaled = self.x_scaler.transform(X)
            predictions_scaled = self.model.predict(X_scaled)
            # Inverse transform predictions
            predictions = self.y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
            return predictions
        except Exception as e:
            print(f"Error predicting with SVR: {str(e)}")
            return None

