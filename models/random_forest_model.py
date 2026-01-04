"""
Random Forest Regressor Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os


class RandomForestModel:
    """Random Forest Regressor for stock price prediction"""
    
    def __init__(self, n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_data(self, df, feature_columns, target_column='close'):
        """Prepare data for training"""
        # Remove NaN values
        df_clean = df.dropna()
        
        if df_clean.empty:
            return None, None, None, None
        
        # Select features
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        # Create target (next period's close price)
        if len(y) > 1:
            y = y[1:]  # Shift target
            X = X[:-1]  # Remove last row
        
        if len(X) == 0:
            return None, None, None, None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df_clean.index[:-1], df_clean.index[1:]
    
    def train(self, X, y):
        """Train the model"""
        if X is None or len(X) == 0:
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Random Forest: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained or X is None:
            return None
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            print(f"Error predicting with Random Forest: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.is_trained:
            return self.model.feature_importances_
        return None

