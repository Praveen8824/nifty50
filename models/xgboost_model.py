"""
XGBoost Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class XGBoostModel:
    """XGBoost Regressor for stock price prediction"""
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            min_child_weight=10,  # Increased to reduce overfitting
            gamma=0.3,  # Increased minimum loss reduction
            reg_alpha=1.0,  # Increased L1 regularization
            reg_lambda=3.0,  # Increased L2 regularization
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=20  # Early stopping to prevent overfitting
        )
        self.scaler = StandardScaler()
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, X, y):
        """Train the model"""
        if X is None or len(X) == 0:
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training XGBoost: {str(e)}")
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
            print(f"Error predicting with XGBoost: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.is_trained:
            return self.model.feature_importances_
        return None

