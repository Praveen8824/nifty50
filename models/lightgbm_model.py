"""
LightGBM Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class LightGBMModel:
    """LightGBM Gradient Boosting for stock price prediction"""
    
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=7):
        self.is_available = LIGHTGBM_AVAILABLE
        if not LIGHTGBM_AVAILABLE:
            self.is_trained = False
            return
        
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
        if not LIGHTGBM_AVAILABLE:
            return False
        
        if X is None or len(X) == 0:
            return False
        
        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Parameters for better accuracy
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': self.learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': self.max_depth,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
            
            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training LightGBM: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not LIGHTGBM_AVAILABLE or not self.is_trained or X is None:
            return None
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error predicting with LightGBM: {str(e)}")
            return None

