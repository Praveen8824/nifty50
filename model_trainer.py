"""
Model Training and Ensemble Module with Enhanced Training
"""
import pandas as pd
import numpy as np
from models.random_forest_model import RandomForestModel
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.svr_model import SVRModel
from models.lstm_model import LSTMModel
from models.lightgbm_model import LightGBMModel
from models.prophet_model import ProphetModel
from models.gradient_boosting_model import GradientBoostingModel
from technical_indicators import TechnicalIndicators
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class to train and manage all prediction models"""
    
    def __init__(self):
        self.rf_model = RandomForestModel()
        self.arima_model = ARIMAModel()
        self.xgb_model = XGBoostModel()
        self.svr_model = SVRModel()
        self.lstm_model = LSTMModel(lookback=60, units=128, dropout=0.3, use_features=True)
        self.lightgbm_model = LightGBMModel(n_estimators=200, learning_rate=0.05, max_depth=7)
        self.prophet_model = ProphetModel()
        self.gb_model = GradientBoostingModel(n_estimators=200, learning_rate=0.05, max_depth=5)
        self.feature_columns = None
        self.is_trained = False
        self.training_data_points = 0  # Track number of data points used for training
    
    def prepare_data(self, df):
        """Prepare data with technical indicators and improved preprocessing"""
        # Add technical indicators
        df_with_indicators = TechnicalIndicators.add_indicators(df)
        
        # Get feature columns
        feature_cols = TechnicalIndicators.get_feature_columns()
        # Filter to only columns that exist in dataframe
        self.feature_columns = [col for col in feature_cols if col in df_with_indicators.columns]
        
        # Remove infinite values and replace with NaN
        df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with forward fill then backward fill
        df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows that still have NaN (should be minimal)
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
    
    def train_all_models(self, df):
        """Train all models"""
        print("Preparing data...")
        df_prepared = self.prepare_data(df)
        
        if df_prepared.empty or len(df_prepared) < 50:
            print("Insufficient data for training")
            return False
        
        # Store training data points count
        self.training_data_points = len(df_prepared)
        
        print("Training Random Forest...")
        X_rf, y_rf, _, _ = self.rf_model.prepare_data(
            df_prepared, self.feature_columns
        )
        if X_rf is not None and len(X_rf) > 10:
            self.rf_model.train(X_rf, y_rf)
        
        print("Training XGBoost...")
        X_xgb, y_xgb = self.xgb_model.prepare_data(
            df_prepared, self.feature_columns
        )
        if X_xgb is not None and len(X_xgb) > 10:
            self.xgb_model.train(X_xgb, y_xgb)
        
        print("Training SVR...")
        X_svr, y_svr = self.svr_model.prepare_data(
            df_prepared, self.feature_columns
        )
        if X_svr is not None and len(X_svr) > 10:
            self.svr_model.train(X_svr, y_svr)
        
        print("Training ARIMA...")
        if 'close' in df_prepared.columns and len(df_prepared) > 30:
            self.arima_model.train(df_prepared['close'])
        
        print("Training LSTM...")
        if hasattr(self.lstm_model, 'is_available') and self.lstm_model.is_available:
            X_lstm, y_lstm = self.lstm_model.prepare_data(
                df_prepared, self.feature_columns
            )
            if X_lstm is not None and len(X_lstm) > 10:
                # Enhanced training with more epochs and better parameters
                self.lstm_model.train(X_lstm, y_lstm, epochs=150, batch_size=32, verbose=0, validation_split=0.2)
        else:
            print("LSTM not available (TensorFlow not installed)")
        
        print("Training LightGBM...")
        if hasattr(self.lightgbm_model, 'is_available') and self.lightgbm_model.is_available:
            X_lgb, y_lgb = self.lightgbm_model.prepare_data(
                df_prepared, self.feature_columns
            )
            if X_lgb is not None and len(X_lgb) > 10:
                self.lightgbm_model.train(X_lgb, y_lgb)
        
        print("Training Prophet...")
        if hasattr(self.prophet_model, 'is_available') and self.prophet_model.is_available:
            prophet_df, y_prophet = self.prophet_model.prepare_data(df_prepared)
            if prophet_df is not None and len(prophet_df) > 30:
                self.prophet_model.train(prophet_df, y_prophet)
        
        print("Training Gradient Boosting...")
        X_gb, y_gb = self.gb_model.prepare_data(df_prepared, self.feature_columns)
        if X_gb is not None and len(X_gb) > 10:
            self.gb_model.train(X_gb, y_gb)
        
        self.is_trained = True
        print("All models trained successfully!")
        return True
    
    def predict_all_models(self, df):
        """Get predictions from all models"""
        if not self.is_trained:
            return {}
        
        df_prepared = self.prepare_data(df)
        
        predictions = {}
        
        # Random Forest
        try:
            X_rf, _, _, _ = self.rf_model.prepare_data(df_prepared, self.feature_columns)
            if X_rf is not None and self.rf_model.is_trained:
                pred_rf = self.rf_model.predict(X_rf)
                if pred_rf is not None:
                    predictions['Random Forest'] = pred_rf
        except:
            pass
        
        # XGBoost
        try:
            X_xgb, _ = self.xgb_model.prepare_data(df_prepared, self.feature_columns)
            if X_xgb is not None and self.xgb_model.is_trained:
                pred_xgb = self.xgb_model.predict(X_xgb)
                if pred_xgb is not None:
                    predictions['XGBoost'] = pred_xgb
        except:
            pass
        
        # SVR
        try:
            X_svr, _ = self.svr_model.prepare_data(df_prepared, self.feature_columns)
            if X_svr is not None and self.svr_model.is_trained:
                pred_svr = self.svr_model.predict(X_svr)
                if pred_svr is not None:
                    predictions['SVR'] = pred_svr
        except:
            pass
        
        # ARIMA
        try:
            if 'close' in df_prepared.columns and self.arima_model.is_trained:
                # ARIMA needs to predict on the same length as training data
                # We'll use fitted values for historical predictions
                pred_arima = self.arima_model.predict_series(df_prepared['close'])
                if pred_arima is not None:
                    # Align with other predictions (shift by 1 to match target)
                    if len(pred_arima) > 1:
                        pred_arima = pred_arima[1:]  # Shift to align with target
                    predictions['ARIMA'] = pred_arima
        except Exception as e:
            print(f"ARIMA prediction error: {str(e)}")
            pass
        
        # LSTM
        try:
            if hasattr(self.lstm_model, 'is_available') and self.lstm_model.is_available:
                X_lstm, _ = self.lstm_model.prepare_data(df_prepared, self.feature_columns)
                if X_lstm is not None and self.lstm_model.is_trained:
                    pred_lstm = self.lstm_model.predict(X_lstm)
                    if pred_lstm is not None:
                        predictions['LSTM'] = pred_lstm
        except:
            pass
        
        # LightGBM
        try:
            if hasattr(self.lightgbm_model, 'is_available') and self.lightgbm_model.is_available:
                X_lgb, _ = self.lightgbm_model.prepare_data(df_prepared, self.feature_columns)
                if X_lgb is not None and self.lightgbm_model.is_trained:
                    pred_lgb = self.lightgbm_model.predict(X_lgb)
                    if pred_lgb is not None:
                        predictions['LightGBM'] = pred_lgb
        except:
            pass
        
        # Prophet
        try:
            if hasattr(self.prophet_model, 'is_available') and self.prophet_model.is_available:
                if self.prophet_model.is_trained:
                    pred_prophet = self.prophet_model.predict(df_prepared)
                    if pred_prophet is not None:
                        predictions['Prophet'] = pred_prophet
        except:
            pass
        
        # Gradient Boosting
        try:
            X_gb, _ = self.gb_model.prepare_data(df_prepared, self.feature_columns)
            if X_gb is not None and self.gb_model.is_trained:
                pred_gb = self.gb_model.predict(X_gb)
                if pred_gb is not None:
                    predictions['Gradient Boosting'] = pred_gb
        except:
            pass
        
        return predictions
    
    def ensemble_predict(self, predictions_dict, metrics_dict=None):
        """Create weighted ensemble prediction based on model performance"""
        if not predictions_dict:
            return None
        
        # Get all prediction arrays
        pred_arrays = []
        model_names = []
        weights = []
        
        for model_name, pred in predictions_dict.items():
            if pred is not None:
                pred_arrays.append(pred)
                model_names.append(model_name)
        
        if not pred_arrays:
            return None
        
        # Align lengths (take minimum length)
        min_len = min(len(arr) for arr in pred_arrays)
        aligned_preds = [arr[:min_len] for arr in pred_arrays]
        
        # Calculate weights based on R2 scores if metrics available
        if metrics_dict:
            weights = []
            for model_name in model_names:
                if model_name in metrics_dict:
                    r2 = metrics_dict[model_name].get('R2_Score', 0.0)
                    # Better handling of negative R2 scores
                    # Use exponential transformation to emphasize positive R2
                    if r2 > 0:
                        weight = r2 ** 2  # Square to emphasize good models
                    elif r2 > -1:
                        weight = 0.1 * (1 + r2)  # Scale negative R2 to small positive
                    else:
                        weight = 0.01  # Very small weight for very negative R2
                    
                    # Also consider other metrics
                    rmse = metrics_dict[model_name].get('RMSE', float('inf'))
                    mae = metrics_dict[model_name].get('MAE', float('inf'))
                    
                    # Penalize high error models
                    if rmse != float('inf') and mae != float('inf'):
                        error_penalty = 1.0 / (1.0 + rmse + mae)
                        weight = weight * error_penalty
                    
                    weights.append(max(0.01, weight))
                else:
                    weights.append(0.01)  # Small weight for models without metrics
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)
        else:
            # Equal weights if no metrics
            weights = [1.0 / len(model_names)] * len(model_names)
        
        # Weighted average
        ensemble_pred = np.zeros(min_len)
        for i, pred in enumerate(aligned_preds):
            ensemble_pred += pred * weights[i]
        
        return ensemble_pred

