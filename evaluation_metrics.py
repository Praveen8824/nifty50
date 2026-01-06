"""
Evaluation Metrics Module for Model Performance
"""
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix
)


class EvaluationMetrics:
    """Class to calculate evaluation metrics for models"""
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        if y_true is None or y_pred is None:
            return {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) == 0:
            return {}
        
        metrics = {}
        
        try:
            metrics['R2_Score'] = r2_score(y_true, y_pred)
        except:
            metrics['R2_Score'] = 0.0
        
        try:
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        except:
            metrics['RMSE'] = 0.0
        
        try:
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        except:
            metrics['MAE'] = 0.0
        
        try:
            metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        except:
            metrics['MAPE'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred):
        """
        Calculate classification metrics for direction prediction
        (Bullish/Bearish/Neutral)
        """
        if y_true is None or y_pred is None:
            return {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) < 2:
            return {}
        
        # Convert to direction based on price changes (1 for up, 0 for down)
        y_true_changes = np.diff(y_true)
        y_pred_changes = np.diff(y_pred)
        
        y_true_dir = np.where(y_true_changes > 0, 1, 0)
        y_pred_dir = np.where(y_pred_changes > 0, 1, 0)
        
        metrics = {}
        
        try:
            metrics['Accuracy'] = accuracy_score(y_true_dir, y_pred_dir)
        except:
            metrics['Accuracy'] = 0.0
        
        try:
            metrics['Precision'] = precision_score(y_true_dir, y_pred_dir, zero_division=0)
        except:
            metrics['Precision'] = 0.0
        
        try:
            metrics['Recall'] = recall_score(y_true_dir, y_pred_dir, zero_division=0)
        except:
            metrics['Recall'] = 0.0
        
        try:
            metrics['F1_Score'] = f1_score(y_true_dir, y_pred_dir, zero_division=0)
        except:
            metrics['F1_Score'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_direction_prediction(current_price, predicted_price):
        """
        Calculate direction prediction based on current vs predicted price
        Returns: Bullish, Bearish, or Hold prediction
        """
        if current_price is None or predicted_price is None:
            return 'Hold'
        
        # Calculate percentage change
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Determine direction based on percentage change
        if price_change_pct > 1.0:  # More than 1% increase = bullish
            return 'Bullish'
        elif price_change_pct < -1.0:  # More than 1% decrease = bearish
            return 'Bearish'
        else:
            return 'Hold'  # Between -1% and 1% = hold
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix for direction prediction"""
        if y_true is None or y_pred is None:
            return None
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) < 2:
            return None
        
        # Convert to direction (1 for up, 0 for down)
        y_true_dir = np.where(np.diff(y_true) > 0, 1, 0)
        y_pred_dir = np.where(np.diff(y_pred) > 0, 1, 0)
        
        try:
            cm = confusion_matrix(y_true_dir, y_pred_dir)
            return cm
        except:
            return None
    
    @staticmethod
    def get_all_metrics(y_true, y_pred):
        """Get all metrics (regression + classification)"""
        regression_metrics = EvaluationMetrics.calculate_regression_metrics(y_true, y_pred)
        classification_metrics = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred)
        
        all_metrics = {**regression_metrics, **classification_metrics}
        return all_metrics

