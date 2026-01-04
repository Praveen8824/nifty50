"""
LSTM Model for Stock Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow - make it optional for deployment
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    Adam = None


class LSTMModel:
    """LSTM Neural Network for stock price prediction"""
    
    def __init__(self, lookback=30, units=32, dropout=0.2):
        self.is_available = TENSORFLOW_AVAILABLE
        if not TENSORFLOW_AVAILABLE:
            self.is_trained = False
            return
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def create_sequences(self, data, lookback):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, feature_columns, target_column='close'):
        """Prepare data for training"""
        df_clean = df.dropna()
        
        if df_clean.empty or len(df_clean) < self.lookback + 1:
            return None, None
        
        # Use close price for LSTM
        close_prices = df_clean[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, self.lookback)
        
        if len(X) == 0:
            return None, None
        
        # Reshape for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(units=self.units),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, verbose=0):
        """Train the model"""
        if not TENSORFLOW_AVAILABLE:
            return False
        if X is None or len(X) == 0:
            return False
        
        try:
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)
            
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not TENSORFLOW_AVAILABLE or not self.is_trained or X is None:
            return None
        
        try:
            predictions_scaled = self.model.predict(X, verbose=0)
            predictions = self.scaler.inverse_transform(predictions_scaled)
            return predictions.ravel()
        except Exception as e:
            print(f"Error predicting with LSTM: {str(e)}")
            return None
    
    def prepare_predict_data(self, last_sequence):
        """Prepare last sequence for prediction"""
        if len(last_sequence) < self.lookback:
            return None
        
        sequence = last_sequence[-self.lookback:]
        scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))
        X = scaled_sequence.reshape((1, self.lookback, 1))
        return X

