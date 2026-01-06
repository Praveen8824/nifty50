"""
Enhanced LSTM Model for Stock Price Prediction with Multi-feature Support
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow - make it optional for deployment
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    BatchNormalization = None
    Adam = None
    EarlyStopping = None
    ReduceLROnPlateau = None


class LSTMModel:
    """Enhanced LSTM Neural Network for stock price prediction with multi-feature support"""
    
    def __init__(self, lookback=60, units=128, dropout=0.3, use_features=True):
        self.is_available = TENSORFLOW_AVAILABLE
        if not TENSORFLOW_AVAILABLE:
            self.is_trained = False
            return
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.use_features = use_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.feature_columns = None
    
    def create_sequences(self, data, lookback):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, feature_columns, target_column='close'):
        """Prepare data for training with multiple features"""
        df_clean = df.dropna()
        
        if df_clean.empty or len(df_clean) < self.lookback + 1:
            return None, None
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        if self.use_features and feature_columns:
            # Use multiple features
            available_features = [col for col in feature_columns if col in df_clean.columns]
            if len(available_features) > 0:
                # Prepare feature data
                feature_data = df_clean[available_features].values
                
                # Scale features
                scaled_features = self.feature_scaler.fit_transform(feature_data)
                
                # Create sequences with multiple features
                X, y = self.create_sequences(scaled_features, self.lookback)
                
                # For target, use close price
                close_prices = df_clean[target_column].values
                target_scaled = self.scaler.fit_transform(close_prices.reshape(-1, 1))
                y_target = target_scaled[self.lookback:]
                
                if len(X) == 0:
                    return None, None
                
                # Reshape for LSTM (samples, time steps, features)
                X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
                
                return X, y_target.ravel()
        
        # Fallback to single feature (close price)
        close_prices = df_clean[target_column].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        X, y = self.create_sequences(scaled_data, self.lookback)
        
        if len(X) == 0:
            return None, None
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        """Build enhanced LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.dropout),
            
            # Second LSTM layer
            LSTM(units=self.units, return_sequences=True),
            BatchNormalization(),
            Dropout(self.dropout),
            
            # Third LSTM layer
            LSTM(units=self.units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout),
            
            # Dense layers
            Dense(units=self.units // 2, activation='relu'),
            Dropout(self.dropout * 0.5),
            Dense(units=self.units // 4, activation='relu'),
            Dropout(self.dropout * 0.5),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Use adaptive learning rate
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
        
        return model
    
    def train(self, X, y, epochs=150, batch_size=32, verbose=0, validation_split=0.2):
        """Train the model with early stopping and learning rate reduction"""
        if not TENSORFLOW_AVAILABLE:
            return False
        if X is None or len(X) == 0:
            return False
        
        try:
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)
            
            # Callbacks for better training
            callbacks = []
            if EarlyStopping:
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                )
                callbacks.append(early_stop)
            
            if ReduceLROnPlateau:
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.00001,
                    verbose=0
                )
                callbacks.append(reduce_lr)
            
            # Train with validation split
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
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
    
    def prepare_predict_data(self, last_sequence, feature_data=None):
        """Prepare last sequence for prediction"""
        if self.use_features and feature_data is not None and self.feature_columns:
            if len(feature_data) < self.lookback:
                return None
            
            sequence = feature_data[-self.lookback:]
            scaled_sequence = self.feature_scaler.transform(sequence)
            X = scaled_sequence.reshape((1, self.lookback, scaled_sequence.shape[1]))
            return X
        else:
            if len(last_sequence) < self.lookback:
                return None
            
            sequence = last_sequence[-self.lookback:]
            scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))
            X = scaled_sequence.reshape((1, self.lookback, 1))
            return X
