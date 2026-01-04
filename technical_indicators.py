"""
Technical Indicators Calculation Module
"""
import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    """Class to calculate technical indicators for stock price prediction"""
    
    @staticmethod
    def add_indicators(df):
        """
        Add technical indicators to dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df.columns:
            return df
        
        # Simple Moving Averages - Multiple timeframes
        df['SMA_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
        df['SMA_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df['close'], window=min(200, len(df))).sma_indicator() if len(df) > 200 else df['close']
        
        # Exponential Moving Averages - Multiple timeframes
        df['EMA_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # EMA/SMA ratios for trend strength
        df['EMA_12_SMA_20'] = df['EMA_12'] / df['SMA_20']
        df['EMA_26_SMA_50'] = df['EMA_26'] / df['SMA_50']
        df['Price_SMA_20'] = df['close'] / df['SMA_20']
        df['Price_EMA_12'] = df['close'] / df['EMA_12']
        
        # RSI (Relative Strength Index) - Multiple timeframes
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['RSI_9'] = ta.momentum.RSIIndicator(df['close'], window=9).rsi()
        
        # MACD - Enhanced
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # ADX (Average Directional Index)
        df['ADX'] = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close'], window=14
        ).adx()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        )
        df['Stoch'] = stoch.stoch()
        df['Stoch_signal'] = stoch.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_Change'] = df['close'].pct_change()
        df['Price_Change_5'] = df['close'].pct_change(periods=5)
        
        # Fill NaN values
        df = df.bfill().ffill()
        
        return df
    
    @staticmethod
    def get_feature_columns():
        """Get list of feature columns for model training"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_10', 'SMA_20', 'SMA_50', 
            'EMA_9', 'EMA_12', 'EMA_26', 'EMA_50',
            'EMA_12_SMA_20', 'EMA_26_SMA_50', 'Price_SMA_20', 'Price_EMA_12',
            'RSI', 'RSI_9',
            'MACD', 'MACD_signal', 'MACD_diff', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'ATR', 'ADX', 'Stoch', 'Stoch_signal',
            'Volume_SMA', 'Volume_Ratio',
            'Price_Change', 'Price_Change_5'
        ]

