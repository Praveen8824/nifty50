"""
Configuration file for Stock Prediction Application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'pub_95be0e8e27524c7394c6030e95a33381')
NEWS_DATA_API_KEY = os.getenv('NEWS_DATA_API_KEY', 'pub_95be0e8e27524c7394c6030e95a33381')

# Nifty 50 Stock Symbols
NIFTY_50_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'SUNPHARMA.NS',
    'BAJFINANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
    'TECHM.NS', 'ADANIENT.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'BAJAJFINSV.NS',
    'INDUSINDBK.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'TATAMOTORS.NS',
    'APOLLOHOSP.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'EICHERMOT.NS',
    'DRREDDY.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ADANIPORTS.NS',
    'TATACONSUM.NS', 'BRITANNIA.NS', 'MARICO.NS', 'PIDILITIND.NS', 'GODREJCP.NS'
]

# Time Frame Mappings
TIME_FRAMES = {
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1hour': '1h',
    '1day': '1d',
    '1week': '1wk',
    '1month': '1mo',
    '6months': '6mo',
    '1year': '1y'
}

# Model Configuration
TRAIN_TEST_SPLIT = 0.8
LOOKBACK_PERIOD = 60  # For LSTM
PREDICTION_HORIZON = 1  # Predict next period

# Technical Indicators
INDICATORS = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'ADX']

