"""
Data fetching module for stock market data using Yahoo Finance API
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Class to fetch stock market data from Yahoo Finance"""
    
    def __init__(self):
        self.time_frame_map = {
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
    
    def get_historical_data(self, symbol, time_frame, period=None):
        """
        Fetch historical stock data
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            time_frame: Time frame string (e.g., '5min', '1day')
            period: Period for data (e.g., '1y', '2y'). If None, auto-calculated
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            tf = self.time_frame_map.get(time_frame, '1d')
            
            # Calculate period based on time frame
            if period is None:
                period = self._calculate_period(time_frame)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=tf)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Clean and prepare data
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns in data")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_period(self, time_frame):
        """Calculate appropriate period based on time frame"""
        period_map = {
            '5min': '5d',
            '15min': '10d',
            '30min': '30d',
            '1hour': '60d',
            '1day': '2y',
            '1week': '5y',
            '1month': '10y',
            '6months': '10y',
            '1year': '10y'
        }
        return period_map.get(time_frame, '2y')
    
    def get_latest_price(self, symbol):
        """Get latest stock price with full info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try to get live price from info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            if current_price:
                return {
                    'price': current_price,
                    'change': info.get('regularMarketChange', 0),
                    'changePercent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'high': info.get('dayHigh', current_price),
                    'low': info.get('dayLow', current_price),
                    'open': info.get('open', current_price)
                }
            
            # Fallback to historical data
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2]['Close'] if len(data) > 1 else latest['Close']
                change = latest['Close'] - prev_close
                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
                
                return {
                    'price': latest['Close'],
                    'change': change,
                    'changePercent': change_pct,
                    'volume': latest['Volume'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'open': latest['Open']
                }
            return None
        except Exception as e:
            print(f"Error fetching latest price: {str(e)}")
            return None
    
    def get_realtime_data(self, symbol, time_frame='1d'):
        """Get real-time data for chart updates"""
        try:
            tf = self.time_frame_map.get(time_frame, '1d')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval=tf)
            
            if data.empty:
                return None
            
            # Clean and prepare data
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            return data
        except Exception as e:
            print(f"Error fetching realtime data: {str(e)}")
            return None

