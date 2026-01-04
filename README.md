# 📈 Nifty 50 Stock Price Prediction System

A comprehensive stock price prediction system for Nifty 50 stocks using ensemble learning with multiple machine learning models, technical indicators, and sentiment analysis.

## 🚀 Features

- **5 Machine Learning Models**:
  - Random Forest Regressor
  - ARIMA (AutoRegressive Integrated Moving Average)
  - XGBoost
  - Support Vector Regressor (SVR)
  - LSTM (Long Short-Term Memory)

- **Multiple Time Frames**: 5min, 15min, 30min, 1hour, 1day, 1week, 1month, 6months, 1year

- **Technical Indicators**: 
  - Moving Averages (SMA, EMA)
  - RSI, MACD
  - Bollinger Bands
  - ATR, ADX
  - Stochastic Oscillator
  - Volume indicators

- **Sentiment Analysis**:
  - Stock-specific news sentiment
  - Indian market sentiment
  - Global market sentiment
  - Focus on relevant news (Q1-Q4 results, earnings, etc.)

- **Comprehensive Metrics**:
  - R2 Score
  - F1 Score
  - Precision
  - Recall
  - Accuracy
  - RMSE, MAE, MAPE

- **Interactive UI**: Built with Streamlit for easy deployment

## requisites

- Python 3.8 or higher
- News API key (optional, for sentiment analysis)

## 🛠️ 

⚠️ **Warning:** This will install packages globally. Use virtual environment for better isolation.

## 🎯 Usage

1. **Activate virtual environment** (if using venv):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py ---
   ```

For detailed setup instructions, see `SETUP.md`

2. **In the web interface**:
   - Select a Nifty 50 stock from the dropdown
   - Choose your desired time frame
   - Click "Fetch Data & Train Models"
   - View predictions, metrics, and sentiment analysis

## 📁 Project Structure

```
Stock_predicition/
│
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration file
├── data_fetcher.py             # Yahoo Finance data fetching
├── technical_indicators.py     # Technical indicators calculation
├── model_trainer.py            # Model training and ensemble
├── evaluation_metrics.py       # Metrics calculation
├── sentiment_analysis.py       # News sentiment analysis
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # This file
│
└── models/
    ├── __init__.py
    ├── random_forest_model.py  # Random Forest implementation
    ├── arima_model.py          # ARIMA implementation
    ├── xgboost_model.py        # XGBoost implementation
    ├── svr_model.py            # SVR implementation
    └── lstm_model.py           # LSTM implementation
```

## 🔧 Configuration

Edit `config.py` to:
- Add/remove Nifty 50 stocks
- Adjust time frame mappings
- Modify model parameters
- Configure technical indicators

## 📊 Model Details

### Random Forest
- Ensemble of decision trees
- Good for non-linear relationships
- Handles multiple features well

### ARIMA
- Time series forecasting model
- Auto-detects optimal parameters
- Good for trend analysis

### XGBoost
- Gradient boosting algorithm
- High performance
- Handles missing values

### SVR
- Support Vector Machine for regression
- Good for non-linear patterns
- Robust to outliers

### LSTM
- Deep learning model
- Captures long-term dependencies
- Best for sequential data

## 🎨 UI Features

- **Interactive Charts**: Plotly-based visualizations
- **Real-time Updates**: Fetch latest data on demand
- **Metrics Dashboard**: Compare model performance
- **Sentiment Indicators**: Visual sentiment display
- **Responsive Design**: Works on different screen sizes

## 📈 Deployment

This application is designed to run on Streamlit Cloud for free:

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repository
4. Set environment variables in Streamlit Cloud settings
5. Deploy!

## ⚠️ Important Notes

- **Data Availability**: Yahoo Finance data availability depends on market hours and API limits
- **News API**: Free tier has rate limits (100 requests/day)
- **Model Training**: Training time depends on data size and time frame
- **Predictions**: These are for educational purposes only, not financial advice


