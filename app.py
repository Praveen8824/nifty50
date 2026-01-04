"""
Main Streamlit Application for Stock Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from config import NIFTY_50_STOCKS, TIME_FRAMES
from data_fetcher import DataFetcher
from model_trainer import ModelTrainer
from evaluation_metrics import EvaluationMetrics
from sentiment_analysis import SentimentAnalyzer
from market_status import MarketStatus

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .bullish {
        background-color: #d4edda;
        color: #155724;
    }
    .bearish {
        background-color: #f8d7da;
        color: #721c24;
    }
    .neutral {
        background-color: #fff3cd;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()

if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()

if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

if 'metrics' not in st.session_state:
    st.session_state.metrics = {}


def get_stock_name(symbol):
    """Get stock name from symbol"""
    name_map = {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'ICICIBANK.NS': 'ICICI Bank',
        'INFY.NS': 'Infosys',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'ITC.NS': 'ITC Limited',
        'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank'
    }
    return name_map.get(symbol, symbol.replace('.NS', ''))


def create_price_chart(df, predictions_dict, model_name=None):
    """Create interactive price chart with predictions"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Prediction', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Actual price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Predictions for each model
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    color_idx = 0
    
    for model_name, pred in predictions_dict.items():
        if pred is not None and len(pred) > 0:
            # Predictions are for next period, so align with actual[1:]
            actual_prices = df['close'].values
            pred_array = np.array(pred)
            
            # Align: predictions[0] corresponds to actual[1]
            min_len = min(len(pred_array), len(actual_prices) - 1)
            if min_len > 0:
                # Shift actual prices by 1 to align with predictions
                actual_aligned = actual_prices[1:min_len+1]
                pred_aligned = pred_array[:min_len]
                
                # Use indices starting from 1
                pred_indices = df.index[1:min_len+1]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_indices,
                        y=pred_aligned,
                        mode='lines',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[color_idx % len(colors)], width=1.5, dash='dash')
                    ),
                    row=1, col=1
                )
                color_idx += 1
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text="Stock Price Prediction Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def display_metrics_table(metrics_dict):
    """Display metrics in a table format"""
    if not metrics_dict:
        return
    
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df, use_container_width=True)


def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">📈 Nifty 50 Stock Price Prediction</h1>', unsafe_allow_html=True)
    
    # Market status banner at top
    market_status_msg = MarketStatus.get_market_status_message()
    is_market_open, _ = MarketStatus.is_market_open()
    if is_market_open:
        st.success(f"📊 {market_status_msg}")
    else:
        st.warning(f"⚠️ {market_status_msg}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock selection
        stock_options = {get_stock_name(s): s for s in NIFTY_50_STOCKS}
        selected_stock_name = st.selectbox(
            "Select Nifty 50 Stock",
            options=list(stock_options.keys()),
            index=0
        )
        selected_stock = stock_options[selected_stock_name]
        
        # Time frame selection
        time_frame_options = list(TIME_FRAMES.keys())
        selected_time_frame = st.selectbox(
            "Select Time Frame",
            options=time_frame_options,
            index=4  # Default to 1day
        )
        
        st.divider()
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = ['Random Forest', 'XGBoost', 'SVR', 'ARIMA', 'LSTM']
        # Check if LSTM is available (TensorFlow might not be installed)
        try:
            if hasattr(st.session_state.model_trainer.lstm_model, 'is_available'):
                if not st.session_state.model_trainer.lstm_model.is_available:
                    available_models = [m for m in available_models if m != 'LSTM']
        except:
            pass
        selected_models = st.multiselect(
            "Select Models for Prediction",
            options=available_models,
            default=available_models  # All models selected by default
        )
        
        if len(selected_models) == 0:
            st.warning("⚠️ Please select at least one model")
        
        st.divider()
        
        # Action buttons
        if st.button("🔄 Fetch Data & Train Models", type="primary", use_container_width=True):
            with st.spinner("Fetching data and training models..."):
                try:
                    # Fetch data
                    df = st.session_state.data_fetcher.get_historical_data(
                        selected_stock, selected_time_frame
                    )
                    
                    if df.empty:
                        st.error("Failed to fetch data. Please try again.")
                        return
                    
                    st.session_state.stock_data = df
                    
                    # Train models
                    success = st.session_state.model_trainer.train_all_models(df)
                    
                    if success:
                        st.success("Models trained successfully!")
                        
                        # Get predictions
                        predictions = st.session_state.model_trainer.predict_all_models(df)
                        st.session_state.predictions = predictions
                        
                        # Calculate metrics
                        metrics_dict = {}
                        actual_prices = df['close'].values[1:]  # Shift for alignment
                        
                        for model_name, pred in predictions.items():
                            if pred is not None and len(pred) > 0:
                                min_len = min(len(actual_prices), len(pred))
                                metrics = EvaluationMetrics.get_all_metrics(
                                    actual_prices[:min_len],
                                    pred[:min_len]
                                )
                                metrics_dict[model_name] = metrics
                        
                        st.session_state.metrics = metrics_dict
                    else:
                        st.error("Failed to train models. Please check the data.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.divider()
        st.info("💡 Click 'Fetch Data & Train Models' to start prediction")
    
    # Main content area
    if st.session_state.stock_data is not None and len(st.session_state.predictions) > 0:
        df = st.session_state.stock_data
        
        # Filter predictions based on selected models
        filtered_predictions = {
            model: pred for model, pred in st.session_state.predictions.items()
            if model in selected_models and pred is not None
        }
        
        if len(filtered_predictions) == 0:
            st.error("⚠️ Selected models are not available. Please train models first or select different models.")
            st.stop()
        
        # Display current stock info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stock", selected_stock_name)
        with col2:
            st.metric("Time Frame", selected_time_frame)
        with col3:
            latest_price = df['close'].iloc[-1]
            st.metric("Latest Price", f"₹{latest_price:.2f}")
        with col4:
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
            st.metric("Change", f"₹{price_change:.2f}")
        
        st.divider()
        
        # Price prediction chart
        st.subheader("📊 Price Prediction Chart")
        fig = create_price_chart(df, filtered_predictions)
        st.plotly_chart(fig, width='stretch')
        
        # Data Analysis Information
        if st.session_state.model_trainer.training_data_points > 0:
            st.info(f"📊 **Analysis Information:** This prediction is based on analysis of **{st.session_state.model_trainer.training_data_points}** historical data points. The predicted price represents the expected price after the next market opening from the last closing price.")
        
        # Prediction section - single model or ensemble
        if len(selected_models) == 1:
            # Single model prediction
            model_name = selected_models[0]
            if model_name in filtered_predictions:
                st.subheader(f"🎯 {model_name} Prediction")
                single_pred = filtered_predictions[model_name]
                if single_pred is not None and len(single_pred) > 0:
                    predicted_price = single_pred[-1]
                else:
                    st.error(f"Prediction not available for {model_name}")
                    st.stop()
            else:
                st.error(f"{model_name} prediction not available")
                st.stop()
        else:
            # Ensemble prediction from selected models
            st.subheader(f"🎯 Ensemble Prediction ({', '.join(selected_models)})")
            # Filter metrics for selected models
            filtered_metrics = {
                model: metrics for model, metrics in st.session_state.metrics.items()
                if model in selected_models
            } if st.session_state.metrics else None
            
            ensemble_pred = st.session_state.model_trainer.ensemble_predict(
                filtered_predictions, 
                filtered_metrics
            )
            
            if ensemble_pred is None or len(ensemble_pred) == 0:
                st.error("Failed to generate ensemble prediction")
                st.stop()
            
            predicted_price = ensemble_pred[-1]
        
        # Get prices (works for both single and ensemble)
        last_close_price = df['close'].iloc[-1]  # Last closing price
        current_price = last_close_price  # Current price (same as last close if market closed)
        open_price = df['open'].iloc[-1] if len(df) > 0 else last_close_price
        price_change_pct = ((predicted_price - last_close_price) / last_close_price) * 100
        
        # Display prices
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Open", f"₹{open_price:.2f}")
        with col2:
            st.metric("Last Close", f"₹{last_close_price:.2f}")
        with col3:
            st.metric("Current Price", f"₹{current_price:.2f}")
        with col4:
            st.metric("Predicted Price (After Next Opening)", f"₹{predicted_price:.2f}", delta=f"{price_change_pct:.2f}%")
        
        # Clarification
        st.caption("💡 **Note:** Predicted price is the expected price after the next market opening from the last closing price.")
        
        # Direction prediction - use last close price vs predicted price
        direction = EvaluationMetrics.calculate_direction_prediction(
            last_close_price, predicted_price
        )
        
        direction_class = direction
        direction_emoji = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "🟡"
        
        st.markdown(
            f'<div class="prediction-box {direction_class}">'
            f'{direction_emoji} Prediction: {direction.upper()}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # Model metrics (only show selected models)
        st.subheader("📈 Model Performance Metrics")
        
        if st.session_state.metrics:
            # Filter metrics for selected models only
            filtered_metrics_data = []
            for model_name in selected_models:
                if model_name in st.session_state.metrics:
                    row = {'Model': model_name}
                    row.update(st.session_state.metrics[model_name])
                    filtered_metrics_data.append(row)
            
            if filtered_metrics_data:
                metrics_df = pd.DataFrame(filtered_metrics_data)
                st.dataframe(metrics_df, width='stretch')
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**R2 Score Comparison**")
                if 'R2_Score' in metrics_df.columns:
                    fig_r2 = go.Figure(data=[
                        go.Bar(x=metrics_df['Model'], y=metrics_df['R2_Score'])
                    ])
                    fig_r2.update_layout(title="R2 Score by Model", yaxis_title="R2 Score")
                    st.plotly_chart(fig_r2, width='stretch')
            
            with col2:
                st.write("**F1 Score Comparison**")
                if 'F1_Score' in metrics_df.columns:
                    fig_f1 = go.Figure(data=[
                        go.Bar(x=metrics_df['Model'], y=metrics_df['F1_Score'])
                    ])
                    fig_f1.update_layout(title="F1 Score by Model", yaxis_title="F1 Score")
                    st.plotly_chart(fig_f1, width='stretch')
            
            # Confusion Matrix for each selected model
            st.subheader("📊 Confusion Matrices")
            actual_prices = df['close'].values[1:]
            
            selected_confusion_models = [
                (model_name, filtered_predictions[model_name])
                for model_name in selected_models
                if model_name in filtered_predictions and filtered_predictions[model_name] is not None
            ]
            
            if selected_confusion_models:
                confusion_cols = st.columns(min(len(selected_confusion_models), 3))
                for idx, (model_name, pred) in enumerate(selected_confusion_models):
                    if len(pred) > 0:
                        min_len = min(len(actual_prices), len(pred))
                        cm = EvaluationMetrics.get_confusion_matrix(
                            actual_prices[:min_len],
                            pred[:min_len]
                        )
                        
                        if cm is not None and idx < len(confusion_cols):
                            with confusion_cols[idx % len(confusion_cols)]:
                                st.write(f"**{model_name}**")
                                fig_cm = go.Figure(data=go.Heatmap(
                                    z=cm,
                                    x=['Down', 'Up'],
                                    y=['Down', 'Up'],
                                    colorscale='Blues',
                                    text=cm,
                                    texttemplate='%{text}',
                                    textfont={"size": 14}
                                ))
                                fig_cm.update_layout(
                                    title=f"{model_name} Confusion Matrix",
                                    width=300,
                                    height=300
                                )
                                st.plotly_chart(fig_cm, width='stretch')
        
        st.divider()
        
        # Sentiment Analysis
        st.subheader("📰 Sentiment Analysis")
        
        sentiment_tabs = st.tabs(["Stock Sentiment", "Indian Market", "Global Market"])
        
        with sentiment_tabs[0]:
            # Stock selection dropdown for sentiment
            stock_options_sentiment = {get_stock_name(s): s for s in NIFTY_50_STOCKS}
            selected_stock_sentiment_name = st.selectbox(
                "Select Stock for Sentiment Analysis",
                options=list(stock_options_sentiment.keys()),
                index=list(stock_options_sentiment.keys()).index(selected_stock_name) if selected_stock_name in stock_options_sentiment else 0,
                key="sentiment_stock"
            )
            selected_stock_sentiment = stock_options_sentiment[selected_stock_sentiment_name]
            
            if st.button("Analyze Stock Sentiment", key="analyze_sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = st.session_state.sentiment_analyzer.get_stock_sentiment(
                        selected_stock_sentiment_name, selected_stock_sentiment
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment_result['sentiment'].upper())
                    with col2:
                        st.metric("Score", f"{sentiment_result['score']:.4f}")
                    with col3:
                        st.metric("Articles Analyzed", sentiment_result['articles_count'])
                    
                    sentiment_class = sentiment_result['sentiment']
                    sentiment_emoji = "🟢" if sentiment_class == "bullish" else "🔴" if sentiment_class == "bearish" else "🟡"
                    
                    st.markdown(
                        f'<div class="prediction-box {sentiment_class}">'
                        f'{sentiment_emoji} Stock Sentiment: {sentiment_class.upper()}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show relevant articles
                    if 'articles' in sentiment_result and sentiment_result['articles']:
                        st.subheader("📰 Relevant Articles")
                        for idx, article in enumerate(sentiment_result['articles'][:10], 1):
                            with st.expander(f"{idx}. {article['title']} ({article['sentiment_label'].upper()})"):
                                if article.get('description'):
                                    st.write(article['description'])
                                if article.get('url'):
                                    st.write(f"🔗 [Read full article]({article['url']})")
                                if article.get('publishedAt'):
                                    st.caption(f"Published: {article['publishedAt']}")
                                st.caption(f"Sentiment Score: {article['sentiment']:.4f}")
        
        with sentiment_tabs[1]:
            if st.button("Analyze Indian Market Sentiment"):
                with st.spinner("Analyzing Indian market sentiment..."):
                    sentiment = st.session_state.sentiment_analyzer.get_market_sentiment('indian')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment['sentiment'].upper())
                    with col2:
                        st.metric("Score", f"{sentiment['score']:.4f}")
                    with col3:
                        st.metric("Articles Analyzed", sentiment['articles_count'])
        
        with sentiment_tabs[2]:
            if st.button("Analyze Global Market Sentiment"):
                with st.spinner("Analyzing global market sentiment..."):
                    sentiment = st.session_state.sentiment_analyzer.get_market_sentiment('global')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment['sentiment'].upper())
                    with col2:
                        st.metric("Score", f"{sentiment['score']:.4f}")
                    with col3:
                        st.metric("Articles Analyzed", sentiment['articles_count'])
        
        # Raw data display
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.tail(100), width='stretch')
    
    else:
        # Welcome screen
        st.info("👈 Please select a stock and time frame, then click 'Fetch Data & Train Models' to start")
        
        # Display instructions
        st.markdown("""
        ### 🚀 How to Use:
        1. **Select a Stock**: Choose from the Nifty 50 stocks in the sidebar
        2. **Select Time Frame**: Choose your desired prediction time frame
        3. **Fetch & Train**: Click the button to fetch data and train all models
        4. **View Results**: See predictions, metrics, and sentiment analysis
        
        ### 📊 Features:
        - **5 Machine Learning Models**: Random Forest, ARIMA, XGBoost, SVR, LSTM
        - **Ensemble Learning**: Combined predictions from all models
        - **Technical Indicators**: Advanced indicators for better accuracy
        - **Sentiment Analysis**: News-based sentiment for stocks and markets
        - **Comprehensive Metrics**: R2, F1, Precision, Recall, Accuracy
        - **Interactive Charts**: Visualize actual vs predicted prices
        """)


if __name__ == "__main__":
    main()

