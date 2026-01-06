"""
Trading Recommendations Module
"""
import numpy as np
import pandas as pd


class TradingRecommendations:
    """Class to provide trading recommendations including stop loss"""
    
    @staticmethod
    def calculate_stop_loss(current_price, predicted_price, risk_tolerance='medium'):
        """
        Calculate stop loss based on price prediction and risk tolerance
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted future price
            risk_tolerance: 'low', 'medium', 'high'
        
        Returns:
            Dictionary with stop loss recommendations
        """
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Risk tolerance multipliers
        risk_multipliers = {
            'low': 0.02,      # 2% stop loss
            'medium': 0.03,   # 3% stop loss
            'high': 0.05      # 5% stop loss
        }
        
        multiplier = risk_multipliers.get(risk_tolerance, 0.03)
        
        # Calculate stop loss based on direction
        if price_change_pct > 0:  # Bullish
            # Stop loss below current price
            stop_loss = current_price * (1 - multiplier)
            take_profit = predicted_price * 1.1  # 10% above predicted
        elif price_change_pct < 0:  # Bearish
            # Stop loss above current price (for short positions)
            stop_loss = current_price * (1 + multiplier)
            take_profit = predicted_price * 0.9  # 10% below predicted
        else:  # Neutral/Hold
            # Conservative stop loss
            stop_loss = current_price * (1 - multiplier)
            take_profit = current_price * (1 + multiplier * 0.5)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward_ratio': round(abs((take_profit - current_price) / abs(stop_loss - current_price)), 2) if abs(stop_loss - current_price) > 0 else 0,
            'risk_tolerance': risk_tolerance
        }
    
    @staticmethod
    def get_trading_signal(current_price, predicted_price, confidence=0.5):
        """
        Get trading signal: BUY, SELL, or HOLD
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted future price
            confidence: Confidence level (0-1)
        
        Returns:
            Dictionary with trading signal and details
        """
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Determine signal based on percentage change and confidence
        if price_change_pct > 2.0 and confidence > 0.6:
            signal = 'BUY'
            sentiment = 'bullish'
        elif price_change_pct < -2.0 and confidence > 0.6:
            signal = 'SELL'
            sentiment = 'bearish'
        elif abs(price_change_pct) > 0.5:
            signal = 'HOLD'
            sentiment = 'bullish' if price_change_pct > 0 else 'bearish'
        else:
            signal = 'HOLD'
            sentiment = 'neutral'
        
        return {
            'signal': signal,
            'sentiment': sentiment,
            'price_change_pct': round(price_change_pct, 2),
            'confidence': confidence,
            'recommendation': f"{signal} - {sentiment.upper()}"
        }
    
    @staticmethod
    def calculate_position_size(account_balance, risk_per_trade=0.02, stop_loss_price=None, current_price=None):
        """
        Calculate recommended position size based on risk management
        
        Args:
            account_balance: Total account balance
            risk_per_trade: Percentage of account to risk per trade (default 2%)
            stop_loss_price: Stop loss price
            current_price: Current stock price
        
        Returns:
            Dictionary with position size recommendations
        """
        risk_amount = account_balance * risk_per_trade
        
        if stop_loss_price and current_price:
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share > 0:
                shares = int(risk_amount / risk_per_share)
                position_value = shares * current_price
            else:
                shares = 0
                position_value = 0
        else:
            # Default: risk 2% of account
            shares = 0
            position_value = risk_amount * 50  # Rough estimate
        
        return {
            'shares': shares,
            'position_value': round(position_value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percentage': risk_per_trade * 100
        }

