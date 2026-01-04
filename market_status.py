"""
Market Status Checker for Indian Stock Market
"""
from datetime import datetime, time
import pytz


class MarketStatus:
    """Check if Indian stock market is open or closed"""
    
    @staticmethod
    def is_market_open():
        """
        Check if Indian stock market (NSE/BSE) is currently open
        Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        """
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        
        # Check if weekend
        if now_ist.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Market is closed (Weekend)"
        
        # Market hours
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        current_time = now_ist.time()
        
        if market_open <= current_time <= market_close:
            return True, "Market is open"
        elif current_time < market_open:
            return False, f"Market opens at 9:15 AM IST (Current: {current_time.strftime('%I:%M %p')})"
        else:
            return False, f"Market closed at 3:30 PM IST (Current: {current_time.strftime('%I:%M %p')})"
    
    @staticmethod
    def get_market_status_message():
        """Get formatted market status message"""
        is_open, reason = MarketStatus.is_market_open()
        status_icon = "🟢" if is_open else "🔴"
        return f"{status_icon} {reason}"

