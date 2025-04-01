import os
import time
import threading
import logging
import sys
import pyotp
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.dependencies import ALL, MATCH
import dash_bootstrap_components as dbc
from SmartApi import SmartConnect
from datetime import datetime, timedelta
from scipy import stats
import plotly.graph_objs as go
from collections import deque
import requests
import json
import random
from functools import wraps
import re
import warnings
import yfinance as yf

# Check for feedparser (required for news functionality)
try:
    import feedparser
except ImportError:
    print("Warning: feedparser not installed. News functionality will be limited.")
    print("Install with: pip install feedparser")

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY], 
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Options Trading Dashboard"

# Define custom CSS for smooth transitions
app_css = '''
/* Base transitions */
.dynamic-update {
    transition: all 0.3s ease-in-out !important;
}

/* Price updates */
.price-element {
    transition: color 0.3s ease, background-color 0.3s ease !important;
}

/* Badge transitions */
.badge-transition {
    transition: background-color 0.3s ease, color 0.3s ease !important;
}

/* Hardware acceleration to prevent flickering */
.no-flicker {
    transform: translateZ(0);
    backface-visibility: hidden;
}

/* Smooth progress bar updates */
.progress-bar {
    transition: width 0.3s ease-in-out !important;
}

/* Value updates */
.value-update {
    transition: opacity 0.3s, transform 0.3s, color 0.3s;
}

/* Card updates */
.card-update {
    transition: border-color 0.3s ease-in-out;
}

/* Remove blinking animations */
.highlight-positive, .highlight-negative {
    animation: none !important;
    transition: background-color 0.3s ease-in-out;
}

/* Color transitions */
.color-transition {
    transition: color 0.3s ease-in-out;
}

/* Search dropdown */
.search-dropdown {
    position: absolute;
    width: 100%;
    max-height: 200px;
    overflow-y: auto;
    z-index: 1000;
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

.search-item {
    padding: 8px 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.search-item:hover {
    background-color: #333;
}

/* Loading spinner */
.loading-spinner {
    display: inline-block;
    width: 1rem;
    height: 1rem;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s linear infinite;
    margin-left: 0.5rem;
}

@keyframes spin {
    to {transform: rotate(360deg);}
}
'''

# The index string MUST include all required placeholders
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            ''' + app_css + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============ Logging Setup ============
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                          mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("trading_dashboard")

# ============ Rate Limiting and Retry Logic ============
class RateLimitHandler:
    def __init__(self, max_requests=1, time_window=1, initial_backoff=1, max_backoff=60):
        # Updated to 1 request per second per SmartAPI limits
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = {}
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.lock = threading.Lock()
        self.rate_limit_hits = {}
        self.api_call_count = 0
        
    def wait_if_needed(self, endpoint_key="default"):
        with self.lock:
            current_time = time.time()
            if endpoint_key not in self.request_timestamps:
                self.request_timestamps[endpoint_key] = []
            self.request_timestamps[endpoint_key] = [ts for ts in self.request_timestamps[endpoint_key] 
                                      if current_time - ts <= self.time_window]
            
            if endpoint_key in self.rate_limit_hits:
                last_hit, hit_count = self.rate_limit_hits[endpoint_key]
                time_since_hit = current_time - last_hit
                if time_since_hit < 60:
                    backoff_time = min(self.initial_backoff * (2 ** min(hit_count, 5)), self.max_backoff)
                    remaining_wait = backoff_time - time_since_hit
                    if remaining_wait > 0:
                        logger.debug(f"Rate limit backoff for {endpoint_key}: {remaining_wait:.2f}s")
                        time.sleep(remaining_wait)
                        if time_since_hit + remaining_wait >= 60:
                            self.rate_limit_hits.pop(endpoint_key, None)
            
            if len(self.request_timestamps[endpoint_key]) >= self.max_requests:
                oldest_timestamp = min(self.request_timestamps[endpoint_key])
                wait_time = self.time_window - (current_time - oldest_timestamp)
                if wait_time > 0:
                    jitter = random.uniform(0.1, 0.2)  # Reduced jitter for more predictable timing
                    logger.debug(f"Rate limiting {endpoint_key}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time + jitter)
            
            self.request_timestamps[endpoint_key].append(time.time())
            self.api_call_count += 1
    
    def register_rate_limit_hit(self, endpoint_key="default"):
        with self.lock:
            current_time = time.time()
            if endpoint_key in self.rate_limit_hits:
                _, hit_count = self.rate_limit_hits[endpoint_key]
                self.rate_limit_hits[endpoint_key] = (current_time, hit_count + 1)
            else:
                self.rate_limit_hits[endpoint_key] = (current_time, 1)
            logger.warning(f"Rate limit hit for {endpoint_key}, implementing backoff")
    
    def get_call_count(self):
        with self.lock:
            return self.api_call_count
            
    def reset_call_count(self):
        with self.lock:
            prev_count = self.api_call_count
            self.api_call_count = 0
            return prev_count
    
    def retry_with_backoff(self, func, *args, endpoint_key="default", max_retries=3, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        
        while retries <= max_retries:
            try:
                self.wait_if_needed(endpoint_key)
                result = func(*args, **kwargs)
                
                rate_limited = False
                if isinstance(result, dict):
                    error_msg = str(result.get("message", "")).lower()
                    if not result.get("status") and ("access rate" in error_msg or "try after some time" in error_msg or "session expired" in error_msg):
                        rate_limited = True
                if isinstance(result, str) and ("access rate" in result.lower() or "session expired" in result.lower()):
                    rate_limited = True
                
                if rate_limited:
                    self.register_rate_limit_hit(endpoint_key)
                    retries += 1
                    if retries <= max_retries:
                        sleep_time = backoff + random.uniform(0, 0.5)  # Reduced randomness
                        logger.warning(f"Rate limited for {endpoint_key}. Retrying in {sleep_time:.2f}s ({retries}/{max_retries})")
                        time.sleep(sleep_time)
                        backoff = min(backoff * 2, self.max_backoff)
                    else:
                        logger.error(f"Failed after {max_retries} retries for {endpoint_key}")
                        return None
                else:
                    return result
                    
            except Exception as e:
                retries += 1
                if "access rate" in str(e).lower() or "rate limit" in str(e).lower() or "session expired" in str(e).lower():
                    self.register_rate_limit_hit(endpoint_key)
                    if retries <= max_retries:
                        sleep_time = backoff + random.uniform(0, 0.5)
                        logger.warning(f"Rate limited (exception) for {endpoint_key}. Retrying in {sleep_time:.2f}s ({retries}/{max_retries})")
                        time.sleep(sleep_time)
                        backoff = min(backoff * 2, self.max_backoff)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        return None
                else:
                    logger.error(f"Error in API call to {endpoint_key}: {e}")
                    return None

# Updated rate limit handler for new SmartAPI limits (1 request per second)
rate_limit_handler = RateLimitHandler(max_requests=1, time_window=1)

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return rate_limit_handler.retry_with_backoff(func, *args, **kwargs)
    return wrapper

# ============ Configuration ============
class Config:
    def __init__(self):
        self.api_key = os.getenv("SMARTAPI_KEY", "kIa6qesM")  # Replace with your key
        self.username = os.getenv("SMARTAPI_USERNAME", "M243904")  # Replace with your username
        self.password = os.getenv("SMARTAPI_PASSWORD", "1209")  # Replace with your password
        self.totp_secret = os.getenv("SMARTAPI_TOTP", "KSFDSR2QQ5D2VNZF2QKO2HRD5A")
        self.exchange = "NSE"
        self.last_refreshed = None
        self.refresh_token = None
        self.validate_config()
    
    def validate_config(self):
        missing = []
        if not self.api_key:
            missing.append("SMARTAPI_KEY")
        if not self.username:
            missing.append("SMARTAPI_USERNAME")
        if not self.password:
            missing.append("SMARTAPI_PASSWORD")
        if not self.totp_secret:
            missing.append("SMARTAPI_TOTP")
        
        if missing:
            logger.warning(f"Missing required environment variables: {', '.join(missing)}")

config = Config()

# ============ Trading Parameters ============
# Risk management
RISK_PER_TRADE = 1.0  # Risk per trade in percentage of capital
MAX_TRADES_PER_DAY = 40  # Maximum number of trades per day
MAX_LOSS_PERCENTAGE = 5.0  # Maximum loss percentage per day

# Dynamic StopLoss Settings
BASE_TRAILING_SL_ACTIVATION = 0.8  # Base percentage of profit at which to activate trailing stop loss
BASE_TRAILING_SL_PERCENTAGE = 0.4  # Base trailing stop loss percentage
VOLATILITY_SL_MULTIPLIER = 5.0  # Multiplier for volatility-based SL
MOMENTUM_SL_FACTOR = 0.2  # Factor for momentum-based SL adjustment

# Enhanced Trailing SL Settings
ENHANCED_TRAILING_SL = True  # Enable enhanced trailing SL
MIN_PROFIT_FOR_TRAILING = 1.0  # Minimum profit percentage to activate trailing SL
TRAILING_STEP_INCREMENTS = [1.0, 2.0, 3.0, 5.0]  # Profit thresholds for tightening trailing SL
TRAILING_SL_PERCENTAGES = [0.6, 0.5, 0.4, 0.3]  # SL percentages for each threshold

# Enhanced Target Settings
DYNAMIC_TARGET_ADJUSTMENT = True  # Enable dynamic target adjustment
MIN_TARGET_ADJUSTMENT = 0.2  # Minimum target adjustment percentage
MAX_TARGET_ADJUSTMENT = 2.5  # Increased maximum target adjustment percentage for higher profits
TARGET_MOMENTUM_FACTOR = 0.4  # Increased factor for momentum-based target adjustment

# Position Management
MAX_POSITION_HOLDING_TIME_SCALP = 10  # Maximum scalping trade holding time in minutes
MAX_POSITION_HOLDING_TIME_SWING = 120  # Maximum swing trade holding time in minutes
MAX_POSITION_HOLDING_TIME_MOMENTUM = 45  # Maximum momentum trade holding time in minutes
MAX_POSITION_HOLDING_TIME_NEWS = 60  # Maximum news-based trade holding time in minutes

# Prediction Thresholds
MIN_SIGNAL_STRENGTH_SCALP = 3.0  # Minimum signal strength for scalping trades
MIN_SIGNAL_STRENGTH_SWING = 4.0  # Minimum signal strength for swing trades
MIN_SIGNAL_STRENGTH_MOMENTUM = 3.5  # Minimum signal strength for momentum trades
MIN_SIGNAL_STRENGTH_NEWS = 3.0  # Minimum signal strength for news-based trades

# Strategy Settings
strategy_settings = {
    "SCALP_ENABLED": True,
    "SWING_ENABLED": True,
    "MOMENTUM_ENABLED": True,
    "NEWS_ENABLED": True  # Added news-based strategy
}

# Technical Indicators Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
EMA_SHORT = 5
EMA_MEDIUM = 10
EMA_LONG = 20
LONG_WINDOW = 20

# Strategy-specific parameters
SCALP_RSI_PERIOD = 7
SCALP_EMA_SHORT = 3
SCALP_EMA_MEDIUM = 8
SCALP_MOMENTUM_PERIOD = 5

SWING_RSI_PERIOD = 21
SWING_EMA_MEDIUM = 18
SWING_EMA_LONG = 34
SWING_TREND_STRENGTH_PERIOD = 50

MOMENTUM_RSI_PERIOD = 9
MOMENTUM_EMA_MEDIUM = 13
MOMENTUM_LOOKBACK_PERIOD = 30
MOMENTUM_STRENGTH_THRESHOLD = 1.5

# PCR parameters
PCR_BULLISH_THRESHOLD = 0.8
PCR_BEARISH_THRESHOLD = 1.2
PCR_NEUTRAL_RANGE = (0.9, 1.1)
PCR_HISTORY_LENGTH = 100
PCR_TREND_LOOKBACK = 5

# API rate limiting parameters - Updated for new SmartAPI limits
API_UPDATE_INTERVAL = 1  # Seconds between API data updates (set to 1 for real-time)
BULK_FETCH_SIZE = 50  # Maximum symbols per bulk request
INDEX_UPDATE_INTERVAL = 1  # Seconds between index data updates
OPTION_UPDATE_INTERVAL = 1  # Seconds between option data updates
PCR_UPDATE_INTERVAL = 30  # Seconds between PCR updates
DATA_CLEANUP_INTERVAL = 600  # Cleanup old data every 10 minutes
MAX_PRICE_HISTORY_POINTS = 1000  # Maximum number of price history points to store

# Option Selection Parameters
NUM_STRIKES_TO_FETCH = 5  # Number of strikes above and below current price to fetch
OPTION_AUTO_SELECT_INTERVAL = 600  # Automatically update option selection every 10 minutes (increased from 5)

# News Monitoring Parameters - IMPROVED: Reduced interval and increased confidence
NEWS_CHECK_INTERVAL = 30  # Check for news every 30 seconds (reduced from 60)
NEWS_SENTIMENT_THRESHOLD = 0.25  # Threshold for considering news as positive/negative (reduced from 0.3)
NEWS_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to trade on news (reduced from 0.7)
NEWS_MAX_AGE = 1800  # Max age of news in seconds (30 minutes) to consider for trading (reduced from 1 hour)

# Historical Data Paths
NIFTY_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\NIFTY.csv"
BANK_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\BANK.csv"

# Symbol mapping for broker
SYMBOL_MAPPING = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE"
}

# Define a modern color scheme for the dashboard
COLOR_SCHEME = {
    "bg_dark": "#121212",
    "card_bg": "#1E1E1E",
    "card_bg_alt": "#252525",
    "text_light": "#F8F9FA",
    "text_muted": "#ADB5BD",
    "primary": "#0D6EFD",
    "secondary": "#6C757D",
    "success": "#198754",
    "danger": "#DC3545",
    "warning": "#FFC107",
    "info": "#0DCAF0",
    "dark": "#212529",
    "border": "#495057"
}

# Custom CSS for improved UI
custom_css = {
    "background": COLOR_SCHEME["bg_dark"],
    "card": {
        "background-color": COLOR_SCHEME["card_bg"],
        "border-radius": "8px",
        "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "margin-bottom": "1rem",
        "border": f"1px solid {COLOR_SCHEME['border']}"
    },
    "card_alt": {
        "background-color": COLOR_SCHEME["card_bg_alt"],
        "border-radius": "8px",
        "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "margin-bottom": "1rem",
        "border": f"1px solid {COLOR_SCHEME['border']}"
    },
    "header": {
        "background-color": "rgba(0, 0, 0, 0.2)",
        "border-bottom": f"1px solid {COLOR_SCHEME['border']}",
        "border-radius": "8px 8px 0 0"
    },
    "text": {
        "color": COLOR_SCHEME["text_light"]
    },
    "muted": {
        "color": COLOR_SCHEME["text_muted"]
    },
    "alert": {
        "position": "fixed",
        "top": "20px",
        "right": "20px",
        "z-index": 1050,
        "min-width": "300px"
    }
}

# ============ Global Variables ============
smart_api = None
broker_connected = False
broker_error_message = None
broker_connection_retry_time = None
dashboard_initialized = False
data_thread_started = False
lock = threading.Lock()
price_history_lock = threading.Lock()
last_historical_refresh = None
last_historical_date = None
last_option_selection_update = datetime.now() - timedelta(seconds=OPTION_AUTO_SELECT_INTERVAL)
last_pcr_update = datetime.now() - timedelta(seconds=PCR_UPDATE_INTERVAL)
last_news_update = datetime.now() - timedelta(seconds=NEWS_CHECK_INTERVAL)
last_cleanup = datetime.now()
last_connection_time = None
tokens_and_symbols_df = None  # DataFrame to store CSV data

# Thread locks for data structures
stocks_data_lock = threading.Lock()
options_data_lock = threading.Lock()
trading_state_lock = threading.Lock()
pcr_data_lock = threading.Lock()
ui_data_store_lock = threading.Lock()

# Add value store for smoother UI transitions
value_store = {
    'stocks': {},
    'options': {},
    'trades': {},
    'sentiment': {},
    'performance': {
        'total_pnl': 0,
        'daily_pnl': 0,
        'win_rate': 0,
        'trades_today': 0
    },
    'last_values': {}  # Store previous values for comparison
}

# Default stocks to show if no stocks are manually added
DEFAULT_STOCKS = [
    {"symbol": "NIFTY", "token": "26000", "exchange": "NSE", "type": "INDEX"},
    {"symbol": "BANKNIFTY", "token": "26009", "exchange": "NSE", "type": "INDEX"},
    {"symbol": "RELIANCE", "token": "2885", "exchange": "NSE", "type": "STOCK"},
    {"symbol": "LTIM", "token": "17818", "exchange": "NSE", "type": "STOCK"}
]

# Timestamps to track last data updates
last_data_update = {
    "stocks": {},
    "options": {},
    "websocket": None
}

# ============ Dynamic Data Structures ============
# Central repository for all stocks data
stocks_data = {}

# Central repository for all options data
options_data = {}

# Volatility tracking for all stocks
volatility_data = {}

# Market sentiment tracking
market_sentiment = {"overall": "NEUTRAL"}

# PCR data for all symbols
pcr_data = {}

# News data storage
news_data = {
    "items": [],  # List of news items
    "last_updated": None,  # Last time news was updated
    "mentions": {},  # Stock mentions in news
    "trading_signals": []  # Trading signals derived from news
}

# Data storage for UI
ui_data_store = {
    'stocks': {},
    'options': {},
    'pcr': {},
    'trades': {},
    'predicted_strategies': {},  # Added for strategy predictions
    'news': {},  # Added for news data
    'search_results': {}  # Added for search functionality
}

# ============ Trading State Class ============
class TradingState:
    def __init__(self):
        self.active_trades = {}
        self.entry_price = {}
        self.entry_time = {}
        self.stop_loss = {}
        self.initial_stop_loss = {}
        self.target = {}
        self.trailing_sl_activated = {}
        self.pnl = {}
        self.strategy_type = {}
        self.trade_source = {}  # Added to track source of trade (e.g., "NEWS", "TECHNICAL")
        self.total_pnl = 0
        self.daily_pnl = 0
        self.trades_history = []
        self.trades_today = 0
        self.trading_day = datetime.now().date()
        self.stock_entry_price = {}
        self.quantity = {}
        self.capital = 100000  # Initial capital
        self.wins = 0
        self.losses = 0
        
    def add_option(self, option_key):
        """Initialize tracking for a new option"""
        if option_key not in self.active_trades:
            self.active_trades[option_key] = False
            self.entry_price[option_key] = None
            self.entry_time[option_key] = None
            self.stop_loss[option_key] = None
            self.initial_stop_loss[option_key] = None
            self.target[option_key] = None
            self.trailing_sl_activated[option_key] = False
            self.pnl[option_key] = 0
            self.strategy_type[option_key] = None
            self.trade_source[option_key] = None
            self.stock_entry_price[option_key] = None
            self.quantity[option_key] = 0
            return True
        return False

    def remove_option(self, option_key):
        """Remove an option from trading state tracking"""
        if option_key in self.active_trades and not self.active_trades[option_key]:
            del self.active_trades[option_key]
            del self.entry_price[option_key]
            del self.entry_time[option_key]
            del self.stop_loss[option_key]
            del self.initial_stop_loss[option_key]
            del self.target[option_key]
            del self.trailing_sl_activated[option_key]
            del self.pnl[option_key]
            del self.strategy_type[option_key]
            del self.trade_source[option_key]
            if option_key in self.stock_entry_price:
                del self.stock_entry_price[option_key]
            if option_key in self.quantity:
                del self.quantity[option_key]
            return True
        return False

trading_state = TradingState()

# Class for data caching to reduce API calls
class DataCache:
    """
    Cache for API data to reduce unnecessary API calls
    """
    def __init__(self, max_age_seconds=1):
        self.cache = {}
        self.max_age_seconds = max_age_seconds
        self.lock = threading.Lock()
    
    def get(self, key):
        """Get data from cache if fresh enough"""
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if (datetime.now() - timestamp).total_seconds() < self.max_age_seconds:
                    return data
        return None
    
    def set(self, key, data):
        """Store data in cache with current timestamp"""
        with self.lock:
            self.cache[key] = (data, datetime.now())
    
    def clear(self, key=None):
        """Clear specific key or entire cache"""
        with self.lock:
            if key:
                if key in self.cache:
                    del self.cache[key]
            else:
                self.cache.clear()
    
    def cleanup(self):
        """Remove old entries from cache"""
        current_time = datetime.now()
        with self.lock:
            keys_to_remove = []
            for key, (data, timestamp) in self.cache.items():
                if (current_time - timestamp).total_seconds() > self.max_age_seconds * 2:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]

# Create global data cache
api_data_cache = DataCache(max_age_seconds=1)  # 1-second cache for real-time data

# Class for error tracking
class ErrorTracker:
    """
    Track and manage errors for different API endpoints
    """
    def __init__(self, max_errors=5, reset_period=300):
        """
        Initialize error tracker
        
        Args:
            max_errors (int): Maximum number of errors before taking action
            reset_period (int): Period in seconds after which error count resets
        """
        self.max_errors = max_errors
        self.reset_period = reset_period
        self.error_counts = {}
        self.last_errors = {}
        self.first_error_times = {}
        self.lock = threading.Lock()
    
    def record_error(self, endpoint):
        """
        Record an error for a specific endpoint
        
        Args:
            endpoint (str): The API endpoint or operation name
            
        Returns:
            bool: True if max errors exceeded, False otherwise
        """
        with self.lock:
            current_time = datetime.now()
            
            # Record first error time
            if endpoint not in self.first_error_times:
                self.first_error_times[endpoint] = current_time
            
            # Check if we should reset the error count
            if endpoint in self.first_error_times:
                time_since_first_error = (current_time - self.first_error_times[endpoint]).total_seconds()
                if time_since_first_error > self.reset_period:
                    # Reset the error count after the reset period
                    self.error_counts[endpoint] = 1
                    self.first_error_times[endpoint] = current_time
                else:
                    # Increment the error count
                    self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
            else:
                self.error_counts[endpoint] = 1
            
            # Record the current time as the last error time
            self.last_errors[endpoint] = current_time
            
            # Check if max errors exceeded
            return self.error_counts.get(endpoint, 0) >= self.max_errors
    
    def get_error_count(self, endpoint):
        """Get current error count for an endpoint"""
        with self.lock:
            return self.error_counts.get(endpoint, 0)
    
    def reset(self, endpoint=None):
        """
        Reset error count for an endpoint or all endpoints
        
        Args:
            endpoint (str, optional): Specific endpoint to reset. Defaults to None (reset all).
        """
        with self.lock:
            if endpoint:
                self.error_counts.pop(endpoint, None)
                self.first_error_times.pop(endpoint, None)
                self.last_errors.pop(endpoint, None)
            else:
                self.error_counts.clear()
                self.first_error_times.clear()
                self.last_errors.clear()
    
    def should_skip_endpoint(self, endpoint):
        """
        Check if an endpoint should be skipped due to excessive errors
        
        Args:
            endpoint (str): The API endpoint or operation name
            
        Returns:
            bool: True if endpoint should be skipped, False otherwise
        """
        with self.lock:
            if endpoint not in self.error_counts:
                return False
                
            # Skip if max errors exceeded and last error was recent
            current_time = datetime.now()
            last_error_time = self.last_errors.get(endpoint)
            
            if (self.error_counts.get(endpoint, 0) >= self.max_errors and 
                last_error_time and 
                (current_time - last_error_time).total_seconds() < 60):  # Skip for at least 1 minute
                return True
            
            return False

# Create global error tracker
api_error_tracker = ErrorTracker(max_errors=3, reset_period=300)  # 3 errors in 5 minutes

# Class for implementing exponential backoff
class ExponentialBackoff:
    """
    Implements exponential backoff for connection retries
    """
    def __init__(self, initial_delay=1, max_delay=300, factor=2, jitter=0.1):
        """
        Initialize exponential backoff parameters
        
        Args:
            initial_delay (int): Initial delay in seconds
            max_delay (int): Maximum delay in seconds
            factor (int): Multiplicative factor for backoff
            jitter (float): Random jitter factor (0-1)
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self.attempt = 0
        self.reset()
    
    def reset(self):
        """Reset backoff to initial state"""
        self.attempt = 0
        self.next_retry = None
    
    def next(self):
        """Calculate next backoff delay"""
        self.attempt += 1
        delay = min(self.max_delay, self.initial_delay * (self.factor ** (self.attempt - 1)))
        jitter_amount = random.uniform(0, delay * self.jitter)
        total_delay = delay + jitter_amount
        self.next_retry = datetime.now() + timedelta(seconds=total_delay)
        return total_delay
    
    def should_retry(self):
        """Check if we should retry now based on backoff"""
        if self.next_retry is None:
            return True
        return datetime.now() >= self.next_retry

# Create global backoff handler
connection_backoff = ExponentialBackoff(
    initial_delay=1,  # Start with 1 second
    max_delay=300,    # Cap at 5 minutes
    factor=2,         # Double each time
    jitter=0.1        # Add 0-10% random jitter
)

# ============ CSV Handling Functions ============
def load_tokens_and_symbols_from_csv():
    """Load token and symbol mapping from CSV file"""
    global tokens_and_symbols_df
    try:
        csv_path = r"C:\Users\madhu\Pictures\ubuntu\stocks_and_options.csv"
        logger.info(f"Loading tokens and symbols from {csv_path}")
        tokens_and_symbols_df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(tokens_and_symbols_df)} rows from CSV")
        return tokens_and_symbols_df
    except Exception as e:
        logger.error(f"Error loading tokens and symbols from CSV: {e}", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on error

def search_stock_in_csv(symbol):
    """
    Search for a stock in the CSV file by symbol
    
    Args:
        symbol (str): Stock symbol to search for
        
    Returns:
        dict: Stock data if found, None otherwise
    """
    try:
        global tokens_and_symbols_df
        
        # If DataFrame not loaded yet, load it
        if tokens_and_symbols_df is None:
            tokens_and_symbols_df = load_tokens_and_symbols_from_csv()
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data not loaded")
            return None
        
        # Filter to NSE segment only
        stocks = tokens_and_symbols_df[tokens_and_symbols_df['exch_seg'] == 'NSE'].copy()
        
        # Search for exact symbol match (case insensitive)
        symbol = symbol.upper()
        matches = stocks[stocks['symbol'].str.upper() == symbol]
        
        if not matches.empty:
            logger.info(f"Found stock {symbol} in CSV")
            # Convert to dictionary for easier handling
            return matches.iloc[0].to_dict()
        
        # If not found by symbol, try searching by name
        matches = stocks[stocks['name'].str.upper() == symbol]
        
        if not matches.empty:
            logger.info(f"Found stock {symbol} by name in CSV")
            return matches.iloc[0].to_dict()
            
        logger.warning(f"Stock {symbol} not found in CSV")
        return None
        
    except Exception as e:
        logger.error(f"Error searching for stock in CSV: {e}", exc_info=True)
        return None

def search_stocks_in_csv(search_term, limit=20):
    """
    Search for stocks in CSV by partial symbol or name match
    
    Args:
        search_term (str): Term to search for
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of matching stock dictionaries
    """
    try:
        global tokens_and_symbols_df
        
        # If DataFrame not loaded yet, load it
        if tokens_and_symbols_df is None:
            tokens_and_symbols_df = load_tokens_and_symbols_from_csv()
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data not loaded")
            return []
        
        # Filter to NSE segment only
        stocks = tokens_and_symbols_df[tokens_and_symbols_df['exch_seg'] == 'NSE'].copy()
        
        # Search by symbol and name (case insensitive)
        search_term = search_term.upper()
        
        # First try exact match
        exact_symbol_matches = stocks[stocks['symbol'].str.upper() == search_term]
        exact_name_matches = stocks[stocks['name'].str.upper() == search_term]
        
        # Then try contains match
        symbol_matches = stocks[stocks['symbol'].str.upper().str.contains(search_term)]
        name_matches = stocks[stocks['name'].str.upper().str.contains(search_term)]
        
        # Combine results, prioritizing exact matches
        combined = pd.concat([exact_symbol_matches, exact_name_matches, symbol_matches, name_matches])
        # Remove duplicates, keeping first occurrence (which will be from exact matches if present)
        combined = combined.drop_duplicates(subset=['token'])
        
        # Convert to list of dictionaries
        results = []
        for _, row in combined.head(limit).iterrows():
            results.append({
                'token': row.get('token', ''),
                'symbol': row.get('symbol', ''),
                'name': row.get('name', ''),
                'exch_seg': row.get('exch_seg', 'NSE'),
                'type': 'INDEX' if 'NIFTY' in row.get('symbol', '').upper() or 'SENSEX' in row.get('symbol', '').upper() else 'STOCK'
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching for stocks in CSV: {e}", exc_info=True)
        return []

def find_atm_options_in_csv(symbol, current_price):
    """
    Find ATM (At The Money) options in the CSV file based on stock symbol and price
    
    Args:
        symbol (str): Stock/index symbol
        current_price (float): Current price of the stock/index
        
    Returns:
        dict: Dictionary with CE and PE options
    """
    try:
        global tokens_and_symbols_df
        
        # If DataFrame not loaded yet, load it
        if tokens_and_symbols_df is None:
            tokens_and_symbols_df = load_tokens_and_symbols_from_csv()
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data not loaded")
            return {"CE": None, "PE": None}
        
        # Filter to NFO segment only
        options = tokens_and_symbols_df[tokens_and_symbols_df['exch_seg'] == 'NFO'].copy()
        
        # Filter by the underlying stock/index name
        symbol = symbol.upper()
        filtered_options = options[options['name'].str.upper() == symbol]
        
        if filtered_options.empty:
            logger.warning(f"No options found for {symbol}")
            return {"CE": None, "PE": None}
        
        # Calculate the appropriate strike price interval based on price
        if current_price < 100:
            strike_interval = 5
        elif current_price < 1000:
            strike_interval = 10
        elif current_price < 10000:
            strike_interval = 50
        else:
            strike_interval = 100
            
        # Round to nearest appropriate interval
        atm_strike = round(current_price / strike_interval) * strike_interval
        logger.info(f"Calculated ATM strike price: {atm_strike}")
        
        # Extract expiry dates and sort them (closest first)
        try:
            # Convert expiry strings to datetime objects for proper sorting
            filtered_options['expiry_date'] = pd.to_datetime(filtered_options['expiry'], format='%d-%b-%y')
            # Get unique expiry dates sorted by date
            expiry_dates = sorted(filtered_options['expiry'].unique(), 
                                key=lambda x: pd.to_datetime(x, format='%d-%b-%y'))
        except:
            # If datetime conversion fails, just sort the strings
            expiry_dates = sorted(filtered_options['expiry'].unique())
        
        if len(expiry_dates) == 0:
            logger.warning(f"No expiry dates found for {symbol}")
            return {"CE": None, "PE": None}
            
        # Get the closest expiry date
        closest_expiry = expiry_dates[0]
        logger.info(f"Selected expiry date: {closest_expiry}")
        
        # Filter options by the closest expiry
        expiry_options = filtered_options[filtered_options['expiry'] == closest_expiry]
        
        # Extract CE options
        ce_options = expiry_options[expiry_options['symbol'].str.endswith('CE')]
        
        # Extract PE options
        pe_options = expiry_options[expiry_options['symbol'].str.endswith('PE')]
        
        # Find closest strikes to ATM
        closest_ce = None
        closest_pe = None
        
        if not ce_options.empty:
            # Calculate distance to ATM strike
            ce_options['strike_distance'] = abs(ce_options['strike'] - atm_strike)
            closest_ce_row = ce_options.loc[ce_options['strike_distance'].idxmin()]
            closest_ce = closest_ce_row.to_dict()
            logger.info(f"Found CE option: {closest_ce['symbol']} with token {closest_ce['token']} at strike {closest_ce['strike']}")
        
        if not pe_options.empty:
            # Calculate distance to ATM strike
            pe_options['strike_distance'] = abs(pe_options['strike'] - atm_strike)
            closest_pe_row = pe_options.loc[pe_options['strike_distance'].idxmin()]
            closest_pe = closest_pe_row.to_dict()
            logger.info(f"Found PE option: {closest_pe['symbol']} with token {closest_pe['token']} at strike {closest_pe['strike']}")
        
        return {"CE": closest_ce, "PE": closest_pe}
        
    except Exception as e:
        logger.error(f"Error finding ATM options: {e}", exc_info=True)
        return {"CE": None, "PE": None}

def add_stock_from_csv_data(stock_data):
    """
    Add a stock using data from CSV
    
    Args:
        stock_data (dict): Stock data from CSV
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not stock_data:
        logger.warning("Cannot add stock: Empty stock data")
        return False
        
    # Extract required fields
    symbol = stock_data.get('symbol')
    token = stock_data.get('token')
    exchange = stock_data.get('exch_seg', 'NSE')
    name = stock_data.get('name', '')
    
    if not symbol or not token:
        logger.warning("Cannot add stock: Missing symbol or token")
        return False
    
    # Determine stock type (INDEX or STOCK)
    stock_type = "INDEX" if "NIFTY" in symbol or "SENSEX" in symbol else "STOCK"
    
    # Add the stock using the general add_stock function
    success = add_stock(symbol, token, exchange, stock_type)
    
    if success:
        logger.info(f"Added stock {symbol} from CSV data")
    else:
        logger.warning(f"Failed to add stock {symbol} from CSV data")
    
    return success

def add_option_from_csv_data(option_data, timeout=5):
    """
    Add an option using data from CSV with strict timeout
    
    Args:
        option_data (dict): Option data from CSV
        timeout (int): Maximum seconds to spend on this function
        
    Returns:
        bool: True if successful, False otherwise
    """
    global options_data, stocks_data
    
    deadline = time.time() + timeout
    
    # Quick validation first to avoid unnecessary work
    if not option_data:
        logger.warning("Cannot add option: Empty option data")
        return False
    
    # Extract required fields
    symbol = option_data.get('symbol')
    token = option_data.get('token')
    exchange = option_data.get('exch_seg', 'NFO')
    expiry = option_data.get('expiry')
    strike = option_data.get('strike')
    name = option_data.get('name', '')
    
    if not symbol or not token:
        logger.warning("Cannot add option: Missing symbol or token")
        return False
    
    # Check timeout
    if time.time() > deadline:
        logger.warning(f"Timeout while validating option data for {symbol}")
        return False
    
    # Determine option type (CE or PE)
    if symbol.endswith('CE'):
        option_type = 'CE'
    elif symbol.endswith('PE'):
        option_type = 'PE'
    else:
        logger.warning(f"Cannot determine option type for {symbol}")
        return False
    
    # Determine parent symbol (e.g., NIFTY, BANKNIFTY) - limit the search to major symbols
    parent_symbol = None
    major_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "HDFC", "TCS", "LTIM"]
    
    for potential_parent in major_symbols:
        if potential_parent in name:
            parent_symbol = potential_parent
            break
    
    # If no match in major symbols, try all stocks but with a time limit
    if not parent_symbol:
        with stocks_data_lock:
            for potential_parent in stocks_data.keys():
                # Check timeout frequently
                if time.time() > deadline:
                    logger.warning(f"Timeout while finding parent symbol for {symbol}")
                    return False
                    
                if potential_parent in name:
                    parent_symbol = potential_parent
                    break
    
    if not parent_symbol:
        logger.warning(f"Cannot determine parent symbol for {symbol}")
        return False
    
    # Create a unique option key - simplified to avoid regex issues
    try:
        expiry_date = expiry.replace('-', '')
        # Simple uppercase conversion without regex
        expiry_date = expiry_date.upper()
    except:
        # If conversion fails, use expiry as is
        expiry_date = expiry
        
    # Check timeout
    if time.time() > deadline:
        logger.warning(f"Timeout while formatting expiry date for {symbol}")
        return False
        
    # Convert strike to integer to avoid float string issues
    try:
        strike_str = str(int(float(strike)))
    except:
        strike_str = str(strike)
        
    option_key = f"{parent_symbol}_{expiry_date}_{strike_str}_{option_type}"
    
    # Quick check if option already exists
    with options_data_lock:
        if option_key in options_data:
            logger.info(f"Option {option_key} already exists, skipping add")
            return True
    
    # Create minimal option entry first
    with options_data_lock:
        options_data[option_key] = {
            "symbol": symbol,
            "token": token,
            "exchange": exchange,
            "parent_symbol": parent_symbol,
            "expiry": expiry,
            "strike": strike,
            "option_type": option_type,
            "ltp": None,
            "previous": None,
            "high": None,
            "low": None,
            "open": None,
            "change_percent": 0,
            "movement_pct": 0,
            "signal": 0,
            "strength": 0,
            "trend": "NEUTRAL",
            "price_history": pd.DataFrame(columns=['timestamp', 'price', 'volume', 'open', 'high', 'low']),
            "last_updated": None,
            "is_fallback": False
        }
    
    # Check timeout again
    if time.time() > deadline:
        # Even if we timeout here, the basic option data is created
        logger.warning(f"Timeout after creating option entry for {option_key}")
        return True
    
    # Add to parent stock's options list
    with stocks_data_lock:
        if parent_symbol in stocks_data:
            if option_type not in stocks_data[parent_symbol]["options"]:
                stocks_data[parent_symbol]["options"][option_type] = []
            
            if option_key not in stocks_data[parent_symbol]["options"][option_type]:
                stocks_data[parent_symbol]["options"][option_type].append(option_key)
    
    # Add to trading state - no thread safety needed for initialization
    trading_state.add_option(option_key)
    
    logger.info(f"Added option {option_key} from CSV data")
    return True
def initialize_from_csv():
    """
    Initialize with default stocks only, without loading everything from CSV
    """
    # Initialize global DataFrame for CSV data
    global tokens_and_symbols_df
    tokens_and_symbols_df = None
    
    # Load CSV data
    load_tokens_and_symbols_from_csv()
    
    # Add default stocks
    for stock in DEFAULT_STOCKS:
        add_stock_from_csv_data({
            'token': stock["token"],
            'symbol': stock["symbol"],
            'name': stock["symbol"],  # Use symbol as name for default stocks
            'exch_seg': stock["exchange"]
        })
        
    logger.info(f"Initialization complete with default stocks")

def find_and_add_options(symbol):
    """
    Find and add appropriate options for a stock
    
    Args:
        symbol (str): Stock or index symbol
        
    Returns:
        dict: Dictionary with CE and PE option keys
    """
    # Use the current price to find ATM options from CSV
    if symbol not in stocks_data:
        logger.warning(f"Cannot find options: Stock {symbol} not found")
        return {"CE": None, "PE": None}

    current_price = stocks_data[symbol].get("ltp")
    if current_price is None or current_price <= 0:
        logger.warning(f"Cannot find options: Invalid price for {symbol}")
        return {"CE": None, "PE": None}
    
    # Find ATM options in CSV
    return find_and_add_atm_options(symbol, current_price)

def find_and_add_atm_options(symbol, current_price, timeout=10):
    """
    Find and add ATM options for a stock with strict timeout
    
    Args:
        symbol (str): Stock or index symbol
        current_price (float): Current price of the stock/index
        timeout (int): Maximum seconds to spend on this function
        
    Returns:
        dict: Dictionary with CE and PE option keys
    """
    deadline = time.time() + timeout
    added_options = {"CE": None, "PE": None}
    
    if current_price is None or current_price <= 0:
        logger.error(f"Cannot find options: Invalid price {current_price} for {symbol}")
        return added_options
    
    try:
        # Find ATM options in CSV - allow 1/3 of the total time for this step
        search_timeout = timeout / 3
        logger.info(f"Searching for ATM options for {symbol} at price {current_price}")
        search_deadline = time.time() + search_timeout
        atm_options = find_atm_options_in_csv(symbol, current_price)
        
        # Check if we're past the search deadline
        if time.time() > search_deadline:
            logger.warning(f"ATM option search took too long for {symbol}, may have incomplete results")
        
        # Check overall deadline
        if time.time() > deadline:
            logger.error(f"Timeout finding ATM options for {symbol}")
            return added_options
        
        # Process CE option with timeout
        ce_option_data = atm_options.get("CE")
        if ce_option_data:
            logger.info(f"Adding CE option for {symbol}")
            try:
                # Allow 1/3 of remaining time for CE option
                remaining_time = max(1, (deadline - time.time()) / 2)
                ce_success = add_option_from_csv_data(ce_option_data, timeout=remaining_time)
                
                if ce_success:
                    # Create option key based on CE data
                    try:
                        expiry_date = ce_option_data.get('expiry', '').replace('-', '').upper()
                        strike = str(int(float(ce_option_data.get('strike', 0))))
                        ce_key = f"{symbol}_{expiry_date}_{strike}_CE"
                        
                        # Set as primary CE option 
                        with stocks_data_lock:
                            if symbol in stocks_data:
                                stocks_data[symbol]["primary_ce"] = ce_key
                                logger.info(f"Set {ce_key} as primary CE option for {symbol}")
                        
                        added_options["CE"] = ce_key
                        logger.info(f"Successfully added CE option {ce_key}")
                    except Exception as key_err:
                        logger.error(f"Error setting CE key: {key_err}")
                else:
                    logger.warning(f"Failed to add CE option for {symbol}")
            except Exception as ce_err:
                logger.error(f"Error processing CE option: {ce_err}")
        
        # Check deadline
        if time.time() > deadline:
            logger.error(f"Timeout after processing CE option for {symbol}")
            return added_options
            
        # Process PE option with remaining time
        pe_option_data = atm_options.get("PE")
        if pe_option_data:
            logger.info(f"Adding PE option for {symbol}")
            try:
                # Use remaining time for PE option
                remaining_time = max(1, deadline - time.time())
                pe_success = add_option_from_csv_data(pe_option_data, timeout=remaining_time)
                
                if pe_success:
                    # Create option key based on PE data
                    try:
                        expiry_date = pe_option_data.get('expiry', '').replace('-', '').upper()
                        strike = str(int(float(pe_option_data.get('strike', 0))))
                        pe_key = f"{symbol}_{expiry_date}_{strike}_PE"
                        
                        # Set as primary PE option
                        with stocks_data_lock:
                            if symbol in stocks_data:
                                stocks_data[symbol]["primary_pe"] = pe_key
                                logger.info(f"Set {pe_key} as primary PE option for {symbol}")
                        
                        added_options["PE"] = pe_key
                        logger.info(f"Successfully added PE option {pe_key}")
                    except Exception as key_err:
                        logger.error(f"Error setting PE key: {key_err}")
                else:
                    logger.warning(f"Failed to add PE option for {symbol}")
            except Exception as pe_err:
                logger.error(f"Error processing PE option: {pe_err}")
        
        return added_options
        
    except Exception as e:
        logger.error(f"Error in find_and_add_atm_options for {symbol}: {e}")
        return added_options

def search_and_validate_option_token(parent_symbol, strike_price, option_type, expiry):
    """
    Search and validate option token using CSV data instead of API
    
    Args:
        parent_symbol (str): Parent stock/index symbol
        strike_price (float): Strike price
        option_type (str): Option type (ce/pe)
        expiry (str): Expiry date
    
    Returns:
        dict: Token info if found, None otherwise
    """
    # Use CSV data to search for the option token
    try:
        global tokens_and_symbols_df
        
        # If DataFrame not loaded yet, load it
        if tokens_and_symbols_df is None:
            tokens_and_symbols_df = load_tokens_and_symbols_from_csv()
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data not loaded")
            return None
        
        # Filter to NFO segment only
        options = tokens_and_symbols_df[tokens_and_symbols_df['exch_seg'] == 'NFO'].copy()
        
        # Filter by the underlying stock/index name
        filtered_options = options[options['name'].str.upper().str.contains(parent_symbol.upper())]
        
        if filtered_options.empty:
            logger.warning(f"No options found for {parent_symbol}")
            return None
        
        # Filter by option type
        option_suffix = option_type.upper()
        type_options = filtered_options[filtered_options['symbol'].str.endswith(option_suffix)]
        
        if type_options.empty:
            logger.warning(f"No {option_type.upper()} options found for {parent_symbol}")
            return None
        
        # Find closest strike
        if 'strike' in type_options.columns:
            type_options['strike_diff'] = abs(type_options['strike'] - strike_price)
            closest_strike = type_options.loc[type_options['strike_diff'].idxmin()]
            
            result = {
                'token': closest_strike.get('token'),
                'symbol': closest_strike.get('symbol'),
                'strike': closest_strike.get('strike'),
                'is_fallback': False
            }
            
            logger.info(f"Found matching option: {result['symbol']} with token {result['token']}")
            return result
        
        logger.warning(f"Could not find matching option for {parent_symbol} {strike_price} {option_type}")
        return None
        
    except Exception as e:
        logger.error(f"Error searching for option token: {e}")
        return None

# ============ Stock Management Functions ============
def add_stock(symbol, token=None, exchange="NSE", stock_type="STOCK", with_options=True):
    """Add a new stock to track with improved CSV-based token matching"""
    global stocks_data, volatility_data, market_sentiment, pcr_data

    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    # If already exists, just return
    with stocks_data_lock:
        if symbol in stocks_data:
            return True
    
    # If token is None, try to find the token using CSV
    if token is None:
        # Search in CSV data
        stock_csv_data = search_stock_in_csv(symbol)
        
        if stock_csv_data:
            token = stock_csv_data.get("token")
            exchange = stock_csv_data.get("exch_seg", exchange)
            stock_type = "INDEX" if "NIFTY" in symbol or "SENSEX" in symbol else "STOCK"
        
        # If still no token, use placeholder
        if token is None:
            logger.warning(f"Could not find token for {symbol} in CSV, using placeholder")
            token = str(hash(symbol) % 100000)
    
    # Create new stock entry with thread safety
    with stocks_data_lock:
        stocks_data[symbol] = {
            "token": token,
            "exchange": exchange,
            "symbol": symbol,
            "type": stock_type,
            "ltp": None,
            "previous": None,
            "high": None,
            "low": None,
            "open": None,
            "change_percent": 0,
            "movement_pct": 0,
            "price_history": pd.DataFrame(columns=['timestamp', 'price', 'volume', 'open', 'high', 'low']),
            "support_levels": [],
            "resistance_levels": [],
            "last_sr_update": None,  # Track when S/R was last calculated
            "strategy_enabled": {
                "SCALP": True,
                "SWING": True,
                "MOMENTUM": True,
                "NEWS": True
            },
            "options": {
                "CE": [],
                "PE": []
            },
            "primary_ce": None,
            "primary_pe": None,
            "last_updated": None,
            "data_source": "broker",  # Track the source of data
            "predicted_strategy": None,  # Added to track predicted strategy
            "strategy_confidence": 0,  # Added to track strategy confidence
            "history_fetched": False,  # Flag to indicate if historical data has been fetched
            "last_strategy_update": None,  # Track when strategy was last updated
        }
    
    # Initialize volatility tracking
    with threading.Lock():
        volatility_data[symbol] = {
            "window": deque(maxlen=30),
            "current": 0,
            "historical": []
        }
    
    # Initialize market sentiment
    with threading.Lock():
        market_sentiment[symbol] = "NEUTRAL"
    
    # Initialize PCR data
    with pcr_data_lock:
        pcr_data[symbol] = {
            "current": 1.0,
            "history": deque(maxlen=PCR_HISTORY_LENGTH),
            "trend": "NEUTRAL",
            "strength": 0.0,  # Added PCR strength indicator
            "last_updated": None
        }
    
    # Set last update time to force immediate update
    with threading.Lock():
        last_data_update["stocks"][symbol] = None
    
    logger.info(f"Added new stock: {symbol} ({token}) on {exchange}")
    
    # For NIFTY and BANKNIFTY, load historical data from CSV files
    
    # Immediately fetch price data for the new stock
    # Immediately fetch price data for the new stock
    if broker_connected:
        logger.info(f"Fetching initial price data for {symbol}")
        fetch_stock_data(symbol)
        logger.info(f"Initial price data fetch complete for {symbol}")
    
    # Add options if requested and we have price data
    with stocks_data_lock:
        if with_options and symbol in stocks_data and stocks_data[symbol]["ltp"] is not None:
            # Find and add ATM options
            options = find_and_add_options(symbol)
            logger.info(f"Added options for {symbol}: CE={options.get('CE')}, PE={options.get('PE')}")
    return True

def remove_stock(symbol):
    """Remove a stock and its options from tracking"""
    global stocks_data, options_data, volatility_data, market_sentiment, pcr_data
    
    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    with stocks_data_lock:
        if symbol not in stocks_data:
            return False
    
    # First remove all associated options
    with stocks_data_lock:
        if "options" in stocks_data[symbol]:
            for option_type in ["CE", "PE"]:
                for option_key in stocks_data[symbol]["options"].get(option_type, []):
                    if option_key in options_data:
                        # Exit any active trade on this option
                        if trading_state.active_trades.get(option_key, False):
                            exit_trade(option_key, reason="Stock removed")
                        
                        # Remove option data
                        with options_data_lock:
                            del options_data[option_key]
                        
                        # Clean up trading state
                        trading_state.remove_option(option_key)
    
    # Clean up other data structures
    with threading.Lock():
        if symbol in volatility_data:
            del volatility_data[symbol]
    
    with threading.Lock():
        if symbol in market_sentiment:
            del market_sentiment[symbol]
    
    with pcr_data_lock:
        if symbol in pcr_data:
            del pcr_data[symbol]
    
    with threading.Lock():
        if symbol in last_data_update["stocks"]:
            del last_data_update["stocks"][symbol]
    
    # Remove from UI data stores
    with ui_data_store_lock:
        if symbol in ui_data_store['stocks']:
            del ui_data_store['stocks'][symbol]
        
        if symbol in ui_data_store['options']:
            del ui_data_store['options'][symbol]
        
        if symbol in ui_data_store['predicted_strategies']:
            del ui_data_store['predicted_strategies'][symbol]
    
    # Finally, remove the stock itself
    with stocks_data_lock:
        del stocks_data[symbol]
    
    logger.info(f"Removed stock: {symbol}")
    return True

def fetch_stock_data(symbol):
    """
    Fetch data for a single stock directly with improved error handling and timeouts.
    
    Args:
        symbol (str): Stock symbol to fetch data for
        
    Returns:
        bool: True if successful, False otherwise
    """
    global smart_api, broker_connected, stocks_data
    
    # Validate input
    if not symbol:
        logger.warning("Cannot fetch data: Empty symbol")
        return False
        
    # Check if symbol exists with thread safety
    with stocks_data_lock:
        if symbol not in stocks_data:
            logger.warning(f"Cannot fetch data: Symbol {symbol} not found")
            return False
    
    # Ensure broker connection
    if not broker_connected or smart_api is None:
        logger.warning(f"Cannot fetch data for {symbol}: Not connected to broker")
        return False
    
    # Refresh session if needed
    try:
        refresh_session_if_needed()
    except Exception as refresh_err:
        logger.error(f"Session refresh error for {symbol}: {refresh_err}")
        return False
    
    # Set a timeout for the API call
    MAX_API_TIME = 5  # 5 seconds timeout
    
    try:
        logger.debug(f"Making API call for {symbol}")
        
        # Get stock info with thread safety
        with stocks_data_lock:
            if symbol not in stocks_data:
                return False
                
            stock_info = stocks_data[symbol]
            token = stock_info.get("token")
            exchange = stock_info.get("exchange")
        
        # Validate token and exchange
        if not token or not exchange:
            logger.warning(f"Missing token or exchange for {symbol}")
            return False
        
        # Create a thread-safe way to store the result
        result = {"success": False, "data": None, "error": None}
        
        # Define a function to run the API call in a separate thread with timeout
        def api_call():
            try:
                # Fetch LTP data 
                ltp_resp = smart_api.ltpData(exchange, symbol, token)
                result["data"] = ltp_resp
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
                logger.error(f"API call error for {symbol}: {e}")
        
        # Create and start the thread
        api_thread = threading.Thread(target=api_call)
        api_thread.daemon = True
        api_thread.start()
        
        # Wait for thread to complete with timeout
        api_thread.join(timeout=MAX_API_TIME)
        
        # Check if thread is still alive (timed out)
        if api_thread.is_alive():
            logger.error(f"API call for {symbol} timed out after {MAX_API_TIME} seconds")
            return False
        
        # Check if we got an error
        if not result["success"]:
            logger.error(f"API call for {symbol} failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Process successful response
        ltp_resp = result["data"]
        if isinstance(ltp_resp, dict) and ltp_resp.get("status"):
            data = ltp_resp.get("data", {})
            
            # Safely extract LTP with default
            ltp = float(data.get("ltp", 0) or 0)
            
            # Use safe defaults for other fields
            open_price = float(data.get("open", ltp) or ltp)
            high_price = float(data.get("high", ltp) or ltp)
            low_price = float(data.get("low", ltp) or ltp)
            previous_price = float(data.get("previous", ltp) or ltp)
            
            # Ensure non-zero values
            ltp = max(ltp, 0.01)
            
            # Thread-safe update of stock data
            with stocks_data_lock:
                if symbol not in stocks_data:
                    return False
                    
                previous_ltp = stock_info["ltp"]
                stock_info["ltp"] = ltp
                stock_info["open"] = open_price
                stock_info["high"] = high_price
                stock_info["low"] = low_price
                stock_info["previous"] = previous_price
                stock_info["last_updated"] = datetime.now()
                
                # Calculate movement percentage
                if previous_ltp is not None and previous_ltp > 0:
                    stock_info["movement_pct"] = ((ltp - previous_ltp) / previous_ltp) * 100
                
                # Calculate change percentage
                if open_price > 0:
                    stock_info["change_percent"] = ((ltp - open_price) / open_price) * 100
            
            # Add to price history
            timestamp = pd.Timestamp.now()
            
            new_data = {
                'timestamp': timestamp,
                'price': ltp,
                'volume': data.get("tradingSymbol", 0),
                'open': open_price,
                'high': high_price,
                'low': low_price
            }
            
            # Thread-safe price history update
            with price_history_lock:
                with stocks_data_lock:
                    if symbol not in stocks_data:
                        return False
                        
                    stock_info["price_history"] = pd.concat([
                        stock_info["price_history"], 
                        pd.DataFrame([new_data])
                    ], ignore_index=True)
                    
                    # Limit price history size
                    if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                        stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            logger.info(f"Fetched real LTP for {symbol}: {ltp:.2f}")
            return True
        else:
            logger.warning(f"Failed to fetch data for {symbol}: {ltp_resp}")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False
# Improved function for fetching historical data
def load_historical_data(symbol, period="3mo", force_refresh=False):
    """Load historical data with proper validation and error handling"""
    if symbol not in stocks_data:
        logger.warning(f"Cannot load history for unknown symbol: {symbol}")
        return False
    
    # Skip if data is already loaded and not forced to refresh
    with stocks_data_lock:
        if not force_refresh and stocks_data[symbol].get("history_fetched", False):
            logger.info(f"Historical data already loaded for {symbol}, skipping")
            return True
    
    # Fetch history from Yahoo Finance
    try:
        history_df = fetch_history_from_yahoo(symbol, period)
        
        if history_df is None or history_df.empty:
            logger.warning(f"Failed to fetch history for {symbol}")
            return False
        
        # Verify we have valid price data
        if 'price' not in history_df.columns:
            logger.warning(f"No price column in fetched data for {symbol}")
            return False
            
        valid_prices = history_df['price'].dropna()
        if len(valid_prices) < 30:  # Require at least 30 valid price points for better S/R
            logger.warning(f"Insufficient valid price data for {symbol}: {len(valid_prices)} points")
            return False
            
        # Store the data directly, overwriting any existing history
        with price_history_lock:
            with stocks_data_lock:
                if symbol not in stocks_data:
                    return False
                stocks_data[symbol]["price_history"] = history_df
        
        # Update metadata
        with stocks_data_lock:
            if symbol not in stocks_data:
                return False
            stocks_data[symbol]["history_source"] = "yahoo"
            stocks_data[symbol]["history_fetched"] = True
            stocks_data[symbol]["last_history_fetch"] = datetime.now()
        
        # Force recalculation of support/resistance
        calculate_support_resistance(symbol)
        
        with stocks_data_lock:
            if symbol not in stocks_data:
                return False
            stocks_data[symbol]["last_sr_update"] = datetime.now()
        
        # Also recalculate strategy prediction with new historical data
        predict_strategy_for_stock(symbol)
        
        logger.info(f"Successfully loaded historical data for {symbol}: {len(history_df)} points")
        return True
    except Exception as e:
        logger.error(f"Error loading historical data for {symbol}: {e}", exc_info=True)
        return False

def fetch_history_from_yahoo(symbol, period="3mo"):
    """
    Fetch historical data for a symbol from Yahoo Finance with improved error handling
    """
    try:
        # Map symbols to Yahoo format
        if symbol.upper() in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            if symbol.upper() == "NIFTY":
                yahoo_symbol = "^NSEI"
            elif symbol.upper() == "BANKNIFTY":
                yahoo_symbol = "^NSEBANK"
            elif symbol.upper() == "FINNIFTY":
                yahoo_symbol = "NIFTY_FIN_SERVICE.NS"
            else:
                yahoo_symbol = f"{symbol}.NS"
        else:
            yahoo_symbol = f"{symbol}.NS"
        
        logger.info(f"Fetching history for {symbol} (Yahoo: {yahoo_symbol}) for period {period}")
        
        # Fetch data from Yahoo Finance
        history = yf.download(yahoo_symbol, period=period, progress=False)
        
        if history.empty:
            logger.warning(f"No historical data found for {yahoo_symbol}")
            return None
        
        # Create new DataFrame for standardized format
        df = pd.DataFrame()
        df['timestamp'] = history.index
        
        # Handle multi-index columns (tuples) properly
        if isinstance(history.columns, pd.MultiIndex):
            # Extract data using the first level of column names
            close_cols = [col for col in history.columns if col[0] == 'Close']
            if close_cols:
                df['price'] = history[close_cols[0]]
                
            open_cols = [col for col in history.columns if col[0] == 'Open']
            if open_cols:
                df['open'] = history[open_cols[0]]
                
            high_cols = [col for col in history.columns if col[0] == 'High']
            if high_cols:
                df['high'] = history[high_cols[0]]
                
            low_cols = [col for col in history.columns if col[0] == 'Low']
            if low_cols:
                df['low'] = history[low_cols[0]]
                
            volume_cols = [col for col in history.columns if col[0] == 'Volume']
            if volume_cols:
                df['volume'] = history[volume_cols[0]]
        else:
            # Standard columns (not multi-index)
            if 'Close' in history.columns:
                df['price'] = history['Close']
            if 'Open' in history.columns:
                df['open'] = history['Open']
            if 'High' in history.columns:
                df['high'] = history['High']
            if 'Low' in history.columns:
                df['low'] = history['Low']
            if 'Volume' in history.columns:
                df['volume'] = history['Volume']
        
        # Fill missing values with forward fill
        df = df.fillna(method='ffill')
        
        # Validate the data
        if 'price' in df.columns:
            valid_price_count = df['price'].notna().sum()
            logger.info(f"Successfully fetched {len(df)} historical records for {symbol} with {valid_price_count} valid prices")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching history for {symbol} from Yahoo Finance: {e}", exc_info=True)
        return None

def fetch_option_data(option_key):
    """
    Fetch data for a single option with comprehensive error handling and fallbacks
    
    Args:
        option_key (str): Option key to fetch data for
        
    Returns:
        bool: True if successful, False otherwise
    """
    global options_data, broker_connected, smart_api
    
    with options_data_lock:
        if option_key not in options_data:
            logger.warning(f"Cannot fetch data for unknown option key: {option_key}")
            return False
    
        option_info = options_data[option_key]
    
    # Skip if not connected to broker
    if not broker_connected:
        logger.warning(f"Broker not connected, skipping update for {option_key}")
        return False
    
    # Check if this option was recently updated (within last 5 seconds) and has valid data
    with options_data_lock:
        last_update = option_info.get("last_updated")
        if (last_update and (datetime.now() - last_update).total_seconds() < 5 and 
            option_info.get("ltp") is not None and 
            option_info.get("ltp") > 0 and
            not option_info.get("using_fallback", False)):
            logger.debug(f"Option {option_key} has recent valid data, skipping update")
            return True
    
    try:
        # Extract essential option details
        with options_data_lock:
            exchange = option_info.get("exchange", "NFO")
            token = option_info.get("token")
            symbol = option_info.get("symbol")
            parent_symbol = option_info.get("parent_symbol")
            option_type = option_info.get("option_type", "").lower()
            strike = option_info.get("strike")
            expiry = option_info.get("expiry")
        
        # Skip invalid options
        if not token or not symbol:
            logger.warning(f"Skipping option {option_key}: Missing token or symbol")
            return False
        
        # Use parent stock price and strike price for fallback calculations
        parent_price = None
        strike_price = float(strike) if strike else 0
        
        with stocks_data_lock:
            if parent_symbol in stocks_data:
                parent_price = stocks_data[parent_symbol].get("ltp")
        
        # Track if we're using a fallback price
        using_fallback = False
        fallback_reason = None
        
        try:
            # Attempt to fetch LTP data
            logger.info(f"Fetching LTP for {symbol} (Token: {token}, Exchange: {exchange})")
            
            ltp_resp = smart_api.ltpData(exchange, symbol, token)
            
            # Analyze response with detailed logging
            if isinstance(ltp_resp, dict):
                if not ltp_resp.get("status"):
                    # Handle error cases
                    error_msg = ltp_resp.get("message", "Unknown error")
                    error_code = ltp_resp.get("errorcode", "N/A")
                    logger.warning(f"LTP fetch failed for {symbol}. Error: {error_msg} (Code: {error_code})")
                    
                    # Check specific error types
                    if "token" in error_msg.lower() or "symbol" in error_msg.lower() or error_code == "AB1018":
                        # Token or symbol issue - try to recover
                        logger.info(f"Attempting to refresh token for {symbol}")
                        
                        # Try to search for the symbol directly
                        with options_data_lock:
                            if option_info.get("is_fallback", False):
                                # For fallback tokens, we need a more advanced search
                                new_token_info = search_and_validate_option_token(
                                    parent_symbol, strike_price, option_type, expiry
                                )
                                
                                if new_token_info and new_token_info.get("token") != token:
                                    new_token = new_token_info.get("token")
                                    new_symbol = new_token_info.get("symbol", symbol)
                                    
                                    logger.info(f"Found new token for {symbol}: {new_token}")
                                    
                                    # Update the option info
                                    option_info["token"] = new_token
                                    option_info["symbol"] = new_symbol
                                    option_info["is_fallback"] = new_token_info.get("is_fallback", False)
                                    
                                    # Try again with the new token
                                    try:
                                        new_resp = smart_api.ltpData(exchange, new_symbol, new_token)
                                        if new_resp.get("status"):
                                            ltp_resp = new_resp
                                            logger.info(f"Successfully fetched data with new token")
                                        else:
                                            logger.warning(f"Still failed with new token: {new_resp}")
                                            using_fallback = True
                                            fallback_reason = "Token refresh failed"
                                    except Exception as e:
                                        logger.warning(f"Error using new token: {e}")
                                        using_fallback = True
                                        fallback_reason = "Error with new token"
                                else:
                                    using_fallback = True
                                    fallback_reason = "No new token found"
                            else:
                                # For non-fallback tokens, search for the symbol in CSV
                                using_fallback = True
                                fallback_reason = "Could not find token in CSV"
                    else:
                        # Other types of errors
                        using_fallback = True
                        fallback_reason = error_msg
                    
                    # If we need to use fallback, calculate theoretical price
                    if using_fallback:
                        ltp = None
                        if parent_price and strike_price:
                            # More sophisticated theoretical option pricing
                            time_to_expiry = 7/365  # Default 7 days to expiry
                            volatility = 0.3  # Default volatility estimate (30%)
                            
                            if option_type == "ce":
                                # Simple call option pricing - intrinsic value + time value
                                intrinsic = max(0, parent_price - strike_price)
                                time_value = parent_price * volatility * time_to_expiry
                                ltp = max(intrinsic + time_value, 5)
                            else:  # PE
                                # Simple put option pricing - intrinsic value + time value
                                intrinsic = max(0, strike_price - parent_price)
                                time_value = parent_price * volatility * time_to_expiry
                                ltp = max(intrinsic + time_value, 5)
                                
                            logger.info(f"Using theoretical price for {symbol}: {ltp} (Reason: {fallback_reason})")
                        else:
                            # Use previous price or default
                            with options_data_lock:
                                ltp = option_info.get("ltp", 50)
                            logger.info(f"Using previous/default price for {symbol}: {ltp} (Reason: {fallback_reason})")
                else:
                    # Extract data with validation
                    data = ltp_resp.get("data", {})
                    ltp = float(data.get("ltp", 0) or 0)
                    if ltp <= 0:
                        # Try theoretical price
                        using_fallback = True
                        fallback_reason = "API returned zero/negative price"
                        
                        if parent_price and strike_price:
                            if option_type == "ce":
                                ltp = max(parent_price - strike_price, 5) if parent_price > strike_price else 5
                            else:  # PE
                                ltp = max(strike_price - parent_price, 5) if strike_price > parent_price else 5
                        else:
                            with options_data_lock:
                                ltp = option_info.get("ltp", 50)
                    else:
                        logger.info(f"Valid LTP received for {symbol}: {ltp}")
            else:
                # Handle invalid response format
                logger.error(f"Invalid response format for {symbol}")
                using_fallback = True
                fallback_reason = "Invalid API response format"
                
                if parent_price and strike_price:
                    if option_type == "ce":
                        ltp = max(parent_price - strike_price, 5) if parent_price > strike_price else 5
                    else:  # PE
                        ltp = max(strike_price - parent_price, 5) if strike_price > parent_price else 5
                else:
                    with options_data_lock:
                        ltp = option_info.get("ltp", 50)
            
            # Ensure valid, non-zero LTP
            ltp = max(float(ltp), 0.01)
        
        except Exception as fetch_err:
            # Comprehensive error logging
            logger.error(f"Critical error fetching LTP for {symbol}: {fetch_err}", exc_info=True)
            using_fallback = True
            fallback_reason = str(fetch_err)
            
            # Fallback to parent stock's price
            if parent_symbol and parent_symbol in stocks_data:
                with stocks_data_lock:
                    stock_price = stocks_data[parent_symbol].get("ltp", 50)
                
                with options_data_lock:    
                    strike = float(option_info.get("strike", 0))
                
                # Simple option pricing model as fallback
                if strike > 0:
                    if option_type == "ce":  # Call option
                        ltp = max(stock_price - strike, 5) if stock_price > strike else 5
                    else:  # Put option
                        ltp = max(strike - stock_price, 5) if strike > stock_price else 5
                else:
                    ltp = 50
                logger.warning(f"Using fallback pricing for {option_key}: {ltp}")
            else:
                ltp = 50
                logger.warning(f"No parent stock data for fallback pricing, using default price: {ltp}")
        
        # Final validation to ensure we have a valid price
        ltp = max(float(ltp), 0.01)
        
        # Update option data with safe calculations
        with options_data_lock:
            if option_key not in options_data:
                return False
                
            current_open = option_info.get("open")
            current_high = option_info.get("high")
            current_low = option_info.get("low")
            
            # Update option info
            previous_price = option_info.get("ltp", ltp)  # Store current as previous
            option_info.update({
                "previous": previous_price,
                "ltp": ltp,
                "open": current_open if current_open is not None and current_open > 0 else ltp,
                "high": max(current_high if current_high is not None else 0, ltp),
                "low": min(current_low if current_low is not None and current_low > 0 else float('inf'), ltp),
                "change_percent": ((ltp / previous_price) - 1) * 100 if previous_price and previous_price > 0 else 0,
                "last_updated": datetime.now(),
                "using_fallback": using_fallback,
                "fallback_reason": fallback_reason if using_fallback else None
            })
        
        # Add to price history
        timestamp = pd.Timestamp.now()
        
        new_data = {
            'timestamp': timestamp,
            'price': ltp,
            'volume': 0,
            'open_interest': 0,
            'change': option_info.get("change_percent", 0),
            'open': option_info["open"],
            'high': option_info["high"],
            'low': option_info["low"],
            'is_fallback': using_fallback
        }
        
        # Thread-safe price history update
        with price_history_lock:
            with options_data_lock:
                if option_key not in options_data:
                    return False
                    
                # Ensure all columns exist
                for col in new_data.keys():
                    if col not in option_info["price_history"].columns:
                        option_info["price_history"][col] = np.nan
                
                option_info["price_history"] = pd.concat([
                    option_info["price_history"], 
                    pd.DataFrame([new_data])
                ], ignore_index=True)
                
                # Limit history size
                if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                    option_info["price_history"] = option_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
        
        # Generate signals with fallback
        try:
            generate_option_signals(option_key)
        except Exception as signal_err:
            logger.warning(f"Signal generation failed for {option_key}: {signal_err}")
        
        # Update last data update timestamp
        with threading.Lock():
            last_data_update["options"][option_key] = datetime.now()
        
        # Update UI data store for primary options
        with ui_data_store_lock:
            if parent_symbol and option_type:
                # Check if this is the primary option for the stock
                with stocks_data_lock:
                    if parent_symbol in stocks_data:
                        primary_option_key = stocks_data[parent_symbol].get(f"primary_{option_type}")
                        
                        if primary_option_key == option_key:
                            # Ensure the parent symbol exists in UI data store
                            if parent_symbol not in ui_data_store['options']:
                                ui_data_store['options'][parent_symbol] = {}
                            
                            with options_data_lock:
                                # Update UI data
                                ui_data_store['options'][parent_symbol][option_type] = {
                                    'strike': option_info.get("strike", "N/A"),
                                    'symbol': option_info.get("symbol", "N/A"),
                                    'price': ltp,
                                    'signal': option_info.get("signal", 0),
                                    'strength': option_info.get("strength", 0),
                                    'trend': option_info.get("trend", "NEUTRAL"),
                                    'using_fallback': using_fallback
                                }
                            
                            logger.info(f"Updated UI data for {parent_symbol} {option_type} option")
        
        return True
    
    except Exception as main_err:
        logger.error(f"Comprehensive error updating option {option_key}: {main_err}", exc_info=True)
        return False

def update_all_stocks():
    """Update all stocks using bulk API where possible."""
    global stocks_data
    
    # Skip if not connected to broker
    if not broker_connected:
        return
    
    # Get all symbols to update
    with stocks_data_lock:
        symbols_to_update = list(stocks_data.keys())
    
    # Process in batches of BULK_FETCH_SIZE to respect API limits
    for i in range(0, len(symbols_to_update), BULK_FETCH_SIZE):
        batch = symbols_to_update[i:i+BULK_FETCH_SIZE]
        
        # Fetch data for this batch
        bulk_results = fetch_bulk_stock_data(batch)
        
        # Update each stock with the fetched data
        for symbol, data in bulk_results.items():
            with stocks_data_lock:
                if symbol in stocks_data and data:
                    stock_info = stocks_data[symbol]
                    
                    # Update with fetched data
                    previous_ltp = stock_info["ltp"]
                    stock_info["ltp"] = data.get("ltp")
                    stock_info["open"] = data.get("open")
                    stock_info["high"] = data.get("high")
                    stock_info["low"] = data.get("low")
                    stock_info["previous"] = data.get("previous")
                    
                    # Calculate movement percentage
                    if previous_ltp is not None and previous_ltp > 0:
                        stock_info["movement_pct"] = ((data.get("ltp", 0) - previous_ltp) / previous_ltp) * 100
                    
                    # Calculate change percentage
                    if stock_info["open"] and stock_info["open"] > 0:
                        stock_info["change_percent"] = ((data.get("ltp", 0) - stock_info["open"]) / stock_info["open"]) * 100
            
            # Add to price history
            timestamp = pd.Timestamp.now()
            
            new_data = {
                'timestamp': timestamp,
                'price': data.get("ltp", 0),
                'volume': data.get("volume", 0),
                'open': data.get("open", data.get("ltp", 0)),
                'high': data.get("high", data.get("ltp", 0)),
                'low': data.get("low", data.get("ltp", 0))
            }
            
            # Thread-safe price history update
            with price_history_lock:
                with stocks_data_lock:
                    if symbol in stocks_data:
                        stock_info = stocks_data[symbol]
                        stock_info["price_history"] = pd.concat([
                        stock_info["price_history"], 
                            pd.DataFrame([new_data])
                        ], ignore_index=True)
                        
                        # Limit price history size
                        if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                            stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            # Update volatility
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    previous_ltp = stock_info.get("previous")
                    if previous_ltp is not None and previous_ltp > 0:
                        pct_change = (data.get("ltp", 0) - previous_ltp) / previous_ltp * 100
                        update_volatility(symbol, pct_change)
            
            # Update support/resistance levels periodically to save processing
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    if stock_info.get("last_sr_update") is None or \
                       (datetime.now() - stock_info.get("last_sr_update")).total_seconds() > 300:  # Every 5 minutes
                        calculate_support_resistance(symbol)
                        stock_info["last_sr_update"] = datetime.now()
                    
                    # Predict best strategy for this stock periodically
                    if stock_info.get("last_strategy_update") is None or \
                       (datetime.now() - stock_info.get("last_strategy_update")).total_seconds() > 300:  # Every 5 minutes
                        predict_strategy_for_stock(symbol)
                        stock_info["last_strategy_update"] = datetime.now()
                    
                    # Update last update time
                    stock_info["last_updated"] = datetime.now()
            
            # Update UI data store
            with ui_data_store_lock:
                ui_data_store['stocks'][symbol] = {
                    'price': data.get("ltp"),
                    'change': stocks_data[symbol]["change_percent"],
                    'ohlc': {
                        'open': stocks_data[symbol]["open"],
                        'high': stocks_data[symbol]["high"],
                        'low': stocks_data[symbol]["low"],
                        'previous': stocks_data[symbol]["previous"]
                    },
                    'last_updated': datetime.now().strftime('%H:%M:%S')
                }
            
            # Update last data update timestamp
            with threading.Lock():
                last_data_update["stocks"][symbol] = datetime.now()
        
        # Short delay between batch requests to avoid overwhelming the API
        time.sleep(0.2)

def update_all_options():
    """
    Update all options with a more efficient prioritized approach using caching
    """
    global options_data, broker_connected, smart_api, ui_data_store
    
    # Skip if not connected to broker
    if not broker_connected:
        logger.warning("Broker not connected. Skipping options update.")
        return
    
    # Prioritize options to update
    priority_options = []
    regular_options = []
    
    # Create a default datetime for options without last_updated
    default_time = datetime.min
    
    # Thread-safe access to options_data
    with options_data_lock:
        for option_key, option_info in options_data.items():
            parent_symbol = option_info.get("parent_symbol")
            option_type = option_info.get("option_type", "").lower()
            
            # Check if this is a primary option for any stock
            is_primary = False
            with stocks_data_lock:
                if parent_symbol in stocks_data:
                    primary_key = stocks_data[parent_symbol].get(f"primary_{option_type}")
                    if primary_key == option_key:
                        is_primary = True
            
            # Check if this option is in an active trade
            is_active_trade = trading_state.active_trades.get(option_key, False)
            
            # Determine update priority
            if is_primary or is_active_trade:
                priority_options.append(option_key)
            else:
                regular_options.append(option_key)
    
    # Update priority options first
    logger.info(f"Updating {len(priority_options)} priority options")
    for option_key in priority_options:
        fetch_option_data(option_key)
        # Small delay between requests
        time.sleep(0.01)
    
    # Sort regular options by last update time, oldest first 
    # Use a custom key function to handle None values
    def get_update_time(option_key):
        with options_data_lock:
            last_updated = options_data[option_key].get("last_updated")
            return last_updated if last_updated is not None else default_time
    
    sorted_regular_options = sorted(regular_options, key=get_update_time)
    
    # Update a limited number of regular options each cycle
    max_regular_updates = min(len(sorted_regular_options), 10)  # Limit to 10 options per cycle
    if max_regular_updates > 0:
        options_to_update = sorted_regular_options[:max_regular_updates]
        logger.info(f"Updating {len(options_to_update)} of {len(sorted_regular_options)} regular options")
        for option_key in options_to_update:
            fetch_option_data(option_key)
            # Small delay between requests
            time.sleep(0.01)
    
    # Log overall update status
    logger.info(f"Options update completed. {len(priority_options)} priority options, {max_regular_updates} regular options.")
def update_option_selection(force_update=False):
    """
    Update option selection for all stocks based on current prices
    """
    global stocks_data, last_option_selection_update
    
    current_time = datetime.now()
    
    # Only update periodically unless forced
    if not force_update and (current_time - last_option_selection_update).total_seconds() < OPTION_AUTO_SELECT_INTERVAL:
        return
    
    logger.info("Updating option selection based on current stock prices")
    
    # Set overall timeout
    start_time = time.time()
    MAX_FUNCTION_TIME = 30  # Maximum 30 seconds for entire function
    
    # Process each stock
    with stocks_data_lock:
        stock_symbols = list(stocks_data.keys())
    
    for symbol in stock_symbols:
        # Check timeout for entire function
        if time.time() - start_time > MAX_FUNCTION_TIME:
            logger.warning(f"Option selection timeout reached after processing {symbol}")
            break
            
        try:
            with stocks_data_lock:
                if symbol not in stocks_data:
                    continue
                stock_info = stocks_data[symbol]
                current_price = stock_info.get("ltp")
                
                # Skip if we don't have valid price data
                if current_price is None or current_price <= 0:
                    logger.info(f"Skipping option selection for {symbol}: No valid price data yet")
                    continue
                    
                # Check if we need new options
                need_new_options = False
                
                # Check current primary options
                primary_ce = stock_info.get("primary_ce")
                primary_pe = stock_info.get("primary_pe")
                
                if primary_ce is None or primary_pe is None:
                    need_new_options = True
                    logger.info(f"Missing primary options for {symbol}")
                elif primary_ce in options_data and primary_pe in options_data:
                    # Check if current options are still ATM
                    with options_data_lock:
                        ce_strike = float(options_data[primary_ce].get("strike", 0))
                        pe_strike = float(options_data[primary_pe].get("strike", 0))
                    
                    # Calculate appropriate strike interval
                    if current_price < 100:
                        strike_interval = 5
                    elif current_price < 1000:
                        strike_interval = 10
                    elif current_price < 10000:
                        strike_interval = 50
                    else:
                        strike_interval = 100
                    
                    # Check if strikes are still close to current price
                    ce_distance = abs(current_price - ce_strike)
                    pe_distance = abs(current_price - pe_strike)
                    
                    if ce_distance > strike_interval * 2 or pe_distance > strike_interval * 2:
                        need_new_options = True
                        logger.info(f"Options for {symbol} too far from current price (CE: {ce_distance:.2f}, PE: {pe_distance:.2f})")
                    
                    # Check if options are using fallback data
                    with options_data_lock:
                        if any(options_data[opt].get("using_fallback", False) for opt in [primary_ce, primary_pe]):
                            need_new_options = True
                            logger.info(f"Options for {symbol} are using fallback data, finding better options")
                else:
                    need_new_options = True
                    logger.info(f"Options data missing for {symbol}")
            
            # Get new options if needed
            if need_new_options:
                with stocks_data_lock:
                    if symbol in stocks_data:
                        current_price = stocks_data[symbol].get("ltp")
                        if current_price is not None and current_price > 0:
                            logger.info(f"Finding ATM options for {symbol} at price {current_price}")
                            find_and_add_atm_options(symbol, current_price)
                        else:
                            logger.info(f"Cannot find options for {symbol}: Price data invalid or missing")
        except Exception as e:
            logger.error(f"Error processing options for {symbol}: {e}")
            # Continue to next stock
    
    # Update timestamp
    last_option_selection_update = current_time
    logger.info("Option selection update completed")
def fetch_bulk_stock_data(symbols):
    """
    Fetch data for multiple stocks using the ltpData method.
    
    Args:
        symbols (list): List of stock symbols to fetch data for
    
    Returns:
        dict: Dictionary of fetched stock data
    """
    global smart_api, broker_connected, stocks_data
    
    # Validate input
    if not symbols:
        return {}
    
    # Ensure broker connection
    if not broker_connected or smart_api is None:
        if not connect_to_broker():
            logger.warning("Cannot fetch bulk data: Not connected to broker")
            return {}
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    results = {}
    
    for symbol in symbols:
        try:
            # Validate symbol exists in stocks_data
            with stocks_data_lock:
                if symbol not in stocks_data:
                    continue
                
                # Retrieve stock details
                token = stocks_data[symbol].get("token")
                exchange = stocks_data[symbol].get("exchange")
            
            # Validate token and exchange
            if not token or not exchange:
                logger.warning(f"Missing token or exchange for {symbol}")
                continue
            
            # Fetch LTP data
            ltp_resp = smart_api.ltpData(exchange, symbol, token)
            
            # Process successful response
            if isinstance(ltp_resp, dict) and ltp_resp.get("status"):
                data = ltp_resp.get("data", {})
                
                # Safely extract LTP with default
                ltp = float(data.get("ltp", 0) or 0)
                
                # Use safe defaults for other fields
                open_price = float(data.get("open", ltp) or ltp)
                high_price = float(data.get("high", ltp) or ltp)
                low_price = float(data.get("low", ltp) or ltp)
                previous_price = float(data.get("previous", ltp) or ltp)
                
                # Ensure non-zero values
                ltp = max(ltp, 0.01)
                open_price = max(open_price, 0.01)
                high_price = max(high_price, 0.01)
                low_price = max(low_price, 0.01)
                previous_price = max(previous_price, 0.01)
                
                results[symbol] = {
                    "ltp": ltp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "previous": previous_price,
                    "volume": data.get("tradingSymbol", 0)
                }
                
                logger.info(f"Fetched real LTP for {symbol}: {ltp:.2f}")
            
            # Small delay between requests to avoid rate limiting
            time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    return results

def fetch_bulk_data_comprehensive(symbols):
    """
    Comprehensive bulk data fetching for stocks, options, and PCR
    
    Args:
        symbols (list): List of stock symbols to fetch data for
    
    Returns:
        dict: Comprehensive dictionary of fetched data
    """
    global smart_api, broker_connected, stocks_data, options_data, pcr_data
    
    # Validate input and broker connection
    if not symbols or not broker_connected or smart_api is None:
        logger.warning("Cannot fetch bulk data: Invalid input or not connected")
        return {}
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    results = {
        "stocks": {},
        "options": {},
        "pcr": {}
    }
    
    # Fetch PCR data once for all symbols
    all_pcr_values = fetch_pcr_data()
    
    for symbol in symbols:
        try:
            # Skip if symbol not in stocks data
            with stocks_data_lock:
                if symbol not in stocks_data:
                    continue
                
                stock_info = stocks_data[symbol]
                token = stock_info.get("token")
                exchange = stock_info.get("exchange")
            
            # Validate token and exchange
            if not token or not exchange:
                logger.warning(f"Missing token or exchange for {symbol}")
                continue
            
            # Fetch stock LTP data
            ltp_resp = smart_api.ltpData(exchange, symbol, token)
            
            if isinstance(ltp_resp, dict) and ltp_resp.get("status"):
                data = ltp_resp.get("data", {})
                
                # Safely extract and normalize stock prices
                ltp = max(float(data.get("ltp", 0) or 0), 0.01)
                open_price = max(float(data.get("open", ltp) or ltp), 0.01)
                high_price = max(float(data.get("high", ltp) or ltp), 0.01)
                low_price = max(float(data.get("low", ltp) or ltp), 0.01)
                previous_price = max(float(data.get("previous", ltp) or ltp), 0.01)
                
                results["stocks"][symbol] = {
                    "ltp": ltp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "previous": previous_price,
                    "volume": data.get("tradingSymbol", 0)
                }
                
                # Add PCR data if available
                if symbol in all_pcr_values:
                    pcr_value = all_pcr_values[symbol]
                    results["pcr"][symbol] = {
                        "current": pcr_value,
                        "trend": pcr_data.get(symbol, {}).get("trend", "NEUTRAL"),
                        "strength": pcr_data.get(symbol, {}).get("strength", 0.0)
                    }
                
                # Fetch primary options
                with stocks_data_lock:
                    if symbol in stocks_data:
                        ce_key = stocks_data[symbol].get("primary_ce")
                        pe_key = stocks_data[symbol].get("primary_pe")
                
                for option_key in [ce_key, pe_key]:
                    if option_key and option_key in options_data:
                        with options_data_lock:
                            option_info = options_data[option_key]
                            option_token = option_info.get("token")
                            option_exchange = option_info.get("exchange")
                            option_symbol = option_info.get("symbol")
                        
                        if option_token and option_exchange and option_symbol:
                            try:
                                option_ltp_resp = smart_api.ltpData(option_exchange, option_symbol, option_token)
                                
                                if isinstance(option_ltp_resp, dict) and option_ltp_resp.get("status"):
                                    option_data = option_ltp_resp.get("data", {})
                                    
                                    # Safely extract and normalize option prices
                                    option_ltp = max(float(option_data.get("ltp", 0) or 0), 0.01)
                                    option_open = max(float(option_data.get("open", option_ltp) or option_ltp), 0.01)
                                    option_high = max(float(option_data.get("high", option_ltp) or option_ltp), 0.01)
                                    option_low = max(float(option_data.get("low", option_ltp) or option_ltp), 0.01)
                                    
                                    results["options"][option_key] = {
                                        "ltp": option_ltp,
                                        "open": option_open,
                                        "high": option_high,
                                        "low": option_low,
                                        "symbol": option_symbol,
                                        "option_type": option_info.get("option_type"),
                                        "strike": option_info.get("strike")
                                    }
                            except Exception as option_error:
                                logger.error(f"Error fetching option {option_key}: {option_error}")
            
            # Small delay to respect API limits
            time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    return results

def update_all_data_comprehensive():
    """
    Comprehensive update of all data (stocks, options, PCR) in a single function
    with optimized API calls
    """
    global stocks_data, options_data, pcr_data, broker_connected, smart_api
    
    if not broker_connected or smart_api is None:
        logger.warning("Cannot update data: Not connected to broker")
        return
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    # Get all symbols to update
    with stocks_data_lock:
        symbols_to_update = list(stocks_data.keys())
    
    # Track which stocks have active trades for prioritization
    active_trade_symbols = set()
    for option_key, is_active in trading_state.active_trades.items():
        if is_active and option_key in options_data:
            with options_data_lock:
                parent_symbol = options_data[option_key].get("parent_symbol")
                if parent_symbol:
                    active_trade_symbols.add(parent_symbol)
    
    # Prioritize stocks with active trades
    prioritized_symbols = list(active_trade_symbols) + [s for s in symbols_to_update if s not in active_trade_symbols]
    
    # Process in batches to respect API limits
    for i in range(0, len(prioritized_symbols), BULK_FETCH_SIZE):
        batch = prioritized_symbols[i:i+BULK_FETCH_SIZE]
        
        # Fetch comprehensive data for this batch
        all_data = fetch_bulk_data_comprehensive(batch)
        
        # Process stock data
        for symbol, data in all_data.get("stocks", {}).items():
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    
                    # Update with fetched data
                    previous_ltp = stock_info["ltp"]
                    stock_info["ltp"] = data.get("ltp")
                    stock_info["open"] = data.get("open")
                    stock_info["high"] = data.get("high") 
                    stock_info["low"] = data.get("low")
                    stock_info["previous"] = data.get("previous")
                    
                    # Calculate movement percentage
                    if previous_ltp is not None and previous_ltp > 0:
                        stock_info["movement_pct"] = ((data.get("ltp", 0) - previous_ltp) / previous_ltp) * 100
                    
                    # Calculate change percentage
                    if stock_info["open"] and stock_info["open"] > 0:
                        stock_info["change_percent"] = ((data.get("ltp", 0) - stock_info["open"]) / stock_info["open"]) * 100
            
            # Add to price history
            timestamp = pd.Timestamp.now()
            
            new_data = {
                'timestamp': timestamp,
                'price': data.get("ltp", 0),
                'volume': data.get("volume", 0),
                'open': data.get("open", data.get("ltp", 0)),
                'high': data.get("high", data.get("ltp", 0)),
                'low': data.get("low", data.get("ltp", 0))
            }
            
            # Thread-safe price history update
            with price_history_lock:
                with stocks_data_lock:
                    if symbol in stocks_data:
                        stock_info = stocks_data[symbol]
                        stock_info["price_history"] = pd.concat([
                            stock_info["price_history"], 
                            pd.DataFrame([new_data])
                        ], ignore_index=True)
                        
                        # Limit price history size
                        if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                            stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            # Update volatility
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    if previous_ltp is not None and previous_ltp > 0:
                        pct_change = (data.get("ltp", 0) - previous_ltp) / previous_ltp * 100
                        update_volatility(symbol, pct_change)
            
            # Update last update time
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    stock_info["last_updated"] = datetime.now()
            
            # Update UI data store
            with ui_data_store_lock:
                ui_data_store['stocks'][symbol] = {
                    'price': data.get("ltp"),
                    'change': stocks_data[symbol]["change_percent"],
                    'ohlc': {
                        'open': stocks_data[symbol]["open"],
                        'high': stocks_data[symbol]["high"],
                        'low': stocks_data[symbol]["low"],
                        'previous': stocks_data[symbol]["previous"]
                    },
                    'last_updated': stocks_data[symbol]["last_updated"].strftime('%H:%M:%S')
                }
            
            with threading.Lock():
                last_data_update["stocks"][symbol] = datetime.now()
        
        # Process option data
        for option_key, data in all_data.get("options", {}).items():
            with options_data_lock:
                if option_key in options_data:
                    option_info = options_data[option_key]
                    
                    # Update option data
                    previous_ltp = option_info["ltp"]
                    option_info["ltp"] = data.get("ltp")
                    option_info["open"] = data.get("open")
                    option_info["high"] = data.get("high")
                    option_info["low"] = data.get("low")
                    
                    # Calculate change percentage
                    if previous_ltp is not None and previous_ltp > 0:
                        option_info["change_percent"] = ((data.get("ltp", 0) - previous_ltp) / previous_ltp) * 100
                    
                    # Update last update time
                    option_info["last_updated"] = datetime.now()
            
            # Add to price history
            timestamp = pd.Timestamp.now()
            
            new_data = {
                'timestamp': timestamp,
                'price': data.get("ltp", 0),
                'volume': 0,
                'open_interest': 0,
                'change': option_info.get("change_percent", 0),
                'open': data.get("open"),
                'high': data.get("high"),
                'low': data.get("low"),
                'is_fallback': False
            }
            
            # Thread-safe price history update
            with price_history_lock:
                with options_data_lock:
                    if option_key in options_data:
                        option_info = options_data[option_key]
                        
                        # Ensure all columns exist
                        for col in new_data.keys():
                            if col not in option_info["price_history"].columns:
                                option_info["price_history"][col] = np.nan
                        
                        option_info["price_history"] = pd.concat([
                            option_info["price_history"], 
                            pd.DataFrame([new_data])
                        ], ignore_index=True)
                        
                        # Limit history size
                        if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                            option_info["price_history"] = option_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            # Generate signals
            try:
                generate_option_signals(option_key)
            except Exception as e:
                logger.error(f"Error generating signals for {option_key}: {e}")
            
            # Update last update time
            with threading.Lock():
                last_data_update["options"][option_key] = datetime.now()
        
        # Process PCR data
        for symbol, data in all_data.get("pcr", {}).items():
            with pcr_data_lock:
                if symbol in pcr_data:
                    pcr_data[symbol]["current"] = data.get("current", 1.0)
                    pcr_data[symbol]["history"].append(data.get("current", 1.0))
                    pcr_data[symbol]["trend"] = data.get("trend", "NEUTRAL")
                    pcr_data[symbol]["strength"] = data.get("strength", 0.0)
                    pcr_data[symbol]["last_updated"] = datetime.now()
        
        # Short delay between batch requests
        time.sleep(0.05)

# ============ Technical Indicators ============
def calculate_rsi(data, period=RSI_PERIOD):
    """Calculate RSI technical indicator with improved error handling."""
    try:
        if len(data) < period + 1:
            return 50  # Default neutral RSI when not enough data
        
        # Get price differences
        delta = data.diff().fillna(0)
        
        # Get gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Use SMA for first calculation
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Then use the Wilder's smoothing method
        avg_gain = avg_gain.fillna(0)
        avg_loss = avg_loss.fillna(0)
        
        # Calculate RS
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Get the latest value, handling NaN
        latest_rsi = rsi[-1]
        if pd.isna(latest_rsi):
            return 50
        return latest_rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50  # Return neutral RSI in case of error

def calculate_macd(data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD technical indicator with improved error handling."""
    try:
        if len(data) < slow + signal:
            return 0, 0, 0  # Default neutral MACD when not enough data
        
        # Calculate EMAs
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Return the latest values
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return 0, 0, 0  # Return neutral values in case of error

def calculate_bollinger_bands(data, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
    """Calculate Bollinger Bands technical indicator with improved error handling."""
    try:
        if len(data) < period:
            return data.iloc[-1], data.iloc[-1], data.iloc[-1]  # Default when not enough data
        
        # Calculate middle band (SMA)
        middle_band = data.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Get latest values
        return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        # In case of error, return the latest price for all bands
        latest_price = data.iloc[-1] if len(data) > 0 else 0
        return latest_price, latest_price, latest_price

def calculate_atr(data, high=None, low=None, period=ATR_PERIOD):
    """Calculate Average True Range (ATR) technical indicator with improved error handling."""
    try:
        if len(data) < period + 1:
            return 1.0  # Default ATR when not enough data
        
        # If high and low are provided, use them
        if high is not None and low is not None:
            # True range calculations
            tr1 = high - low
            tr2 = abs(high - data.shift())
            tr3 = abs(low - data.shift())
            
            # Find true range
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            # Use price changes as a proxy for true range
            tr = data.diff().abs()
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        # Return latest value
        latest_atr = atr.iloc[-1]
        return float(latest_atr) if not pd.isna(latest_atr) else 1.0
    
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return 1.0  # Return default ATR in case of error

def calculate_ema(data, span):
    """Calculate Exponential Moving Average (EMA) with improved error handling."""
    try:
        if len(data) < span:
            return data.iloc[-1] if len(data) > 0 else 0  # Default when not enough data
        
        ema = data.ewm(span=span, adjust=False).mean()
        return ema.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return data.iloc[-1] if len(data) > 0 else 0  # Return last value in case of error

def calculate_momentum(data, period=10):
    """Calculate price momentum over a period with improved error handling."""
    try:
        if len(data) < period:
            return 0  # Default when not enough data
        
        # Handle division by zero
        if data.iloc[-period] == 0:
            return 0
        
        # Ensure we have valid data
        if pd.isna(data.iloc[-1]) or pd.isna(data.iloc[-period]):
            return 0
            
        return (data.iloc[-1] - data.iloc[-period]) / data.iloc[-period] * 100
    except Exception as e:
        logger.error(f"Error calculating momentum: {e}")
        return 0  # Return neutral momentum in case of error

def calculate_trend_strength(data, period=20):
    """Calculate trend strength using linear regression slope and r-squared with improved error handling."""
    try:
        if len(data) < period:
            return 0, 0
        
        # Use the last 'period' data points
        y = data.tail(period).values
        x = np.arange(period)
        
        # Handle missing values
        valid_indices = ~np.isnan(y)
        if sum(valid_indices) < 5:  # Need at least 5 valid points
            return 0, 0
            
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        
        # Calculate r-squared (coefficient of determination)
        r_squared = r_value ** 2
        
        # Normalize slope to percentage
        avg_price = np.mean(y_valid)
        if avg_price > 0:
            normalized_slope = (slope * period) / avg_price * 100
        else:
            normalized_slope = 0
        
        return normalized_slope, r_squared
    except Exception as e:
        logger.error(f"Error calculating trend strength: {e}")
        return 0, 0  # Return neutral trend in case of error

def calculate_support_resistance(symbol):
    """
    Calculate support and resistance levels with advanced algorithms and error handling
    """
    with stocks_data_lock:
        if symbol not in stocks_data:
            logger.warning(f"Cannot calculate S/R: Symbol {symbol} not found")
            return False
        
        stock_info = stocks_data[symbol]
    
    try:
        # Get the price history with proper validation
        with stocks_data_lock:
            df = stock_info["price_history"]
        
        if 'price' not in df.columns:
            logger.warning(f"Price column missing in historical data for {symbol}")
            return False
            
        # Get valid price data points (no NaN)
        prices = df['price'].dropna()
        
        if len(prices) < 30:  # Need at least 30 data points for reliable S/R
            logger.warning(f"Not enough valid price data for {symbol} to calculate S/R: {len(prices)} points")
            return False
        
        # 1. Identify local maxima and minima using rolling window approach
        window_size = min(21, len(prices) // 10)  # Adaptive window size
        
        # Find peaks (local maxima)
        price_series = prices.copy()
        peaks_indices = []
        
        for i in range(window_size, len(price_series) - window_size):
            window = price_series[i-window_size:i+window_size+1]
            if price_series[i] == max(window):
                peaks_indices.append(i)
                
        # Find troughs (local minima)
        troughs_indices = []
        for i in range(window_size, len(price_series) - window_size):
            window = price_series[i-window_size:i+window_size+1]
            if price_series[i] == min(window):
                troughs_indices.append(i)
        
        # 2. Cluster peaks and troughs to find significant levels using price-based clustering
        resistance_points = [price_series.iloc[i] for i in peaks_indices]
        support_points = [price_series.iloc[i] for i in troughs_indices]
        
        # Define clustering range based on price magnitude
        with stocks_data_lock:
            current_price = stock_info.get("ltp", price_series.iloc[-1])
        
        if current_price < 100:
            cluster_range = 1.0  # 1 rupee for low-priced stocks
        elif current_price < 500:
            cluster_range = 2.0  # 2 rupees for mid-priced stocks
        elif current_price < 1000:
            cluster_range = 5.0  # 5 rupees for higher-priced stocks
        elif current_price < 5000:
            cluster_range = 10.0  # 10 rupees for high-valued stocks/indices
        else:
            cluster_range = 20.0  # 20 rupees for very high-valued stocks/indices
        
        # Perform clustering for resistance levels
        resistance_clusters = []
        for point in sorted(resistance_points):
            # Check if the point belongs to an existing cluster
            found_cluster = False
            for i, cluster in enumerate(resistance_clusters):
                if abs(point - sum(cluster) / len(cluster)) < cluster_range:
                    resistance_clusters[i].append(point)
                    found_cluster = True
                    break
            
            # If no suitable cluster found, create a new one
            if not found_cluster:
                resistance_clusters.append([point])
        
        # Perform clustering for support levels
        support_clusters = []
        for point in sorted(support_points):
            # Check if the point belongs to an existing cluster
            found_cluster = False
            for i, cluster in enumerate(support_clusters):
                if abs(point - sum(cluster) / len(cluster)) < cluster_range:
                    support_clusters[i].append(point)
                    found_cluster = True
                    break
            
            # If no suitable cluster found, create a new one
            if not found_cluster:
                support_clusters.append([point])
        
        # 3. Calculate the average of each cluster to determine the S/R level
        resistance_levels = [round(sum(cluster) / len(cluster), 2) for cluster in resistance_clusters
                           if len(cluster) >= 2]  # Consider only significant clusters (2+ points)
        
        support_levels = [round(sum(cluster) / len(cluster), 2) for cluster in support_clusters
                        if len(cluster) >= 2]  # Consider only significant clusters (2+ points)
        
        # 4. Sort levels and filter them to focus on the most relevant ones
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)
        
        # Cap the number of levels to avoid information overload
        resistance_levels = resistance_levels[:3]  # Keep only the 3 nearest resistance levels
        support_levels = support_levels[:3]  # Keep only the 3 nearest support levels
        
        # Update stock data with the calculated S/R levels
        with stocks_data_lock:
            if symbol in stocks_data:
                stocks_data[symbol]["support_levels"] = support_levels
                stocks_data[symbol]["resistance_levels"] = resistance_levels
        
        # Update UI data store
        with ui_data_store_lock:
            if 'stocks' not in ui_data_store:
                ui_data_store['stocks'] = {}
            if symbol not in ui_data_store['stocks']:
                ui_data_store['stocks'][symbol] = {}
            
            ui_data_store['stocks'][symbol]['support_levels'] = support_levels
            ui_data_store['stocks'][symbol]['resistance_levels'] = resistance_levels
        
        logger.info(f"Updated S/R for {symbol}: Support={support_levels}, Resistance={resistance_levels}")
        return True
    
    except Exception as e:
        logger.error(f"Error calculating S/R for {symbol}: {e}", exc_info=True)
        
        # Set empty values to avoid cascading errors
        with stocks_data_lock:
            if symbol in stocks_data:
                stocks_data[symbol]["support_levels"] = []
                stocks_data[symbol]["resistance_levels"] = []
        
        return False

# ============ Volatility Calculation ============
def update_volatility(symbol, latest_pct_change):
    """Update the volatility window with the latest percentage change."""
    global volatility_data
    
    # Check if the symbol exists in volatility data
    if symbol not in volatility_data:
        volatility_data[symbol] = {"window": deque(maxlen=30), "current": 0, "historical": []}
    
    # Add the latest percentage change to the window
    volatility_data[symbol]["window"].append(latest_pct_change)
    
    # Calculate current volatility
    if len(volatility_data[symbol]["window"]) >= 5:
        volatility_data[symbol]["current"] = np.std(list(volatility_data[symbol]["window"]))

def calculate_volatility(symbol):
    """Get the current market volatility for the specified symbol."""
    if symbol in volatility_data:
        return volatility_data[symbol]["current"]
    else:
        return 0

# ============ Signal Generation ============
def generate_option_signals(option_key):
    """Generate trading signals for the specified option with improved algorithms and error handling."""
    with options_data_lock:
        if option_key not in options_data:
            return
    
        option_info = options_data[option_key]
        price_history = option_info["price_history"]
    
    # Skip if not enough data
    if len(price_history) <= LONG_WINDOW:
        return
    
    try:
        price_series = price_history['price']
        is_call = option_info["option_type"] == "CE"
        parent_symbol = option_info.get("parent_symbol")
        
        # Calculate multiple technical indicators
        rsi = calculate_rsi(price_series)
        macd_line, signal_line, histogram = calculate_macd(price_series)
        upper_band, middle_band, lower_band = calculate_bollinger_bands(price_series)
        atr = calculate_atr(price_series)
        ema_short = calculate_ema(price_series, EMA_SHORT)
        ema_medium = calculate_ema(price_series, EMA_MEDIUM)
        ema_long = calculate_ema(price_series, EMA_LONG)
        momentum = calculate_momentum(price_series, 5)
        trend_slope, trend_r2 = calculate_trend_strength(price_series, 10)
        
        # Get current price
        current_price = price_series.iloc[-1] if not price_series.empty else 0
        
        # Get PCR and market sentiment for the parent symbol
        pcr_value = 1.0
        pcr_strength = 0.0
        market_trend = "NEUTRAL"
        
        with pcr_data_lock:
            if parent_symbol and parent_symbol in pcr_data:
                pcr_value = pcr_data[parent_symbol].get("current", 1.0)
                pcr_strength = pcr_data[parent_symbol].get("strength", 0.0)
                pcr_trend = pcr_data[parent_symbol].get("trend", "NEUTRAL")
                
                # Convert trend to numeric value
                if pcr_trend == "RISING":
                    market_trend = "BEARISH"  # Rising PCR typically bearish
                elif pcr_trend == "FALLING":
                    market_trend = "BULLISH"  # Falling PCR typically bullish
        
        # Initialize signal components with weightings
        signal_components = {
            "rsi": 0,
            "macd": 0,
            "bollinger": 0,
            "ema": 0,
            "momentum": 0,
            "pcr": 0,
            "trend": 0
        }
        
        # RSI signals (weight: 2.0)
        if rsi < RSI_OVERSOLD:
            signal_components["rsi"] = 2.0  # Oversold - bullish
        elif rsi > RSI_OVERBOUGHT:
            signal_components["rsi"] = -2.0  # Overbought - bearish
        else:
            # Proportional signal between oversold and overbought
            normalized_rsi = (rsi - 50) / (RSI_OVERBOUGHT - 50)
            signal_components["rsi"] = -normalized_rsi * 1.5  # Scaled down for middle range
        
        # MACD signals (weight: 2.5)
        if macd_line > signal_line:
            # Stronger signal if MACD is positive and rising
            strength = 1.0 + (0.5 if macd_line > 0 else 0) + (0.5 if histogram > 0 else 0)
            signal_components["macd"] = 2.5 * strength
        elif macd_line < signal_line:
            # Stronger signal if MACD is negative and falling
            strength = 1.0 + (0.5 if macd_line < 0 else 0) + (0.5 if histogram < 0 else 0)
            signal_components["macd"] = -2.5 * strength
        
        # Bollinger Bands signals (weight: 1.5)
        if current_price <= lower_band:
            signal_components["bollinger"] = 1.5  # Price at or below lower band - bullish
        elif current_price >= upper_band:
            signal_components["bollinger"] = -1.5  # Price at or above upper band - bearish
        else:
            # Position within the bands
            band_range = upper_band - lower_band
            if band_range > 0:
                position = (current_price - lower_band) / band_range
                # Normalize to -1.0 to 1.0 (middle = 0)
                signal_components["bollinger"] = (0.5 - position) * 1.5
        
        # EMA signals (weight: 2.0)
        if ema_short > ema_medium and ema_medium > ema_long:
            signal_components["ema"] = 2.0  # Strong uptrend
        elif ema_short < ema_medium and ema_medium < ema_long:
            signal_components["ema"] = -2.0  # Strong downtrend
        elif ema_short > ema_medium:
            signal_components["ema"] = 1.0  # Weak uptrend
        elif ema_short < ema_medium:
            signal_components["ema"] = -1.0  # Weak downtrend
        
        # Momentum signals (weight: 1.5)
        if momentum > 1.0:
            signal_components["momentum"] = min(momentum / 2, 3.0) * 0.5  # Cap at 3.0
        elif momentum < -1.0:
            signal_components["momentum"] = max(momentum / 2, -3.0) * 0.5  # Cap at -3.0
        
        # PCR signals (weight: 1.0)
        signal_components["pcr"] = pcr_strength * 1.0
        
        # Trend strength (weight: 1.5)
        if trend_r2 > 0.6:  # Only consider strong trends
            signal_components["trend"] = (trend_slope / 2) * 1.5  # Scale trend slope
        
        # Combine all signals
        signal = sum(signal_components.values())
        
        # Adjust signal for option type (invert for puts)
        if not is_call:
            signal = -signal
        
        # Calculate signal strength (0-10 scale)
        strength_components = [abs(val) for val in signal_components.values()]
        signal_strength = min(sum(strength_components) * 0.8, 10)  # Scale and cap at 10
        
        # Save signal and strength
        with options_data_lock:
            if option_key in options_data:
                options_data[option_key]["signal"] = signal
                options_data[option_key]["strength"] = signal_strength
                options_data[option_key]["signal_components"] = signal_components  # Store components for analysis
                
                # Determine trend based on signal and strength
                if signal > 2:
                    options_data[option_key]["trend"] = "BULLISH"
                elif signal > 1:
                    options_data[option_key]["trend"] = "MODERATELY BULLISH"
                elif signal < -2:
                    options_data[option_key]["trend"] = "BEARISH"
                elif signal < -1:
                    options_data[option_key]["trend"] = "MODERATELY BEARISH"
                else:
                    options_data[option_key]["trend"] = "NEUTRAL"
            
        logger.debug(f"Generated signal for {option_key}: {signal:.2f} (strength: {signal_strength:.2f})")
    
    except Exception as e:
        logger.error(f"Error generating option signals for {option_key}: {e}")
        
        # Set defaults in case of error
        with options_data_lock:
            if option_key in options_data:
                options_data[option_key]["signal"] = 0
                options_data[option_key]["strength"] = 0
                options_data[option_key]["trend"] = "NEUTRAL"

# ============ PCR Analysis ============
@rate_limited
def fetch_pcr_data():
    """
    Fetch PCR data - tries to get live data first, falls back to calculation if not available.
    
    Returns:
        dict: Dictionary with symbol as key and PCR value as value.
    """
    global smart_api, broker_connected
    
    pcr_dict = {}
    
    # Check if broker is connected
    if not broker_connected or smart_api is None:
        logger.warning("Cannot fetch PCR data: Not connected to broker")
        return pcr_dict
    
    # Try to fetch live PCR data first
    try:
        logger.info("Attempting to fetch live PCR data from broker API")
        
        # Get the session token from smart_api - depending on the API version, this might be stored differently
        session_token = None
        
        # Try different possible attributes where the token might be stored
        if hasattr(smart_api, 'session_token'):
            session_token = smart_api.session_token
        elif hasattr(smart_api, '_SmartConnect__session_token'):
            session_token = smart_api._SmartConnect__session_token
        elif hasattr(smart_api, 'session'):
            session_token = smart_api.session
        
        if session_token:
            import requests
            
            # Prepare headers using the session token
            headers = {
                'Authorization': f'Bearer {session_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                # API key required in many Angel Broking API calls
                'X-PrivateKey': config.api_key  
            }
            
            # Make the API request
            response = requests.get(
                'https://apiconnect.angelbroking.com/rest/secure/angelbroking/marketData/v1/putCallRatio',
                headers=headers
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                pcr_resp = response.json()
                
                if pcr_resp.get("status"):
                    pcr_data = pcr_resp.get("data", [])
                    
                    for item in pcr_data:
                        trading_symbol = item.get("tradingSymbol", "")
                        pcr_value = item.get("pcr", 1.0)
                        
                        # Extract base symbol from trading symbol (e.g., NIFTY from NIFTY25JAN24FUT)
                        base_symbol_match = re.match(r"([A-Z]+)(?:\d|\w)+", trading_symbol)
                        if base_symbol_match:
                            symbol = base_symbol_match.group(1)
                            pcr_dict[symbol] = pcr_value
                            logger.info(f"Live PCR for {symbol}: {pcr_value:.2f}")
                    
                    logger.info(f"Successfully fetched live PCR data for {len(pcr_dict)} symbols")
                    return pcr_dict  # Return the live data
                else:
                    logger.warning(f"PCR API returned error: {pcr_resp.get('message', 'Unknown error')}")
            else:
                logger.warning(f"PCR API request failed with status {response.status_code}")
                
    except Exception as e:
        logger.warning(f"Failed to fetch live PCR data: {e}")
    
    # If we get here, live PCR data fetch failed - calculate PCR from option data
    logger.info("Falling back to calculated PCR from option data")
    
    # Calculate PCR from option data
    with stocks_data_lock:
        stock_symbols = list(stocks_data.keys())
    
    for symbol in stock_symbols:
        # Get all CE and PE options for this symbol
        ce_options = []
        pe_options = []
        
        with stocks_data_lock:
            if symbol in stocks_data:
                ce_keys = stocks_data[symbol].get("options", {}).get("CE", [])
                pe_keys = stocks_data[symbol].get("options", {}).get("PE", [])
        
        with options_data_lock:
            for key in ce_keys:
                if key in options_data:
                    ce_options.append(options_data[key])
            
            for key in pe_keys:
                if key in options_data:
                    pe_options.append(options_data[key])
        
        # Calculate total open interest or use LTP as proxy
        total_ce_oi = 0
        total_pe_oi = 0
        
        for option in ce_options:
            if option.get("ltp") is not None:
                total_ce_oi += option["ltp"]
        
        for option in pe_options:
            if option.get("ltp") is not None:
                total_pe_oi += option["ltp"]
        
        # Calculate PCR
        if total_ce_oi > 0:
            pcr_value = total_pe_oi / total_ce_oi
        else:
            pcr_value = 1.0  # Default neutral value
        
        # Add to result
        pcr_dict[symbol] = pcr_value
        logger.info(f"Calculated PCR for {symbol}: {pcr_value:.2f}")
    
    # Add common indices with typical values if not already present
    default_pcrs = {
        "NIFTY": 1.04,
        "BANKNIFTY": 0.95,
        "FINNIFTY": 1.02
    }
    
    for idx, val in default_pcrs.items():
        if idx not in pcr_dict:
            pcr_dict[idx] = val
            logger.info(f"Added default PCR for {idx}: {val:.2f}")
    
    return pcr_dict

def determine_pcr_trend(symbol):
    """Determine the trend of PCR based on recent history."""
    with pcr_data_lock:
        if symbol not in pcr_data:
            return
        
        # Only determine trend if we have enough history
        if len(pcr_data[symbol]["history"]) >= PCR_TREND_LOOKBACK:
            recent_pcr = list(pcr_data[symbol]["history"])[-PCR_TREND_LOOKBACK:]
            if recent_pcr[-1] > recent_pcr[0] * 1.05:
                pcr_data[symbol]["trend"] = "RISING"
            elif recent_pcr[-1] < recent_pcr[0] * 0.95:
                pcr_data[symbol]["trend"] = "FALLING"
            else:
                pcr_data[symbol]["trend"] = "NEUTRAL"

def calculate_pcr_strength(symbol):
    """
    Calculate PCR strength based on current value and trend
    
    Args:
        symbol (str): Symbol to calculate PCR strength for
    """
    with pcr_data_lock:
        if symbol not in pcr_data:
            return
        
        pcr_val = pcr_data[symbol]["current"]
        pcr_trend = pcr_data[symbol]["trend"]
        
        # Base strength calculation
        if pcr_val < PCR_BULLISH_THRESHOLD:
            # Bullish indication (lower PCR)
            deviation = (PCR_BULLISH_THRESHOLD - pcr_val) / PCR_BULLISH_THRESHOLD
            strength = min(deviation * 10, 1.0)  # Scale to max 1.0
        elif pcr_val > PCR_BEARISH_THRESHOLD:
            # Bearish indication (higher PCR)
            deviation = (pcr_val - PCR_BEARISH_THRESHOLD) / PCR_BEARISH_THRESHOLD
            strength = -min(deviation * 10, 1.0)  # Negative for bearish, scale to max -1.0
        else:
            # Neutral zone
            mid_point = (PCR_BULLISH_THRESHOLD + PCR_BEARISH_THRESHOLD) / 2
            if pcr_val < mid_point:
                # Slightly bullish
                strength = 0.2
            elif pcr_val > mid_point:
                # Slightly bearish
                strength = -0.2
            else:
                strength = 0
        
        # Enhance strength based on trend
        if pcr_trend == "RISING" and strength < 0:
            # Strengthening bearish signal
            strength *= 1.2
        elif pcr_trend == "FALLING" and strength > 0:
            # Strengthening bullish signal
            strength *= 1.2
        elif pcr_trend == "RISING" and strength > 0:
            # Weakening bullish signal
            strength *= 0.8
        elif pcr_trend == "FALLING" and strength < 0:
            # Weakening bearish signal
            strength *= 0.8
        
        # Update PCR strength
        pcr_data[symbol]["strength"] = strength

def update_pcr_data(symbol):
    """Update PCR data for a symbol using real data when possible, falling back to simulation."""
    global pcr_data
    
    with pcr_data_lock:
        if symbol not in pcr_data:
            pcr_data[symbol] = {
                "current": 1.0,
                "history": deque(maxlen=PCR_HISTORY_LENGTH),
                "trend": "NEUTRAL",
                "strength": 0.0,
                "last_updated": None
            }
    
    # Get real PCR data
    all_pcr_values = fetch_pcr_data()
    
    # Use real PCR if available for this symbol
    if symbol in all_pcr_values:
        pcr = all_pcr_values[symbol]
        
        with pcr_data_lock:
            pcr_data[symbol]["current"] = pcr
            pcr_data[symbol]["history"].append(pcr)
            pcr_data[symbol]["last_updated"] = datetime.now()
        
        logger.info(f"Updated PCR for {symbol} with real data: {pcr:.2f}")
        determine_pcr_trend(symbol)
        calculate_pcr_strength(symbol)
        return True
    
    # Fall back to simulation if real data isn't available
    try:
        # Fetch option chain (can be simulated depending on implementation)
        option_chain = fetch_option_chain(symbol)
        
        if option_chain and len(option_chain) > 0:
            # Calculate PCR from option chain
            total_ce_oi = option_chain[-1].get("totalCEOI", 0)
            total_pe_oi = option_chain[-1].get("totalPEOI", 0)
            
            # Calculate PCR
            if total_ce_oi > 0:
                pcr = total_pe_oi / total_ce_oi
            else:
                pcr = 1.0  # Default neutral value
            
            # Update PCR data
            with pcr_data_lock:
                pcr_data[symbol]["current"] = pcr
                pcr_data[symbol]["history"].append(pcr)
                pcr_data[symbol]["last_updated"] = datetime.now()
            
            # Determine trend and strength
            determine_pcr_trend(symbol)
            calculate_pcr_strength(symbol)
            
            logger.info(f"Updated PCR for {symbol} with simulated data: {pcr:.2f}")
            return True
    except Exception as e:
        logger.error(f"Error updating PCR data for {symbol}: {e}")
    
    # If all else fails, use the previous value
    return False

def update_all_pcr_data():
    """Update PCR data for all stocks more efficiently with timeout protection."""
    global pcr_data, last_pcr_update
    
    current_time = datetime.now()
    if (current_time - last_pcr_update).total_seconds() < PCR_UPDATE_INTERVAL:
        return
    
    # Set deadline for entire function
    start_time = time.time()
    MAX_PCR_TIME = 15  # Maximum 15 seconds for entire PCR update
    
    try:
        # Fetch PCR data for all symbols in a single API call with timeout
        logger.info("Fetching PCR data for all symbols...")
        pcr_fetch_start = time.time()
        PCR_FETCH_TIMEOUT = 5  # 5 seconds max for PCR fetch
        
        # Run the fetch in a separate thread with timeout
        pcr_values = {}
        fetch_completed = [False]
        
        def fetch_pcr_thread():
            try:
                nonlocal pcr_values
                pcr_values = fetch_pcr_data()
                fetch_completed[0] = True
            except Exception as e:
                logger.error(f"Error in PCR fetch thread: {e}")
        
        fetch_thread = threading.Thread(target=fetch_pcr_thread)
        fetch_thread.daemon = True
        fetch_thread.start()
        fetch_thread.join(timeout=PCR_FETCH_TIMEOUT)
        
        if not fetch_completed[0]:
            logger.warning(f"PCR fetch timed out after {PCR_FETCH_TIMEOUT}s, using fallback values")
            # Create fallback PCR values
            with stocks_data_lock:
                pcr_values = {symbol: 1.0 for symbol in stocks_data.keys()}
        
        # Process PCR data with timeout checks
        logger.info(f"Processing PCR data for {len(pcr_values)} symbols...")
        with stocks_data_lock:
            stock_symbols = list(stocks_data.keys())
        
        # Process a limited number of stocks if we're running out of time
        for symbol in stock_symbols:
            # Check overall timeout
            if time.time() - start_time > MAX_PCR_TIME:
                logger.warning(f"PCR update timeout reached after processing {symbol}")
                break
                
            try:
                with pcr_data_lock:
                    if symbol not in pcr_data:
                        pcr_data[symbol] = {
                            "current": 1.0,
                            "history": deque(maxlen=PCR_HISTORY_LENGTH),
                            "trend": "NEUTRAL",
                            "strength": 0.0,
                            "last_updated": None
                        }
                    
                    # Use real PCR if available
                    pcr_value = pcr_values.get(symbol, 1.0)
                    pcr_data[symbol]["current"] = pcr_value
                    pcr_data[symbol]["history"].append(pcr_value)
                    pcr_data[symbol]["last_updated"] = current_time
                
                # Determine trend and strength (quick calculations)
                determine_pcr_trend(symbol)
                calculate_pcr_strength(symbol)
                
                # Skip detailed calculations that might be time-consuming
                
            except Exception as e:
                logger.error(f"Error updating PCR for {symbol}: {e}")
        
        # Update market sentiment quickly
        try:
            if time.time() - start_time <= MAX_PCR_TIME - 1:
                quick_update_market_sentiment()
            else:
                logger.warning("Skipping market sentiment update due to time constraints")
        except Exception as e:
            logger.error(f"Error updating market sentiment: {e}")
        
        # Update timestamp
        last_pcr_update = current_time
        logger.info(f"PCR update completed in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Error in PCR update: {e}")
        return False

def quick_update_market_sentiment():
    """A faster version of market sentiment update."""
    global market_sentiment
    
    # Just calculate a simple average sentiment
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    
    for symbol, sentiment in market_sentiment.items():
        if symbol == "overall":
            continue
            
        if "BULLISH" in sentiment:
            bullish_count += 1
        elif "BEARISH" in sentiment:
            bearish_count += 1
        else:
            neutral_count += 1
    
    total_count = bullish_count + bearish_count + neutral_count
    
    if total_count > 0:
        if bullish_count > bearish_count and bullish_count > neutral_count:
            market_sentiment["overall"] = "BULLISH"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            market_sentiment["overall"] = "BEARISH"
        else:
            market_sentiment["overall"] = "NEUTRAL"
    else:
        market_sentiment["overall"] = "NEUTRAL"
        
    logger.info(f"Quick market sentiment update: {market_sentiment['overall']}")
@rate_limited
def fetch_option_chain(symbol):
    """Generate simulated option chain for PCR calculation with realistic characteristics"""
    try:
        # Get current price for the symbol
        current_price = None
        with stocks_data_lock:
            if symbol in stocks_data:
                current_price = stocks_data[symbol].get("ltp")
        
        if current_price is None:
            current_price = 100  # Default value
        
        # Generate strikes around current price with proper intervals
        if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            # For indices, use 50-point interval
            base_strike = round(current_price / 50) * 50
            strike_interval = 50
            num_strikes = 10  # 5 strikes below, 5 above
        elif current_price > 1000:
            # For high-priced stocks
            base_strike = round(current_price / 20) * 20
            strike_interval = 20
            num_strikes = 10
        else:
            # For regular stocks
            base_strike = round(current_price / 10) * 10
            strike_interval = 10
            num_strikes = 10
        
        strikes = [base_strike + (i - num_strikes//2) * strike_interval for i in range(num_strikes)]
        
        # Generate option chain
        option_chain = []
        total_ce_oi = 0
        total_pe_oi = 0
        
        for strike in strikes:
            # Calculate distance from ATM
            distance_factor = abs(strike - current_price) / current_price
            
            # For realistic OI distribution - higher near ATM
            atm_factor = max(0.5, 1 - distance_factor * 3)
            
            # Slight bias based on whether strike is above or below current price
            if strike > current_price:
                ce_bias = 0.9  # Less CE OI above current price
                pe_bias = 1.1  # More PE OI above current price
            else:
                ce_bias = 1.1  # More CE OI below current price
                pe_bias = 0.9  # Less PE OI below current price
            
            # Generate OI data with randomness
            ce_oi = int(random.random() * 10000 * atm_factor * ce_bias)
            pe_oi = int(random.random() * 10000 * atm_factor * pe_bias)
            
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            
            # Calculate option prices with realistic characteristics
            if strike > current_price:
                # Out of the money CE, in the money PE
                ce_price = max(0.1, (current_price * 0.03) * (1 - distance_factor * 0.8))
                pe_price = max(0.1, strike - current_price + (current_price * 0.02))
            else:
                # In the money CE, out of the money PE
                ce_price = max(0.1, current_price - strike + (current_price * 0.02))
                pe_price = max(0.1, (current_price * 0.03) * (1 - distance_factor * 0.8))
            
            # CE option
            ce_option = {
                "optionType": "CE",
                "strikePrice": strike,
                "openInterest": ce_oi,
                "lastPrice": ce_price
            }
            option_chain.append(ce_option)
            
            # PE option
            pe_option = {
                "optionType": "PE",
                "strikePrice": strike,
                "openInterest": pe_oi,
                "lastPrice": pe_price
            }
            option_chain.append(pe_option)
        
        # Add total OI to the option chain data for easier PCR calculation
        option_chain.append({"totalCEOI": total_ce_oi, "totalPEOI": total_pe_oi})
        
        return option_chain
    except Exception as e:
        logger.error(f"Error generating simulated option chain for {symbol}: {e}")
        return None

def get_pcr_signal(symbol):
    """Generate trading signal based on PCR value and trend."""
    with pcr_data_lock:
        if symbol not in pcr_data:
            return "NEUTRAL", "badge bg-secondary"
        
        pcr_value = pcr_data[symbol]["current"]
        trend = pcr_data[symbol]["trend"]
        
        if pcr_value > PCR_BEARISH_THRESHOLD and trend == "RISING":
            return "BEARISH", "badge bg-danger"
        elif pcr_value < PCR_BULLISH_THRESHOLD and trend == "FALLING":
            return "BULLISH", "badge bg-success"
        else:
            return "NEUTRAL", "badge bg-secondary"

def update_market_sentiment():
    """Update market sentiment based on technical indicators and PCR with improved algorithms."""
    global market_sentiment, stocks_data
    
    # Process each stock individually
    with stocks_data_lock:
        stock_symbols = list(stocks_data.keys())
    
    for symbol in stock_symbols:
        # Skip if not enough data
        with stocks_data_lock:
            if symbol not in stocks_data:
                continue
                
            price_history = stocks_data[symbol]["price_history"]
            
            if len(price_history) < 20:
                continue
                
            price_series = price_history['price']
        
        # Calculate technical indicators
        rsi = calculate_rsi(price_series)
        macd_line, signal_line, histogram = calculate_macd(price_series)
        
        with pcr_data_lock:
            pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
            pcr_strength = pcr_data.get(symbol, {}).get("strength", 0)
        
        trend_slope, trend_r2 = calculate_trend_strength(price_series, 20)
        volatility = calculate_volatility(symbol)
        
        # Initialize sentiment score with more factors and weighted components
        sentiment_components = {
            "rsi": 0,
            "macd": 0,
            "pcr": 0,
            "trend": 0,
            "volatility": 0,
            "price_action": 0
        }
        
        # RSI indicator (weight: 1.0)
        if rsi > 70:
            sentiment_components["rsi"] = -1.0  # Overbought - bearish
        elif rsi < 30:
            sentiment_components["rsi"] = 1.0  # Oversold - bullish
        else:
            # Normalize RSI between 30-70 to -0.5 to 0.5
            normalized_rsi = (rsi - 50) / 20
            sentiment_components["rsi"] = -normalized_rsi * 0.5
            
        # MACD indicator (weight: 1.5)
        macd_shift = pd.Series(histogram).shift(1)
        prev_histogram = macd_shift.iloc[-1] if len(macd_shift) > 0 and not pd.isna(macd_shift.iloc[-1]) else 0
            
        if histogram > 0 and histogram > prev_histogram:
            sentiment_components["macd"] = 1.5  # Positive and increasing - strong bullish
        elif histogram > 0:
            sentiment_components["macd"] = 1.0  # Positive - bullish
        elif histogram < 0 and histogram < prev_histogram:
            sentiment_components["macd"] = -1.5  # Negative and decreasing - strong bearish
        else:
            sentiment_components["macd"] = -1.0  # Negative - bearish
            
        # PCR indicator (weight: 1.0)
        sentiment_components["pcr"] = pcr_strength * 1.0
            
        # Trend indicator (weight: 1.5)
        if trend_r2 > 0.6:  # Only consider strong trends
            sentiment_components["trend"] = (trend_slope / 2) * 1.5
            
        # Volatility indicator (weight: 0.5)
        # High volatility slightly favors bearish sentiment
        if volatility > 1.0:
            sentiment_components["volatility"] = -0.5
        elif volatility < 0.3:
            sentiment_components["volatility"] = 0.3  # Low volatility slightly bullish
            
        # Recent price action (weight: 1.0)
        try:
            if len(price_series) >= 3:
                recent_change = (price_series.iloc[-1] - price_series.iloc[-3]) / price_series.iloc[-3] * 100
                if recent_change > 2:
                    sentiment_components["price_action"] = 1.0  # Strong recent gain - bullish
                elif recent_change < -2:
                    sentiment_components["price_action"] = -1.0  # Strong recent loss - bearish
                else:
                    sentiment_components["price_action"] = recent_change / 2  # Scaled lower weight for smaller changes
        except Exception:
            sentiment_components["price_action"] = 0
        
        # Calculate total sentiment score
        sentiment_score = sum(sentiment_components.values())
        
        # Set sentiment based on score with more gradations
        sentiment = "NEUTRAL"
        if sentiment_score >= 3:
            sentiment = "STRONGLY BULLISH"
        elif sentiment_score >= 1.5:
            sentiment = "BULLISH"
        elif sentiment_score <= -3:
            sentiment = "STRONGLY BEARISH"
        elif sentiment_score <= -1.5:
            sentiment = "BEARISH"
        elif sentiment_score > 0.5:
            sentiment = "MODERATELY BULLISH"
        elif sentiment_score < -0.5:
            sentiment = "MODERATELY BEARISH"
        
        # Update market sentiment
        market_sentiment[symbol] = sentiment
    
    # Calculate overall market sentiment with more weight to major indices
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    total_weight = 0
    
    for symbol, sentiment in market_sentiment.items():
        if symbol == "overall":
            continue
            
        # Assign weight based on symbol importance
        if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            weight = 3.0  # Higher weight for major indices
        elif symbol in ["RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY"]:
            weight = 2.0  # Higher weight for major stocks
        else:
            weight = 1.0  # Standard weight for other stocks
            
        total_weight += weight
        
        # Count weighted sentiment
        if "BULLISH" in sentiment:
            bullish_factor = 2 if "STRONGLY" in sentiment else 1
            bullish_count += weight * bullish_factor
        elif "BEARISH" in sentiment:
            bearish_factor = 2 if "STRONGLY" in sentiment else 1
            bearish_count += weight * bearish_factor
        else:
            neutral_count += weight
    
    # Determine overall sentiment with more emphasis on strong signals
    if total_weight > 0:
        bullish_percentage = bullish_count / total_weight
        bearish_percentage = bearish_count / total_weight
        
        if bullish_percentage > 0.5:
            market_sentiment["overall"] = "BULLISH"
        elif bearish_percentage > 0.5:
            market_sentiment["overall"] = "BEARISH"
        elif bullish_percentage > 0.3 and bullish_percentage > bearish_percentage:
            market_sentiment["overall"] = "MODERATELY BULLISH"
        elif bearish_percentage > 0.3 and bearish_percentage > bullish_percentage:
            market_sentiment["overall"] = "MODERATELY BEARISH"
        else:
            market_sentiment["overall"] = "NEUTRAL"
    else:
        market_sentiment["overall"] = "NEUTRAL"
        
    logger.info(f"Updated market sentiment. Overall: {market_sentiment['overall']}")

# ============ Strategy Prediction ============
def predict_strategy_for_stock(symbol):
    """
    Predict the most suitable trading strategy for a stock based on its 
    characteristics and current market conditions with improved algorithms
    
    Args:
        symbol (str): Stock symbol to predict strategy for
    """
    with stocks_data_lock:
        if symbol not in stocks_data:
            return None
        
        stock_info = stocks_data[symbol]
    
    try:
        # Skip if we don't have enough price history
        with stocks_data_lock:
            price_history = stock_info["price_history"]
            
            if len(price_history) < 30:
                logger.info(f"Not enough price history for {symbol} to predict strategy")
                return None
            
            if 'price' not in price_history.columns:
                logger.warning(f"Price column missing in historical data for {symbol}")
                return None
                
            price_series = price_history['price'].dropna()
        
        if len(price_series) < 30:
            logger.info(f"Not enough valid price points for {symbol} to predict strategy")
            return None
        
        # Ensure price_series is sorted chronologically
        if 'timestamp' in price_history.columns:
            price_series = price_history.sort_values('timestamp')['price'].dropna()
        
        # Calculate key technical metrics with error handling
        try:
            rsi = calculate_rsi(price_series)
            rsi_7 = calculate_rsi(price_series, period=7)  # Short-term RSI
            rsi_21 = calculate_rsi(price_series, period=21)  # Long-term RSI
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            rsi = 50  # Default neutral value
            rsi_7 = 50
            rsi_21 = 50
            
        volatility = calculate_volatility(symbol)
        
        try:
            momentum_short = calculate_momentum(price_series, 5)  # 5-period momentum
            momentum_medium = calculate_momentum(price_series, 10)  # 10-period momentum
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {e}")
            momentum_short = 0  # Default neutral value
            momentum_medium = 0
            
        try:
            atr = calculate_atr(price_series)
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            atr = 1.0  # Default value
            
        try:
            trend_slope, trend_r2 = calculate_trend_strength(price_series, 20)
            trend_slope_short, trend_r2_short = calculate_trend_strength(price_series, 10)
        except Exception as e:
            logger.error(f"Error calculating trend strength for {symbol}: {e}")
            trend_slope, trend_r2 = 0, 0  # Default neutral values
            trend_slope_short, trend_r2_short = 0, 0
        
        # Calculate EMAs
        ema_short = calculate_ema(price_series, EMA_SHORT)
        ema_medium = calculate_ema(price_series, EMA_MEDIUM)
        ema_long = calculate_ema(price_series, EMA_LONG)
        
        # MACD indicator
        macd_line, signal_line, histogram = calculate_macd(price_series)
        
        # Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(price_series)
        
        # Calculate band width (volatility indicator)
        band_width = (upper_band - lower_band) / middle_band
        
        # Get PCR data
        with pcr_data_lock:
            pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
            pcr_trend = pcr_data.get(symbol, {}).get("trend", "NEUTRAL")
            pcr_strength = pcr_data.get(symbol, {}).get("strength", 0.0)
        
        # Get market sentiment
        sentiment = market_sentiment.get(symbol, "NEUTRAL")
        overall_sentiment = market_sentiment.get("overall", "NEUTRAL")
        
        # Check news mentions for this stock
        news_mentions = news_data.get("mentions", {}).get(symbol, [])
        has_recent_news = len(news_mentions) > 0
        
        # ======= IMPROVED STRATEGY SCORING ALGORITHM =======
        # Initialize strategy scores with more comprehensive metrics
        strategy_scores = {
            "SCALP": 0,
            "SWING": 0,
            "MOMENTUM": 0,
            "NEWS": 0
        }
        
        # ===== SCALPING SUITABILITY FACTORS =====
        # 1. Volatility is key for scalping - moderate is best
        if 0.2 < volatility < 0.8:
            strategy_scores["SCALP"] += 2.0  # Ideal volatility range
        elif 0.8 <= volatility < 1.2:
            strategy_scores["SCALP"] += 1.0  # Still good but less optimal
        elif volatility >= 1.2:
            strategy_scores["SCALP"] -= 1.0  # Too volatile for reliable scalping
            
        # 2. Tight Bollinger Bands are good for scalping
        if band_width < 0.03:
            strategy_scores["SCALP"] += 1.5  # Very tight bands
        elif band_width < 0.05:
            strategy_scores["SCALP"] += 1.0  # Tight bands
            
        # 3. Short-term RSI deviations are good for scalping
        if (rsi_7 < 30) or (rsi_7 > 70):
            strategy_scores["SCALP"] += 1.5  # Good for mean reversion scalping
            
        # 4. Price near band edges
        current_price = price_series.iloc[-1]
        if current_price <= lower_band * 1.02 or current_price >= upper_band * 0.98:
            strategy_scores["SCALP"] += 1.0  # Near band edge - potential reversal
            
        # 5. Low trend strength is better for scalping (range-bound)
        if trend_r2 < 0.4:
            strategy_scores["SCALP"] += 1.0  # Low trend strength
        elif trend_r2 > 0.7:
            strategy_scores["SCALP"] -= 1.0  # Too trendy for scalping
            
        # 6. Small recent price moves
        if abs(momentum_short) < 2.0:
            strategy_scores["SCALP"] += 1.0  # Small recent moves
            
        # 7. Higher liquidity stocks are better for scalping
        if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "ICICIBANK", "HDFCBANK", "INFY"]:
            strategy_scores["SCALP"] += 1.0  # Higher liquidity
            
        # ===== SWING TRADING SUITABILITY FACTORS =====
        # 1. Strong trend with good r-squared is ideal for swing
        if trend_r2 > 0.7:
            strategy_scores["SWING"] += 2.0  # Strong clear trend
        elif trend_r2 > 0.5:
            strategy_scores["SWING"] += 1.0  # Moderate trend
            
        # 2. Medium volatility is good for swing
        if 0.5 <= volatility <= 1.5:
            strategy_scores["SWING"] += 1.5
            
        # 3. Trend direction aligned with longer-term indicators
        trend_aligned = False
        if trend_slope > 0 and ema_short > ema_medium and ema_medium > ema_long:
            trend_aligned = True  # Bullish alignment
            strategy_scores["SWING"] += 1.5
        elif trend_slope < 0 and ema_short < ema_medium and ema_medium < ema_long:
            trend_aligned = True  # Bearish alignment
            strategy_scores["SWING"] += 1.5
            
        # 4. RSI approaching extreme levels can signal swing opportunities
        if (rsi_21 < 35 and trend_slope > 0) or (rsi_21 > 65 and trend_slope < 0):
            strategy_scores["SWING"] += 1.5  # Potential reversal points
            
        # 5. PCR indication strong
        if abs(pcr_strength) > 0.5:
            strategy_scores["SWING"] += 1.0
            
        # 6. Market sentiment aligned with trend
        sentiment_aligned = False
        if (sentiment == "BULLISH" and trend_slope > 0) or (sentiment == "BEARISH" and trend_slope < 0):
            sentiment_aligned = True
            strategy_scores["SWING"] += 1.0
            
        # 7. Retracement from trend
        if trend_aligned and ((trend_slope > 0 and momentum_short < 0) or (trend_slope < 0 and momentum_short > 0)):
            strategy_scores["SWING"] += 1.0  # Recent counter-trend move - potential entry
            
        # ===== MOMENTUM TRADING SUITABILITY FACTORS =====
        # 1. Strong recent momentum is essential
        if abs(momentum_short) > 5.0:
            strategy_scores["MOMENTUM"] += 2.5  # Very strong momentum
        elif abs(momentum_short) > 3.0:
            strategy_scores["MOMENTUM"] += 1.5  # Strong momentum
        elif abs(momentum_short) < 1.0:
            strategy_scores["MOMENTUM"] -= 1.0  # Too weak for momentum strategy
            
        # 2. Trend should be strong and recent
        if trend_r2_short > 0.7 and abs(trend_slope_short) > abs(trend_slope):
            strategy_scores["MOMENTUM"] += 2.0  # Strong recent trend
            
        # 3. Volume increasing (proxy with higher volatility)
        if volatility > 0.8:
            strategy_scores["MOMENTUM"] += 1.0  # Higher volatility often means higher volume
            
        # 4. MACD line direction
        macd_series = pd.Series(macd_line)
        macd_diff = macd_series.diff(1)
        macd_direction = macd_diff.iloc[-1] if len(macd_diff) > 0 and not pd.isna(macd_diff.iloc[-1]) else 0
            
        if (macd_line > 0 and macd_direction > 0) or (macd_line < 0 and macd_direction < 0):
            strategy_scores["MOMENTUM"] += 1.0  # MACD trending strongly
            
        # 5. RSI alignment with momentum
        if (momentum_short > 0 and rsi_7 > 55) or (momentum_short < 0 and rsi_7 < 45):
            strategy_scores["MOMENTUM"] += 1.0  # RSI confirms momentum
            
        # 6. Price breaking through Bollinger Bands
        if (current_price > upper_band * 1.01 and momentum_short > 0) or (current_price < lower_band * 0.99 and momentum_short < 0):
            strategy_scores["MOMENTUM"] += 1.5  # Breaking through bands with momentum
            
        # 7. Market conditions favorable for momentum
        if overall_sentiment != "NEUTRAL":
            strategy_scores["MOMENTUM"] += 1.0  # Directional market better for momentum
            
        # ===== NEWS TRADING SUITABILITY FACTORS =====
        # 1. Recent news is essential for news trading
        if has_recent_news:
            strategy_scores["NEWS"] += 3.0  # Recent news mentions
            
            # 2. Increased volatility after news
            if volatility > 0.8:
                strategy_scores["NEWS"] += 1.0
                
            # 3. Stock with higher retail interest better for news trading
            if symbol in ["RELIANCE", "TATAMOTORS", "ICICIBANK", "INFY", "ADANI", "TATA"]:
                strategy_scores["NEWS"] += 1.0
                
            # 4. Strong sentiment in news
            avg_news_sentiment = sum(mention['sentiment'] for mention in news_mentions) / len(news_mentions)
            if abs(avg_news_sentiment) > 0.5:
                strategy_scores["NEWS"] += 1.5  # Strong news sentiment
                
            # 5. Recent price move aligned with news sentiment
            if (avg_news_sentiment > 0 and momentum_short > 2) or (avg_news_sentiment < 0 and momentum_short < -2):
                strategy_scores["NEWS"] += 1.5  # Price already moving with news
                
            # 6. Higher liquidity better for news trading
            if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "ICICIBANK", "HDFCBANK"]:
                strategy_scores["NEWS"] += 1.0  # Higher liquidity
        else:
            # No recent news - news strategy unlikely to be suitable
            strategy_scores["NEWS"] = 0  # Force to zero with no news
        
        # Check if the stock's strategy is enabled in global settings
        for strategy in strategy_scores.keys():
            if not strategy_settings.get(f"{strategy}_ENABLED", True):
                strategy_scores[strategy] *= 0.5  # Reduce score for disabled strategies
        
        # Find the best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        # Calculate confidence based on relative score difference
        strategies_by_score = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        next_best_strategy = strategies_by_score[1][0] if len(strategies_by_score) > 1 else best_strategy
        next_best_score = strategies_by_score[1][1] if len(strategies_by_score) > 1 else 0
        
        if best_score > 0:
            score_difference = best_score - next_best_score
            relative_difference = score_difference / best_score
            confidence = min(0.5 + relative_difference * 0.5, 1.0)  # 0.5-1.0 range
        else:
            confidence = 0  # No good strategy
        
        # Look for strong signals
        threshold = 5.0  # Minimum score to consider the strategy strong enough
        
        if best_score > threshold and confidence > 0.6:
            logger.info(f"Predicted {best_strategy} strategy for {symbol} with {confidence:.2f} confidence (score: {best_score:.1f})")
            
            # Store the prediction
            with stocks_data_lock:
                if symbol in stocks_data:
                    stocks_data[symbol]["predicted_strategy"] = best_strategy
                    stocks_data[symbol]["strategy_confidence"] = confidence
                    stocks_data[symbol]["strategy_score"] = best_score
                    stocks_data[symbol]["last_strategy_update"] = datetime.now()
            
            # Update UI data store
            with ui_data_store_lock:
                if 'predicted_strategies' not in ui_data_store:
                    ui_data_store['predicted_strategies'] = {}
                
                ui_data_store['predicted_strategies'][symbol] = {
                    'strategy': best_strategy,
                    'confidence': confidence,
                    'score': strategy_scores
                }
            
            return best_strategy
        else:
            # Not enough confidence for any strategy
            logger.info(f"No strong strategy signal for {symbol}. Best: {best_strategy} with {confidence:.2f} confidence (score: {best_score:.1f})")
            
            with stocks_data_lock:
                if symbol in stocks_data:
                    stocks_data[symbol]["predicted_strategy"] = None
                    stocks_data[symbol]["strategy_confidence"] = 0
                    stocks_data[symbol]["last_strategy_update"] = datetime.now()
            
            # Update UI data store
            with ui_data_store_lock:
                if 'predicted_strategies' not in ui_data_store:
                    ui_data_store['predicted_strategies'] = {}
                
                ui_data_store['predicted_strategies'][symbol] = {
                    'strategy': "NONE",
                    'confidence': 0,
                    'score': strategy_scores
                }
            
            return None
            
    except Exception as e:
        logger.error(f"Error predicting strategy for {symbol}: {e}", exc_info=True)
        return None

# ============ Trading Functions ============
def should_enter_trade(option_key):
    """Determine if we should enter a trade for the given option with enhanced decision making."""
    with options_data_lock:
        if option_key not in options_data:
            return False, None, 0
        
        option_info = options_data[option_key]
        parent_symbol = option_info["parent_symbol"]
        option_type = option_info["option_type"]
    
    # Don't enter trades if broker is not connected
    if not broker_connected:
        return False, None, 0
    
    # Make sure at least one strategy is enabled
    if not any(strategy_settings.values()):
        return False, None, 0
    
    # Check if we've hit the maximum trades for the day
    if trading_state.trades_today >= MAX_TRADES_PER_DAY:
        return False, None, 0
    
    # Check if we've hit the maximum loss percentage for the day
    if trading_state.daily_pnl <= -MAX_LOSS_PERCENTAGE * trading_state.capital / 100:
        return False, None, 0
    
    # Check if we're already in a trade for this option
    if trading_state.active_trades.get(option_key, False):
        return False, None, 0
    
    # Get signal and strength
    with options_data_lock:
        signal = option_info.get("signal", 0)
        strength = option_info.get("strength", 0)
        
        # Get signal components for more detailed analysis
        signal_components = option_info.get("signal_components", {})
    
    # Check the market sentiment for the parent symbol
    sentiment = market_sentiment.get(parent_symbol, "NEUTRAL")
    overall_sentiment = market_sentiment.get("overall", "NEUTRAL")
    
    # Check PCR data
    with pcr_data_lock:
            pcr_value = pcr_data.get(parent_symbol, {}).get("current", 1.0)
            pcr_strength = pcr_data.get(parent_symbol, {}).get("strength", 0.0)
    
    # Check if there's a predicted strategy for this stock
    with stocks_data_lock:
        if parent_symbol in stocks_data:
            predicted_strategy = stocks_data.get(parent_symbol, {}).get("predicted_strategy")
            strategy_confidence = stocks_data.get(parent_symbol, {}).get("strategy_confidence", 0)
            strategy_score = stocks_data.get(parent_symbol, {}).get("strategy_score", 0)
        else:
            predicted_strategy = None
            strategy_confidence = 0
            strategy_score = 0
    
    # Check for news signals - only use news from last 30 minutes
    current_time = datetime.now()
    news_signals = []
    for s in news_data.get("trading_signals", []):
        if s.get("stock") == parent_symbol and not s.get("executed", False):
            signal_time = s.get("timestamp")
            if isinstance(signal_time, datetime) and (current_time - signal_time).total_seconds() < 1800:  # 30 minutes
                news_signals.append(s)
    
    news_signal = news_signals[0] if news_signals else None
    
    # Default to determining strategy based on signals
    strategy_type = None
    should_enter = False
    entry_confidence = 0
    
    # For a call option, we want positive signal; for a put, we want negative signal
    signal_aligned = (option_type == "CE" and signal > 0) or (option_type == "PE" and signal < 0)
    
    # For a call option, we want bullish sentiment; for a put, we want bearish sentiment
    sentiment_aligned = False
    if "BULLISH" in sentiment and option_type == "CE":
        sentiment_aligned = True
    elif "BEARISH" in sentiment and option_type == "PE":
        sentiment_aligned = True
    
    # For a call option, we want falling PCR; for a put, we want rising PCR
    pcr_aligned = (option_type == "CE" and pcr_strength > 0) or (option_type == "PE" and pcr_strength < 0)
    
    # Check for news-based trade first
    if strategy_settings["NEWS_ENABLED"] and news_signal:
        action = news_signal.get("action", "")
        confidence = news_signal.get("confidence", 0)
        
        if ((action == "BUY_CE" and option_type == "CE") or (action == "BUY_PE" and option_type == "PE")) and confidence > NEWS_CONFIDENCE_THRESHOLD:
            strategy_type = "NEWS"
            should_enter = True
            entry_confidence = confidence
            logger.info(f"News-based {strategy_type} trade signal for {option_key} (confidence: {confidence:.2f})")
            return should_enter, strategy_type, entry_confidence
    
    # Prioritize the predicted strategy if it has good confidence
    if predicted_strategy and strategy_confidence > 0.7 and strategy_settings[f"{predicted_strategy}_ENABLED"]:
        # Check if the signal is aligned with the predicted strategy
        if signal_aligned:
            min_strength = globals()[f"MIN_SIGNAL_STRENGTH_{predicted_strategy}"]
            
            if strength >= min_strength:
                strategy_type = predicted_strategy
                should_enter = True
                entry_confidence = strategy_confidence * (strength / 10)  # Combine strategy and signal confidence
                logger.info(f"Using predicted strategy {strategy_type} for {option_key} (confidence: {entry_confidence:.2f})")
                return should_enter, strategy_type, entry_confidence
    
    # If no predicted strategy or not qualified, fall back to normal strategy selection
    # Check each strategy with more detailed criteria
    
    # Check scalping conditions
    if strategy_settings["SCALP_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_SCALP:
        # Additional scalping-specific criteria
        rsi_component = signal_components.get("rsi", 0)
        bollinger_component = signal_components.get("bollinger", 0)
        
        # Scalping often works better with RSI and Bollinger band signals
        if abs(rsi_component) + abs(bollinger_component) > 1.5:
            strategy_type = "SCALP"
            should_enter = True
            entry_confidence = strength / 10
            logger.info(f"Scalping trade signal for {option_key} (confidence: {entry_confidence:.2f})")
            return should_enter, strategy_type, entry_confidence
    
    # Check momentum conditions
    if strategy_settings["MOMENTUM_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_MOMENTUM:
        # Additional momentum-specific criteria
        momentum_component = signal_components.get("momentum", 0)
        trend_component = signal_components.get("trend", 0)
        
        # Momentum strategies need strong trend and momentum signals
        if abs(momentum_component) + abs(trend_component) > 2.0:
            strategy_type = "MOMENTUM"
            should_enter = True
            entry_confidence = strength / 10
            logger.info(f"Momentum trade signal for {option_key} (confidence: {entry_confidence:.2f})")
            return should_enter, strategy_type, entry_confidence
    
    # Check swing trading conditions
    if strategy_settings["SWING_ENABLED"] and signal_aligned and sentiment_aligned and strength >= MIN_SIGNAL_STRENGTH_SWING:
        # Additional swing-specific criteria
        ema_component = signal_components.get("ema", 0)
        pcr_component = signal_components.get("pcr", 0)
        
        # Swing trades work better with aligned indicators
        swing_factors = signal_aligned + sentiment_aligned + pcr_aligned
        
        if swing_factors >= 2 and abs(ema_component) > 1.0:
            strategy_type = "SWING"
            should_enter = True
            entry_confidence = strength / 10
            logger.info(f"Swing trade signal for {option_key} (confidence: {entry_confidence:.2f})")
            return should_enter, strategy_type, entry_confidence
    
    # No valid strategy found
    return False, None, 0

def enter_trade(option_key, strategy_type, trade_source="TECHNICAL"):
    """Enter a trade for the given option with the specified strategy."""
    global trading_state, options_data, stocks_data
    
    with options_data_lock:
        if option_key not in options_data:
            logger.warning(f"Cannot enter trade: Option {option_key} not found")
            return False
        
        option_info = options_data[option_key]
        current_price = option_info["ltp"]
    
    if current_price is None or current_price <= 0:
        logger.warning(f"Cannot enter trade: Invalid price {current_price}")
        return False
    
    # Get the parent symbol
    parent_symbol = option_info["parent_symbol"]
    
    # Calculate position size based on risk management
    total_capital = trading_state.capital
    risk_amount = total_capital * (RISK_PER_TRADE / 100)
    
    # Use ATR or percentage-based stop loss depending on the strategy
    option_type = option_info["option_type"]
    
    # Adjust strategy-based parameters
    if strategy_type == "SCALP":
        stop_loss_pct = 0.015  # 1.5% for scalping
        target_pct = 0.035     # 3.5% for scalping (increased from 3%)
    elif strategy_type == "MOMENTUM":
        stop_loss_pct = 0.02   # 2% for momentum
        target_pct = 0.06      # 6% for momentum (increased from 5%)
    elif strategy_type == "NEWS":
        stop_loss_pct = 0.025  # 2.5% for news
        target_pct = 0.12      # 12% for news (increased from 10%)
    else:  # SWING
        stop_loss_pct = 0.03   # 3% for swing
        target_pct = 0.10      # 10% for swing (increased from 8%)
    
    # Calculate stop loss amount
    stop_loss_amount = current_price * stop_loss_pct
    
    # Calculate quantity based on risk per trade
    quantity = max(int(risk_amount / stop_loss_amount), 1)
    
    # Cap quantity to avoid too large positions
    max_quantity = int(total_capital * 0.1 / current_price)  # Max 10% of capital
    quantity = min(quantity, max_quantity)
    
    # Calculate actual stop loss and target
    if option_type == "CE":
        stop_loss = current_price * (1 - stop_loss_pct)
        target = current_price * (1 + target_pct)
    else:  # PE
        stop_loss = current_price * (1 + stop_loss_pct)
        target = current_price * (1 - target_pct)
    
    # Update trading state with thread safety
    with trading_state_lock:
        trading_state.active_trades[option_key] = True
        trading_state.entry_price[option_key] = current_price
        trading_state.entry_time[option_key] = datetime.now()
        trading_state.stop_loss[option_key] = stop_loss
        trading_state.initial_stop_loss[option_key] = stop_loss
        trading_state.target[option_key] = target
        trading_state.trailing_sl_activated[option_key] = False
        trading_state.quantity[option_key] = quantity
        trading_state.strategy_type[option_key] = strategy_type
        trading_state.trade_source[option_key] = trade_source
        
        # Store the parent stock price
        with stocks_data_lock:
            if parent_symbol in stocks_data:
                trading_state.stock_entry_price[option_key] = stocks_data[parent_symbol]["ltp"]
        
        # Increment trades today counter
        trading_state.trades_today += 1
    
    logger.info(
        f"Entered {option_key} {strategy_type} trade: "
        f"Price={current_price:.2f}, SL={stop_loss:.2f}, Target={target:.2f}, "
        f"Qty={quantity}, Source={trade_source}"
    )
    return True

def should_exit_trade(option_key):
    """Determine if we should exit a trade for the given option with enhanced exit criteria."""
    with options_data_lock:
        if option_key not in options_data:
            return False, None
    
    with trading_state_lock:
        if not trading_state.active_trades.get(option_key, False):
            return False, None
    
    with options_data_lock:
        option_info = options_data[option_key]
        current_price = option_info["ltp"]
    
    if current_price is None or current_price <= 0:
        return False, None
    
    # Get trading parameters with thread safety
    with trading_state_lock:
        entry_price = trading_state.entry_price[option_key]
        stop_loss = trading_state.stop_loss[option_key]
        target = trading_state.target[option_key]
        option_type = option_info["option_type"]
        strategy_type = trading_state.strategy_type[option_key]
        entry_time = trading_state.entry_time[option_key]
    
    # Calculate current P&L percentage
    if option_type == "CE":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:  # PE
        pnl_pct = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
    
    # Check for stop loss hit
    if (option_type == "CE" and current_price <= stop_loss) or (option_type == "PE" and current_price >= stop_loss):
        return True, "Stop Loss"
    
    # Check for target hit
    if (option_type == "CE" and current_price >= target) or (option_type == "PE" and current_price <= target):
        return True, "Target"
    
    # Check for signal reversal
    with options_data_lock:
        signal = option_info["signal"]
        strength = option_info["strength"]
    
    # Check for strong reversal signals
    if ((option_type == "CE" and signal < -2 and strength > 6) or 
        (option_type == "PE" and signal > 2 and strength > 6)):
        return True, "Strong Signal Reversal"
    
    # Check time-based exit
    current_time = datetime.now()
    holding_time_minutes = (current_time - entry_time).total_seconds() / 60 if entry_time else 0
    
    # Different max holding times based on strategy
    if strategy_type == "SCALP" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_SCALP:
        return True, "Max Holding Time (Scalp)"
    elif strategy_type == "MOMENTUM" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_MOMENTUM:
        return True, "Max Holding Time (Momentum)"
    elif strategy_type == "NEWS" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_NEWS:
        return True, "Max Holding Time (News)"
    elif strategy_type == "SWING" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_SWING:
        return True, "Max Holding Time (Swing)"
    
    # Check for deteriorating profits (peak drawdown)
    # If we were up more than 4% but have lost more than half of those gains
    with trading_state_lock:
        trailing_sl_activated = trading_state.trailing_sl_activated.get(option_key, False)
    
    if pnl_pct > 0 and trailing_sl_activated:
        # Get maximum potential target
        max_target_pct = 10 if strategy_type == "SWING" else 6  # Higher for swing
        
        # If we've reached 75% of the way to max target and are still holding
        if pnl_pct > max_target_pct * 0.75:
            return True, "Taking Profits (Near Max)"
    
    # Check for extended unprofitable trade
    if pnl_pct < -1 and holding_time_minutes > 15:
        # Calculate time-weighted expectation
        time_weight = min(holding_time_minutes / 60, 1.0)  # Scale up to 1 hour
        
        # Time-weighted stop-loss - the longer we're underwater, the tighter the exit
        if pnl_pct < -2 * time_weight:
            return True, "Time-weighted Exit (Not Performing)"
    
    return False, None

def update_dynamic_stop_loss(option_key):
    """
    Update stop loss dynamically based on price action with enhanced algorithm
    that adapts to changing market conditions
    """
    with trading_state_lock:
        if not trading_state.active_trades.get(option_key, False):
            return
    
    with options_data_lock:
        if option_key not in options_data:
            return
            
        option_info = options_data[option_key]
        current_price = option_info["ltp"]
        
        if current_price is None or current_price <= 0:
            return
    
    with trading_state_lock:
        entry_price = trading_state.entry_price[option_key]
        current_stop_loss = trading_state.stop_loss[option_key]
        initial_stop_loss = trading_state.initial_stop_loss[option_key]
        option_type = option_info["option_type"]
        strategy_type = trading_state.strategy_type[option_key]
    
    # Calculate current profit percentage
    if option_type == "CE":
        current_profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    else:  # PE
        current_profit_pct = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0

    if ENHANCED_TRAILING_SL:
        with trading_state_lock:
            trailing_sl_activated = trading_state.trailing_sl_activated[option_key]

            if not trailing_sl_activated and current_profit_pct >= MIN_PROFIT_FOR_TRAILING:
                trading_state.trailing_sl_activated[option_key] = True
                
                if strategy_type == "SCALP":
                    new_stop_loss = entry_price + (current_price - entry_price) * 0.3 if option_type == "CE" else entry_price - (entry_price - current_price) * 0.3
                elif strategy_type == "MOMENTUM":
                    new_stop_loss = entry_price + (current_price - entry_price) * 0.2 if option_type == "CE" else entry_price - (entry_price - current_price) * 0.2
                elif strategy_type == "NEWS":
                    new_stop_loss = entry_price + (current_price - entry_price) * 0.25 if option_type == "CE" else entry_price - (entry_price - current_price) * 0.25
                else:  # SWING
                    new_stop_loss = entry_price
                
                trading_state.stop_loss[option_key] = new_stop_loss
                logger.info(f"{option_key} enhanced trailing stop loss activated at {current_profit_pct:.2f}%, SL={new_stop_loss:.2f}")

            elif trailing_sl_activated:
                for i, threshold in enumerate(TRAILING_STEP_INCREMENTS):
                    if current_profit_pct >= threshold:
                        sl_percentage = TRAILING_SL_PERCENTAGES[i]
                        potential_sl = current_price * (1 - sl_percentage / 100) if option_type == "CE" else current_price * (1 + sl_percentage / 100)
                        
                        if (option_type == "CE" and potential_sl > current_stop_loss) or (option_type == "PE" and potential_sl < current_stop_loss):
                            trading_state.stop_loss[option_key] = potential_sl
                            logger.info(f"{option_key} trailing SL updated to {potential_sl:.2f} at {threshold}% profit level")
                            break
    else:
        with trading_state_lock:
            trailing_sl_activated = trading_state.trailing_sl_activated[option_key]
            
            if current_profit_pct > 2.0 and not trailing_sl_activated:
                trading_state.trailing_sl_activated[option_key] = True
                trading_state.stop_loss[option_key] = entry_price
                logger.info(f"{option_key} trailing stop loss activated, moved to breakeven")
            
            elif trailing_sl_activated:
                trail_price = current_price * 0.99 if option_type == "CE" else current_price * 1.01
                
                if (option_type == "CE" and trail_price > current_stop_loss) or (option_type == "PE" and trail_price < current_stop_loss):
                    trading_state.stop_loss[option_key] = trail_price
                    logger.info(f"{option_key} trailing stop loss updated to {trail_price}")

def update_dynamic_target(option_key):
    """
    Update target price dynamically based on market conditions and price momentum
    to maximize profit potential
    """
    if not DYNAMIC_TARGET_ADJUSTMENT:
        return
    
    with trading_state_lock:
        if not trading_state.active_trades.get(option_key, False):
            return
    
    with options_data_lock:
        if option_key not in options_data:
            return
            
        option_info = options_data[option_key]
        current_price = option_info["ltp"]
        
        if current_price is None or current_price <= 0:
            return
    
    with trading_state_lock:
        entry_price = trading_state.entry_price[option_key]
        current_target = trading_state.target[option_key]
        option_type = option_info["option_type"]
        parent_symbol = option_info["parent_symbol"]
        strategy_type = trading_state.strategy_type[option_key]
    
    try:
        # Calculate current profit percentage
        if option_type == "CE":
            current_profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        else:  # PE
            current_profit_pct = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0
        
        # Only adjust target if we're in profit
        if current_profit_pct < 1.0:
            return
        
        # Calculate momentum factors
        with options_data_lock:
            price_history = option_info["price_history"]
            
            if len(price_history) < 10:
                return
            
            price_series = price_history['price']
            
        current_rsi = calculate_rsi(price_series, 7)  # Use shorter RSI period for sensitivity
        current_momentum = calculate_momentum(price_series, 3)  # Short-term momentum
        
        # Calculate parent stock momentum if available
        parent_momentum = 0
        with stocks_data_lock:
            if parent_symbol in stocks_data:
                parent_price_history = stocks_data[parent_symbol]["price_history"]
                if len(parent_price_history) >= 10:
                    parent_price_series = parent_price_history['price']
                    parent_momentum = calculate_momentum(parent_price_series, 5)
        
        # Get current signal strength and trend
        with options_data_lock:
            signal = option_info.get("signal", 0)
            strength = option_info.get("strength", 0)
            option_trend = option_info.get("trend", "NEUTRAL")
        
        # Determine adjustment factor based on strategy and market conditions
        adjustment_factor = 0
        
        # Base adjustment on strategy
        if strategy_type == "SCALP":
            # Scalp trades should have quick, smaller targets
            base_adjustment = 0.3
            
            # For scalping, we want to take profits quicker as they become available
            if current_profit_pct > 2:
                # Increase target adjustment to capture more profit
                base_adjustment = 0.4
        elif strategy_type == "MOMENTUM":
            # Momentum trades can have more aggressive targets
            base_adjustment = 0.7
            
            # For momentum trades, we want to let profits run with the trend
            if "BULLISH" in option_trend and current_momentum > 0:
                base_adjustment = 0.9  # Strong continuation
        elif strategy_type == "NEWS":
            # News trades often have quick, dramatic moves
            base_adjustment = 0.6
            
            # For news trades, we want to adapt to the volatility
            if abs(current_momentum) > 5:
                base_adjustment = 0.8  # High momentum
        else:  # SWING
            # Swing trades need balanced targets
            base_adjustment = 0.5
            
            # For swing trades, we want to account for longer-term trend
            if current_profit_pct > 5:
                base_adjustment = 0.7  # Capture more profit on bigger moves
        
        # Adjust based on RSI
        if option_type == "CE":
            # For call options, higher RSI suggests more upside momentum
            if current_rsi > 70:
                rsi_factor = 0.4  # Strong momentum, increase target
            elif current_rsi < 30:
                rsi_factor = -0.2  # Weak momentum, decrease target
            else:
                rsi_factor = 0.1  # Neutral
        else:  # PE
            # For put options, lower RSI suggests more downside momentum
            if current_rsi < 30:
                rsi_factor = 0.4  # Strong momentum, increase target
            elif current_rsi > 70:
                rsi_factor = -0.2  # Weak momentum, decrease target
            else:
                rsi_factor = 0.1  # Neutral
        
        # Adjust based on recent momentum
        momentum_factor = min(abs(current_momentum) * TARGET_MOMENTUM_FACTOR, 0.5)
        if (option_type == "CE" and current_momentum < 0) or (option_type == "PE" and current_momentum > 0):
            momentum_factor = -momentum_factor  # Negative adjustment for counter-trend momentum
        
        # Consider parent stock momentum (with lower weight)
        parent_factor = min(abs(parent_momentum) * 0.05, 0.2)
        if (option_type == "CE" and parent_momentum < 0) or (option_type == "PE" and parent_momentum > 0):
            parent_factor = -parent_factor  # Negative adjustment for counter-trend parent momentum
        
        # Consider signal strength (stronger signals suggest more potential)
        signal_factor = min(strength / 20, 0.3)  # Max 0.3 contribution
        
        # Calculate total adjustment factor
        adjustment_factor = base_adjustment + rsi_factor + momentum_factor + parent_factor + signal_factor
        
        # Limit adjustment within reasonable bounds
        adjustment_factor = max(MIN_TARGET_ADJUSTMENT, min(adjustment_factor, MAX_TARGET_ADJUSTMENT))
        
        # Calculate new target
        if option_type == "CE":
            # For call options, target is above entry price
            target_price_diff = (current_price - entry_price) * adjustment_factor
            new_target = current_price + target_price_diff
            
            # Only adjust target upward
            with trading_state_lock:
                if new_target > current_target:
                    trading_state.target[option_key] = new_target
                    logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
        else:  # PE
            # For put options, target is below entry price
            target_price_diff = (entry_price - current_price) * adjustment_factor
            new_target = current_price - target_price_diff
            
            # Only adjust target downward
            with trading_state_lock:
                if new_target < current_target:
                    trading_state.target[option_key] = new_target
                    logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
                
    except Exception as e:
        logger.error(f"Error updating dynamic target for {option_key}: {e}")

def exit_trade(option_key, reason="Manual"):
    """Exit a trade for the given option with enhanced P&L tracking."""
    with trading_state_lock:
        if not trading_state.active_trades.get(option_key, False):
            return False
    
    with options_data_lock:
        if option_key not in options_data:
            logger.warning(f"Cannot exit trade: Option {option_key} not found")
            return False
            
        option_info = options_data[option_key]
        current_price = option_info["ltp"]
    
    if current_price is None or current_price <= 0:
        logger.warning(f"Cannot exit trade: Invalid price {current_price}")
        return False
    
    # Calculate P&L with thread safety
    with trading_state_lock:
        entry_price = trading_state.entry_price[option_key]
        quantity = trading_state.quantity[option_key]
        
        # Calculate P&L in both absolute and percentage terms
        if option_info["option_type"] == "CE":
            pnl = (current_price - entry_price) * quantity
            pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        else:  # PE
            pnl = (entry_price - current_price) * quantity
            pnl_pct = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0
        
        # Update trading state
        trading_state.active_trades[option_key] = False
        trading_state.pnl[option_key] = pnl
        trading_state.total_pnl += pnl
        trading_state.daily_pnl += pnl
        
        # Update win/loss counter
        if pnl > 0:
            trading_state.wins += 1
        else:
            trading_state.losses += 1
        
        # Calculate trade duration
        entry_time = trading_state.entry_time[option_key]
        exit_time = datetime.now()
        duration_seconds = (exit_time - entry_time).total_seconds() if entry_time else 0
        duration_minutes = duration_seconds / 60
        
        # Add to trade history
        trade_record = {
            'option_key': option_key,
            'parent_symbol': option_info.get("parent_symbol", ""),
            'strategy_type': trading_state.strategy_type[option_key],
            'trade_source': trading_state.trade_source.get(option_key, "TECHNICAL"),
            'option_type': option_info.get("option_type", ""),
            'strike': option_info.get("strike", ""),
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_minutes': duration_minutes,
            'entry_price': entry_price,
            'exit_price': current_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        trading_state.trades_history.append(trade_record)
        
        # Reset entry and stop loss values
        trading_state.entry_price[option_key] = None
        trading_state.entry_time[option_key] = None
        trading_state.stop_loss[option_key] = None
        trading_state.initial_stop_loss[option_key] = None
        trading_state.target[option_key] = None
        trading_state.trailing_sl_activated[option_key] = False
        trading_state.stock_entry_price[option_key] = None
        trading_state.quantity[option_key] = 0
        trading_state.strategy_type[option_key] = None
        trading_state.trade_source[option_key] = None
    
    logger.info(
        f"Exited {option_key} trade: "
        f"Price={current_price:.2f}, P&L={pnl:.2f} ({pnl_pct:.2f}%), "
        f"Duration={duration_minutes:.1f}min, Reason={reason}"
    )
    return True

def apply_trading_strategy():
    """Apply the trading strategy with improved timeout handling."""
    # Skip if all strategies are disabled or broker not connected
    if (not any(strategy_settings.values())) or not broker_connected:
        return
    
    # Set deadline for entire function
    start_time = time.time()
    MAX_STRATEGY_TIME = 8  # Maximum 8 seconds for strategy application
    
    try:
        # Update news data with timeout
        if strategy_settings["NEWS_ENABLED"]:
            news_start = time.time()
            NEWS_TIMEOUT = 2  # 2 seconds max for news update
            
            try:
                logger.info("Checking for news updates...")
                
                def news_update_thread():
                    try:
                        update_news_data()
                    except Exception as e:
                        logger.error(f"Error in news update thread: {e}")
                
                news_thread = threading.Thread(target=news_update_thread)
                news_thread.daemon = True
                news_thread.start()
                news_thread.join(timeout=NEWS_TIMEOUT)
                
                logger.info(f"News update took {time.time() - news_start:.2f}s")
            except Exception as e:
                logger.error(f"Error starting news update: {e}")
        
        # Check for trade exits first (most important)
        logger.info("Checking for trade exits...")
        
        with trading_state_lock:
            active_trade_keys = [key for key, is_active in trading_state.active_trades.items() if is_active]
        
        # Process limited number of trades if we're running out of time
        for i, option_key in enumerate(active_trade_keys):
            # Check if we've used more than half our total time
            if time.time() - start_time > MAX_STRATEGY_TIME / 2:
                logger.warning(f"Stopping trade exit checks - reached time limit after {i} trades")
                break
                
            try:
                should_exit, reason = should_exit_trade(option_key)
                if should_exit:
                    exit_trade(option_key, reason=reason)
                else:
                    # Only update SL/target if we have time
                    if time.time() - start_time <= MAX_STRATEGY_TIME * 0.6:
                        update_dynamic_stop_loss(option_key)
                        update_dynamic_target(option_key)
            except Exception as e:
                logger.error(f"Error checking exit for {option_key}: {e}")
        
        # Check for trade entries if time allows
        remaining_time = MAX_STRATEGY_TIME - (time.time() - start_time)
        if remaining_time >= 2:  # Only if we have at least 2 seconds left
            logger.info("Checking for trade entries...")
            
            # Get potential entries quickly
            potential_entries = []
            max_options_to_check = 20  # Limit the number of options to check
            
            with options_data_lock:
                option_keys = list(options_data.keys())[:max_options_to_check]
            
            # Check a limited number of options
            for option_key in option_keys:
                # Stop if running out of time
                if time.time() - start_time > MAX_STRATEGY_TIME * 0.9:
                    break
                    
                with trading_state_lock:
                    is_active = trading_state.active_trades.get(option_key, False)
                    
                    if not is_active:
                        try:
                            should_enter, strategy_type, confidence = should_enter_trade(option_key)
                            if should_enter and strategy_type:
                                potential_entries.append((option_key, strategy_type, confidence))
                        except Exception as e:
                            logger.error(f"Error checking entry for {option_key}: {e}")
            
            # Sort by confidence score, highest first
            potential_entries.sort(key=lambda x: x[2], reverse=True)
            
            # Take the top entries, limited by remaining trades for the day
            with trading_state_lock:
                remaining_trades = MAX_TRADES_PER_DAY - trading_state.trades_today
            
            # Enter a limited number of trades
            for i, (option_key, strategy_type, confidence) in enumerate(potential_entries):
                if i >= min(3, remaining_trades) or time.time() - start_time > MAX_STRATEGY_TIME * 0.95:
                    break
                    
                if confidence > 0.6:  # Only take high confidence trades
                    # Determine the source of the trade
                    trade_source = "TECHNICAL"
                    if strategy_type == "NEWS":
                        trade_source = "NEWS"
                    
                    enter_trade(option_key, strategy_type, trade_source)
                    
                    # Brief delay between entries
                    time.sleep(0.05)
        
        logger.info(f"Strategy application completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in trading strategy application: {e}")
# ============ News Monitoring Functions ============
def fetch_news_from_sources():
    """
    Fetch financial news from various free sources
    Returns a list of news items with title, description, source, and timestamp
    """
    news_items = []
    
    try:
        # Check if feedparser is available
        if 'feedparser' in sys.modules:
            # Fetch from Yahoo Finance RSS
            yahoo_feed = feedparser.parse('https://finance.yahoo.com/news/rssindex')
            for entry in yahoo_feed.entries[:15]:  # Get top 15 news (increased from 10)
                news_items.append({
                    'title': entry.title,
                    'description': entry.get('description', ''),
                    'source': 'Yahoo Finance',
                    'timestamp': datetime.now(),
                    'url': entry.link
                })
            
            # Add Moneycontrol news (India specific)
            try:
                mc_feed = feedparser.parse('https://www.moneycontrol.com/rss/latestnews.xml')
                for entry in mc_feed.entries[:15]:  # Get top 15 news (increased from 10)
                    news_items.append({
                        'title': entry.title,
                        'description': entry.get('description', ''),
                        'source': 'Moneycontrol',
                        'timestamp': datetime.now(),
                        'url': entry.link
                    })
            except Exception as e:
                logger.warning(f"Error fetching Moneycontrol news: {e}")
                
            # Add Economic Times news (India specific)
            try:
                et_feed = feedparser.parse('https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms')
                for entry in et_feed.entries[:15]:  # Get top 15 news
                    news_items.append({
                        'title': entry.title,
                        'description': entry.get('description', ''),
                        'source': 'Economic Times',
                        'timestamp': datetime.now(),
                        'url': entry.link
                    })
            except Exception as e:
                logger.warning(f"Error fetching Economic Times news: {e}")
        
        else:
            # Fallback to a simplified approach if feedparser is not available
            # Fetch some placeholder news or latest headlines
            logger.warning("feedparser not available, using placeholder news")
            news_items.append({
                'title': 'Market Update: NIFTY shows strong momentum',
                'description': 'Market analysis indicates bullish trend for NIFTY',
                'source': 'Dashboard',
                'timestamp': datetime.now(),
                'url': '#'
            })
        
        logger.info(f"Fetched {len(news_items)} news items")
        return news_items
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

def analyze_news_for_stocks(news_items, stock_universe):
    """
    Analyze news items to find mentions of stocks and determine sentiment
    
    Args:
        news_items: List of news items with title and description
        stock_universe: List of stock symbols to look for
    
    Returns:
        dict: Dictionary of stocks with their sentiment scores
    """
    stock_mentions = {}
    
    try:
        # Import regex module
        import re
        
        for item in news_items:
            title = item.get('title', '')
            description = item.get('description', '')
            full_text = f"{title} {description}"
            
            # Enhanced sentiment analysis with more keywords and weightings
            positive_words = {
                'surge': 1.5, 'jump': 1.5, 'rise': 1.0, 'gain': 1.0, 'profit': 1.2, 
                'up': 0.8, 'higher': 1.0, 'bull': 1.3, 'positive': 1.0, 
                'outperform': 1.4, 'rally': 1.5, 'strong': 1.2, 'beat': 1.3, 
                'exceed': 1.4, 'growth': 1.2, 'soar': 1.7, 'boost': 1.2, 
                'upgrade': 1.5, 'breakthrough': 1.6, 'opportunity': 1.1,
                'record high': 1.8, 'buy': 1.3, 'accumulate': 1.2
            }
            
            negative_words = {
                'fall': 1.5, 'drop': 1.5, 'decline': 1.0, 'loss': 1.2, 'down': 0.8, 
                'lower': 1.0, 'bear': 1.3, 'negative': 1.0, 'underperform': 1.4, 
                'weak': 1.2, 'miss': 1.3, 'disappointing': 1.4, 'sell-off': 1.6, 
                'crash': 1.8, 'downgrade': 1.5, 'warning': 1.3, 'risk': 1.0,
                'trouble': 1.4, 'concern': 1.1, 'caution': 0.9, 'burden': 1.2,
                'record low': 1.8, 'sell': 1.3, 'reduce': 1.2
            }
            
            # Calculate weighted sentiment score
            pos_score = 0
            neg_score = 0
            
            # Check for positive words with proximity bonus
            for word, weight in positive_words.items():
                if word.lower() in full_text.lower():
                    # Count occurrences
                    count = full_text.lower().count(word.lower())
                    # Add to score with weight
                    pos_score += count * weight
                    
                    # Title bonus - words in title have more impact
                    if word.lower() in title.lower():
                        pos_score += 0.5 * weight
            
            # Check for negative words with proximity bonus
            for word, weight in negative_words.items():
                if word.lower() in full_text.lower():
                    # Count occurrences
                    count = full_text.lower().count(word.lower())
                    # Add to score with weight
                    neg_score += count * weight
                    
                    # Title bonus - words in title have more impact
                    if word.lower() in title.lower():
                        neg_score += 0.5 * weight
            
            # Calculate sentiment score (-1 to 1)
            if pos_score + neg_score > 0:
                sentiment = (pos_score - neg_score) / (pos_score + neg_score)
            else:
                sentiment = 0  # Neutral if no sentiment words found
            
            # Improved stock mention detection
            for stock in stock_universe:
                # Look for exact stock symbol with word boundaries
                pattern = r'\b' + re.escape(stock) + r'\b'
                
                # Also check for common variations of the stock name
                stock_variations = [stock]
                
                # Add variations for common Indian stocks
                if stock == "RELIANCE":
                    stock_variations.extend(["Reliance Industries", "RIL"])
                elif stock == "INFY":
                    stock_variations.extend(["Infosys", "Infosys Technologies"])
                elif stock == "TCS":
                    stock_variations.extend(["Tata Consultancy", "Tata Consultancy Services"])
                elif stock == "HDFCBANK":
                    stock_variations.extend(["HDFC Bank", "Housing Development Finance Corporation"])
                elif stock == "SBIN":
                    stock_variations.extend(["State Bank of India", "SBI"])
                
                # Check for any variation
                found = False
                for variation in stock_variations:
                    var_pattern = r'\b' + re.escape(variation) + r'\b'
                    if re.search(var_pattern, full_text, re.IGNORECASE):
                        found = True
                        break
                
                if found:
                    if stock not in stock_mentions:
                        stock_mentions[stock] = []
                    
                    stock_mentions[stock].append({
                        'sentiment': sentiment,
                        'title': title,
                        'source': item.get('source', 'Unknown'),
                        'timestamp': item.get('timestamp', datetime.now()),
                        'url': item.get('url', '')
                    })
                    logger.info(f"Found mention of {stock} in news with sentiment {sentiment:.2f}")
        
        logger.info(f"Found mentions of {len(stock_mentions)} stocks in news")
        return stock_mentions
    except Exception as e:
        logger.error(f"Error analyzing news: {e}")
        return {}

def generate_news_trading_signals(stock_mentions):
    """
    Generate trading signals based on news mentions and sentiment
    
    Args:
        stock_mentions: Dictionary of stocks with their sentiment scores
    
    Returns:
        list: List of trading signals
    """
    trading_signals = []
    
    for stock, mentions in stock_mentions.items():
        if not mentions:
            continue
        
        # Calculate average sentiment
        avg_sentiment = sum(mention['sentiment'] for mention in mentions) / len(mentions)
        
        # Determine confidence based on number of mentions and sentiment strength
        mentions_factor = min(len(mentions) / 3, 1.0)  # Scale up to 3 mentions
        sentiment_factor = min(abs(avg_sentiment) * 1.5, 1.0)  # Scale sentiment impact
        
        # Combined confidence score
        confidence = (mentions_factor * 0.6) + (sentiment_factor * 0.4)  # 60% mentions, 40% sentiment strength
        
        # Determine trading action based on sentiment
        if avg_sentiment > NEWS_SENTIMENT_THRESHOLD:  # Strong positive sentiment
            action = 'BUY_CE'  # Buy Call option
            
            trading_signals.append({
                'stock': stock,
                'action': action,
                'sentiment': avg_sentiment,
                'confidence': confidence,
                'news_count': len(mentions),
                'latest_news': mentions[0]['title'],
                'source': mentions[0]['source'],
                'timestamp': mentions[0]['timestamp']
            })
            logger.info(f"Generated BUY_CE signal for {stock} with confidence {confidence:.2f}")
            
        elif avg_sentiment < -NEWS_SENTIMENT_THRESHOLD:  # Strong negative sentiment
            action = 'BUY_PE'  # Buy Put option
            
            trading_signals.append({
                'stock': stock,
                'action': action,
                'sentiment': avg_sentiment,
                'confidence': confidence,
                'news_count': len(mentions),
                'latest_news': mentions[0]['title'],
                'source': mentions[0]['source'],
                'timestamp': mentions[0]['timestamp']
            })
            logger.info(f"Generated BUY_PE signal for {stock} with confidence {confidence:.2f}")
    
    # Sort signals by confidence (highest first)
    trading_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return trading_signals

def update_news_data():
    """Update news data and generate trading signals"""
    global news_data, last_news_update
    
    current_time = datetime.now()
    if (current_time - last_news_update).total_seconds() < NEWS_CHECK_INTERVAL:
        return
    
    # Get all tracked stocks as the universe
    with stocks_data_lock:
        stock_universe = list(stocks_data.keys())
    
    # Add a list of common Indian stocks/indices
    common_stocks = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "WIPRO", "SBIN", 
        "TATAMOTORS", "BAJFINANCE", "ADANIENT", "ADANIPORTS", "HINDUNILVR",
        "AXISBANK", "SUNPHARMA", "KOTAKBANK", "ONGC", "MARUTI", "BHARTIARTL"
    ]
    
    for stock in common_stocks:
        if stock not in stock_universe:
            stock_universe.append(stock)
    
    # Add major indices
    indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    for idx in indices:
        if idx not in stock_universe:
            stock_universe.append(idx)
    
    # Fetch news
    news_items = fetch_news_from_sources()
    
    # Only update if we got new news items
    if news_items:
        # Analyze news for stock mentions
        stock_mentions = analyze_news_for_stocks(news_items, stock_universe)
        
        # Generate trading signals
        trading_signals = generate_news_trading_signals(stock_mentions)
        
        # Update news data
        news_data["items"] = news_items
        news_data["mentions"] = stock_mentions
        news_data["trading_signals"] = trading_signals
        news_data["last_updated"] = current_time
        
        # Update UI data store
        with ui_data_store_lock:
            ui_data_store['news'] = {
                'items': news_items[:5],  # Store the 5 most recent items
                'mentions': stock_mentions,
                'signals': trading_signals,
                'last_updated': current_time.strftime('%H:%M:%S')
            }
        
        # Add stocks mentioned in news
        add_news_mentioned_stocks()
        
        # Try to execute trades if any signals were generated
        if trading_signals and broker_connected:
            execute_news_based_trades()
    
    last_news_update = current_time

def add_news_mentioned_stocks():
    """Add stocks that are mentioned in news but not currently tracked"""
    if not news_data.get("mentions"):
        return
    
    # Get all mentioned stocks
    mentioned_stocks = set(news_data["mentions"].keys())
    
    # Get currently tracked stocks
    with stocks_data_lock:
        tracked_stocks = set(stocks_data.keys())
    
    # Find stocks to add (mentioned but not tracked)
    stocks_to_add = mentioned_stocks - tracked_stocks
    
    # Add each stock
    for stock in stocks_to_add:
        # Skip if it doesn't look like a valid stock symbol
        if not stock.isalpha() or len(stock) < 2:
            continue
            
        logger.info(f"Adding stock {stock} based on news mentions")
        add_stock(stock, None, "NSE", "STOCK")
        
        # Try to fetch data immediately
        if broker_connected:
            fetch_stock_data(stock)

def execute_news_based_trades():
    """Execute trades based on news analysis"""
    if not strategy_settings["NEWS_ENABLED"]:
        logger.info("News-based trading is disabled")
        return
    
    if not broker_connected:
        logger.warning("Cannot execute news-based trades: Not connected to broker")
        return
    
    # Check if we've hit the maximum trades for the day
    with trading_state_lock:
        if trading_state.trades_today >= MAX_TRADES_PER_DAY:
            logger.info("Maximum trades for the day reached, skipping news-based trades")
            return
        
        # Check if we've hit the maximum loss percentage for the day
        if trading_state.daily_pnl <= -MAX_LOSS_PERCENTAGE * trading_state.capital / 100:
            logger.info("Maximum daily loss reached, skipping news-based trades")
            return
    
    # Limit to top 3 highest confidence signals
    top_signals = sorted(news_data["trading_signals"], key=lambda x: x['confidence'], reverse=True)[:3]
    
    for signal in top_signals:
        stock = signal['stock']
        action = signal['action']
        confidence = signal['confidence']
        news_title = signal['latest_news']
        
        # Skip trades with low confidence
        if confidence < NEWS_CONFIDENCE_THRESHOLD:
            logger.info(f"Skipping news-based trade for {stock} due to low confidence ({confidence:.2f})")
            continue
        
        # Skip if the news is too old
        signal_time = signal['timestamp']
        if isinstance(signal_time, datetime):
            if (datetime.now() - signal_time).total_seconds() > NEWS_MAX_AGE:
                logger.info(f"Skipping news-based trade for {stock} - news is too old")
                continue
        
        # Add stock if not already tracking
        with stocks_data_lock:
            if stock not in stocks_data:
                logger.info(f"Adding stock {stock} based on news")
                add_stock(stock, None, "NSE", "STOCK")
        
        # Wait for stock data to be loaded
        attempt = 0
        while attempt < 10:
            with stocks_data_lock:
                if stock in stocks_data and stocks_data[stock]["ltp"] is not None:
                    break
            logger.info(f"Waiting for {stock} data to load... (attempt {attempt+1})")
            time.sleep(0.5)
            attempt += 1
        
        with stocks_data_lock:
            if stock not in stocks_data or stocks_data[stock]["ltp"] is None:
                logger.warning(f"Could not get price data for {stock}, skipping trade")
                continue
        
        # Determine option type
        option_type = "CE" if action == "BUY_CE" else "PE"
        
        # Find appropriate option
        options = find_and_add_options(stock)
        option_key = options.get(option_type)
        
        if not option_key:
            logger.warning(f"Could not find suitable {option_type} option for {stock}")
            continue
        
        # Check if option is already in active trade
        with trading_state_lock:
            if trading_state.active_trades.get(option_key, False):
                logger.info(f"Already in a trade for {stock} {option_type}, skipping")
                continue
        
        # Execute the trade
        strategy_type = "NEWS"  # Special strategy type for news-based trades
        success = enter_trade(option_key, strategy_type, "NEWS")
        
        if success:
            logger.info(f"Executed news-based trade for {stock} {option_type} based on: {news_title}")
            
            # Update news signal to avoid duplicate trades
            signal['executed'] = True
            signal['execution_time'] = datetime.now()
            signal['option_key'] = option_key
            
            # We only take a few news-based trades at a time
            break

# ============ Broker Connection Functions ============
def connect_to_broker():
    """Connect to the broker API with improved error handling and retry logic."""
    global smart_api, broker_connected, broker_error_message, broker_connection_retry_time, config, last_connection_time
    
    if broker_connected and smart_api is not None:
        return True
        
    # Check if we need to wait before retrying
    current_time = datetime.now()
    if broker_connection_retry_time and current_time < broker_connection_retry_time:
        wait_time = (broker_connection_retry_time - current_time).total_seconds()
        logger.debug(f"Waiting {wait_time:.1f} seconds before retrying broker connection")
        return False
    
    max_retries = 3
    base_delay = 2  # Base delay in seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            # Calculate delay with exponential backoff and jitter
            delay = min(300, base_delay * (2 ** (attempt - 1)))  # Cap at 5 minutes
            jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
            total_delay = delay + jitter
            
            if attempt > 1:
                logger.info(f"Connecting to broker (attempt {attempt}/{max_retries}, delay: {total_delay:.1f}s)...")
                time.sleep(total_delay)
            else:
                logger.info("Connecting to broker...")
            
            # Get TOTP code
            totp = pyotp.TOTP(config.totp_secret)
            totp_value = totp.now()
            
            # Initialize the API
            smart_api = SmartConnect(api_key=config.api_key)
            
            # Login to SmartAPI
            login_resp = smart_api.generateSession(config.username, config.password, totp_value)
            
            # Check login response thoroughly
            if isinstance(login_resp, dict):
                if login_resp.get("status"):
                    # Store refresh token for later use
                    config.refresh_token = login_resp.get("data", {}).get("refreshToken", "")
                    config.last_refreshed = datetime.now()
                    
                    broker_connected = True
                    broker_error_message = None
                    broker_connection_retry_time = None
                    last_connection_time = datetime.now()
                    logger.info("Successfully connected to broker")
                    
                    # Verify connection with a test API call
                    try:
                        # Get user profile - requires refresh token in newer API versions
                        profile = smart_api.getProfile(refreshToken=config.refresh_token)
                        if isinstance(profile, dict) and profile.get("status"):
                            logger.info(f"Connected as {profile.get('data', {}).get('name', 'User')}")
                            return True
                        else:
                            logger.warning(f"Profile verification returned unexpected response: {profile}")
                            smart_api = None
                            broker_connected = False
                            continue
                    except Exception as e:
                        logger.error(f"Profile verification failed: {e}")
                        # If the error is about missing refreshToken, it's likely an API version mismatch
                        if "refreshToken" in str(e):
                            logger.warning("API may require refreshToken. Check API version compatibility.")
                        smart_api = None
                        broker_connected = False
                        continue
                else:
                    error_msg = login_resp.get("message", "Unknown error")
                    error_code = login_resp.get("errorCode", "")
                    
                    # Handle specific error cases
                    if any(x in error_msg.lower() for x in ["invalid", "incorrect", "wrong"]):
                        broker_error_message = "Invalid credentials. Please check API key and login details."
                        broker_connection_retry_time = current_time + timedelta(minutes=15)
                        logger.error(f"Authentication failed: {error_msg}")
                        break  # Don't retry for invalid credentials
                        
                    elif "block" in error_msg.lower() or error_code == "AB1006":
                        broker_error_message = "Account blocked for trading. Please contact broker support."
                        broker_connection_retry_time = current_time + timedelta(hours=12)
                        logger.error(f"Account blocked: {error_msg}")
                        logger.warning("==========================================")
                        logger.warning("ACCOUNT BLOCKED - REQUIRES MANUAL ACTION:")
                        logger.warning("1. Contact AngelOne customer support")
                        logger.warning("2. Verify KYC and account status")
                        logger.warning("3. Resolve any pending issues with your account")
                        logger.warning("==========================================")
                        break  # Don't retry for blocked accounts
                        
                    elif "limit" in error_msg.lower():
                        broker_error_message = f"Rate limit exceeded: {error_msg}"
                        broker_connection_retry_time = current_time + timedelta(minutes=5)
                        logger.warning(broker_error_message)
                        break  # Don't retry for rate limits
                        
                    else:
                        broker_error_message = f"Login failed: {error_msg}"
                        logger.warning(f"Failed to connect (attempt {attempt}/{max_retries}): {error_msg}")
            else:
                broker_error_message = "Invalid response from broker API"
                logger.error(f"Invalid login response type: {type(login_resp)}")
            
            # Reset connection variables
            smart_api = None
            broker_connected = False
            
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str:
                broker_error_message = "Connection timeout. Check your internet connection."
            elif "connection" in error_str:
                broker_error_message = "Network error. Check your internet connection."
            else:
                broker_error_message = f"Connection error: {str(e)}"
            
            logger.error(f"Error connecting to broker (attempt {attempt}/{max_retries}): {e}")
            smart_api = None
            broker_connected = False
    
    # If all retries failed, set a longer retry time
    if not broker_connected and not broker_connection_retry_time:
        broker_connection_retry_time = current_time + timedelta(minutes=5)
    
    return False

def refresh_session_if_needed():
    """Refresh broker session if it's been more than 1 hour since last refresh"""
    global smart_api, broker_connected, config
    
    if not broker_connected or not smart_api or not config.refresh_token:
        return False
    
    # Check if we need to refresh (every 6 hours)
    current_time = datetime.now()
    if config.last_refreshed and (current_time - config.last_refreshed).total_seconds() < 21600:  # 6 hours
        return True  # No need to refresh yet
    
    try:
        logger.info("Refreshing broker session...")
        refresh_resp = smart_api.generateSession(config.username, config.password, refreshToken=config.refresh_token)
        
        if isinstance(refresh_resp, dict) and refresh_resp.get("status"):
            # Update refresh token
            config.refresh_token = refresh_resp.get("data", {}).get("refreshToken", "")
            config.last_refreshed = current_time
            logger.info("Session refreshed successfully")
            return True
        else:
            logger.warning("Failed to refresh session, will attempt reconnection")
            broker_connected = False
            smart_api = None
            return connect_to_broker()
    except Exception as e:
        logger.error(f"Error refreshing session: {e}")
        broker_connected = False
        smart_api = None
        return connect_to_broker()

def safe_api_call(func_name, func, *args, **kwargs):
    """
    Safely call an API function with error tracking and backoff
    
    Args:
        func_name (str): Function name for logging and tracking
        func (callable): The API function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call, or None if an error occurred
    """
    # Check if we should skip this endpoint due to excessive errors
    if api_error_tracker.should_skip_endpoint(func_name):
        logger.warning(f"Skipping {func_name} due to excessive errors")
        return None
    
    try:
        # Apply rate limiting
        rate_limit_handler.wait_if_needed(func_name)
        
        # Make the API call
        result = func(*args, **kwargs)
        
        # Check for error in result
        if isinstance(result, dict) and not result.get("status"):
            error_msg = result.get("message", "Unknown error")
            error_code = result.get("errorCode", "")
            
            logger.warning(f"API error in {func_name}: {error_msg} (Code: {error_code})")
            
            # Check for rate limit errors
            if ("access rate" in error_msg.lower() or 
                "try after some time" in error_msg.lower() or 
                "session expired" in error_msg.lower()):
                
                rate_limit_handler.register_rate_limit_hit(func_name)
                
                # Record the error
                exceeded = api_error_tracker.record_error(func_name)
                if exceeded:
                    logger.error(f"Excessive errors for {func_name}, will temporarily skip this endpoint")
                
                return None
            
            # Record the error
            api_error_tracker.record_error(func_name)
            
            return None
        
        # Reset error count on success
        api_error_tracker.reset(func_name)
        
        return result
        
    except Exception as e:
        logger.error(f"Exception in {func_name}: {e}")
        
        # Check for rate limit errors
        if "access rate" in str(e).lower() or "rate limit" in str(e).lower() or "session expired" in str(e).lower():
            rate_limit_handler.register_rate_limit_hit(func_name)
        
        # Record the error
        exceeded = api_error_tracker.record_error(func_name)
        if exceeded:
            logger.error(f"Excessive errors for {func_name}, will temporarily skip this endpoint")
        
        return None

def connect_to_broker_with_backoff():
    """Connect to the broker API with improved backoff and error handling"""
    global smart_api, broker_connected, broker_error_message, last_connection_time, connection_backoff
    
    if broker_connected and smart_api is not None:
        return True
    
    # Check if we should retry based on backoff
    if not connection_backoff.should_retry():
        return False
    
    try:
        logger.info(f"Connecting to broker (attempt {connection_backoff.attempt + 1})...")
        
        # Get TOTP code
        totp = pyotp.TOTP(config.totp_secret)
        totp_value = totp.now()
        
        # Initialize the API
        smart_api = SmartConnect(api_key=config.api_key)
        
        # Login to SmartAPI
        login_resp = smart_api.generateSession(config.username, config.password, totp_value)
        
        # Check login response
        if isinstance(login_resp, dict):
            if login_resp.get("status"):
                # Store refresh token for later use
                config.refresh_token = login_resp.get("data", {}).get("refreshToken", "")
                config.last_refreshed = datetime.now()
                
                broker_connected = True
                broker_error_message = None
                last_connection_time = datetime.now()
                
                # Verify connection with a test API call
                profile = smart_api.getProfile(refreshToken=config.refresh_token)
                
                if isinstance(profile, dict) and profile.get("status"):
                    logger.info(f"Connected as {profile.get('data', {}).get('name', 'User')}")
                    
                    # Connection successful, reset backoff
                    connection_backoff.reset()
                    return True
                else:
                    # Profile validation failed
                    logger.warning(f"Profile verification returned unexpected response: {profile}")
                    smart_api = None
                    broker_connected = False
                    # Calculate next retry time
                    backoff_time = connection_backoff.next()
                    logger.info(f"Will retry in {backoff_time:.1f} seconds")
                    return False
            else:
                # Login failed
                error_msg = login_resp.get("message", "Unknown error")
                error_code = login_resp.get("errorCode", "")
                
                # Handle specific error cases
                if any(x in error_msg.lower() for x in ["invalid", "incorrect", "wrong"]):
                    broker_error_message = "Invalid credentials. Please check API key and login details."
                    # Longer backoff for invalid credentials
                    connection_backoff.attempt = 5  # Simulate higher attempt count for longer delay
                    backoff_time = connection_backoff.next()
                    logger.error(f"Authentication failed: {error_msg}. Will retry in {backoff_time:.1f} seconds")
                    
                elif "block" in error_msg.lower() or error_code == "AB1006":
                    broker_error_message = "Account blocked for trading. Please contact broker support."
                    # Very long backoff for blocked accounts
                    connection_backoff.attempt = 10  # Simulate higher attempt count for longer delay
                    backoff_time = connection_backoff.next()
                    logger.error(f"Account blocked: {error_msg}. Will retry in {backoff_time:.1f} seconds")
                    logger.warning("==========================================")
                    logger.warning("ACCOUNT BLOCKED - REQUIRES MANUAL ACTION:")
                    logger.warning("1. Contact AngelOne customer support")
                    logger.warning("2. Verify KYC and account status")
                    logger.warning("3. Resolve any pending issues with your account")
                    logger.warning("==========================================")
                    
                elif "limit" in error_msg.lower():
                    broker_error_message = f"Rate limit exceeded: {error_msg}"
                    backoff_time = connection_backoff.next()
                    logger.warning(f"Rate limit exceeded. Will retry in {backoff_time:.1f} seconds")
                    
                else:
                    broker_error_message = f"Login failed: {error_msg}"
                    backoff_time = connection_backoff.next()
                    logger.warning(f"Failed to connect: {error_msg}. Will retry in {backoff_time:.1f} seconds")
                
                # Reset connection variables
                smart_api = None
                broker_connected = False
                return False
        else:
            broker_error_message = "Invalid response from broker API"
            logger.error(f"Invalid login response type: {type(login_resp)}")
            backoff_time = connection_backoff.next()
            logger.warning(f"Will retry in {backoff_time:.1f} seconds")
            
            # Reset connection variables
            smart_api = None
            broker_connected = False
            return False
            
    except Exception as e:
        error_str = str(e).lower()
        if "timeout" in error_str:
            broker_error_message = "Connection timeout. Check your internet connection."
        elif "connection" in error_str:
            broker_error_message = "Network error. Check your internet connection."
        else:
            broker_error_message = f"Connection error: {str(e)}"
        
        logger.error(f"Error connecting to broker: {e}")
        
        # Get next backoff time
        backoff_time = connection_backoff.next()
        logger.warning(f"Will retry in {backoff_time:.1f} seconds")
        
        smart_api = None
        broker_connected = False
        return False

# ============ Maintenance Functions ============
def check_day_rollover():
    """Check if trading day has changed and reset daily stats."""
    global trading_state
    
    current_date = datetime.now().date()
    with trading_state_lock:
        if current_date != trading_state.trading_day:
            logger.info(f"New trading day detected: {current_date}. Resetting daily stats.")
            trading_state.trading_day = current_date
            trading_state.trades_today = 0
            trading_state.daily_pnl = 0

def should_remove_stock(symbol):
    """
    Determine if a stock should be automatically removed.
    """
    with stocks_data_lock:
        if symbol not in stocks_data:
            return True
            
        stock_info = stocks_data[symbol]
    
    current_time = datetime.now()
    
    # Skip default/index stocks from automatic removal
    if symbol in [s["symbol"] for s in DEFAULT_STOCKS]:
        return False
    
    # Check if the stock hasn't been updated in 15 minutes
    with stocks_data_lock:
        last_update = stock_info.get("last_updated")
        
    if last_update is None or (current_time - last_update).total_seconds() > 900:  # 15 minutes
        logger.warning(f"Stock {symbol} has not been updated for >15 minutes")
        return True
    
    # Only remove stocks with completely invalid price data
    with stocks_data_lock:
        if stock_info.get("ltp") is None:
            logger.warning(f"Stock {symbol} has no price data")
            return True
    
    return False

def cleanup_inactive_stocks():
    """
    Remove stocks that are inactive or have invalid data.
    """
    with stocks_data_lock:
        symbols_to_check = list(stocks_data.keys())
    
    for symbol in symbols_to_check:
        if should_remove_stock(symbol):
            remove_stock(symbol)

def cleanup_old_data():
    """Clean up old data to prevent memory bloat with improved optimization."""
    global last_cleanup, stocks_data, options_data
    
    current_time = datetime.now()
    if (current_time - last_cleanup).total_seconds() < DATA_CLEANUP_INTERVAL:
        return
    
    logger.info("Running data cleanup...")
    
    # Optimize price history data instead of just truncating
    optimize_price_history()
    
    # Clean up news data (keep only recent news)
    if "items" in news_data:
        # Keep only news from last 24 hours
        recent_news = []
        for item in news_data["items"]:
            timestamp = item.get("timestamp")
            if timestamp and isinstance(timestamp, datetime):
                if (current_time - timestamp).total_seconds() < 86400:  # 24 hours
                    recent_news.append(item)
        
        news_data["items"] = recent_news
    
    # Run Python garbage collection
    import gc
    gc.collect()
    
    last_cleanup = current_time
    logger.info("Data cleanup completed")

def optimize_price_history(symbol=None):
    """
    Optimize price history storage by resampling older data points
    to reduce memory usage while preserving recent data at full resolution.
    
    Args:
        symbol (str, optional): Symbol to optimize. If None, optimize all symbols.
    """
    global stocks_data, options_data
    
    try:
        # Process specific symbol if provided
        if symbol:
            with stocks_data_lock:
                if symbol in stocks_data:
                    stock_info = stocks_data[symbol]
                    # Only process if we have significant data
                    if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                        # Keep last 200 points at full resolution
                        recent_data = stock_info["price_history"].tail(200)
                        
                        # Resample older data to 5-minute intervals
                        older_data = stock_info["price_history"].iloc[:-200]
                        if 'timestamp' in older_data.columns and len(older_data) > 100:
                            older_data['timestamp'] = pd.to_datetime(older_data['timestamp'])
                            older_data = older_data.set_index('timestamp')
                            resampled = older_data.resample('5T').mean().reset_index()
                            
                            # Combine resampled older data with recent data
                            stock_info["price_history"] = pd.concat([resampled, recent_data], ignore_index=True)
            
            # Also process options for this symbol
            with options_data_lock:
                for option_key, option_info in options_data.items():
                    if option_info.get("parent_symbol") == symbol:
                        if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                            # Keep last 200 points at full resolution
                            recent_data = option_info["price_history"].tail(200)
                            
                            # Resample older data to 5-minute intervals
                            older_data = option_info["price_history"].iloc[:-200]
                            if 'timestamp' in older_data.columns and len(older_data) > 100:
                                older_data['timestamp'] = pd.to_datetime(older_data['timestamp'])
                                older_data = older_data.set_index('timestamp')
                                resampled = older_data.resample('5T').mean().reset_index()
                                
                                # Combine resampled older data with recent data
                                option_info["price_history"] = pd.concat([resampled, recent_data], ignore_index=True)
        
        # Process all symbols if none specified
        else:
            with stocks_data_lock:
                for symbol, stock_info in stocks_data.items():
                    # Only process if we have significant data
                    if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                        # Keep last 200 points at full resolution
                        recent_data = stock_info["price_history"].tail(200)
                        
                        # Resample older data to 5-minute intervals
                        older_data = stock_info["price_history"].iloc[:-200]
                        if 'timestamp' in older_data.columns and len(older_data) > 100:
                            older_data['timestamp'] = pd.to_datetime(older_data['timestamp'])
                            older_data = older_data.set_index('timestamp')
                            resampled = older_data.resample('5T').mean().reset_index()
                            
                            # Combine resampled older data with recent data
                            stock_info["price_history"] = pd.concat([resampled, recent_data], ignore_index=True)
            
            with options_data_lock:
                for option_key, option_info in options_data.items():
                    if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                        # Keep last 200 points at full resolution
                        recent_data = option_info["price_history"].tail(200)
                        
                        # Resample older data to 5-minute intervals
                        older_data = option_info["price_history"].iloc[:-200]
                        if 'timestamp' in older_data.columns and len(older_data) > 100:
                            older_data['timestamp'] = pd.to_datetime(older_data['timestamp'])
                            older_data = older_data.set_index('timestamp')
                            resampled = older_data.resample('5T').mean().reset_index()
                            
                            # Combine resampled older data with recent data
                            option_info["price_history"] = pd.concat([resampled, recent_data], ignore_index=True)
        
        logger.info(f"Price history optimization completed for {'all symbols' if symbol is None else symbol}")
    except Exception as e:
        logger.error(f"Error optimizing price history: {e}")

def check_system_health():
    """
    Check the overall health of the system and recover if needed
    
    Returns:
        dict: Health status information
    """
    health = {
        "status": "healthy",
        "broker_connected": broker_connected,
        "data_thread_running": data_thread_started,
        "dashboard_initialized": dashboard_initialized,
        "memory_usage_mb": 0,
        "uptime_seconds": 0,
        "errors": {
            "connection": 0,
            "api": 0,
            "data_processing": 0
        },
        "warnings": []
    }
    
    # Check broker connection
    if not broker_connected:
        health["status"] = "degraded"
        health["warnings"].append("Broker not connected")
        health["errors"]["connection"] = api_error_tracker.get_error_count("connect_to_broker")
    
    # Check API errors
    total_api_errors = sum(count for endpoint, count in api_error_tracker.error_counts.items() if "connect" not in endpoint)
    health["errors"]["api"] = total_api_errors
    
    if total_api_errors > 10:
        health["status"] = "degraded"
        health["warnings"].append(f"High API error count: {total_api_errors}")
    
    # Check thread health
    if not data_thread_started:
        health["status"] = "critical"
        health["warnings"].append("Data thread not running")
    
    # Check memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        health["memory_usage_mb"] = round(memory_mb, 2)
        
        if memory_mb > 500:  # Over 500 MB is high
            health["status"] = "degraded"
            health["warnings"].append(f"High memory usage: {round(memory_mb)} MB")
    except:
        health["memory_usage_mb"] = -1
    
    # Check last data updates
    current_time = datetime.now()
    with threading.Lock():
        for symbol, last_update in last_data_update["stocks"].items():
            if last_update and (current_time - last_update).total_seconds() > 300:  # 5 minutes
                health["warnings"].append(f"Stale data for {symbol}: Last update {(current_time - last_update).total_seconds():.0f}s ago")
    
    # Count active tasks
    with trading_state_lock:
        active_trades_count = sum(1 for v in trading_state.active_trades.values() if v)
    health["active_trades"] = active_trades_count
    
    # Auto-recovery for critical status
    if health["status"] == "critical":
        logger.warning("Critical system health detected, attempting recovery")
        recovery_result = attempt_recovery()
        health["recovery_attempted"] = True
        health["recovery_success"] = recovery_result
    
    return health

def attempt_recovery():
    """
    Attempt to recover from a critical error state
    
    Returns:
        bool: True if recovery was successful, False otherwise
    """
    global smart_api, broker_connected, broker_error_message
    
    logger.warning("Attempting system recovery...")
    
    # 1. Reset connection state
    smart_api = None
    broker_connected = False
    broker_error_message = "Recovery in progress..."
    
    # 2. Clear API error tracker
    api_error_tracker.reset()
    
    # 3. Clear rate limit handler
    rate_limit_handler.rate_limit_hits.clear()
    
    # 4. Attempt to reconnect to broker
    try:
        logger.info("Attempting broker reconnection...")
        connection_success = connect_to_broker_with_backoff()
        
        if connection_success:
            logger.info("Recovery successful: Broker reconnected")
            
            # 5. Update all data
            update_all_stocks()
            update_all_options()
            update_all_pcr_data()
            
            return True
        else:
            logger.warning("Recovery partial: Broker reconnection failed")
            return False
            
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return False

def fetch_with_error_handling(fetch_func, item_key, max_retries=3, retry_delay=1.0):
    """
    Generic wrapper for fetch functions with consistent error handling
    
    Args:
        fetch_func (callable): The fetch function to call
        item_key (str): Key/name of the item being fetched (for logging)
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        retry_delay (float, optional): Delay between retries in seconds. Defaults to 1.0.
        
    Returns:
        The result of the fetch function, or None if all retries failed
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = fetch_func(item_key)
            if result:
                # Reset error tracker on success
                api_error_tracker.reset(f"fetch_{item_key}")
                return result
                
            # If result is falsy (False, None, etc.), it's an error
            logger.warning(f"Fetch failed for {item_key} (attempt {attempt}/{max_retries})")
            
            if attempt < max_retries:
                # Add some jitter to the retry delay
                jitter = random.uniform(0, 0.2 * retry_delay)
                sleep_time = retry_delay + jitter
                time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Exception fetching {item_key} (attempt {attempt}/{max_retries}): {e}")
            
            # Record error
            api_error_tracker.record_error(f"fetch_{item_key}")
            
            if attempt < max_retries:
                # Add some jitter to the retry delay
                jitter = random.uniform(0, 0.2 * retry_delay)
                sleep_time = retry_delay + jitter
                time.sleep(sleep_time)
    
    # All retries failed
    return None

def batch_process_data_items(items, process_func, batch_size=10, delay=0.01):
    """
    Process a list of items in batches with a small delay between batches
    to avoid overwhelming the system or APIs
    
    Args:
        items (list): List of items to process
        process_func (callable): Function to process each item
        batch_size (int, optional): Batch size. Defaults to 10.
        delay (float, optional): Delay between batches in seconds. Defaults to 0.01.
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        
        for item in batch:
            try:
                process_func(item)
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
        
        # Small delay between batches
        if i + batch_size < len(items):
            time.sleep(delay)

# ============ UI Components ============
def create_notification_container():
    return html.Div(
        [
            dbc.Toast(
                id="notification-toast",
                header="Notification",
                is_open=False,
                dismissable=True,
                duration=4000,
                icon="primary",
                style={
                    "position": "fixed", 
                    "top": 20, 
                    "right": 20, 
                    "width": "350px", 
                    "z-index": 1000,
                    "box-shadow": "0 4px 10px rgba(0, 0, 0, 0.3)"
                }
            ),
        ]
    )

def create_stock_control_card():
    return dbc.Card(
        [
            dbc.CardHeader(html.H4("Add Stock/Index", className="text-warning mb-0"), 
                          style=custom_css["header"]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Input(
                                id="add-stock-symbol", 
                                placeholder="Symbol (e.g., NIFTY, RELIANCE)", 
                                type="text",
                                className="border-0 bg-dark text-light mb-2"
                            ),
                            html.Div(id="search-results", className="search-dropdown d-none")
                        ], className="position-relative"),
                    ], width=6),
                    dbc.Col([
                        dbc.Select(
                            id="add-stock-type",
                            options=[
                                {"label": "INDEX", "value": "INDEX"},
                                {"label": "STOCK", "value": "STOCK"}
                            ],
                            value="STOCK",
                            className="border-0 bg-dark text-light"
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Add", id="add-stock-button", color="primary", className="w-100"),
                    ], width=3),
                ]),
                html.Div(id="add-stock-message", className="mt-2"),
                html.Div(id="search-status", className="mt-2 text-muted small")
            ], className="px-4 py-3")
        ], 
        style=custom_css["card"],
        className="mb-3 border-warning"
    )

# Continuation of the UI components section

def create_stock_option_card(symbol):
    """Create a card for a stock with its options."""
    stock_info = stocks_data.get(symbol, {})
    stock_type = stock_info.get("type", "STOCK")
    
    # Get predicted strategy if available
    predicted_strategy = stock_info.get("predicted_strategy", None)
    strategy_confidence = stock_info.get("strategy_confidence", 0)
    history_fetched = stock_info.get("history_fetched", False)

    # Create strategy badge if a strategy is predicted with confidence
    strategy_badge = ""
    if predicted_strategy and strategy_confidence > 0.6:
        badge_class = "badge bg-success ml-2" if strategy_confidence > 0.8 else "badge bg-info ml-2"
        strategy_badge = html.Span(f"Best: {predicted_strategy} ({strategy_confidence:.1%})", className=badge_class)
    
    history_badge = ""
    if history_fetched:
        last_fetch = stock_info.get("last_history_fetch")
        if last_fetch:
            fetch_time = last_fetch.strftime("%H:%M:%S")
            history_badge = html.Span(f"History: {fetch_time}", className="badge bg-info ms-2 small")
        else:
            history_badge = html.Span("History: Yes", className="badge bg-info ms-2 small")
    
    # Check if this stock has news
    news_mentions = news_data.get("mentions", {}).get(symbol, [])
    news_badge = ""
    if news_mentions:
        news_badge = html.Span(f"News: {len(news_mentions)}", className="badge bg-warning ms-2 small")
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H4(symbol, className="text-info d-inline me-1 mb-0"),
                html.Span(f"({stock_type})", className="text-muted small me-2"),
                html.Span(id={"type": "data-source-badge", "index": symbol}, className="badge ms-2 smooth-transition"),
                strategy_badge,
                history_badge,
                news_badge,
                dbc.Button("×", id={"type": "remove-stock-btn", "index": symbol}, 
                         color="danger", size="sm", className="float-end"),
                dbc.Button("Fetch History", id={"type": "fetch-history-btn", "index": symbol},
                         color="primary", size="sm", className="float-end me-2")
            ])
        ], style=custom_css["header"]),

        dbc.CardBody([
            # Stock Information Row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Price: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-price", "index": symbol}, className="fs-4 text-light dynamic-update price-element")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Change: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-change", "index": symbol}, className="text-light dynamic-update price-element")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("OHLC: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-ohlc", "index": symbol}, className="text-light small smooth-transition")
                    ], className="mb-1"),
                ], width=4),
                
                dbc.Col([
                    html.Div([
                        html.Span("PCR: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-pcr", "index": symbol}, className="text-light smooth-transition numeric-value"),
                        html.Span(id={"type": "pcr-strength", "index": symbol}, className="ms-2 small smooth-transition")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Sentiment: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-sentiment", "index": symbol}, className="badge smooth-transition")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("S/R Levels: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-sr-levels", "index": symbol}, className="text-light small smooth-transition")
                    ])
                ], width=4),
                
                dbc.Col([
                    html.Div([
                        html.Span("Volatility: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-volatility", "index": symbol}, className="text-light smooth-transition")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-signal", "index": symbol}, className="badge smooth-transition")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Last Update: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-last-update", "index": symbol}, className="text-light small smooth-transition")
                    ])
                ], width=4)
            ], className="mb-3"),
            
            # Options Row
            dbc.Row([
                # CE Option Column
                dbc.Col([
                    html.Div([
                        html.H5("Call Option", className="d-inline"),
                        html.Span(id={"type": "ce-strike", "index": symbol}, className="badge bg-secondary ms-2"),
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Span("Price: ", className="text-muted me-2"),
                        html.Span(id={"type": "ce-price", "index": symbol}, className="fs-5 text-light dynamic-update price-element")
                    ], className="mb-1"),
                    
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-2"),
                        html.Span(id={"type": "ce-signal", "index": symbol}, className="badge smooth-transition badge-transition")
                    ], className="mb-1"),
                    
                    html.Div([
                        html.Span("Strength: ", className="text-muted me-2"),
                        html.Div([
                            dbc.Progress(
                                id={"type": "ce-strength", "index": symbol},
                                value=0, 
                                className="progress-bar"
                            )
                        ], className="w-75 d-inline-block")
                    ], className="mb-1"),
                    
                    html.Div([
                        dbc.Button(
                            "Trade", 
                            id={"type": "trade-ce-btn", "index": symbol},
                            color="success", 
                            size="sm",
                            className="mt-1",
                            disabled=not broker_connected
                        )
                    ])
                ], width=6),
                
                # PE Option Column
                dbc.Col([
                    html.Div([
                        html.H5("Put Option", className="d-inline"),
                        html.Span(id={"type": "pe-strike", "index": symbol}, className="badge bg-secondary ms-2"),
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Span("Price: ", className="text-muted me-2"),
                        html.Span(id={"type": "pe-price", "index": symbol}, className="fs-5 text-light dynamic-update price-element")
                    ], className="mb-1"),
                    
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-2"),
                        html.Span(id={"type": "pe-signal", "index": symbol}, className="badge smooth-transition badge-transition")
                    ], className="mb-1"),
                    
                    html.Div([
                        html.Span("Strength: ", className="text-muted me-2"),
                        html.Div([
                            dbc.Progress(
                                id={"type": "pe-strength", "index": symbol},
                                value=0, 
                                className="progress-bar"
                            )
                        ], className="w-75 d-inline-block")
                    ], className="mb-1"),
                    
                    html.Div([
                        dbc.Button(
                            "Trade", 
                            id={"type": "trade-pe-btn", "index": symbol},
                            color="success", 
                            size="sm",
                            className="mt-1",
                            disabled=not broker_connected
                        )
                    ])
                ], width=6)
            ])
        ], className="px-3 py-3")
    ], style=custom_css["card"], className="mb-3")

def create_broker_connection_card():
    """Create a card for broker connection status."""
    connection_status = "Connected" if broker_connected else "Disconnected"
    status_class = "text-success" if broker_connected else "text-danger"
    
    return dbc.Card([
        dbc.CardHeader(html.H4("Broker Connection", className="text-primary mb-0"), style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Status: ", className="text-muted me-2"),
                        html.Span(id="broker-status", className=f"fs-5 {status_class}")
                    ], className="mb-2"),
                    html.Div(id="broker-error-message", className="text-warning small"),
                ], width=7),
                dbc.Col([
                    dbc.Button(
                        "Connect", 
                        id="connect-broker-button", 
                        color="success", 
                        className="w-100 mb-2",
                        disabled=broker_connected
                    ),
                    dbc.Button(
                        "Disconnect", 
                        id="disconnect-broker-button", 
                        color="danger", 
                        className="w-100",
                        disabled=not broker_connected
                    )
                ], width=5)
            ])
        ], className="px-4 py-3")
    ], style=custom_css["card"], className="mb-3")

def create_trade_control_card():
    """Create a card for trading control settings."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Trading Controls", className="text-success mb-0"), style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="toggle-scalp",
                        label="Scalping",
                        value=strategy_settings["SCALP_ENABLED"],
                        className="mb-2"
                    ),
                    dbc.Switch(
                        id="toggle-momentum",
                        label="Momentum",
                        value=strategy_settings["MOMENTUM_ENABLED"],
                        className="mb-2"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Switch(
                        id="toggle-swing",
                        label="Swing",
                        value=strategy_settings["SWING_ENABLED"],
                        className="mb-2"
                    ),
                    dbc.Switch(
                        id="toggle-news",
                        label="News-based",
                        value=strategy_settings["NEWS_ENABLED"],
                        className="mb-2"
                    ),
                ], width=6)
            ]),
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Risk per trade: ", className="text-muted me-2"),
                        html.Span(f"{RISK_PER_TRADE}%", className="text-light")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Max trades/day: ", className="text-muted me-2"),
                        html.Span(id="max-trades-display", className="text-light")
                    ], className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Max daily loss: ", className="text-muted me-2"),
                        html.Span(f"{MAX_LOSS_PERCENTAGE}%", className="text-light")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Trades today: ", className="text-muted me-2"),
                        html.Span(id="trades-today-display", className="text-light")
                    ], className="mb-2"),
                ], width=6)
            ])
        ], className="px-4 py-3")
    ], style=custom_css["card"], className="mb-3")

def create_active_trades_card():
    """Create a card for active trades."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Active Trades", className="text-warning mb-0 d-inline"),
            dbc.Badge(id="active-trades-count", color="light", className="ms-2 text-dark")
        ], style=custom_css["header"]),
        dbc.CardBody([
            html.Div(id="active-trades-container", className="trades-container")
        ], className="px-2 py-2")
    ], style=custom_css["card"], className="mb-3")

def create_trade_history_card():
    """Create a card for trade history."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Trade History", className="text-info mb-0 d-inline"),
            dbc.Badge(id="trade-history-count", color="light", className="ms-2 text-dark")
        ], style=custom_css["header"]),
        dbc.CardBody([
            html.Div(id="trade-history-container", className="trades-container")
        ], className="px-2 py-2")
    ], style=custom_css["card"], className="mb-3")

def create_performance_card():
    """Create a card for trading performance metrics."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Trading Performance", className="text-primary mb-0"), style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Total P&L: ", className="text-muted me-2"),
                        html.Span(id="total-pnl", className="fs-4 dynamic-update value-update")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Daily P&L: ", className="text-muted me-2"),
                        html.Span(id="daily-pnl", className="fs-5 dynamic-update value-update")
                    ], className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Win Rate: ", className="text-muted me-2"),
                        html.Span(id="win-rate", className="fs-5 text-light")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Win/Loss: ", className="text-muted me-2"),
                        html.Span(id="win-loss-ratio", className="text-light")
                    ], className="mb-2"),
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Avg Win: ", className="text-muted me-2"),
                        html.Span(id="avg-win", className="text-light")
                    ], className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Avg Loss: ", className="text-muted me-2"),
                        html.Span(id="avg-loss", className="text-light")
                    ], className="mb-2"),
                ], width=6)
            ])
        ], className="px-4 py-3")
    ], style=custom_css["card"], className="mb-3")

def create_news_card():
    """Create a card for news feed."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Market News", className="text-success mb-0 d-inline"),
            dbc.Badge(id="news-count", color="light", className="ms-2 text-dark")
        ], style=custom_css["header"]),
        dbc.CardBody([
            html.Div(id="news-container", className="news-container")
        ], className="px-2 py-2 overflow-auto", style={"max-height": "300px"})
    ], style=custom_css["card"], className="mb-3")

def create_market_overview_card():
    """Create a card for market overview."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Market Overview", className="text-info mb-0"), style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Market Sentiment: ", className="text-muted me-2"),
                        html.Span(id="overall-sentiment", className="badge badge-transition")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Active Signals: ", className="text-muted me-2"),
                        html.Span(id="active-signals-count", className="text-light")
                    ], className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("System Health: ", className="text-muted me-2"),
                        html.Span(id="system-health", className="badge badge-transition")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Last Update: ", className="text-muted me-2"),
                        html.Span(id="last-update-time", className="text-light small")
                    ], className="mb-2"),
                ], width=6)
            ])
        ], className="px-4 py-3")
    ], style=custom_css["card"], className="mb-3")

def create_active_trade_item(option_key):
    """Create an item for an active trade."""
    with options_data_lock:
        if option_key not in options_data:
            return html.Div()
        
        option_info = options_data[option_key]
        parent_symbol = option_info.get("parent_symbol", "")
        option_type = option_info.get("option_type", "")
        strike = option_info.get("strike", "")
    
    with trading_state_lock:
        entry_price = trading_state.entry_price.get(option_key, 0)
        entry_time = trading_state.entry_time.get(option_key, datetime.now())
        stop_loss = trading_state.stop_loss.get(option_key, 0)
        target = trading_state.target.get(option_key, 0)
        quantity = trading_state.quantity.get(option_key, 0)
        strategy = trading_state.strategy_type.get(option_key, "")
        
    # Calculate duration
    duration_min = (datetime.now() - entry_time).total_seconds() / 60 if entry_time else 0
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span(f"{parent_symbol} {strike} {option_type}", className="fs-5 text-light"),
                        html.Span(strategy, className="badge bg-info ms-2")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Entry: ", className="text-muted small"),
                        html.Span(f"₹{entry_price:.2f} x {quantity}", className="text-light small"),
                        html.Span(f" ({entry_time.strftime('%H:%M:%S')})", className="text-muted small")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Duration: ", className="text-muted small"),
                        html.Span(id={"type": "trade-duration", "index": option_key}, className="text-light small")
                    ], className="mb-1"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Current: ", className="text-muted small"),
                        html.Span(id={"type": "trade-current", "index": option_key}, className="text-light small")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("P&L: ", className="text-muted small"),
                        html.Span(id={"type": "trade-pnl", "index": option_key}, className="small dynamic-update")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("SL: ", className="text-muted small"),
                        html.Span(f"₹{stop_loss:.2f}", className="text-danger small"),
                        html.Span(" | Target: ", className="text-muted small"),
                        html.Span(f"₹{target:.2f}", className="text-success small")
                    ], className="mb-1"),
                ], width=4),
                dbc.Col([
                    dbc.Button("Exit", id={"type": "exit-trade-btn", "index": option_key}, 
                             color="danger", size="sm", className="w-100 mb-1"),
                ], width=2)
            ])
        ], className="p-2")
    ], style=custom_css["card_alt"], className="mb-2 trade-item card-update")

def create_trade_history_item(trade):
    """Create an item for a historical trade."""
    option_key = trade.get('option_key', '')
    parent_symbol = trade.get('parent_symbol', '')
    strategy_type = trade.get('strategy_type', '')
    option_type = trade.get('option_type', '')
    strike = trade.get('strike', '')
    
    entry_time = trade.get('entry_time', datetime.now())
    exit_time = trade.get('exit_time', datetime.now())
    duration_min = trade.get('duration_minutes', 0)
    
    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    quantity = trade.get('quantity', 0)
    pnl = trade.get('pnl', 0)
    pnl_pct = trade.get('pnl_pct', 0)
    reason = trade.get('reason', 'Unknown')
    
    # Determine text color based on P&L
    pnl_color = "text-success" if pnl > 0 else "text-danger" if pnl < 0 else "text-light"
    duration_str = f"{duration_min:.1f}m" if duration_min < 60 else f"{duration_min/60:.1f}h"
    
    # Create the trade history item
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span(f"{parent_symbol} {strike} {option_type}", className="text-light"),
                        html.Span(strategy_type, className="badge bg-info ms-2 small")
                    ], className="mb-1"),
                    html.Div([
                        html.Span(f"{entry_time.strftime('%H:%M')} → {exit_time.strftime('%H:%M')} ({duration_str})", 
                                 className="text-muted small")
                    ], className="mb-1"),
                ], width=5),
                dbc.Col([
                    html.Div([
                        html.Span(f"₹{entry_price:.2f} → ₹{exit_price:.2f}", className="text-light small")
                    ], className="mb-1"),
                    html.Div([
                        html.Span(f"{reason}", className="badge bg-secondary small")
                    ], className="mb-1"),
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Span(f"₹{pnl:.2f}", className=f"{pnl_color}")
                    ], className="mb-1"),
                    html.Div([
                        html.Span(f"{pnl_pct:.2f}%", className=f"{pnl_color} small")
                    ], className="mb-1"),
                ], width=3)
            ])
        ], className="p-2")
    ], style=custom_css["card_alt"], className="mb-2 trade-history-item")

def create_news_item(news):
    """Create an item for a news feed."""
    title = news.get('title', '')
    source = news.get('source', 'Unknown')
    timestamp = news.get('timestamp', datetime.now())
    url = news.get('url', '#')
    
    # Format timestamp
    time_str = timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else timestamp
    
    # Calculate how recent the news is
    recency = None
    if isinstance(timestamp, datetime):
        recency = (datetime.now() - timestamp).total_seconds() / 60  # in minutes
    
    # Determine if news is recent (less than 30 minutes old)
    is_recent = recency is not None and recency < 30
    recency_badge = html.Span("NEW", className="badge bg-danger ms-2") if is_recent else ""
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.A(title, href=url, target="_blank", className="text-light text-decoration-none"),
                recency_badge
            ], className="mb-1"),
            html.Div([
                html.Span(source, className="badge bg-secondary me-2"),
                html.Span(time_str, className="text-muted small")
            ])
        ], className="p-2")
    ], style=custom_css["card_alt"], className="mb-2 news-item")

# Define the layout of the app
app.layout = html.Div([
    dbc.Container([
        # Header
        html.Div([
            html.H2("Options Trading Dashboard", className="text-light mb-0"),
            html.P("Real-time market analysis and automated trading signals", className="text-muted")
        ], className="py-3 mb-3"),
        
        # Main content
        dbc.Row([
            # Left Column
            dbc.Col([
                create_broker_connection_card(),
                create_trade_control_card(),
                create_stock_control_card(),
                create_performance_card(),
                create_market_overview_card(),
            ], md=4),
            
            # Center Column
            dbc.Col([
                # Stocks Container
                html.Div(id="stocks-container", className="stocks-container"),
            ], md=5),
            
            # Right Column
            dbc.Col([
                create_active_trades_card(),
                create_trade_history_card(),
                create_news_card(),
            ], md=3),
        ]),
    ], fluid=True, className="px-4"),
    
    # Notification area
    create_notification_container(),
    
    # Interval for updating data
    dcc.Interval(
        id='update-interval',
        interval=1000,  # Update every 1 second
        n_intervals=0
    ),
    
    # Store components for holding data across callbacks
    dcc.Store(id='stocks-data-store'),
    dcc.Store(id='options-data-store'),
    dcc.Store(id='trading-state-store'),
    dcc.Store(id='pcr-data-store'),
    dcc.Store(id='news-data-store'),
    dcc.Store(id='last-update-store'),
], style={"backgroundColor": COLOR_SCHEME["bg_dark"], "minHeight": "100vh"})

# ============ UI Callbacks ============
# Callback to add stock
@app.callback(
    [Output("add-stock-message", "children"),
     Output("add-stock-symbol", "value")],
    [Input("add-stock-button", "n_clicks")],
    [State("add-stock-symbol", "value"),
     State("add-stock-type", "value")]
)
def add_stock_callback(n_clicks, symbol, stock_type):
    if not n_clicks or not symbol:
        return "", None
    
    symbol = symbol.upper().strip()
    
    # Check if symbol already exists
    with stocks_data_lock:
        if symbol in stocks_data:
            return html.Span(f"Symbol {symbol} already exists", className="text-warning"), symbol
    
    # Try to add the stock
    stock_data = search_stock_in_csv(symbol)
    if stock_data:
        success = add_stock_from_csv_data(stock_data)
    else:
        success = add_stock(symbol, None, "NSE", stock_type)
    
    if success:
        # Try to fetch data if broker connected
        if broker_connected:
            fetch_stock_data(symbol)
        
        return html.Span(f"Added {symbol} successfully", className="text-success"), ""
    else:
        return html.Span(f"Failed to add {symbol}", className="text-danger"), symbol

# Callback to remove stock
@app.callback(
    Output({"type": "remove-stock-btn", "index": MATCH}, "disabled"),
    Input({"type": "remove-stock-btn", "index": MATCH}, "n_clicks"),
    State({"type": "remove-stock-btn", "index": MATCH}, "id")
)
def remove_stock_callback(n_clicks, id_dict):
    if not n_clicks:
        return False
    
    symbol = id_dict["index"]
    success = remove_stock(symbol)
    
    # Return True to disable the button after removal
    return success

# Callback to search stocks while typing
@app.callback(
    [Output("search-results", "children"),
     Output("search-results", "className"),
     Output("search-status", "children")],
    [Input("add-stock-symbol", "value")],
    prevent_initial_call=True
)
def search_stocks_callback(search_term):
    if not search_term or len(search_term) < 2:
        return [], "search-dropdown d-none", ""
    
    # Search stocks in CSV
    search_term = search_term.upper().strip()
    results = search_stocks_in_csv(search_term, limit=8)
    
    if not results:
        return [], "search-dropdown d-none", html.Span("No matching stocks found", className="text-warning")
    
    # Create result items
    result_items = []
    for stock in results:
        symbol = stock.get('symbol', '')
        name = stock.get('name', '')
        stock_type = "INDEX" if "NIFTY" in symbol or "SENSEX" in symbol else "STOCK"
        
        result_items.append(
            html.Div(
                html.Div([
                    html.Span(symbol, className="text-light"),
                    html.Span(f" ({stock_type})", className="text-muted small"),
                    html.Span(f" - {name}", className="text-muted small d-none d-md-inline")
                ]),
                className="search-item",
                id={"type": "search-result-item", "index": symbol}
            )
        )
    
    search_status = html.Span(f"Found {len(results)} matching stocks", className="text-muted")
    
    return result_items, "search-dropdown", search_status

# Callback for search result click
@app.callback(
    Output("add-stock-symbol", "value", allow_duplicate=True),
    Input({"type": "search-result-item", "index": ALL}, "n_clicks"),
    State({"type": "search-result-item", "index": ALL}, "id"),
    prevent_initial_call=True
)
def search_result_click(n_clicks_list, id_list):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Find which item was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        button_id_dict = json.loads(button_id)
        symbol = button_id_dict["index"]
        return symbol
    except:
        return dash.no_update

# Callback to connect to broker
@app.callback(
    [Output("broker-status", "children"),
     Output("broker-status", "className"),
     Output("broker-error-message", "children"),
     Output("connect-broker-button", "disabled"),
     Output("disconnect-broker-button", "disabled")],
    [Input("connect-broker-button", "n_clicks"),
     Input("disconnect-broker-button", "n_clicks"),
     Input("update-interval", "n_intervals")]
)
def broker_connection_callback(connect_clicks, disconnect_clicks, n_intervals):
    global smart_api, broker_connected, broker_error_message
    
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == "connect-broker-button" and connect_clicks:
        connect_to_broker()
    elif triggered_id == "disconnect-broker-button" and disconnect_clicks:
        # Disconnect logic
        smart_api = None
        broker_connected = False
        broker_error_message = None
    
    # Check connection status
    status_text = "Connected" if broker_connected else "Disconnected"
    status_class = "fs-5 text-success" if broker_connected else "fs-5 text-danger"
    error_message = broker_error_message if broker_error_message else ""
    
    return status_text, status_class, error_message, broker_connected, not broker_connected

# Callback to toggle trading strategies
@app.callback(
    [Output("toggle-scalp", "value"),
     Output("toggle-swing", "value"),
     Output("toggle-momentum", "value"),
     Output("toggle-news", "value")],
    [Input("toggle-scalp", "value"),
     Input("toggle-swing", "value"),
     Input("toggle-momentum", "value"),
     Input("toggle-news", "value")]
)
def toggle_strategies_callback(scalp, swing, momentum, news):
    global strategy_settings
    
    # Update strategy settings
    strategy_settings["SCALP_ENABLED"] = scalp
    strategy_settings["SWING_ENABLED"] = swing
    strategy_settings["MOMENTUM_ENABLED"] = momentum
    strategy_settings["NEWS_ENABLED"] = news
    
    logger.info(f"Strategy settings updated: {strategy_settings}")
    
    return scalp, swing, momentum, news

@app.callback(
    Output({"type": "fetch-history-btn", "index": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "fetch-history-btn", "index": MATCH}, "n_clicks"),
    State({"type": "fetch-history-btn", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def fetch_history_callback(n_clicks, id_dict):
    if not n_clicks:
        return False
    
    symbol = id_dict["index"]
    success = load_historical_data(symbol, force_refresh=True)
    
    # Disable button temporarily to prevent spam clicks
    return True
# Callback to enable history button after timeout
@app.callback(
    Output({"type": "fetch-history-btn", "index": MATCH}, "disabled"),
    Input("update-interval", "n_intervals"),
    State({"type": "fetch-history-btn", "index": MATCH}, "disabled"),
    State({"type": "fetch-history-btn", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def enable_history_button(n_intervals, is_disabled, id_dict):
    if not is_disabled:
        return False
    
    # Re-enable after checking if history was fetched
    symbol = id_dict["index"]
    with stocks_data_lock:
        if symbol in stocks_data:
            last_fetch = stocks_data[symbol].get("last_history_fetch")
            if last_fetch and (datetime.now() - last_fetch).total_seconds() > 5:
                return False
    
    return is_disabled

# Callback to exit a trade
@app.callback(
    Output({"type": "exit-trade-btn", "index": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "exit-trade-btn", "index": MATCH}, "n_clicks"),
    State({"type": "exit-trade-btn", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def exit_trade_callback(n_clicks, id_dict):
    if not n_clicks:
        return False
    
    option_key = id_dict["index"]
    success = exit_trade(option_key, reason="Manual Exit")
    
    # Disable button after exit
    return True

# Callback to initiate a trade
@app.callback(
    Output({"type": "trade-ce-btn", "index": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "trade-ce-btn", "index": MATCH}, "n_clicks"),
    State({"type": "trade-ce-btn", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def trade_ce_callback(n_clicks, id_dict):
    if not n_clicks or not broker_connected:
        return False
    
    symbol = id_dict["index"]
    
    # Find the CE option key for this symbol
    with stocks_data_lock:
        if symbol in stocks_data:
            option_key = stocks_data[symbol].get("primary_ce")
            if option_key:
                # Check if already in a trade
                with trading_state_lock:
                    is_active = trading_state.active_trades.get(option_key, False)
                    if is_active:
                        return True
                
                # Determine best strategy for this option
                with stocks_data_lock:
                    predicted_strategy = stocks_data[symbol].get("predicted_strategy")
                
                strategy_type = predicted_strategy if predicted_strategy else "MANUAL"
                success = enter_trade(option_key, strategy_type, "MANUAL")
                
                # Disable button temporarily to prevent spam clicks
                return True
    
    return False

# Similar callback for PE trades
@app.callback(
    Output({"type": "trade-pe-btn", "index": MATCH}, "disabled", allow_duplicate=True),
    Input({"type": "trade-pe-btn", "index": MATCH}, "n_clicks"),
    State({"type": "trade-pe-btn", "index": MATCH}, "id"),
    prevent_initial_call=True
)
def trade_pe_callback(n_clicks, id_dict):
    if not n_clicks or not broker_connected:
        return False
    
    symbol = id_dict["index"]
    
    # Find the PE option key for this symbol
    with stocks_data_lock:
        if symbol in stocks_data:
            option_key = stocks_data[symbol].get("primary_pe")
            if option_key:
                # Check if already in a trade
                with trading_state_lock:
                    is_active = trading_state.active_trades.get(option_key, False)
                    if is_active:
                        return True
                
                # Determine best strategy for this option
                with stocks_data_lock:
                    predicted_strategy = stocks_data[symbol].get("predicted_strategy")
                
                strategy_type = predicted_strategy if predicted_strategy else "MANUAL"
                success = enter_trade(option_key, strategy_type, "MANUAL")
                
                # Disable button temporarily to prevent spam clicks
                return True
    
    return False

# Callback to re-enable trade buttons after timeout
@app.callback(
    [Output({"type": "trade-ce-btn", "index": MATCH}, "disabled"),
     Output({"type": "trade-pe-btn", "index": MATCH}, "disabled")],
    Input("update-interval", "n_intervals"),
    [State({"type": "trade-ce-btn", "index": MATCH}, "disabled"),
     State({"type": "trade-pe-btn", "index": MATCH}, "disabled"),
     State({"type": "trade-ce-btn", "index": MATCH}, "id")]
)
def enable_trade_buttons(n_intervals, ce_disabled, pe_disabled, id_dict):
    if not ce_disabled and not pe_disabled:
        return False, False
    
    # Re-enable after checking if not in a trade
    symbol = id_dict["index"]
    
    with stocks_data_lock:
        if symbol in stocks_data:
            ce_key = stocks_data[symbol].get("primary_ce")
            pe_key = stocks_data[symbol].get("primary_pe")
            
            ce_is_active = False
            pe_is_active = False
            
            with trading_state_lock:
                if ce_key:
                    ce_is_active = trading_state.active_trades.get(ce_key, False)
                if pe_key:
                    pe_is_active = trading_state.active_trades.get(pe_key, False)
            
            return ce_is_active or not broker_connected, pe_is_active or not broker_connected
    
    return not broker_connected, not broker_connected

# Main callback to update all stock cards in the UI
@app.callback(
    Output("stocks-container", "children"),
    [Input("update-interval", "n_intervals"),
     Input("stocks-data-store", "data")]
)
def update_stocks_container(n_intervals, stocks_data_json):
    # Get the list of stocks
    with stocks_data_lock:
        stock_symbols = list(stocks_data.keys())
    
    # Create a card for each stock
    stock_cards = [create_stock_option_card(symbol) for symbol in stock_symbols]
    
    if not stock_cards:
        return html.Div(
            html.P("Add stocks above to start monitoring...", className="text-center text-muted mt-4"),
            className="text-center"
        )
    
    return stock_cards

# Callback to update the active trades container
@app.callback(
    [Output("active-trades-container", "children"),
     Output("active-trades-count", "children")],
    [Input("update-interval", "n_intervals"),
     Input("trading-state-store", "data")]
)
def update_active_trades_container(n_intervals, trading_state_json):
    # Get active trades
    with trading_state_lock:
        active_trades = [key for key, is_active in trading_state.active_trades.items() if is_active]
    
    # Create a card for each active trade
    active_trade_cards = [create_active_trade_item(option_key) for option_key in active_trades]
    active_count = len(active_trades)
    
    if not active_trade_cards:
        return html.Div(
            html.P("No active trades", className="text-center text-muted mt-3"),
            className="text-center"
        ), "0"
    
    return active_trade_cards, str(active_count)

# Callback to update the trade history container
@app.callback(
    [Output("trade-history-container", "children"),
     Output("trade-history-count", "children")],
    [Input("update-interval", "n_intervals"),
     Input("trading-state-store", "data")]
)
def update_trade_history_container(n_intervals, trading_state_json):
    # Get trade history
    with trading_state_lock:
        # Get most recent 10 trades
        trade_history = trading_state.trades_history[-10:] if trading_state.trades_history else []
    
    # Create a card for each historical trade
    trade_history_cards = [create_trade_history_item(trade) for trade in reversed(trade_history)]
    history_count = len(trading_state.trades_history)
    
    if not trade_history_cards:
        return html.Div(
            html.P("No trade history", className="text-center text-muted mt-3"),
            className="text-center"
        ), "0"
    
    return trade_history_cards, str(history_count)

# Callback to update the news container
@app.callback(
    [Output("news-container", "children"),
     Output("news-count", "children")],
    [Input("update-interval", "n_intervals"),
     Input("news-data-store", "data")]
)
def update_news_container(n_intervals, news_data_json):
    # Get news items
    news_items = news_data.get("items", [])
    
    # Create an item for each news
    news_cards = [create_news_item(news) for news in news_items[:10]]  # Show only 10 most recent
    news_count = len(news_items)
    
    if not news_cards:
        return html.Div(
            html.P("No recent news", className="text-center text-muted mt-3"),
            className="text-center"
        ), "0"
    
    return news_cards, str(news_count)

# Callback to update performance metrics
@app.callback(
    [Output("total-pnl", "children"),
     Output("total-pnl", "className"),
     Output("daily-pnl", "children"),
     Output("daily-pnl", "className"),
     Output("win-rate", "children"),
     Output("win-loss-ratio", "children"),
     Output("avg-win", "children"),
     Output("avg-loss", "children"),
     Output("trades-today-display", "children"),
     Output("max-trades-display", "children")],
    [Input("update-interval", "n_intervals"),
     Input("trading-state-store", "data")]
)
def update_performance_metrics(n_intervals, trading_state_json):
    with trading_state_lock:
        total_pnl = trading_state.total_pnl
        daily_pnl = trading_state.daily_pnl
        trades_today = trading_state.trades_today
        wins = trading_state.wins
        losses = trading_state.losses
        trade_history = trading_state.trades_history
    
    # Calculate win rate
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    win_loss_ratio = f"{wins}/{losses}" if losses > 0 else f"{wins}/0"
    
    # Calculate average win and loss
    win_trades = [trade['pnl'] for trade in trade_history if trade['pnl'] > 0]
    loss_trades = [trade['pnl'] for trade in trade_history if trade['pnl'] < 0]
    
    avg_win = sum(win_trades) / len(win_trades) if win_trades else 0
    avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
    
    # Format outputs
    total_pnl_str = f"₹{total_pnl:.2f}"
    daily_pnl_str = f"₹{daily_pnl:.2f}"
    win_rate_str = f"{win_rate:.1f}%"
    avg_win_str = f"₹{avg_win:.2f}" if avg_win else "N/A"
    avg_loss_str = f"₹{avg_loss:.2f}" if avg_loss else "N/A"
    
    # Define classes
    total_pnl_class = "fs-4 text-success dynamic-update value-update" if total_pnl >= 0 else "fs-4 text-danger dynamic-update value-update"
    daily_pnl_class = "fs-5 text-success dynamic-update value-update" if daily_pnl >= 0 else "fs-5 text-danger dynamic-update value-update"
    
    return (
        total_pnl_str, total_pnl_class,
        daily_pnl_str, daily_pnl_class,
        win_rate_str, win_loss_ratio,
        avg_win_str, avg_loss_str,
        str(trades_today), str(MAX_TRADES_PER_DAY)
    )

# Callback to update market overview
@app.callback(
    [Output("overall-sentiment", "children"),
     Output("overall-sentiment", "className"),
     Output("active-signals-count", "children"),
     Output("system-health", "children"),
     Output("system-health", "className"),
     Output("last-update-time", "children")],
    [Input("update-interval", "n_intervals")]
)
def update_market_overview(n_intervals):
    # Get overall sentiment
    overall_sentiment = market_sentiment.get("overall", "NEUTRAL")
    
    # Set sentiment badge color
    sentiment_class = "badge "
    if "BULLISH" in overall_sentiment:
        sentiment_class += "bg-success"
    elif "BEARISH" in overall_sentiment:
        sentiment_class += "bg-danger"
    else:
        sentiment_class += "bg-secondary"
    
    # Count active signals
    active_signals = 0
    with options_data_lock:
        for option_info in options_data.values():
            signal = option_info.get("signal", 0)
            strength = option_info.get("strength", 0)
            if abs(signal) > 2 and strength > 5:
                active_signals += 1
    
    # Check system health
    health = check_system_health()
    health_status = health["status"]
    
    # Set health badge color
    health_class = "badge "
    if health_status == "healthy":
        health_class += "bg-success"
    elif health_status == "degraded":
        health_class += "bg-warning"
    else:
        health_class += "bg-danger"
    
    # Get current time
    current_time = datetime.now().strftime("%H:%M:%S")
    
    return (
        overall_sentiment, sentiment_class,
        str(active_signals),
        health_status.title(), health_class,
        current_time
    )

# Dynamic callbacks to update each stock's data in real-time
@app.callback(
    [Output({"type": "stock-price", "index": MATCH}, "children"),
     Output({"type": "stock-price", "index": MATCH}, "className"),
     Output({"type": "stock-change", "index": MATCH}, "children"),
     Output({"type": "stock-change", "index": MATCH}, "className"),
     Output({"type": "stock-ohlc", "index": MATCH}, "children"),
     Output({"type": "stock-pcr", "index": MATCH}, "children"),
     Output({"type": "pcr-strength", "index": MATCH}, "children"),
     Output({"type": "stock-sentiment", "index": MATCH}, "children"),
     Output({"type": "stock-sentiment", "index": MATCH}, "className"),
     Output({"type": "stock-sr-levels", "index": MATCH}, "children"),
     Output({"type": "stock-volatility", "index": MATCH}, "children"),
     Output({"type": "stock-signal", "index": MATCH}, "children"),
     Output({"type": "stock-signal", "index": MATCH}, "className"),
     Output({"type": "stock-last-update", "index": MATCH}, "children"),
     Output({"type": "data-source-badge", "index": MATCH}, "children"),
     Output({"type": "data-source-badge", "index": MATCH}, "className")],
    [Input("update-interval", "n_intervals")],
    [State({"type": "stock-price", "index": MATCH}, "id")]
)
def update_stock_data(n_intervals, id_dict):
    symbol = id_dict["index"]
    
    # Check if the stock exists
    with stocks_data_lock:
        if symbol not in stocks_data:
            # Return default values if stock doesn't exist
            return (
                "N/A", "fs-4 text-light dynamic-update price-element",
                "N/A", "text-light dynamic-update price-element",
                "N/A", "N/A", "", "NEUTRAL", "badge bg-secondary",
                "N/A", "N/A", "NEUTRAL", "badge bg-secondary",
                "N/A", "Unknown", "badge bg-secondary ms-2"
            )
        
        # Get stock data
        stock_info = stocks_data[symbol]
        price = stock_info.get("ltp")
        change = stock_info.get("change_percent")
        open_price = stock_info.get("open")
        high_price = stock_info.get("high")
        low_price = stock_info.get("low")
        prev_close = stock_info.get("previous")
        last_updated = stock_info.get("last_updated")
        data_source = stock_info.get("data_source", "unknown")
        support_levels = stock_info.get("support_levels", [])
        resistance_levels = stock_info.get("resistance_levels", [])
    
    # Get PCR data
    with pcr_data_lock:
        pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
        pcr_trend = pcr_data.get(symbol, {}).get("trend", "NEUTRAL")
        pcr_strength = pcr_data.get(symbol, {}).get("strength", 0.0)
    
    # Get sentiment
    sentiment = market_sentiment.get(symbol, "NEUTRAL")
    
    # Calculate volatility
    volatility = calculate_volatility(symbol)
    
    # Determine signal based on sentiment and PCR
    signal, signal_class = get_pcr_signal(symbol)
    
    # Format outputs
    price_str = f"₹{price:.2f}" if price is not None else "N/A"
    change_str = f"{change:.2f}%" if change is not None else "N/A"
    ohlc_str = f"O: {open_price:.2f} H: {high_price:.2f} L: {low_price:.2f} P: {prev_close:.2f}" if all(x is not None for x in [open_price, high_price, low_price, prev_close]) else "N/A"
    pcr_str = f"{pcr_value:.2f}" if pcr_value is not None else "N/A"
    
    # Format support/resistance levels
    sr_levels_str = ""
    if support_levels:
        sr_levels_str += f"S: {', '.join([str(round(level, 1)) for level in support_levels[:2]])}"
    if resistance_levels:
        sr_levels_str += f" R: {', '.join([str(round(level, 1)) for level in resistance_levels[:2]])}"
    if not sr_levels_str:
        sr_levels_str = "N/A"
    
    volatility_str = f"{volatility:.2f}" if volatility is not None else "N/A"
    last_update_str = last_updated.strftime("%H:%M:%S") if last_updated else "N/A"
    
    # Format PCR strength
    pcr_strength_str = ""
    if pcr_strength > 0.2:
        pcr_strength_str = "↓ Bullish" if pcr_strength > 0 else "↑ Bearish"
    
    # Determine class for price and change
    price_class = "fs-4 text-light dynamic-update price-element"
    change_class = "text-success dynamic-update price-element" if change and change >= 0 else "text-danger dynamic-update price-element"
    
    # Determine sentiment badge
    sentiment_class = "badge "
    if "BULLISH" in sentiment:
        sentiment_class += "bg-success"
    elif "BEARISH" in sentiment:
        sentiment_class += "bg-danger"
    elif "MODERATELY BULLISH" in sentiment:
        sentiment_class += "bg-success bg-opacity-50"
    elif "MODERATELY BEARISH" in sentiment:
        sentiment_class += "bg-danger bg-opacity-50"
    else:
        sentiment_class += "bg-secondary"
    
    # Determine data source badge
    data_source_badge = data_source.title()
    data_source_class = "badge ms-2 "
    if data_source == "broker":
        data_source_class += "bg-success"
    elif data_source == "yahoo":
        data_source_class += "bg-primary"
    elif data_source == "fallback":
        data_source_class += "bg-warning"
    else:
        data_source_class += "bg-secondary"
    
    return (
        price_str, price_class,
        change_str, change_class,
        ohlc_str,
        pcr_str, pcr_strength_str,
        sentiment, sentiment_class,
        sr_levels_str,
        volatility_str,
        signal, signal_class,
        last_update_str,
        data_source_badge, data_source_class
    )

# Callback to update option data for a stock
@app.callback(
    [Output({"type": "ce-strike", "index": MATCH}, "children"),
     Output({"type": "ce-price", "index": MATCH}, "children"),
     Output({"type": "ce-price", "index": MATCH}, "className"),
     Output({"type": "ce-signal", "index": MATCH}, "children"),
     Output({"type": "ce-signal", "index": MATCH}, "className"),
     Output({"type": "ce-strength", "index": MATCH}, "value"),
     Output({"type": "ce-strength", "index": MATCH}, "color"),
     Output({"type": "pe-strike", "index": MATCH}, "children"),
     Output({"type": "pe-price", "index": MATCH}, "children"),
     Output({"type": "pe-price", "index": MATCH}, "className"),
     Output({"type": "pe-signal", "index": MATCH}, "children"),
     Output({"type": "pe-signal", "index": MATCH}, "className"),
     Output({"type": "pe-strength", "index": MATCH}, "value"),
     Output({"type": "pe-strength", "index": MATCH}, "color")],
    [Input("update-interval", "n_intervals")],
    [State({"type": "ce-strike", "index": MATCH}, "id")]
)
def update_option_data(n_intervals, id_dict):
    symbol = id_dict["index"]
    
    # Check if the stock exists
    with stocks_data_lock:
        if symbol not in stocks_data:
            # Return default values if stock doesn't exist
            return (
                "N/A", "N/A", "fs-5 text-light dynamic-update price-element",
                "NEUTRAL", "badge bg-secondary", 0, "primary",
                "N/A", "N/A", "fs-5 text-light dynamic-update price-element",
                "NEUTRAL", "badge bg-secondary", 0, "primary"
            )
        
        # Get option keys
        ce_key = stocks_data[symbol].get("primary_ce")
        pe_key = stocks_data[symbol].get("primary_pe")
    
    # CE option data
    ce_strike, ce_price, ce_price_class = "N/A", "N/A", "fs-5 text-light dynamic-update price-element"
    ce_signal, ce_signal_class, ce_strength, ce_strength_color = "NEUTRAL", "badge bg-secondary", 0, "primary"
    
    if ce_key and ce_key in options_data:
        with options_data_lock:
            ce_info = options_data[ce_key]
            ce_strike = ce_info.get("strike", "N/A")
            ce_price = ce_info.get("ltp")
            ce_signal_val = ce_info.get("signal", 0)
            ce_strength_val = ce_info.get("strength", 0)
            ce_trend = ce_info.get("trend", "NEUTRAL")
            ce_using_fallback = ce_info.get("using_fallback", False)
        
        # Format CE data
        ce_strike = f"Strike: {ce_strike}"
        ce_price = f"₹{ce_price:.2f}" if ce_price is not None else "N/A"
        ce_price_class = "fs-5 text-light dynamic-update price-element"
        if ce_using_fallback:
            ce_price_class += " text-opacity-50"  # Dim the price if using fallback data
        
        ce_signal = ce_trend
        ce_signal_class = "badge "
        if "BULLISH" in ce_trend:
            ce_signal_class += "bg-success badge-transition"
            ce_strength_color = "success"
        elif "BEARISH" in ce_trend:
            ce_signal_class += "bg-danger badge-transition"
            ce_strength_color = "danger"
        else:
            ce_signal_class += "bg-secondary badge-transition"
            ce_strength_color = "primary"
        
        ce_strength = min(ce_strength_val * 10, 100)  # Convert 0-10 scale to 0-100 for progress bar
    
    # PE option data
    pe_strike, pe_price, pe_price_class = "N/A", "N/A", "fs-5 text-light dynamic-update price-element"
    pe_signal, pe_signal_class, pe_strength, pe_strength_color = "NEUTRAL", "badge bg-secondary", 0, "primary"
    
    if pe_key and pe_key in options_data:
        with options_data_lock:
            pe_info = options_data[pe_key]
            pe_strike = pe_info.get("strike", "N/A")
            pe_price = pe_info.get("ltp")
            pe_signal_val = pe_info.get("signal", 0)
            pe_strength_val = pe_info.get("strength", 0)
            pe_trend = pe_info.get("trend", "NEUTRAL")
            pe_using_fallback = pe_info.get("using_fallback", False)
        
        # Format PE data
        pe_strike = f"Strike: {pe_strike}"
        pe_price = f"₹{pe_price:.2f}" if pe_price is not None else "N/A"
        pe_price_class = "fs-5 text-light dynamic-update price-element"
        if pe_using_fallback:
            pe_price_class += " text-opacity-50"  # Dim the price if using fallback data
        
        pe_signal = pe_trend
        pe_signal_class = "badge "
        if "BULLISH" in pe_trend:
            pe_signal_class += "bg-success badge-transition"
            pe_strength_color = "success"
        elif "BEARISH" in pe_trend:
            pe_signal_class += "bg-danger badge-transition"
            pe_strength_color = "danger"
        else:
            pe_signal_class += "bg-secondary badge-transition"
            pe_strength_color = "primary"
        
        pe_strength = min(pe_strength_val * 10, 100)  # Convert 0-10 scale to 0-100 for progress bar
    
    return (
        ce_strike, ce_price, ce_price_class,
        ce_signal, ce_signal_class, ce_strength, ce_strength_color,
        pe_strike, pe_price, pe_price_class,
        pe_signal, pe_signal_class, pe_strength, pe_strength_color
    )

# Callback to update active trade details
@app.callback(
    [Output({"type": "trade-duration", "index": MATCH}, "children"),
     Output({"type": "trade-current", "index": MATCH}, "children"),
     Output({"type": "trade-pnl", "index": MATCH}, "children"),
     Output({"type": "trade-pnl", "index": MATCH}, "className")],
    [Input("update-interval", "n_intervals")],
    [State({"type": "trade-duration", "index": MATCH}, "id")]
)
def update_active_trade_details(n_intervals, id_dict):
    option_key = id_dict["index"]
    
    # Check if the trade is active
    with trading_state_lock:
        is_active = trading_state.active_trades.get(option_key, False)
        if not is_active:
            return "N/A", "N/A", "N/A", "small dynamic-update"
        
        entry_price = trading_state.entry_price.get(option_key, 0)
        entry_time = trading_state.entry_time.get(option_key, datetime.now())
        quantity = trading_state.quantity.get(option_key, 0)
    
    # Get current price
    with options_data_lock:
        if option_key not in options_data:
            return "N/A", "N/A", "N/A", "small dynamic-update"
        
        option_info = options_data[option_key]
        current_price = option_info.get("ltp", 0)
        option_type = option_info.get("option_type", "")
    
    # Calculate duration
    current_time = datetime.now()
    duration = (current_time - entry_time).total_seconds() if entry_time else 0
    duration_str = f"{duration / 60:.1f}min"
    
    # Calculate P&L
    if option_type == "CE":
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
    else:  # PE
        pnl = (entry_price - current_price) * quantity
        pnl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0
    
    current_str = f"₹{current_price:.2f}"
    pnl_str = f"₹{pnl:.2f} ({pnl_pct:.1f}%)"
    
    # Determine P&L class
    pnl_class = "small text-success dynamic-update" if pnl >= 0 else "small text-danger dynamic-update"
    
    return duration_str, current_str, pnl_str, pnl_class

# Callback to update the data stores
@app.callback(
    [Output("stocks-data-store", "data"),
     Output("options-data-store", "data"),
     Output("trading-state-store", "data"),
     Output("pcr-data-store", "data"),
     Output("news-data-store", "data"),
     Output("last-update-store", "data")],
    Input("update-interval", "n_intervals")
)
def update_data_stores(n_intervals):
    # Convert data structures to JSON-serializable format
    stocks_data_json = {}
    with stocks_data_lock:
        for symbol, data in stocks_data.items():
            stocks_data_json[symbol] = {
                "ltp": data.get("ltp"),
                "change_percent": data.get("change_percent"),
                "primary_ce": data.get("primary_ce"),
                "primary_pe": data.get("primary_pe"),
                "predicted_strategy": data.get("predicted_strategy"),
                "strategy_confidence": data.get("strategy_confidence"),
                "last_updated": data.get("last_updated").isoformat() if data.get("last_updated") else None
            }
    
    options_data_json = {}
    with options_data_lock:
        for key, data in options_data.items():
            options_data_json[key] = {
                "ltp": data.get("ltp"),
                "signal": data.get("signal"),
                "strength": data.get("strength"),
                "trend": data.get("trend"),
                "using_fallback": data.get("using_fallback", False)
            }
    
    trading_state_json = {}
    with trading_state_lock:
        trading_state_json = {
            "active_trades": trading_state.active_trades,
            "total_pnl": trading_state.total_pnl,
            "daily_pnl": trading_state.daily_pnl,
            "trades_today": trading_state.trades_today,
            "wins": trading_state.wins,
            "losses": trading_state.losses,
            "trades_history": len(trading_state.trades_history)
        }
    
    pcr_data_json = {}
    with pcr_data_lock:
        for symbol, data in pcr_data.items():
            pcr_data_json[symbol] = {
                "current": data.get("current"),
                "trend": data.get("trend"),
                "strength": data.get("strength")
            }
    
    news_data_json = {
        "count": len(news_data.get("items", [])),
        "last_updated": news_data.get("last_updated").isoformat() if news_data.get("last_updated") else None
    }
    
    last_update_json = {
        "timestamp": datetime.now().isoformat()
    }
    
    return stocks_data_json, options_data_json, trading_state_json, pcr_data_json, news_data_json, last_update_json

# Callback for showing notifications
@app.callback(
    [Output("notification-toast", "is_open"),
     Output("notification-toast", "header"),
     Output("notification-toast", "children"),
     Output("notification-toast", "icon")],
    [Input("trading-state-store", "data")]
)
def show_trade_notifications(trading_state_json):
    # Check if there are new trades
    if not hasattr(show_trade_notifications, "prev_trade_count"):
        show_trade_notifications.prev_trade_count = 0
    
    current_trade_count = trading_state_json.get("trades_history", 0)
    
    if current_trade_count > show_trade_notifications.prev_trade_count:
        # New trade executed, show notification
        with trading_state_lock:
            if trading_state.trades_history:
                latest_trade = trading_state.trades_history[-1]
                
                # Get trade details
                symbol = latest_trade.get("parent_symbol", "")
                option_type = latest_trade.get("option_type", "")
                pnl = latest_trade.get("pnl", 0)
                reason = latest_trade.get("reason", "")
                
                # Create notification
                header = "Trade Completed"
                message = f"{symbol} {option_type} trade exited: ₹{pnl:.2f} ({reason})"
                icon = "success" if pnl > 0 else "danger"
                
                # Update previous count
                show_trade_notifications.prev_trade_count = current_trade_count
                
                return True, header, message, icon
    
    # Check for new active trades
    if not hasattr(show_trade_notifications, "prev_active_count"):
        show_trade_notifications.prev_active_count = 0
    
    current_active_count = sum(1 for v in trading_state.active_trades.values() if v)
    
    if current_active_count > show_trade_notifications.prev_active_count:
        # New trade entered, show notification
        with trading_state_lock:
            # Find the new active trade
            for option_key, is_active in trading_state.active_trades.items():
                if is_active:
                    with options_data_lock:
                        if option_key in options_data:
                            parent_symbol = options_data[option_key].get("parent_symbol", "")
                            option_type = options_data[option_key].get("option_type", "")
                            
                            # Create notification
                            header = "Trade Initiated"
                            message = f"Entered new {parent_symbol} {option_type} trade"
                            icon = "primary"
                            
                            # Update previous count
                            show_trade_notifications.prev_active_count = current_active_count
                            
                            return True, header, message, icon
    
    # If no new trades, don't show notification
    show_trade_notifications.prev_trade_count = current_trade_count
    show_trade_notifications.prev_active_count = current_active_count
    return False, "", "", "primary"

# ============ Data Thread Functions ============

def data_thread_function():
    """Main thread function for data collection and trading with bulk API usage."""
    global data_thread_started, dashboard_initialized
    
    data_thread_started = True
    logger.info("Data thread started")
    
    # Initialize state
    initialize_from_csv()
    dashboard_initialized = True
    
    # Setup heartbeat thread for monitoring
    heartbeat_active = True
    def heartbeat_function():
        counter = 0
        while heartbeat_active:
            logger.info(f"Data thread heartbeat: {counter}")
            counter += 1
            time.sleep(10)  # Log every 10 seconds
            
    heartbeat_thread = threading.Thread(target=heartbeat_function, daemon=True)
    heartbeat_thread.start()
    
    # Slow startup to allow the dashboard to load
    time.sleep(3)
    
    # Main loop
    try:
        iteration = 0
        while True:
            try:
                # Log the start of each cycle with timestamp
                current_time = datetime.now().strftime("%H:%M:%S")
                logger.info(f"[{current_time}] Starting data cycle iteration {iteration}")
                
                # Try to connect to broker if not already connected
                if not broker_connected:
                    logger.info("Attempting broker connection...")
                    connect_to_broker_with_backoff()
                    logger.info("Broker connection attempt completed")
                
                # Simple day rollover check - unlikely to hang
                logger.info("Checking day rollover...")
                check_day_rollover()
                logger.info("Day rollover check completed")
                
                # Process basic data if broker is connected
                if broker_connected:
                    try:
                        # Use bulk API to fetch all data in one go with timeout
                        bulk_start_time = time.time()
                        BULK_TIMEOUT = 15  # 15 seconds timeout for bulk data fetching
                        
                        logger.info("Fetching all data using bulk API...")
                        with stocks_data_lock:
                            symbols_to_fetch = list(stocks_data.keys())
                        
                        # Run the bulk fetch in a separate thread with timeout
                        all_data = {}
                        fetch_completed = [False]
                        
                        def bulk_fetch_thread():
                            try:
                                nonlocal all_data
                                all_data = fetch_bulk_data_comprehensive(symbols_to_fetch)
                                fetch_completed[0] = True
                            except Exception as e:
                                logger.error(f"Error in bulk fetch thread: {e}")
                        
                        bulk_thread = threading.Thread(target=bulk_fetch_thread)
                        bulk_thread.daemon = True
                        bulk_thread.start()
                        bulk_thread.join(timeout=BULK_TIMEOUT)
                        
                        if not fetch_completed[0]:
                            logger.warning(f"Bulk data fetch timed out after {BULK_TIMEOUT}s")
                            # Fall back to individual fetches for critical data
                            logger.info("Falling back to individual stock fetches...")
                            for symbol in symbols_to_fetch[:4]:  # Limit to the first few important stocks
                                fetch_stock_data(symbol)
                                time.sleep(0.1)
                        else:
                            # Process the bulk data
                            logger.info(f"Processing bulk data from {len(all_data.get('stocks', {}))} stocks")
                            
                            # Update stock data
                            for symbol, data in all_data.get("stocks", {}).items():
                                with stocks_data_lock:
                                    if symbol in stocks_data:
                                        stock_info = stocks_data[symbol]
                                        
                                        # Update with fetched data
                                        previous_ltp = stock_info["ltp"]
                                        stock_info["ltp"] = data.get("ltp")
                                        stock_info["open"] = data.get("open")
                                        stock_info["high"] = data.get("high") 
                                        stock_info["low"] = data.get("low")
                                        stock_info["previous"] = data.get("previous")
                                        
                                        # Calculate movement percentage
                                        if previous_ltp is not None and previous_ltp > 0:
                                            stock_info["movement_pct"] = ((data.get("ltp", 0) - previous_ltp) / previous_ltp) * 100
                                        
                                        # Calculate change percentage
                                        if stock_info["open"] and stock_info["open"] > 0:
                                            stock_info["change_percent"] = ((data.get("ltp", 0) - stock_info["open"]) / stock_info["open"]) * 100
                                        
                                        # Update last update time
                                        stock_info["last_updated"] = datetime.now()
                                
                                # Update price history
                                timestamp = pd.Timestamp.now()
                                
                                new_data = {
                                    'timestamp': timestamp,
                                    'price': data.get("ltp", 0),
                                    'volume': data.get("volume", 0),
                                    'open': data.get("open", data.get("ltp", 0)),
                                    'high': data.get("high", data.get("ltp", 0)),
                                    'low': data.get("low", data.get("ltp", 0))
                                }
                                
                                # Thread-safe price history update
                                with price_history_lock:
                                    with stocks_data_lock:
                                        if symbol in stocks_data:
                                            stock_info = stocks_data[symbol]
                                            stock_info["price_history"] = pd.concat([
                                                stock_info["price_history"], 
                                                pd.DataFrame([new_data])
                                            ], ignore_index=True)
                                            
                                            # Limit price history size
                                            if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                                                stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
                                
                                # Update volatility
                                with stocks_data_lock:
                                    if symbol in stocks_data:
                                        stock_info = stocks_data[symbol]
                                        if previous_ltp is not None and previous_ltp > 0:
                                            pct_change = (data.get("ltp", 0) - previous_ltp) / previous_ltp * 100
                                            update_volatility(symbol, pct_change)
                                
                                # Update UI data store
                                with ui_data_store_lock:
                                    ui_data_store['stocks'][symbol] = {
                                        'price': data.get("ltp"),
                                        'change': stocks_data[symbol]["change_percent"],
                                        'ohlc': {
                                            'open': stocks_data[symbol]["open"],
                                            'high': stocks_data[symbol]["high"],
                                            'low': stocks_data[symbol]["low"],
                                            'previous': stocks_data[symbol]["previous"]
                                        },
                                        'last_updated': datetime.now().strftime('%H:%M:%S')
                                    }
                            
                            # Update option data
                            for option_key, data in all_data.get("options", {}).items():
                                with options_data_lock:
                                    if option_key in options_data:
                                        option_info = options_data[option_key]
                                        
                                        # Update with fetched data
                                        previous_ltp = option_info.get("ltp")
                                        option_info["ltp"] = data.get("ltp")
                                        option_info["open"] = data.get("open")
                                        option_info["high"] = data.get("high")
                                        option_info["low"] = data.get("low")
                                        
                                        # Calculate change percentage
                                        if previous_ltp is not None and previous_ltp > 0:
                                            option_info["change_percent"] = ((data.get("ltp", 0) - previous_ltp) / previous_ltp) * 100
                                        
                                        # Update last update time
                                        option_info["last_updated"] = datetime.now()
                                        option_info["using_fallback"] = False
                            
                            # Update PCR data
                            for symbol, data in all_data.get("pcr", {}).items():
                                with pcr_data_lock:
                                    if symbol in pcr_data:
                                        pcr_data[symbol]["current"] = data.get("current", 1.0)
                                        pcr_data[symbol]["history"].append(data.get("current", 1.0))
                                        pcr_data[symbol]["trend"] = data.get("trend", "NEUTRAL")
                                        pcr_data[symbol]["strength"] = data.get("strength", 0.0)
                                        pcr_data[symbol]["last_updated"] = datetime.now()
                        
                        logger.info(f"Bulk data processing completed in {time.time() - bulk_start_time:.2f}s")
                        
                        # Process options for stocks that need them
                        logger.info("Checking for stocks that need options...")
                        option_start_time = time.time()
                        MAX_OPTION_TIME = 20  # Max 20 seconds for option processing
                        
                        for symbol in symbols_to_fetch:
                            # Check timeout
                            if time.time() - option_start_time > MAX_OPTION_TIME:
                                logger.warning(f"Option processing timeout reached after {symbol}")
                                break
                                
                            try:
                                # Check if we need options for this stock
                                with stocks_data_lock:
                                    if symbol not in stocks_data:
                                        continue
                                        
                                    stock_info = stocks_data[symbol]
                                    current_price = stock_info.get("ltp")
                                    
                                    # Skip if no valid price
                                    if current_price is None or current_price <= 0:
                                        logger.info(f"Skipping option checks for {symbol}: No valid price")
                                        continue
                                    
                                    # Only process options if needed
                                    need_options = (stock_info.get("primary_ce") is None or 
                                                  stock_info.get("primary_pe") is None)
                                
                                if need_options:
                                    logger.info(f"Finding options for {symbol}...")
                                    remaining_time = MAX_OPTION_TIME - (time.time() - option_start_time)
                                    if remaining_time <= 0:
                                        break
                                        
                                    find_and_add_atm_options(symbol, current_price, timeout=min(10, remaining_time))
                            except Exception as option_error:
                                logger.error(f"Error processing options for {symbol}: {option_error}")
                        
                        logger.info(f"Option processing completed in {time.time() - option_start_time:.2f}s")
                        
                        # Update PCR and market sentiment periodically
                        if iteration % 30 == 0:  # Every 30 iterations
                            pcr_start_time = time.time()
                            MAX_PCR_TIME = 10  # 10 seconds max for PCR and sentiment updates
                            
                            try:
                                logger.info("Updating PCR data...")
                                
                                def pcr_thread_func():
                                    try:
                                        update_all_pcr_data()
                                    except Exception as e:
                                        logger.error(f"Error in PCR thread: {e}")
                                
                                pcr_thread = threading.Thread(target=pcr_thread_func)
                                pcr_thread.daemon = True
                                pcr_thread.start()
                                pcr_thread.join(timeout=MAX_PCR_TIME)
                                
                                logger.info(f"PCR update took {time.time() - pcr_start_time:.2f}s")
                            except Exception as pcr_error:
                                logger.error(f"Error starting PCR update: {pcr_error}")
                        
                        # Apply trading strategies
                        strategy_start_time = time.time()
                        MAX_STRATEGY_TIME = 10  # 10 seconds max for strategy application
                        
                        try:
                            logger.info("Applying trading strategy...")
                            
                            def strategy_thread_func():
                                try:
                                    apply_trading_strategy()
                                except Exception as e:
                                    logger.error(f"Error in strategy thread: {e}")
                            
                            strategy_thread = threading.Thread(target=strategy_thread_func)
                            strategy_thread.daemon = True
                            strategy_thread.start()
                            strategy_thread.join(timeout=MAX_STRATEGY_TIME)
                            
                            logger.info(f"Strategy application took {time.time() - strategy_start_time:.2f}s")
                        except Exception as strategy_error:
                            logger.error(f"Error starting strategy application: {strategy_error}")
                        
                        # Run cleanup periodically
                        if iteration % 600 == 0:  # Every 600 iterations
                            try:
                                logger.info("Running data cleanup...")
                                cleanup_old_data()
                                cleanup_inactive_stocks()
                                logger.info("Cleanup completed")
                            except Exception as cleanup_error:
                                logger.error(f"Error in cleanup: {cleanup_error}")
                    
                    except Exception as data_error:
                        logger.error(f"Error in data processing: {data_error}", exc_info=True)
                
                # Log completion of cycle
                logger.info(f"Completed data cycle iteration {iteration}")
                
                # Increment iteration counter
                iteration += 1
                if iteration > 10000:  # Prevent integer overflow
                    iteration = 0
                
                # Sleep before next cycle
                time.sleep(1)
                
            except Exception as cycle_error:
                logger.error(f"Error in data thread cycle: {cycle_error}", exc_info=True)
                time.sleep(5)  # Sleep longer on error
    
    except KeyboardInterrupt:
        logger.info("Data thread stopped by user")
    except Exception as fatal_error:
        logger.error(f"Fatal error in data thread: {fatal_error}", exc_info=True)
    finally:
        heartbeat_active = False  # Stop the heartbeat
        data_thread_started = False
        logger.info("Data thread stopped")
        
if __name__ == "__main__":
    logger.info("Starting Options Trading Dashboard")
    
    # Start the data thread
    data_thread = threading.Thread(target=data_thread_function, daemon=True)
    data_thread.start()
    
    # Start the Dash app
    app.run_server(debug=True, use_reloader=False, port=8050)
