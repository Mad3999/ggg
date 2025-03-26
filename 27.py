import os
import time
import threading
import logging
import sys
import pyotp
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
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

# Suppress warnings
warnings.filterwarnings('ignore')

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
    
    def register_rate_limit_hit(self, endpoint_key="default"):
        with self.lock:
            current_time = time.time()
            if endpoint_key in self.rate_limit_hits:
                _, hit_count = self.rate_limit_hits[endpoint_key]
                self.rate_limit_hits[endpoint_key] = (current_time, hit_count + 1)
            else:
                self.rate_limit_hits[endpoint_key] = (current_time, 1)
            logger.warning(f"Rate limit hit for {endpoint_key}, implementing backoff")
    
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
MAX_TARGET_ADJUSTMENT = 2.0  # Maximum target adjustment percentage
TARGET_MOMENTUM_FACTOR = 0.3  # Factor for momentum-based target adjustment

# Position Management
MAX_POSITION_HOLDING_TIME_SCALP = 10  # Maximum scalping trade holding time in minutes
MAX_POSITION_HOLDING_TIME_SWING = 120  # Maximum swing trade holding time in minutes
MAX_POSITION_HOLDING_TIME_MOMENTUM = 45  # Maximum momentum trade holding time in minutes

# Prediction Thresholds
MIN_SIGNAL_STRENGTH_SCALP = 3.0  # Minimum signal strength for scalping trades
MIN_SIGNAL_STRENGTH_SWING = 4.0  # Minimum signal strength for swing trades
MIN_SIGNAL_STRENGTH_MOMENTUM = 3.5  # Minimum signal strength for momentum trades

# Strategy Settings
strategy_settings = {
    "SCALP_ENABLED": True,
    "SWING_ENABLED": True,
    "MOMENTUM_ENABLED": True
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
API_UPDATE_INTERVAL = 1  # Seconds between API data updates
BULK_FETCH_SIZE = 50  # Maximum symbols per bulk request
INDEX_UPDATE_INTERVAL = 1  # Seconds between index data updates
OPTION_UPDATE_INTERVAL = 1  # Seconds between option data updates
PCR_UPDATE_INTERVAL = 30  # Seconds between PCR updates
DATA_CLEANUP_INTERVAL = 600  # Cleanup old data every 10 minutes
MAX_PRICE_HISTORY_POINTS = 1000  # Maximum number of price history points to store

# Option Selection Parameters
NUM_STRIKES_TO_FETCH = 5  # Number of strikes above and below current price to fetch
OPTION_AUTO_SELECT_INTERVAL = 300  # Automatically update option selection every 5 minutes

# Historical Data Paths
NIFTY_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\NIFTY.csv"
BANK_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\BANK.csv"

# Symbol mapping for broker
SYMBOL_MAPPING = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE"
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
last_cleanup = datetime.now()
last_connection_time = None

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

# Data storage for UI
ui_data_store = {
    'stocks': {},
    'options': {},
    'pcr': {},
    'trades': {},
    'predicted_strategies': {}  # Added for strategy predictions
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
            if option_key in self.stock_entry_price:
                del self.stock_entry_price[option_key]
            if option_key in self.quantity:
                del self.quantity[option_key]
            return True
        return False

trading_state = TradingState()

# ============ Stock Management Functions ============
def fetch_stock_data(symbol):
    """
    Fetch data for a single stock directly
    
    Args:
        symbol (str): Stock symbol to fetch data for
        
    Returns:
        bool: True if successful, False otherwise
    """
    global smart_api, broker_connected, stocks_data
    
    # Validate input
    if not symbol or symbol not in stocks_data:
        return False
    
    # Ensure broker connection
    if not broker_connected or smart_api is None:
        if not connect_to_broker():
            logger.warning(f"Cannot fetch data for {symbol}: Not connected to broker")
            return False
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    try:
        # Retrieve stock details
        stock_info = stocks_data[symbol]
        token = stock_info.get("token")
        exchange = stock_info.get("exchange")
        
        # Validate token and exchange
        if not token or not exchange:
            logger.warning(f"Missing token or exchange for {symbol}")
            return False
        
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
            
            # Update stock data
            previous_ltp = stock_info["ltp"]
            stock_info["ltp"] = ltp
            stock_info["open"] = open_price
            stock_info["high"] = high_price
            stock_info["low"] = low_price
            stock_info["previous"] = previous_price
            
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
            
            # Append to price history DataFrame with proper index handling
            with price_history_lock:
                stock_info["price_history"] = pd.concat([
                    stock_info["price_history"], 
                    pd.DataFrame([new_data])
                ], ignore_index=True)
                
                # Limit price history size
                if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                    stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            # Update volatility
            if previous_ltp is not None and previous_ltp > 0:
                pct_change = (ltp - previous_ltp) / previous_ltp * 100
                update_volatility(symbol, pct_change)
            
            # Update support/resistance levels
            calculate_support_resistance(symbol)
            
            # Update last update time
            stock_info["last_updated"] = datetime.now()
            last_data_update["stocks"][symbol] = datetime.now()
            
            # Update UI data store
            ui_data_store['stocks'][symbol] = {
                'price': ltp,
                'change': stock_info["change_percent"],
                'ohlc': {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'previous': previous_price
                },
                'last_updated': stock_info["last_updated"].strftime('%H:%M:%S')
            }
            
            # Predict the most suitable strategy for this stock
            predict_strategy_for_stock(symbol)
            
            logger.info(f"Fetched real LTP for {symbol}: {ltp:.2f}")
            return True
        else:
            logger.warning(f"Failed to fetch data for {symbol}: {ltp_resp}")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False
def fetch_history_from_yahoo(symbol, period="3mo"):
    """
    Fetch historical data for a symbol from Yahoo Finance with handling for multi-index columns.
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
        
        # Log raw data structure for debugging
        logger.info(f"Raw data sample: {history.head(1).to_dict()}")
        
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
        
        # Validate the data
        if 'price' in df.columns:
            valid_price_count = df['price'].notna().sum()
            logger.info(f"Successfully fetched {len(df)} historical records for {symbol} with {valid_price_count} valid prices")
            logger.info(f"Sample processed data: {df.iloc[0].to_dict() if not df.empty else 'Empty DataFrame'}")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching history for {symbol} from Yahoo Finance: {e}", exc_info=True)
        return None
def load_historical_data(symbol, period="3mo", force_refresh=False):
    """Load historical data with proper validation"""
    if symbol not in stocks_data:
        logger.warning(f"Cannot load history for unknown symbol: {symbol}")
        return False
    
    # Fetch history with our fixed function
    history_df = fetch_history_from_yahoo(symbol, period)
    
    if history_df is None or history_df.empty:
        logger.warning(f"Failed to fetch history for {symbol}")
        return False
    
    # Verify we have valid price data
    if 'price' not in history_df.columns:
        logger.warning(f"No price column in fetched data for {symbol}")
        return False
        
    valid_prices = history_df['price'].dropna()
    if len(valid_prices) < 10:  # Require at least 10 valid price points
        logger.warning(f"Insufficient valid price data for {symbol}: {len(valid_prices)} points")
        return False
        
    # Store the data directly
    stocks_data[symbol]["price_history"] = history_df
    
    # Update metadata
    stocks_data[symbol]["history_source"] = "yahoo"
    stocks_data[symbol]["history_fetched"] = True
    stocks_data[symbol]["last_history_fetch"] = datetime.now()
    
    # Force recalculation of support/resistance
    calculate_support_resistance(symbol)
    
    return True
def find_stock_token_in_json(symbol):
    """
    Find stock token in the script master data with exact symbol matching
    
    Args:
        symbol (str): Stock symbol to search for
        
    Returns:
        str: Token if found, None otherwise
    """
    global script_master_data
    
    # Make sure script master is loaded
    if not script_master_loaded and not load_script_master():
        logger.error("Cannot search for stock token: Script master data not loaded")
        return None
    
    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    # First look for exact matches in the symbol field
    for entry in script_master_data:
        trading_symbol = entry.get("symbol", "").upper()
        name = entry.get("name", "").upper()
        
        # Only match exact symbols - not partial matches
        if symbol == trading_symbol or symbol == name:
            return entry.get("token")
    
    # If no exact match found, log a warning
    logger.warning(f"No exact matching token found in script master for stock {symbol}")
    return None

def add_stock(symbol, token=None, exchange="NSE", stock_type="STOCK", with_options=True):
    """Add a new stock to track with improved token matching"""
    global stocks_data, volatility_data, market_sentiment, pcr_data

    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    # If already exists, just return
    if symbol in stocks_data:
        return True
    
    # If token is None, try to find the token using exact matching
    if token is None:
        # Try to find the exact token in script master first
        token = find_stock_token_in_json(symbol)
        
        # If not found in script master, try to search using broker API if connected
        if token is None and broker_connected:
            matches = search_symbols(symbol)
            
            # Find the exact match only
            for match in matches:
                match_symbol = match.get("symbol", "").upper()
                match_name = match.get("name", "").upper()
                
                if match_symbol == symbol or match_name == symbol:
                    token = match.get("token")
                    exchange = match.get("exch_seg", exchange)
                    stock_type = "INDEX" if "NIFTY" in symbol or "SENSEX" in symbol else "STOCK"
                    break
        
        # If still no token, use placeholder
        if token is None:
            logger.warning(f"Could not find exact token for {symbol}, using placeholder")
            token = str(hash(symbol) % 100000)
    
    # Create new stock entry
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
        "strategy_enabled": {
            "SCALP": True,
            "SWING": True,
            "MOMENTUM": True
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
    }
    
    # Initialize volatility tracking
    volatility_data[symbol] = {
        "window": deque(maxlen=30),
        "current": 0,
        "historical": []
    }
    
    # Initialize market sentiment
    market_sentiment[symbol] = "NEUTRAL"
    
    # Initialize PCR data
    pcr_data[symbol] = {
        "current": 1.0,
        "history": deque(maxlen=PCR_HISTORY_LENGTH),
        "trend": "NEUTRAL",
        "strength": 0.0,  # Added PCR strength indicator
        "last_updated": None
    }
    
    # Set last update time to force immediate update
    last_data_update["stocks"][symbol] = None
    
    logger.info(f"Added new stock: {symbol} ({token}) on {exchange}")
    
    # For NIFTY and BANKNIFTY, load historical data from CSV files
    if symbol in ["NIFTY", "BANKNIFTY"]:
        load_historical_data(symbol)
    
    # Immediately fetch price data for the new stock
    if broker_connected:
        fetch_stock_data(symbol)
    
    # Add options if requested and we have price data
    if with_options and stocks_data[symbol]["ltp"] is not None:
        # Find and add ATM options
        options = find_and_add_options(symbol)
        logger.info(f"Added options for {symbol}: CE={options.get('CE')}, PE={options.get('PE')}")
    
    return True

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
    for symbol in stocks_data.keys():
        # Get all CE and PE options for this symbol
        ce_options = []
        pe_options = []
        
        if symbol in stocks_data:
            ce_keys = stocks_data[symbol].get("options", {}).get("CE", [])
            pe_keys = stocks_data[symbol].get("options", {}).get("PE", [])
            
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
def update_pcr_data(symbol):
    """Update PCR data for a symbol using real data when possible, falling back to simulation."""
    global pcr_data
    
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
        pcr_data[symbol]["current"] = pcr
        pcr_data[symbol]["history"].append(pcr)
        pcr_data[symbol]["last_updated"] = datetime.now()
        logger.info(f"Updated PCR for {symbol} with real data: {pcr:.2f}")
        determine_pcr_trend(symbol)
        calculate_pcr_strength(symbol)
        return True
    
    # Fall back to previous implementation if real data isn't available
    # (Your existing option chain logic and simulated PCR code)
    # ...rest of your existing update_pcr_data function...
def calculate_pcr_strength(symbol):
    """
    Calculate PCR strength based on current value and trend
    
    Args:
        symbol (str): Symbol to calculate PCR strength for
    """
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

@rate_limited
def fetch_option_data(option_key):
    """
    Fetch data for a single option with comprehensive error handling and fallbacks
    
    Args:
        option_key (str): Option key to fetch data for
        
    Returns:
        bool: True if successful, False otherwise
    """
    global options_data, broker_connected, smart_api
    
    if option_key not in options_data:
        logger.warning(f"Cannot fetch data for unknown option key: {option_key}")
        return False
    
    option_info = options_data[option_key]
    
    # Skip if not connected to broker
    if not broker_connected:
        logger.warning(f"Broker not connected, skipping update for {option_key}")
        return False
    
    # Check if this option was recently updated (within last 5 seconds)
    last_update = option_info.get("last_updated")
    if last_update and (datetime.now() - last_update).total_seconds() < 5:
        logger.debug(f"Option {option_key} was recently updated, skipping")
        return True
    
    try:
        # Extract essential option details
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
                            # For non-fallback tokens, try direct search
                            matches = search_symbols(symbol)
                            if matches and len(matches) > 0:
                                new_token = matches[0].get("token")
                                if new_token and new_token != token:
                                    logger.info(f"Updated token for {symbol}: {new_token}")
                                    option_info["token"] = new_token
                                    
                                    # Try again with the new token
                                    try:
                                        new_resp = smart_api.ltpData(exchange, symbol, new_token)
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
                                using_fallback = True
                                fallback_reason = "Symbol search failed"
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
                stock_price = stocks_data[parent_symbol].get("ltp", 50)
                strike = float(option_info.get("strike", 0))
                
                # Simple option pricing model as fallback
                if strike > 0:
                    if option_type == "ce":  # Call option
                        ltp = max(stock_price - strike, 5) if stock_price > strike else 5
                    else:  # Put option
                        ltp = max(strike - stock_price, 5) if strike > stock_price else 5
                else:
                    ltp = 50
            else:
                ltp = 50
                logger.warning(f"No parent stock data for fallback pricing, using default price: {ltp}")
        
        # Final validation to ensure we have a valid price
        ltp = max(float(ltp), 0.01)
        
        # Update option data with safe calculations
        current_open = option_info.get("open")
        current_high = option_info.get("high")
        current_low = option_info.get("low")
        
        # Update option info
        option_info.update({
            "ltp": ltp,
            "open": current_open if current_open is not None and current_open > 0 else ltp,
            "high": max(current_high if current_high is not None else 0, ltp),
            "low": min(current_low if current_low is not None and current_low > 0 else float('inf'), ltp),
            "previous": option_info.get("ltp", ltp),  # Use previous LTP as the previous price
            "change_percent": ((ltp / option_info.get("ltp", ltp)) - 1) * 100 if option_info.get("ltp") else 0,
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
        last_data_update["options"][option_key] = datetime.now()
        
        # Update UI data store for primary options
        if parent_symbol and option_type:
            # Check if this is the primary option for the stock
            primary_option_key = stocks_data.get(parent_symbol, {}).get(f"primary_{option_type}")
            
            if primary_option_key == option_key:
                # Ensure the parent symbol exists in UI data store
                if parent_symbol not in ui_data_store['options']:
                    ui_data_store['options'][parent_symbol] = {}
                
                # Update UI data
                ui_data_store['options'][parent_symbol][option_type] = {
                    'strike': option_info.get("strike", "N/A"),
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
    last_updated_times = {}
    
    for option_key, option_info in options_data.items():
        parent_symbol = option_info.get("parent_symbol")
        option_type = option_info.get("option_type", "").lower()
        last_updated = option_info.get("last_updated")
        last_updated_times[option_key] = last_updated
        
        # Check if this is a primary option for any stock
        is_primary = False
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
        time.sleep(0.05)
    
    # Sort regular options by last update time, oldest first
    sorted_regular_options = sorted(
        regular_options,
        key=lambda k: last_updated_times.get(k, datetime.min)
    )
    
    # Update a limited number of regular options each cycle
    max_regular_updates = min(len(sorted_regular_options), 10)  # Limit to 10 options per cycle
    if max_regular_updates > 0:
        options_to_update = sorted_regular_options[:max_regular_updates]
        logger.info(f"Updating {len(options_to_update)} of {len(sorted_regular_options)} regular options")
        for option_key in options_to_update:
            fetch_option_data(option_key)
            # Small delay between requests
            time.sleep(0.05)
    
    # Log overall update status
    logger.info(f"Options update completed. {len(priority_options)} priority options updated, {max_regular_updates} regular options updated.")

def remove_stock(symbol):
    """Remove a stock and its options from tracking"""
    global stocks_data, options_data, volatility_data, market_sentiment, pcr_data
    
    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    if symbol not in stocks_data:
        return False
    
    # First remove all associated options
    if "options" in stocks_data[symbol]:
        for option_type in ["CE", "PE"]:
            for option_key in stocks_data[symbol]["options"].get(option_type, []):
                if option_key in options_data:
                    # Exit any active trade on this option
                    if trading_state.active_trades.get(option_key, False):
                        exit_trade(option_key, reason="Stock removed")
                    
                    # Remove option data
                    del options_data[option_key]
                    
                    # Clean up trading state
                    trading_state.remove_option(option_key)
    
    # Clean up other data structures
    if symbol in volatility_data:
        del volatility_data[symbol]
    
    if symbol in market_sentiment:
        del market_sentiment[symbol]
    
    if symbol in pcr_data:
        del pcr_data[symbol]
    
    if symbol in last_data_update["stocks"]:
        del last_data_update["stocks"][symbol]
    
    # Remove from UI data stores
    if symbol in ui_data_store['stocks']:
        del ui_data_store['stocks'][symbol]
    
    if symbol in ui_data_store['options']:
        del ui_data_store['options'][symbol]
    
    if symbol in ui_data_store['predicted_strategies']:
        del ui_data_store['predicted_strategies'][symbol]
    
    # Finally, remove the stock itself
    del stocks_data[symbol]
    
    logger.info(f"Removed stock: {symbol}")
    return True

# Global variables for script master data
SCRIPT_MASTER_PATH = r"C:\Users\madhu\Pictures\ubuntu\OpenAPIScripMaster.json"
script_master_data = None
script_master_loaded = False
option_token_cache = {}

# ============ Script Master Data Management ============
SCRIPT_MASTER_PATH = r"C:\Users\madhu\Pictures\ubuntu\OpenAPIScripMaster.json"
script_master_data = None
script_master_loaded = False
option_token_cache = {}

def load_script_master():
    """Load the script master data from the local JSON file with advanced error handling"""
    global script_master_data, script_master_loaded
    
    if script_master_loaded and script_master_data:
        logger.info("Using already loaded script master data")
        return True
        
    try:
        logger.info(f"Loading script master data from {SCRIPT_MASTER_PATH}")
        with open(SCRIPT_MASTER_PATH, 'r') as f:
            script_master_data = json.load(f)
        
        if not script_master_data:
            logger.error("Script master data is empty")
            return False
            
        script_master_loaded = True
        logger.info(f"Successfully loaded script master data with {len(script_master_data)} entries")
        
        # Preprocess script master data for faster searches
        preprocess_script_master()
        return True
    except FileNotFoundError:
        logger.error(f"Script master file not found at {SCRIPT_MASTER_PATH}")
        script_master_data = {}
        return False
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in script master file")
        script_master_data = {}
        return False
    except Exception as e:
        logger.error(f"Failed to load script master data: {e}")
        script_master_data = {}
        return False

# Script master index for faster lookups
script_master_index = {
    'by_symbol': {},
    'by_token': {}
}

def preprocess_script_master():
    """Preprocess script master data to create indices for faster lookups"""
    global script_master_data, script_master_index
    
    if not script_master_data:
        return
    
    logger.info("Preprocessing script master data for efficient lookups...")
    
    # Clear existing indices
    script_master_index = {
        'by_symbol': {},
        'by_token': {}
    }
    
    # Create indices
    for entry in script_master_data:
        symbol = entry.get('symbol', '').upper()
        name = entry.get('name', '').upper()
        token = entry.get('token')
        
        if symbol:
            if symbol not in script_master_index['by_symbol']:
                script_master_index['by_symbol'][symbol] = []
            script_master_index['by_symbol'][symbol].append(entry)
            
        if name:
            if name not in script_master_index['by_symbol']:
                script_master_index['by_symbol'][name] = []
            script_master_index['by_symbol'][name].append(entry)
            
        if token:
            script_master_index['by_token'][token] = entry
    
    logger.info(f"Preprocessing complete. Created indices for {len(script_master_index['by_symbol'])} symbols and {len(script_master_index['by_token'])} tokens")

def get_next_expiry_date(symbol=None):
    """
    Get the next expiry date for options using improved logic.
    
    Args:
        symbol (str, optional): Symbol for which to get expiry date. Defaults to None.
        
    Returns:
        str: Expiry date in format "DDMMMYY" (e.g., "27MAR25")
    """
    try:
        # If broker is connected, try to get real expiry dates from the broker API
        if broker_connected and smart_api is not None:
            try:
                # Convert symbol if necessary
                api_symbol = SYMBOL_MAPPING.get(symbol, symbol) if symbol else "NIFTY"
                
                # This is a placeholder - in a real implementation, you would make an API call
                # to fetch the actual expiry dates from the broker
                # Example: expiry_resp = smart_api.get_option_chain(api_symbol)
                
                # For now, we're using a fixed expiry date but logging this clearly
                logger.info(f"Using predefined expiry date for {api_symbol} (would fetch from API in production)")
                
                # Standardized expiry date based on current date
                today = datetime.now()
                
                # For testing hardcoding to 27MAR25
                return "27MAR25"
                
                # In real implementation we would use code like this:
                # Find the next Thursday (weekly expiry for many indices)
                days_until_thursday = (3 - today.weekday()) % 7
                next_thursday = today + timedelta(days=days_until_thursday)
                
                # If today is Thursday and it's past market hours, use next Thursday
                if days_until_thursday == 0 and today.hour >= 15:
                    next_thursday = today + timedelta(days=7)
                
                # Format the date as DDMMMYY
                expiry_date = next_thursday.strftime("%d%b%y").upper()
                
                logger.info(f"Calculated upcoming expiry date for {api_symbol}: {expiry_date}")
                return expiry_date
                
            except Exception as e:
                logger.warning(f"Error fetching expiry date from broker API: {e}")
                # Fall through to default handling
        
        # Symbol-specific default expiry dates if API fails or broker not connected
        # This should be updated regularly based on the exchange calendar
        if symbol == "NIFTY" or symbol == "BANKNIFTY" or symbol == "FINNIFTY":
            default_expiry = "27MAR25"
        else:
            default_expiry = "27MAR25"  # Monthly expiry for stocks
        
        logger.info(f"Using default expiry date for {symbol}: {default_expiry}")
        return default_expiry
        
    except Exception as e:
        # Log any errors and fall back to a safe default
        logger.error(f"Error determining expiry date for {symbol}: {e}")
        default_expiry = "27MAR25"  # Hardcoded safe default
        logger.info(f"Using safe default expiry date: {default_expiry}")
        return default_expiry

def build_option_symbol(index_name, expiry_date, strike, option_type):
    """
    Build option symbol in the exact format expected by the broker
    
    Args:
        index_name (str): Index/Stock name (e.g., NIFTY, BANKNIFTY)
        expiry_date (str): Expiry date in format "DDMMMYY" (e.g., "27MAR25")
        strike (float/int): Strike price
        option_type (str): Option type (CE/PE)
        
    Returns:
        str: Formatted option symbol
    """
    # Standardize inputs
    index_name = index_name.upper()
    expiry_date = expiry_date.upper()
    option_type = option_type.upper()
    
    # Ensure strike is an integer with no decimal
    strike = int(float(strike))
    
    # Build the symbol in the exact format expected by Angel Broking
    # Format: SYMBOLDDMMMYYSTRIKECE/PE (e.g., NIFTY27MAR2519000CE)
    symbol = f"{index_name}{expiry_date}{strike}{option_type}"
    
    logger.debug(f"Built option symbol: {symbol}")
    return symbol

def generate_option_symbol_variations(symbol, expiry_date, strike, option_type):
    """
    Generate various formats of the option symbol to increase search match chances
    
    Args:
        symbol (str): Index/Stock name
        expiry_date (str): Expiry date in format "DDMMMYY"
        strike (str/int): Strike price
        option_type (str): Option type (CE/PE)
        
    Returns:
        list: List of possible symbol formats
    """
    symbol = symbol.upper()
    expiry_date = expiry_date.upper()
    option_type = option_type.upper()
    strike = str(int(float(strike)))
    
    variations = [
        f"{symbol}{expiry_date}{strike}{option_type}",  # NIFTY27MAR2519000CE
        f"{symbol} {strike} {option_type}",             # NIFTY 19000 CE
        f"{symbol}{strike}{option_type}",               # NIFTY19000CE
        f"{symbol}-{strike}-{option_type}",             # NIFTY-19000-CE
        f"{symbol} {expiry_date} {strike} {option_type}", # NIFTY 27MAR25 19000 CE
    ]
    
    # Add variations with different expiry date formats
    # Try to convert expiry from DDMMMYY to DD-MMM-YYYY format if possible
    try:
        expiry_obj = datetime.strptime(expiry_date, "%d%b%y")
        alt_expiry1 = expiry_obj.strftime("%d-%b-%Y").upper()
        alt_expiry2 = expiry_obj.strftime("%d%b%Y").upper()
        
        variations.extend([
            f"{symbol}{alt_expiry1}{strike}{option_type}",
            f"{symbol}{alt_expiry2}{strike}{option_type}",
        ])
    except:
        # If date conversion fails, proceed with the base variations
        pass
    
    return variations

def find_option_token_in_json(symbol, strike, option_type, expiry_date=None):
    """
    Find option token in the local JSON file with exact symbol matching
    
    Args:
        symbol (str): Stock/index symbol (e.g., NIFTY, BANKNIFTY)
        strike (int/float): Strike price
        option_type (str): Option type (CE/PE)
        expiry_date (str, optional): Expiry date in format "DDMMMYY"
        
    Returns:
        dict: Option information with token if found, None otherwise
    """
    global script_master_data
    
    # Make sure script master is loaded
    if not script_master_loaded and not load_script_master():
        logger.error("Cannot search for option token: Script master data not loaded")
        return None
    
    # Get expiry date if not provided
    if expiry_date is None or expiry_date == '28MAY25':
        expiry_date = get_next_expiry_date()
    
    # Format parameters correctly
    symbol = symbol.upper()
    option_type = option_type.upper()
    
    # Format strike correctly without decimals
    if isinstance(strike, float):
        strike = int(strike)
    strike = str(strike)
    
    # Create the exact symbol format we expect to find
    exact_symbol = f"{symbol}{expiry_date}{strike}{option_type}"
    
    # Search through the script master data for an exact match
    for entry in script_master_data:
        # Check the trading symbol or name field for exact match
        trading_symbol = entry.get("symbol", "").upper()
        name = entry.get("name", "").upper()
        
        # Look for exact match with the symbol we constructed
        if exact_symbol == trading_symbol or exact_symbol == name:
            return {
                "token": entry.get("token"),
                "symbol": trading_symbol,
                "strike": strike,
                "expiry": expiry_date
            }
    
    # If we couldn't find an exact match, log a warning
    logger.warning(f"No exact matching token found in script master for {exact_symbol}")
    return None

def get_option_token_from_cache_or_search(symbol, strike, option_type, expiry_date=None):
    """
    Get option token from cache if available, otherwise search for it
    with enhanced caching and error handling
    
    Args:
        symbol (str): Stock/index symbol (e.g., NIFTY, BANKNIFTY)
        strike (int/float): Strike price
        option_type (str): Option type (CE/PE)
        expiry_date (str, optional): Expiry date in format "DDMMMYY"
        
    Returns:
        dict: Option information with token
    """
    global option_token_cache
    
    # Get expiry date if not provided
    if not expiry_date or expiry_date == '28MAY25':
        expiry_date = get_next_expiry_date(symbol)
    
    # Standardize parameters
    symbol = symbol.upper()
    option_type = option_type.upper()
    strike_int = int(float(strike))
    
    # Create the exact symbol format
    option_symbol = build_option_symbol(symbol, expiry_date, strike_int, option_type)
    
    # Check if we have this in our cache
    cache_key = f"{symbol}_{expiry_date}_{strike_int}_{option_type}"
    if cache_key in option_token_cache:
        cache_entry = option_token_cache[cache_key]
        cache_age = datetime.now() - cache_entry.get("timestamp", datetime.min)
        
        # Use cache if it's not too old (less than 24 hours)
        if cache_age.total_seconds() < 86400:
            logger.info(f"Using cached token for {option_symbol}: {cache_entry.get('token')}")
            return cache_entry
        else:
            logger.info(f"Cache entry for {option_symbol} is stale, refreshing")
    
    # If not in cache or cache is stale, search for it in script master first
    option_info = find_option_token_in_json(symbol, strike_int, option_type, expiry_date)
    
    # If found in script master, cache it and return
    if option_info and option_info.get("token"):
        # Add timestamp to cache entry
        option_info["timestamp"] = datetime.now()
        option_token_cache[cache_key] = option_info
        logger.info(f"Cached token from script master for {option_symbol}: {option_info.get('token')}")
        return option_info
    
    # If not found in script master and we're connected to broker, search directly
    if broker_connected and smart_api is not None:
        try:
            logger.info(f"Searching for option token via broker API: {option_symbol}")
            
            # Generate variation for more thorough search
            search_variations = [option_symbol]
            
            # Add a common variation with spaces
            if len(symbol) <= 8:  # Only for shorter symbols to avoid false matches
                parts = re.match(r"([A-Z]+)(\d+[A-Z]{3}\d+)(\d+)([CPE]{2})", option_symbol)
                if parts:
                    sym, exp, strk, opt = parts.groups()
                    search_variations.append(f"{sym} {strk} {opt}")
            
            # Try each variation
            for search_text in search_variations:
                matches = search_symbols(search_text)
                
                # Process matches with detailed logging
                if matches:
                    logger.info(f"Found {len(matches)} potential matches for {search_text}")
                    
                    # Look for exact match first
                    exact_matches = []
                    for match in matches:
                        match_name = match.get("name", "").upper()
                        match_symbol = match.get("symbol", "").upper()
                        
                        logger.debug(f"Evaluating match: {match_name} / {match_symbol}")
                        
                        # Check for exact symbol match
                        if match_name == option_symbol or match_symbol == option_symbol:
                            exact_matches.append(match)
                    
                    # If we have exact matches, use the first one
                    if exact_matches:
                        best_match = exact_matches[0]
                        result = {
                            "token": best_match.get("token"),
                            "symbol": best_match.get("symbol", option_symbol),
                            "strike": strike_int,
                            "expiry": expiry_date,
                            "timestamp": datetime.now()
                        }
                        
                        # Cache the result
                        option_token_cache[cache_key] = result
                        logger.info(f"Found and cached exact match token for {option_symbol}: {result['token']}")
                        return result
                    
                    # No exact match, try partial matching with scoring
                    best_match = None
                    best_score = 0
                    
                    for match in matches:
                        match_name = match.get("name", "").upper()
                        match_symbol = match.get("symbol", "").upper()
                        
                        score = 0
                        
                        # Check if all key components are present
                        if symbol in match_name or symbol in match_symbol:
                            score += 10
                        
                        if str(strike_int) in match_name or str(strike_int) in match_symbol:
                            score += 10
                            
                        if option_type in match_name or option_type in match_symbol:
                            score += 10
                            
                        # Check for expiry components
                        if expiry_date in match_name or expiry_date in match_symbol:
                            score += 20
                        else:
                            # Check for month part
                            month_part = expiry_date[2:5]  # e.g., "MAR" from "27MAR25"
                            if month_part in match_name or month_part in match_symbol:
                                score += 10
                        
                        if score > best_score:
                            best_score = score
                            best_match = match
                    
                    # If we have a good match (score of 30 or higher)
                    if best_match and best_score >= 30:
                        result = {
                            "token": best_match.get("token"),
                            "symbol": best_match.get("symbol", option_symbol),
                            "strike": strike_int,
                            "expiry": expiry_date,
                            "timestamp": datetime.now()
                        }
                        
                        # Cache the result
                        option_token_cache[cache_key] = result
                        logger.info(f"Found and cached partial match token for {option_symbol} (score {best_score}): {result['token']}")
                        return result
        
        except Exception as e:
            logger.error(f"Error searching for option token via API: {e}")
    
    # If all searches fail, generate a consistent fallback token
    dummy_token = str(abs(hash(option_symbol)) % 100000)
    logger.warning(f"Could not find token for {option_symbol}, using fallback token: {dummy_token}")
    
    result = {
        "token": dummy_token,
        "symbol": option_symbol,
        "strike": strike_int,
        "expiry": expiry_date,
        "timestamp": datetime.now(),
        "is_fallback": True  # Flag to indicate this is a fallback token
    }
    
    # Cache fallback tokens too, but with a shorter expiry (1 hour)
    option_token_cache[cache_key] = result
    return result

def calculate_rounded_strike(current_price):
    """
    Calculate rounded strike price based on price magnitude:
    - Under 100: Round to nearest 5
    - 100-1000: Round to nearest 10
    - 1000-10000: Round to nearest 50
    - 10000-100000: Round to nearest 100
    - Above 100000: Round to nearest 100
    
    Maximum rounding adjustment is capped at 100 points
    
    Args:
        current_price (float): Current price of the security
        
    Returns:
        float: Rounded strike price
    """
    if current_price < 100:
        # Under 100: Round to nearest 5
        rounded_strike = round(current_price / 5) * 5
    elif current_price < 1000:
        # 100-1000: Round to nearest 10
        rounded_strike = round(current_price / 10) * 10
    elif current_price < 10000:
        # 1000-10000: Round to nearest 50
        rounded_strike = round(current_price / 50) * 50
    elif current_price < 100000:
        # 10000-100000: Round to nearest 100
        rounded_strike = round(current_price / 100) * 100
    else:
        # Above 100000: Round to nearest 100
        rounded_strike = round(current_price / 100) * 100
    
    # Calculate difference between rounded and original price
    diff = abs(rounded_strike - current_price)
    
    # If difference exceeds maximum of 100 points, cap it
    if diff > 100:
        # Move toward original price to limit difference to 100
        if rounded_strike > current_price:
            rounded_strike = current_price + 100
        else:
            rounded_strike = current_price - 100
        
        # Re-round to appropriate interval based on magnitude
        if current_price < 100:
            rounded_strike = round(rounded_strike / 5) * 5
        elif current_price < 1000:
            rounded_strike = round(rounded_strike / 5) * 5
        elif current_price < 10000:
            rounded_strike = round(rounded_strike / 10) * 10
        else:
            rounded_strike = round(rounded_strike / 100) * 100
    
    return rounded_strike

def get_option_token(symbol, current_price=None, option_type="CE", expiry_date=None):
    """
    Get option token with appropriate strike price rounding based on price magnitude
    
    Args:
        symbol (str): Stock/index symbol
        current_price (float, optional): Current price for ATM options
        option_type (str, optional): Option type (CE/PE)
        expiry_date (str, optional): Expiry date in format "DDMMMYY"
        
    Returns:
        dict: Option information with token or None if failed
    """
    # Get current price if not provided
    if current_price is None and symbol in stocks_data:
        current_price = stocks_data[symbol].get("ltp")
    
    if current_price is None:
        logger.error(f"Cannot get option token: No price available for {symbol}")
        return None
    
    # Get expiry date if not provided
    if not expiry_date:
        expiry_date = get_next_expiry_date(symbol)
    
    # Calculate rounded strike appropriate for the current price
    rounded_strike = calculate_rounded_strike(current_price)
    
    logger.info(f"Getting option token for {symbol} {rounded_strike} {option_type} {expiry_date}")
    
    # Get token using our enhanced function with caching
    option_info = get_option_token_from_cache_or_search(symbol, rounded_strike, option_type, expiry_date)
    
    if option_info and option_info.get("is_fallback"):
        logger.warning(f"Using fallback token for {symbol} {rounded_strike} {option_type}")
    
    return option_info

def search_and_validate_option_token(symbol, strike, option_type, expiry_date=None):
    """
    Comprehensive search and validation of option token with multiple fallbacks
    
    Args:
        symbol (str): Stock/index symbol (e.g., NIFTY, BANKNIFTY)
        strike (int/float): Strike price
        option_type (str): Option type (CE/PE)
        expiry_date (str, optional): Expiry date in format "DDMMMYY"
        
    Returns:
        dict: Validated option information with token or None if failed
    """
    if not symbol or not strike or not option_type:
        logger.error("Missing required parameters for option token search")
        return None
    
    # Standardize inputs
    symbol = symbol.upper()
    option_type = option_type.upper()
    strike_int = int(float(strike))
    
    # Get expiry date if not provided
    if not expiry_date or expiry_date == '28MAY25':
        expiry_date = get_next_expiry_date(symbol)
    
    logger.info(f"Searching for token: {symbol} {strike_int} {option_type} {expiry_date}")
    
    # First try: Get from cache or script master
    option_info = get_option_token_from_cache_or_search(symbol, strike_int, option_type, expiry_date)
    
    # If we got a result and it's not a fallback, verify it with a test API call
    if option_info and not option_info.get("is_fallback") and broker_connected and smart_api:
        token = option_info.get("token")
        option_symbol = option_info.get("symbol")
        
        try:
            # Try to get a test quote to validate the token
            logger.info(f"Validating token {token} for {option_symbol}")
            test_quote = smart_api.ltpData("NFO", option_symbol, token)
            
            if isinstance(test_quote, dict) and test_quote.get("status"):
                logger.info(f"Token validation successful for {option_symbol}")
                return option_info
            else:
                logger.warning(f"Token validation failed for {option_symbol}: {test_quote}")
                # Continue to next fallback
        except Exception as e:
            logger.warning(f"Error during token validation: {e}")
            # Continue to next fallback
    
    # Second try: Search for symbol variations
    if broker_connected and smart_api:
        # Prepare the option symbol in all possible formats
        option_symbol = build_option_symbol(symbol, expiry_date, strike_int, option_type)
        variation1 = f"{symbol}{strike_int}{option_type}"  # NIFTY19000CE
        variation2 = f"{symbol} {strike_int} {option_type}"  # NIFTY 19000 CE
        
        for search_text in [option_symbol, variation1, variation2]:
            try:
                logger.info(f"Searching broker API for: {search_text}")
                matches = search_symbols(search_text)
                
                if matches:
                    # Find the best match
                    for match in matches:
                        match_name = match.get("name", "").upper()
                        match_symbol = match.get("symbol", "").upper()
                        match_token = match.get("token")
                        
                        # Check if this is a good match
                        if (symbol in match_name or symbol in match_symbol) and \
                           (str(strike_int) in match_name or str(strike_int) in match_symbol) and \
                           (option_type in match_name or option_type in match_symbol):
                            
                            logger.info(f"Found match via broker search: {match_name} with token {match_token}")
                            
                            # Create and return the option info
                            found_info = {
                                "token": match_token,
                                "symbol": match_symbol or match_name,
                                "strike": strike_int,
                                "expiry": expiry_date,
                                "timestamp": datetime.now()
                            }
                            
                            # Cache this result
                            cache_key = f"{symbol}_{expiry_date}_{strike_int}_{option_type}"
                            option_token_cache[cache_key] = found_info
                            
                            return found_info
            except Exception as e:
                logger.warning(f"Error searching for {search_text}: {e}")
    
    # If all else fails, return the original result (which might be a fallback)
    if option_info:
        return option_info
    
    # Last resort: generate a fallback token
    logger.warning(f"All token search methods failed for {symbol} {strike_int} {option_type}")
    fallback_token = str(abs(hash(f"{symbol}{expiry_date}{strike_int}{option_type}")) % 100000)
    
    return {
        "token": fallback_token,
        "symbol": build_option_symbol(symbol, expiry_date, strike_int, option_type),
        "strike": strike_int,
        "expiry": expiry_date,
        "timestamp": datetime.now(),
        "is_fallback": True
    }

def clear_option_token_cache(older_than=None):
    """
    Clear option token cache to force fresh lookups
    
    Args:
        older_than (int, optional): Clear entries older than this many seconds
    """
    global option_token_cache
    
    if older_than is None:
        # Clear all entries
        logger.info(f"Clearing entire option token cache ({len(option_token_cache)} entries)")
        option_token_cache = {}
        return
    
    # Clear only entries older than specified time
    current_time = datetime.now()
    entries_to_remove = []
    
    for key, entry in option_token_cache.items():
        timestamp = entry.get("timestamp", datetime.min)
        age = (current_time - timestamp).total_seconds()
        
        if age > older_than:
            entries_to_remove.append(key)
    
    # Remove old entries
    for key in entries_to_remove:
        del option_token_cache[key]
    
    logger.info(f"Cleared {len(entries_to_remove)} old entries from option token cache")

# Add this function to build proper option symbols and tokens
# Add a cache for option tokens to avoid redundant lookups
option_token_cache = {}
def update_symbol_token(symbol, new_token=None, new_symbol=None):
    """
    Update symbol and token information comprehensively
    
    Args:
        symbol (str): Existing symbol to update
        new_token (str, optional): New token to replace existing token
        new_symbol (str, optional): New symbol to replace existing symbol
    
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate input
    if symbol not in stocks_data and symbol not in options_data:
        logger.warning(f"Symbol {symbol} not found for update")
        return False
    
    # Determine if it's a stock or option
    is_stock = symbol in stocks_data
    is_option = symbol in options_data
    
    if is_stock:
        stock_info = stocks_data[symbol]
        
        # Update token if provided
        if new_token:
            stock_info['token'] = new_token
            logger.info(f"Updated token for stock {symbol}: {new_token}")
        
        # Update symbol if provided
        if new_symbol:
            # Update the key in stocks_data
            stocks_data[new_symbol] = stocks_data.pop(symbol)
            stock_info['symbol'] = new_symbol
            logger.info(f"Updated stock symbol from {symbol} to {new_symbol}")
        
        return True
    
    if is_option:
        option_info = options_data[symbol]
        
        # Update token if provided
        if new_token:
            option_info['token'] = new_token
            logger.info(f"Updated token for option {symbol}: {new_token}")
        
        # Update symbol if provided
        if new_symbol:
            # Update the key in options_data
            options_data[new_symbol] = options_data.pop(symbol)
            option_info['symbol'] = new_symbol
            logger.info(f"Updated option symbol from {symbol} to {new_symbol}")
        
        return True
    
    return False

def search_and_update_symbol(old_symbol, search_text=None):
    """
    Search for a symbol and update its details
    
    Args:
        old_symbol (str): Existing symbol to search and update
        search_text (str, optional): Text to search, defaults to symbol itself
    
    Returns:
        dict: Updated symbol information or None
    """
    if not search_text:
        search_text = old_symbol
    
    # Search for the symbol using broker API
    matches = search_symbols(search_text)
    
    if matches:
        # Get the first match or find the most relevant
        best_match = matches[0]
        
        # Extract new details
        new_token = best_match.get('token')
        new_symbol = best_match.get('name') or best_match.get('symbol')
        
        # Perform update
        if update_symbol_token(old_symbol, new_token, new_symbol):
            return {
                'old_symbol': old_symbol,
                'new_symbol': new_symbol,
                'new_token': new_token,
                'details': best_match
            }
    
    return None

# Example usage
def update_specific_option(option_symbol):
    """
    Specific function to update an option's details
    """
    # Example: BANKNIFTY28MAR2550300CE
    result = search_and_update_symbol(option_symbol)
    
    if result:
        logger.info(f"Successfully updated option: {result}")
        return result
    else:
        logger.warning(f"Could not update option: {option_symbol}")
        return None

def get_option_token(symbol, current_price=None, option_type="CE", expiry_date=None):
    """Get option token with 100-point rounding"""
    # Get current price if not provided
    if current_price is None and symbol in stocks_data:
        current_price = stocks_data[symbol].get("ltp")
    
    if current_price is None:
        logger.error(f"Cannot get option token: No price available for {symbol}")
        return None
    
    # Get expiry date if not provided
    if expiry_date is None:
        expiry_date = get_next_expiry_date()
    
    # Round price to nearest 100 for all symbols
    rounded_strike = round(current_price / 100) * 100
    
    # Get token using our enhanced function with caching
    option_info = get_option_token_from_cache_or_search(symbol, rounded_strike, option_type, expiry_date)
    
    return option_info

def add_option(symbol, strike, expiry, option_type, token=None, exchange="NFO"):
    """
    Add a new option for a stock with improved token handling
    
    Args:
        symbol (str): Stock symbol 
        strike (str): Option strike price
        expiry (str): Option expiry date
        option_type (str): Option type (CE/PE)
        token (str, optional): Predefined token. Defaults to None.
        exchange (str, optional): Exchange. Defaults to "NFO".
    
    Returns:
        str: Unique option key or None if addition fails
    """
    # Validate and standardize inputs
    symbol = symbol.upper()
    option_type = option_type.upper()
    
    # Check if parent stock exists
    if symbol not in stocks_data:
        logger.error(f"Cannot add option: Parent stock {symbol} does not exist")
        return None
    
    # Create properly formatted option symbol
    option_symbol = build_option_symbol(symbol, expiry, strike, option_type)
    
    # Create unique option key
    option_key = f"{symbol}_{expiry}_{strike}_{option_type}"
    
    # Return existing option key if already exists
    if option_key in options_data:
        logger.info(f"Option {option_key} already exists, returning existing key")
        return option_key
    
    # Retrieve or generate token
    if token is None:
        # Use our improved token lookup
        option_info = search_and_validate_option_token(symbol, strike, option_type, expiry)
        if option_info and "token" in option_info:
            token = option_info.get("token")
            option_symbol = option_info.get("symbol", option_symbol)  # Use the symbol from the search result
            logger.info(f"Using token for {option_symbol}: {token}")
    
    # Double-check token validity
    if token is None or not token:
        logger.warning(f"Could not find valid token for {option_symbol}, using fallback token")
        token = str(hash(option_symbol) % 100000)
        is_fallback = True
    else:
        is_fallback = False
    
    # Prepare option entry
    option_entry = {
        "symbol": option_symbol,
        "token": token,
        "exchange": exchange,
        "ltp": None,
        "high": None,
        "low": None,
        "open": None,
        "previous": None,
        "change_percent": 0,
        "price_history": pd.DataFrame(columns=[
            'timestamp', 'price', 'volume', 'open_interest', 
            'change', 'open', 'high', 'low'
        ]),
        "signal": 0,
        "strength": 0,
        "trend": "NEUTRAL",
        "strike": strike,
        "expiry": expiry,
        "parent_symbol": symbol,
        "option_type": option_type,
        "last_updated": None,
        "data_source": "broker" if not is_fallback else "fallback",
        "is_fallback": is_fallback
    }
    
    # Store option data
    options_data[option_key] = option_entry
    
    # Add to parent stock's options list
    if option_type in stocks_data[symbol]["options"]:
        if option_key not in stocks_data[symbol]["options"][option_type]:
            stocks_data[symbol]["options"][option_type].append(option_key)
    
    # Update trading state
    trading_state.add_option(option_key)
    
    # Mark for immediate update
    last_data_update["options"][option_key] = None
    
    logger.info(f"Added new option: {option_symbol} ({token}) for {symbol}")
    return option_key

def remove_option(option_key):
    """Remove an option from tracking"""
    global options_data, stocks_data
    
    if option_key not in options_data:
        return False
    
    # Get option details
    option_info = options_data[option_key]
    parent_symbol = option_info.get("parent_symbol")
    option_type = option_info.get("option_type")
    
    # Exit any active trade on this option
    if trading_state.active_trades.get(option_key, False):
        exit_trade(option_key, reason="Option removed")
    
    # Remove from parent stock's options list
    if parent_symbol in stocks_data and option_type in stocks_data[parent_symbol]["options"]:
        if option_key in stocks_data[parent_symbol]["options"][option_type]:
            stocks_data[parent_symbol]["options"][option_type].remove(option_key)
        
        # Update primary option if needed
        if stocks_data[parent_symbol].get(f"primary_{option_type.lower()}") == option_key:
            stocks_data[parent_symbol][f"primary_{option_type.lower()}"] = None
    
    # Clean up trading state
    trading_state.remove_option(option_key)
    
    # Remove option data
    del options_data[option_key]
    
    if option_key in last_data_update["options"]:
        del last_data_update["options"][option_key]
    
    logger.info(f"Removed option: {option_key}")
    return True

def find_and_add_options(symbol, current_price=None, expiry_date=None):
    """
    Find and add ATM options for a stock with comprehensive error handling
    
    Args:
        symbol (str): Stock or index symbol
        current_price (float, optional): Current price of the stock/index. Defaults to None.
        expiry_date (str, optional): Expiry date in format "DDMMMYY". Defaults to None.
    
    Returns:
        dict: Dictionary with CE and PE option keys
    """
    # Get current price if not provided
    if current_price is None and symbol in stocks_data:
        current_price = stocks_data[symbol].get("ltp")
    
    if current_price is None:
        logger.error(f"Cannot find options: No price available for {symbol}")
        return {"CE": None, "PE": None}
    
    # Get option tokens for CE and PE using enhanced search
    ce_info = get_option_token(symbol, current_price, "CE", expiry_date)
    pe_info = get_option_token(symbol, current_price, "PE", expiry_date)
    
    # Results to track added options
    added_options = {"CE": None, "PE": None}
    
    # Add CE option
    if ce_info:
        try:
            logger.info(f"Adding CE option for {symbol}: Strike={ce_info['strike']}, Token={ce_info['token']}")
            
            ce_key = add_option(
                symbol,
                str(ce_info["strike"]),
                ce_info["expiry"],
                "CE",
                ce_info["token"]
            )
            added_options["CE"] = ce_key
            
            # Set as primary CE
            if ce_key and symbol in stocks_data:
                stocks_data[symbol]["primary_ce"] = ce_key
                logger.info(f"Set {ce_key} as primary CE option for {symbol}")
                
                # Immediately fetch data for this option
                if broker_connected:
                    fetch_option_data(ce_key)
        except Exception as e:
            logger.error(f"Error adding CE option for {symbol}: {e}")
    else:
        logger.warning(f"Could not get CE option token for {symbol}")
    
    # Add PE option
    if pe_info:
        try:
            logger.info(f"Adding PE option for {symbol}: Strike={pe_info['strike']}, Token={pe_info['token']}")
            
            pe_key = add_option(
                symbol,
                str(pe_info["strike"]),
                pe_info["expiry"],
                "PE",
                pe_info["token"]
            )
            added_options["PE"] = pe_key
            
            # Set as primary PE
            if pe_key and symbol in stocks_data:
                stocks_data[symbol]["primary_pe"] = pe_key
                logger.info(f"Set {pe_key} as primary PE option for {symbol}")
                
                # Immediately fetch data for this option
                if broker_connected:
                    fetch_option_data(pe_key)
        except Exception as e:
            logger.error(f"Error adding PE option for {symbol}: {e}")
    else:
        logger.warning(f"Could not get PE option token for {symbol}")
    
    # Log the results
    logger.info(f"Added options for {symbol}: CE={added_options['CE']}, PE={added_options['PE']}")
    return added_options

def find_options_at_strikes(symbol, strikes, expiry_date=None):
    """
    Find options at specific strike prices
    
    Args:
        symbol (str): Stock or index symbol
        strikes (list): List of strike prices to find
        expiry_date (str, optional): Expiry date in format "DDMMMYY". Defaults to None.
    
    Returns:
        dict: Dictionary with CE and PE options at each strike
    """
    results = {
        "CE": {},
        "PE": {}
    }
    
    if not strikes:
        logger.warning(f"No strikes provided for {symbol}")
        return results
    
    # Get expiry date if not provided
    if not expiry_date:
        expiry_date = get_next_expiry_date(symbol)
    
    logger.info(f"Finding options for {symbol} at strikes {strikes} for expiry {expiry_date}")
    
    # Process each strike
    for strike in strikes:
        # Get CE option
        ce_info = search_and_validate_option_token(symbol, strike, "CE", expiry_date)
        if ce_info:
            try:
                ce_key = add_option(
                    symbol,
                    str(ce_info["strike"]),
                    ce_info["expiry"],
                    "CE",
                    ce_info["token"]
                )
                results["CE"][strike] = ce_key
                logger.info(f"Added CE option for {symbol} at strike {strike}: {ce_key}")
            except Exception as e:
                logger.error(f"Error adding CE option at strike {strike}: {e}")
        
        # Get PE option
        pe_info = search_and_validate_option_token(symbol, strike, "PE", expiry_date)
        if pe_info:
            try:
                pe_key = add_option(
                    symbol,
                    str(pe_info["strike"]),
                    pe_info["expiry"],
                    "PE",
                    pe_info["token"]
                )
                results["PE"][strike] = pe_key
                logger.info(f"Added PE option for {symbol} at strike {strike}: {pe_key}")
            except Exception as e:
                logger.error(f"Error adding PE option at strike {strike}: {e}")
    
    # Return the results
    return results

def find_suitable_options(stock_symbol, stock_price, expiry_days=7):
    """
    Find suitable options for a stock based on its current price.
    
    Args:
        stock_symbol (str): Stock or index symbol
        stock_price (float): Current price of the stock/index
        expiry_days (int): Days to expiry from today

    Returns:
        tuple: CE and PE option keys
    """
    # Find closest expiry date based on expiry_days
    from datetime import datetime, timedelta
    today = datetime.now()
    expiry_date = today + timedelta(days=expiry_days)
    expiry_str = expiry_date.strftime('%d%b%y').upper()  # e.g., "27MAR25"
    
    # Find and add options
    options = find_and_add_options(stock_symbol, stock_price, expiry_str)
    
    if not options:
        return None, None
        
    return options["CE"], options["PE"]

def update_option_selection(force_update=False):
    """
    Update option selection for all stocks based on current prices, with improved handling.
    
    Args:
        force_update (bool, optional): Force update regardless of time interval. Defaults to False.
    """
    global stocks_data, last_option_selection_update, options_data
    
    current_time = datetime.now()
    
    # Only update periodically unless forced
    if not force_update and (current_time - last_option_selection_update).total_seconds() < OPTION_AUTO_SELECT_INTERVAL:
        return
    
    logger.info("Updating option selection based on current stock prices")
    
    # Process each stock
    for symbol, stock_info in stocks_data.items():
        current_price = stock_info.get("ltp")
        
        if current_price is None or current_price <= 0:
            logger.warning(f"Skipping option selection for {symbol}: Invalid price {current_price}")
            continue
            
        # Check if we already have primary options
        primary_ce = stock_info.get("primary_ce")
        primary_pe = stock_info.get("primary_pe")
        
        need_new_options = False
        
        # If we have both options, check if they're still ATM
        if primary_ce and primary_pe and primary_ce in options_data and primary_pe in options_data:
            ce_strike = float(options_data[primary_ce].get("strike", 0))
            pe_strike = float(options_data[primary_pe].get("strike", 0))
            
            # Determine strike interval based on symbol
            if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                strike_interval = 50
            elif symbol in ["RELIANCE", "SBIN", "HDFCBANK", "ICICIBANK", "INFY"]:
                strike_interval = 20
            else:
                strike_interval = 10
            
            # Check if current options are still close to ATM
            ce_distance = abs(current_price - ce_strike)
            pe_distance = abs(current_price - pe_strike)
            
            # If either option is too far from current price
            if ce_distance > strike_interval * 2 or pe_distance > strike_interval * 2:
                need_new_options = True
                logger.info(f"Options for {symbol} are too far from current price (CE: {ce_distance}, PE: {pe_distance})")
            
            # If options are using fallback prices too often
            elif any(options_data[opt].get("using_fallback", False) for opt in [primary_ce, primary_pe]):
                need_new_options = True
                logger.info(f"Options for {symbol} are using fallback prices, trying to find better ones")
        else:
            # If we're missing primary options, we need new ones
            need_new_options = True
            logger.info(f"Missing primary options for {symbol}")
        
        # If we need new options, find suitable ones
        if need_new_options:
            # Get the next expiry date (can be customized by symbol)
            expiry_date = get_next_expiry_date(symbol)
            
            # Find and add options
            options = find_and_add_options(symbol, current_price, expiry_date)
            
            # Update the stock's primary options
            if options["CE"]:
                stock_info["primary_ce"] = options["CE"]
            
            if options["PE"]:
                stock_info["primary_pe"] = options["PE"]
            
            logger.info(f"Selected new options for {symbol}: CE={options.get('CE')}, PE={options.get('PE')}")
    
    # Update timestamp
    last_option_selection_update = current_time

def schedule_option_token_refresh():
    """
    Schedule periodic option token refreshes to ensure tokens are valid
    """
    global option_token_cache
    
    # This function would typically be called in a thread
    # For options trading applications, tokens should be refreshed daily
    
    # Clear tokens older than 12 hours
    clear_option_token_cache(older_than=43200)
    
    # Refresh tokens for active trades
    for option_key, is_active in trading_state.active_trades.items():
        if is_active and option_key in options_data:
            option_info = options_data[option_key]
            
            # Only refresh if using fallback or having issues
            if option_info.get("is_fallback", False) or option_info.get("using_fallback", False):
                logger.info(f"Refreshing token for active trade option: {option_key}")
                
                symbol = option_info.get("parent_symbol")
                strike = option_info.get("strike")
                option_type = option_info.get("option_type")
                expiry = option_info.get("expiry")
                
                if all([symbol, strike, option_type, expiry]):
                    # Try to get a better token
                    new_token_info = search_and_validate_option_token(
                        symbol, strike, option_type, expiry
                    )
                    
                    if new_token_info and not new_token_info.get("is_fallback", False):
                        # Update the option info
                        option_info["token"] = new_token_info.get("token")
                        option_info["symbol"] = new_token_info.get("symbol", option_info.get("symbol"))
                        option_info["is_fallback"] = False
                        
                        logger.info(f"Successfully refreshed token for {option_key}")
    
    logger.info(f"Option token refresh completed. Cache size: {len(option_token_cache)}")

# ============ Technical Indicators ============
def calculate_rsi(data, period=RSI_PERIOD):
    """Calculate RSI technical indicator."""
    if len(data) < period + 1:
        return 50  # Default neutral RSI when not enough data
    
    # Get price differences
    delta = data.diff()
    
    # Get gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss - use SMA for first period, then EMA
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # First period SMA
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    
    # First SMA value
    first_avg_gain = gain.iloc[:period].mean()
    first_avg_loss = loss.iloc[:period].mean()
    
    # Use more efficient vectorized operations
    for i in range(period, len(gain)):
        if i == period:
            avg_gain[i] = first_avg_gain
            avg_loss[i] = first_avg_loss
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = np.zeros_like(avg_gain)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi[-1]  # Return last value

def calculate_macd(data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD technical indicator."""
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
    
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger_bands(data, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
    """Calculate Bollinger Bands technical indicator."""
    if len(data) < period:
        return data.iloc[-1], data.iloc[-1], data.iloc[-1]  # Default when not enough data
    
    # Calculate middle band (SMA)
    middle_band = data.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = data.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1]

def calculate_atr(data, high=None, low=None, period=ATR_PERIOD):
    """Calculate Average True Range (ATR) technical indicator."""
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
    
    return atr.iloc[-1]

def calculate_ema(data, span):
    """Calculate Exponential Moving Average (EMA)."""
    if len(data) < span:
        return data.iloc[-1]  # Default when not enough data
    
    return data.ewm(span=span, adjust=False).mean().iloc[-1]

def calculate_momentum(data, period=10):
    """Calculate price momentum over a period."""
    if len(data) < period:
        return 0  # Default when not enough data
    
    # Handle division by zero
    if data.iloc[-period] == 0:
        return 0
    
    return (data.iloc[-1] - data.iloc[-period]) / data.iloc[-period] * 100

def calculate_trend_strength(data, period=20):
    """Calculate trend strength using linear regression slope and r-squared"""
    if len(data) < period:
        return 0, 0
    
    # Use the last 'period' data points
    y = data.tail(period).values
    x = np.arange(period)
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate r-squared (coefficient of determination)
    r_squared = r_value ** 2
    
    # Normalize slope to percentage
    avg_price = y.mean()
    if avg_price > 0:
        normalized_slope = (slope * period) / avg_price * 100
    else:
        normalized_slope = 0
    
    return normalized_slope, r_squared

# 1. First, let's improve the calculate_support_resistance function to better handle errors:

def calculate_support_resistance(symbol):
    """
    Calculate support and resistance levels with better data validation
    """
    if symbol not in stocks_data:
        logger.warning(f"Cannot calculate S/R: Symbol {symbol} not found")
        return False
    
    stock_info = stocks_data[symbol]
    df = stock_info["price_history"]
    
    logger.info(f"Calculating S/R for {symbol} with {len(df)} data points")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    try:
        # Make sure price column exists
        if 'price' not in df.columns:
            logger.warning(f"Price column missing in historical data for {symbol}")
            return False
            
        # Get valid prices (drop NaN values)
        prices = df['price'].dropna()
        logger.info(f"Found {len(prices)} valid price points for {symbol}")
        
        if len(prices) < 50:  # Need at least 50 valid points for reliable S/R
            logger.warning(f"Not enough valid prices for {symbol} after dropping NaN: {len(prices)}")
            return False
        
        # Make sure prices is sorted chronologically
        if 'timestamp' in df.columns:
            # Create a new DataFrame with valid prices and timestamps
            valid_data = df[['timestamp', 'price']].dropna(subset=['price'])
            prices = valid_data.sort_values('timestamp')['price']
        
        # Find swing points for S/R levels
        # Rest of your existing S/R calculation logic...
        
        # After calculation, make sure to update UI data store
        if 'stocks' not in ui_data_store:
            ui_data_store['stocks'] = {}
        if symbol not in ui_data_store['stocks']:
            ui_data_store['stocks'][symbol] = {}
        
        ui_data_store['stocks'][symbol]['support_levels'] = stock_info["support_levels"]
        ui_data_store['stocks'][symbol]['resistance_levels'] = stock_info["resistance_levels"]
        
        logger.info(f"Updated S/R for {symbol}: S={stock_info['support_levels']}, R={stock_info['resistance_levels']}")
        return True
    except Exception as e:
        logger.error(f"Error calculating S/R for {symbol}: {e}", exc_info=True)
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
    """Generate trading signals for the specified option."""
    if option_key not in options_data:
        return
    
    option_info = options_data[option_key]
    price_history = option_info["price_history"]
    
    # Skip if not enough data
    if len(price_history) <= LONG_WINDOW:
        return
    
    price_series = price_history['price']
    is_call = option_info["option_type"] == "CE"
    
    # Calculate technical indicators
    rsi = calculate_rsi(price_series)
    macd_line, signal_line, histogram = calculate_macd(price_series)
    
    # Initialize signal strength and signal value
    signal_strength = 0
    signal = 0
    
    # RSI signals
    if rsi < RSI_OVERSOLD:
        signal += 1.5
        signal_strength += 1.0
    elif rsi > RSI_OVERBOUGHT:
        signal -= 1.5
        signal_strength += 1.0
    
    # MACD signal
    if macd_line > signal_line:
        signal += 1.5
        signal_strength += 1.0
    elif macd_line < signal_line:
        signal -= 1.5
        signal_strength += 1.0
    
    # Adjust signal based on option type
    if not is_call:
        signal = -signal  # Invert signal for put options
    
    # Normalize signal strength to 0-10 range
    signal_strength = min(max(signal_strength * 1.5, 0), 10)
    
    # Save signal and strength
    option_info["signal"] = signal
    option_info["strength"] = signal_strength
    
    # Determine trend based on signal and strength
    if signal > 2:
        option_info["trend"] = "BULLISH"
    elif signal > 1:
        option_info["trend"] = "MODERATELY BULLISH"
    elif signal < -2:
        option_info["trend"] = "BEARISH"
    elif signal < -1:
        option_info["trend"] = "MODERATELY BEARISH"
    else:
        option_info["trend"] = "NEUTRAL"

# ============ PCR Analysis ============
@rate_limited
def fetch_option_chain(symbol):
    """Generate simulated option chain for PCR calculation"""
    try:
        # Get current price for the symbol
        current_price = None
        if symbol in stocks_data:
            current_price = stocks_data[symbol].get("ltp")
        
        if current_price is None:
            current_price = 100  # Default value
        
        # Generate some dummy strikes around current price
        base_strike = round(current_price / 50) * 50  # Round to nearest 50
        strikes = [base_strike + (i - 5) * 50 for i in range(10)]  # 5 strikes below, 5 above
        
        # Generate dummy option chain
        option_chain = []
        total_ce_oi = 0
        total_pe_oi = 0
        
        for strike in strikes:
            # Generate some OI data with slight bias based on strike price
            ce_oi = int(random.random() * 10000 * (1.1 if strike > current_price else 0.9))
            pe_oi = int(random.random() * 10000 * (0.9 if strike > current_price else 1.1))
            
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            
            # CE option
            ce_option = {
                "optionType": "CE",
                "strikePrice": strike,
                "openInterest": ce_oi,
                "lastPrice": max(0.1, current_price - strike + random.random() * 10)
            }
            option_chain.append(ce_option)
            
            # PE option
            pe_option = {
                "optionType": "PE",
                "strikePrice": strike,
                "openInterest": pe_oi,
                "lastPrice": max(0.1, strike - current_price + random.random() * 10)
            }
            option_chain.append(pe_option)
        
        # Add total OI to the option chain data for easier PCR calculation
        option_chain.append({"totalCEOI": total_ce_oi, "totalPEOI": total_pe_oi})
        
        return option_chain
    except Exception as e:
        logger.error(f"Error generating simulated option chain for {symbol}: {e}")
        return None

def determine_pcr_trend(symbol):
    """Determine the trend of PCR based on recent history."""
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

def update_all_pcr_data():
    """Update PCR data for all stocks more efficiently."""
    global pcr_data, last_pcr_update
    
    current_time = datetime.now()
    if (current_time - last_pcr_update).total_seconds() < PCR_UPDATE_INTERVAL:
        return
    
    # Fetch PCR data for all symbols in a single API call
    all_pcr_values = fetch_pcr_data()
    
    # Update PCR data for each symbol
    for symbol in stocks_data.keys():
        if symbol not in pcr_data:
            pcr_data[symbol] = {
                "current": 1.0,
                "history": deque(maxlen=PCR_HISTORY_LENGTH),
                "trend": "NEUTRAL",
                "strength": 0.0,
                "last_updated": None
            }
        
        # Use real PCR if available
        if symbol in all_pcr_values:
            pcr = all_pcr_values[symbol]
            pcr_data[symbol]["current"] = pcr
            pcr_data[symbol]["history"].append(pcr)
            pcr_data[symbol]["last_updated"] = datetime.now()
            
            # Determine PCR trend
            determine_pcr_trend(symbol)
            calculate_pcr_strength(symbol)
        else:
            # Fall back to existing method for symbols without real PCR data
            update_pcr_data(symbol)
    
    # Update market sentiment
    update_market_sentiment()
    
    last_pcr_update = current_time
def get_pcr_signal(symbol):
    """Generate trading signal based on PCR value and trend."""
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
    """Update market sentiment based on technical indicators and PCR."""
    global market_sentiment, stocks_data
    
    # Process each stock individually
    for symbol, stock_info in stocks_data.items():
        # Skip if not enough data
        if len(stock_info["price_history"]) < 20:
            continue
            
        price_series = stock_info["price_history"]['price']
        
        # Calculate technical indicators
        rsi = calculate_rsi(price_series)
        macd_line, signal_line, histogram = calculate_macd(price_series)
        pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
        
        # Initialize sentiment score
        sentiment_score = 0
        
        # RSI indicator
        if rsi > 70:
            sentiment_score -= 1  # Overbought - bearish
        elif rsi < 30:
            sentiment_score += 1  # Oversold - bullish
            
        # MACD indicator
        if histogram > 0:
            sentiment_score += 1  # Bullish
        else:
            sentiment_score -= 1  # Bearish
            
        # PCR indicator
        if pcr_value > PCR_BEARISH_THRESHOLD:
            sentiment_score -= 1  # Bearish
        elif pcr_value < PCR_BULLISH_THRESHOLD:
            sentiment_score += 1  # Bullish
            
        # Set sentiment based on score
        if sentiment_score >= 2:
            market_sentiment[symbol] = "BULLISH"
        elif sentiment_score <= -2:
            market_sentiment[symbol] = "BEARISH"
        else:
            market_sentiment[symbol] = "NEUTRAL"
    
    # Calculate overall market sentiment
    bullish_count = sum(1 for s in market_sentiment.values() if s == "BULLISH")
    bearish_count = sum(1 for s in market_sentiment.values() if s == "BEARISH")
    
    if bullish_count > bearish_count:
        market_sentiment["overall"] = "BULLISH"
    elif bearish_count > bullish_count:
        market_sentiment["overall"] = "BEARISH"
    else:
        market_sentiment["overall"] = "NEUTRAL"

# ============ Strategy Prediction ============

def predict_strategy_for_stock(symbol):
    """
    Predict the most suitable trading strategy for a stock based on its 
    characteristics and current market conditions
    
    Args:
        symbol (str): Stock symbol to predict strategy for
    """
    if symbol not in stocks_data:
        return None
    
    stock_info = stocks_data[symbol]
    
    try:
        # Skip if we don't have enough price history
        if len(stock_info["price_history"]) < 30:
            logger.info(f"Not enough price history for {symbol} to predict strategy")
            return None
        
        if 'price' not in stock_info["price_history"].columns:
            logger.warning(f"Price column missing in historical data for {symbol}")
            return None
            
        price_series = stock_info["price_history"]['price'].dropna()
        
        if len(price_series) < 30:
            logger.info(f"Not enough valid price points for {symbol} to predict strategy")
            return None
        
        # Ensure price_series is sorted chronologically
        if 'timestamp' in stock_info["price_history"].columns:
            price_series = stock_info["price_history"].sort_values('timestamp')['price'].dropna()
        
        # Calculate key technical metrics with error handling
        try:
            rsi = calculate_rsi(price_series)
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            rsi = 50  # Default neutral value
            
        volatility = calculate_volatility(symbol)
        
        try:
            momentum = calculate_momentum(price_series, 5)
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {e}")
            momentum = 0  # Default neutral value
            
        try:
            atr = calculate_atr(price_series)
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            atr = 1.0  # Default value
            
        try:
            trend_slope, trend_r2 = calculate_trend_strength(price_series, 20)
        except Exception as e:
            logger.error(f"Error calculating trend strength for {symbol}: {e}")
            trend_slope, trend_r2 = 0, 0  # Default neutral values
        
        # Get PCR data
        pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
        pcr_trend = pcr_data.get(symbol, {}).get("trend", "NEUTRAL")
        pcr_strength = pcr_data.get(symbol, {}).get("strength", 0.0)
        
        # Get market sentiment
        sentiment = market_sentiment.get(symbol, "NEUTRAL")
        
        # Initialize strategy scores
        strategy_scores = {
            "SCALP": 0,
            "SWING": 0,
            "MOMENTUM": 0
        }
        
        # Scalping suitability factors
        if volatility > 0.2 and volatility < 1.0:  # Moderate volatility
            strategy_scores["SCALP"] += 2
        if abs(momentum) < 5:  # Not too much short-term momentum
            strategy_scores["SCALP"] += 1
        if rsi > 40 and rsi < 60:  # Not overbought or oversold
            strategy_scores["SCALP"] += 1
        if abs(trend_slope) < 0.05:  # Not a strong trend
            strategy_scores["SCALP"] += 1
        
        # Swing trading suitability factors
        if abs(trend_slope) > 0.05 and trend_r2 > 0.6:  # Strong trend with good r-squared
            strategy_scores["SWING"] += 2
        if volatility > 0.5 and volatility < 2.0:  # Medium volatility
            strategy_scores["SWING"] += 1
        if (rsi < 30 and trend_slope > 0) or (rsi > 70 and trend_slope < 0):  # Potential reversal points
            strategy_scores["SWING"] += 2
        if pcr_strength > 0.5 or pcr_strength < -0.5:  # Strong PCR indication
            strategy_scores["SWING"] += 1
        
        # Momentum trading suitability factors
        if abs(momentum) > 5:  # Strong momentum
            strategy_scores["MOMENTUM"] += 2
        if (trend_slope > 0.1 and rsi < 70) or (trend_slope < -0.1 and rsi > 30):  # Trend with room to run
            strategy_scores["MOMENTUM"] += 2
        if volatility > 1.0:  # Higher volatility
            strategy_scores["MOMENTUM"] += 1
        if sentiment == "BULLISH" and trend_slope > 0:  # Bullish sentiment with uptrend
            strategy_scores["MOMENTUM"] += 1
        if sentiment == "BEARISH" and trend_slope < 0:  # Bearish sentiment with downtrend
            strategy_scores["MOMENTUM"] += 1
        
        # Find the best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = min(strategy_scores[best_strategy] / 5.0, 1.0)  # Normalize to 0-1 range
        
        # Look for strong signals
        if strategy_scores[best_strategy] > 3 and confidence > 0.6:
            logger.info(f"Predicted {best_strategy} strategy for {symbol} with {confidence:.2f} confidence")
            
            # Store the prediction
            stock_info["predicted_strategy"] = best_strategy
            stock_info["strategy_confidence"] = confidence
            
            # Update UI data store
            ui_data_store['predicted_strategies'][symbol] = {
                'strategy': best_strategy,
                'confidence': confidence,
                'score': strategy_scores
            }
            
            return best_strategy
        else:
            # Not enough confidence for any strategy
            logger.info(f"No strong strategy signal for {symbol}. Best: {best_strategy} with {confidence:.2f} confidence")
            stock_info["predicted_strategy"] = None
            stock_info["strategy_confidence"] = 0
            
            # Update UI data store
            ui_data_store['predicted_strategies'][symbol] = {
                'strategy': "NONE",
                'confidence': 0,
                'score': strategy_scores
            }
            
            return None
            
    except Exception as e:
        logger.error(f"Error predicting strategy for {symbol}: {e}", exc_info=True)
        return None

# ============ Trading Strategy ============
def should_enter_trade(option_key):
    """Determine if we should enter a trade for the given option."""
    if option_key not in options_data:
        return False, None
    
    option_info = options_data[option_key]
    parent_symbol = option_info["parent_symbol"]
    option_type = option_info["option_type"]
    
    # Don't enter trades if broker is not connected
    if not broker_connected:
        return False, None
    
    # Check if any trading strategies are enabled
    if not strategy_settings["SCALP_ENABLED"] and not strategy_settings["SWING_ENABLED"] and not strategy_settings["MOMENTUM_ENABLED"]:
        return False, None
    
    # Check if we've hit the maximum trades for the day
    if trading_state.trades_today >= MAX_TRADES_PER_DAY:
        return False, None
    
    # Check if we've hit the maximum loss percentage for the day
    if trading_state.daily_pnl <= -MAX_LOSS_PERCENTAGE * trading_state.capital / 100:
        return False, None
    
    # Check if we're already in a trade for this option
    if trading_state.active_trades.get(option_key, False):
        return False, None
    
    # Get signal and strength
    signal = option_info["signal"]
    strength = option_info["strength"]
    
    # Check the market sentiment for the parent symbol
    sentiment = market_sentiment.get(parent_symbol, "NEUTRAL")
    
    # Check if there's a predicted strategy for this stock
    predicted_strategy = stocks_data.get(parent_symbol, {}).get("predicted_strategy")
    strategy_confidence = stocks_data.get(parent_symbol, {}).get("strategy_confidence", 0)
    
    # Default to determining strategy based on signals
    strategy_type = None
    should_enter = False
    
    # For a call option, we want positive signal; for a put, we want negative signal
    signal_aligned = (option_type == "CE" and signal > 0) or (option_type == "PE" and signal < 0)
    
    # For a call option, we want bullish sentiment; for a put, we want bearish sentiment
    sentiment_aligned = (option_type == "CE" and sentiment == "BULLISH") or (option_type == "PE" and sentiment == "BEARISH")
    
    # Prioritize the predicted strategy if it has good confidence
    if predicted_strategy and strategy_confidence > 0.7 and strategy_settings[f"{predicted_strategy}_ENABLED"]:
        # Check if the signal is aligned with the predicted strategy
        if signal_aligned:
            min_strength = globals()[f"MIN_SIGNAL_STRENGTH_{predicted_strategy}"]
            if strength >= min_strength:
                strategy_type = predicted_strategy
                should_enter = True
                logger.info(f"Using predicted strategy {strategy_type} for {option_key} (confidence: {strategy_confidence:.2f})")
    
    # If no predicted strategy or not qualified, fall back to normal strategy selection
    if not should_enter:
        # Check scalping conditions
        if strategy_settings["SCALP_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_SCALP:
            strategy_type = "SCALP"
            should_enter = True
        
        # Check momentum conditions if scalping isn't viable
        elif not should_enter and strategy_settings["MOMENTUM_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_MOMENTUM:
            strategy_type = "MOMENTUM"
            should_enter = True
        
        # Check swing trading conditions if other strategies aren't viable
        elif not should_enter and strategy_settings["SWING_ENABLED"] and signal_aligned and sentiment_aligned and strength >= MIN_SIGNAL_STRENGTH_SWING:
            strategy_type = "SWING"
            should_enter = True
    
    return should_enter, strategy_type

def enter_trade(option_key, strategy_type):
    """Enter a trade for the given option with the specified strategy."""
    global trading_state, options_data, stocks_data
    
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
    
    # Calculate position size
    quantity = int(1000 / current_price)  # Simple position sizing
    if quantity < 1:
        quantity = 1
    
    # Set stop loss and target based on strategy
    if strategy_type == "SCALP":
        stop_loss_pct = 0.01  # 1% for scalping
        target_pct = 0.015    # 1.5% for scalping
    elif strategy_type == "MOMENTUM":
        stop_loss_pct = 0.015  # 1.5% for momentum
        target_pct = 0.03      # 3% for momentum
    else:  # SWING
        stop_loss_pct = 0.02   # 2% for swing
        target_pct = 0.05      # 5% for swing
    
    # Calculate actual stop loss and target
    option_type = option_info["option_type"]
    if option_type == "CE":
        stop_loss = current_price * (1 - stop_loss_pct)
        target = current_price * (1 + target_pct)
    else:  # PE
        stop_loss = current_price * (1 + stop_loss_pct)
        target = current_price * (1 - target_pct)
    
    # Update trading state
    trading_state.active_trades[option_key] = True
    trading_state.entry_price[option_key] = current_price
    trading_state.entry_time[option_key] = datetime.now()
    trading_state.stop_loss[option_key] = stop_loss
    trading_state.initial_stop_loss[option_key] = stop_loss
    trading_state.target[option_key] = target
    trading_state.trailing_sl_activated[option_key] = False
    trading_state.quantity[option_key] = quantity
    trading_state.strategy_type[option_key] = strategy_type
    
    # Store the parent stock price
    if parent_symbol in stocks_data:
        trading_state.stock_entry_price[option_key] = stocks_data[parent_symbol]["ltp"]
    
    # Increment trades today counter
    trading_state.trades_today += 1
    
    logger.info(f"Entered {option_key} {strategy_type} trade: Price={current_price}, SL={stop_loss}, Target={target}, Qty={quantity}")
    return True

def should_exit_trade(option_key):
    """Determine if we should exit a trade for the given option."""
    if option_key not in options_data or not trading_state.active_trades.get(option_key, False):
        return False, None
    
    option_info = options_data[option_key]
    current_price = option_info["ltp"]
    
    if current_price is None or current_price <= 0:
        return False, None
    
    entry_price = trading_state.entry_price[option_key]
    stop_loss = trading_state.stop_loss[option_key]
    target = trading_state.target[option_key]
    option_type = option_info["option_type"]
    
    # Check for stop loss hit
    if (option_type == "CE" and current_price <= stop_loss) or (option_type == "PE" and current_price >= stop_loss):
        return True, "Stop Loss"
    
    # Check for target hit
    if (option_type == "CE" and current_price >= target) or (option_type == "PE" and current_price <= target):
        return True, "Target"
    
    # Check for signal reversal
    signal = option_info["signal"]
    if (option_type == "CE" and signal < -1) or (option_type == "PE" and signal > 1):
        return True, "Signal Reversal"
    
    return False, None

def update_dynamic_stop_loss(option_key):
    """
    Update stop loss dynamically based on price action with enhanced algorithm
    that adapts to changing market conditions
    """
    if not trading_state.active_trades.get(option_key, False):
        return
    
    option_info = options_data.get(option_key)
    if not option_info:
        return
    
    current_price = option_info["ltp"]
    if current_price is None or current_price <= 0:
        return
    
    entry_price = trading_state.entry_price[option_key]
    current_stop_loss = trading_state.stop_loss[option_key]
    initial_stop_loss = trading_state.initial_stop_loss[option_key]
    option_type = option_info["option_type"]
    strategy_type = trading_state.strategy_type[option_key]
    
    # Calculate current profit percentage
    if option_type == "CE":
        current_profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        if ENHANCED_TRAILING_SL:
            # Enhanced trailing stop loss with multiple thresholds
            if not trading_state.trailing_sl_activated[option_key] and current_profit_pct >= MIN_PROFIT_FOR_TRAILING:
                # Activate trailing stop loss
                trading_state.trailing_sl_activated[option_key] = True
                
                # Set initial trailing stop loss based on strategy
                if strategy_type == "SCALP":
                    # Tighter stop for scalping
                    new_stop_loss = entry_price + (current_price - entry_price) * 0.3
                elif strategy_type == "MOMENTUM":
                    # Moderate stop for momentum
                    new_stop_loss = entry_price + (current_price - entry_price) * 0.2
                else:  # SWING
                    # Wider stop for swing
                    new_stop_loss = entry_price
                
                trading_state.stop_loss[option_key] = new_stop_loss
                logger.info(f"{option_key} enhanced trailing stop loss activated at {current_profit_pct:.2f}%, SL={new_stop_loss:.2f}")
            
            elif trading_state.trailing_sl_activated[option_key]:
                # Progressive trailing stop loss tightens as profit increases
                for i, threshold in enumerate(TRAILING_STEP_INCREMENTS):
                    if current_profit_pct >= threshold:
                        # Calculate new stop loss based on threshold level
                        sl_percentage = TRAILING_SL_PERCENTAGES[i]
                        potential_sl = current_price * (1 - sl_percentage / 100)
                        
                        # Only move stop loss up
                        if potential_sl > current_stop_loss:
                            trading_state.stop_loss[option_key] = potential_sl
                            logger.info(f"{option_key} trailing SL updated to {potential_sl:.2f} at {threshold}% profit level")
                            break
        else:
            # Original trailing stop loss logic
            if current_profit_pct > 2.0 and not trading_state.trailing_sl_activated[option_key]:
                # Activate trailing stop loss at 2% profit
                trading_state.trailing_sl_activated[option_key] = True
                new_stop_loss = entry_price  # Move stop loss to breakeven
                trading_state.stop_loss[option_key] = new_stop_loss
                logger.info(f"{option_key} trailing stop loss activated, moved to breakeven")
                
            elif trading_state.trailing_sl_activated[option_key]:
                # Continue trailing - only move stop loss up
                trail_price = current_price * 0.99  # 1% below current price
                if trail_price > current_stop_loss:
                    trading_state.stop_loss[option_key] = trail_price
                    logger.info(f"{option_key} trailing stop loss updated to {trail_price}")
    
    else:  # PE
        current_profit_pct = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0
        
        if ENHANCED_TRAILING_SL:
            # Enhanced trailing stop loss with multiple thresholds
            if not trading_state.trailing_sl_activated[option_key] and current_profit_pct >= MIN_PROFIT_FOR_TRAILING:
                # Activate trailing stop loss
                trading_state.trailing_sl_activated[option_key] = True
                
                # Set initial trailing stop loss based on strategy
                if strategy_type == "SCALP":
                    # Tighter stop for scalping
                    new_stop_loss = entry_price - (entry_price - current_price) * 0.3
                elif strategy_type == "MOMENTUM":
                    # Moderate stop for momentum
                    new_stop_loss = entry_price - (entry_price - current_price) * 0.2
                else:  # SWING
                    # Wider stop for swing
                    new_stop_loss = entry_price
                
                trading_state.stop_loss[option_key] = new_stop_loss
                logger.info(f"{option_key} enhanced trailing stop loss activated at {current_profit_pct:.2f}%, SL={new_stop_loss:.2f}")
            
            elif trading_state.trailing_sl_activated[option_key]:
                # Progressive trailing stop loss tightens as profit increases
                for i, threshold in enumerate(TRAILING_STEP_INCREMENTS):
                    if current_profit_pct >= threshold:
                        # Calculate new stop loss based on threshold level
                        sl_percentage = TRAILING_SL_PERCENTAGES[i]
                        potential_sl = current_price * (1 + sl_percentage / 100)
                        
                        # Only move stop loss down
                        if potential_sl < current_stop_loss:
                            trading_state.stop_loss[option_key] = potential_sl
                            logger.info(f"{option_key} trailing SL updated to {potential_sl:.2f} at {threshold}% profit level")
                            break
        else:
            # Original trailing stop loss logic
            if current_profit_pct > 2.0 and not trading_state.trailing_sl_activated[option_key]:
                # Activate trailing stop loss at 2% profit
                trading_state.trailing_sl_activated[option_key] = True
                new_stop_loss = entry_price  # Move stop loss to breakeven
                trading_state.stop_loss[option_key] = new_stop_loss
                logger.info(f"{option_key} trailing stop loss activated, moved to breakeven")
                
            elif trading_state.trailing_sl_activated[option_key]:
                # Continue trailing - only move stop loss down
                trail_price = current_price * 1.01  # 1% above current price
                if trail_price < current_stop_loss:
                    trading_state.stop_loss[option_key] = trail_price
                    logger.info(f"{option_key} trailing stop loss updated to {trail_price}")

def update_dynamic_target(option_key):
    """
    Update target price dynamically based on market conditions and price momentum
    to maximize profit potential
    """
    if not DYNAMIC_TARGET_ADJUSTMENT:
        return
    
    if not trading_state.active_trades.get(option_key, False):
        return
    
    option_info = options_data.get(option_key)
    if not option_info:
        return
    
    current_price = option_info["ltp"]
    if current_price is None or current_price <= 0:
        return
    
    entry_price = trading_state.entry_price[option_key]
    current_target = trading_state.target[option_key]
    option_type = option_info["option_type"]
    parent_symbol = option_info["parent_symbol"]
    strategy_type = trading_state.strategy_type[option_key]
    
    try:
        # Calculate current momentum factors
        price_history = option_info["price_history"]
        if len(price_history) < 10:
            return
        
        price_series = price_history['price']
        current_rsi = calculate_rsi(price_series, 7)  # Use shorter RSI period for sensitivity
        current_momentum = calculate_momentum(price_series, 3)  # Short-term momentum
        
        # Calculate parent stock momentum if available
        parent_momentum = 0
        if parent_symbol in stocks_data:
            parent_price_history = stocks_data[parent_symbol]["price_history"]
            if len(parent_price_history) >= 10:
                parent_price_series = parent_price_history['price']
                parent_momentum = calculate_momentum(parent_price_series, 5)
        
        # Determine adjustment factor based on strategy and market conditions
        adjustment_factor = 0
        
        # Base adjustment on strategy
        if strategy_type == "SCALP":
            # Scalp trades should have quick, smaller targets
            base_adjustment = 0.3
        elif strategy_type == "MOMENTUM":
            # Momentum trades can have more aggressive targets
            base_adjustment = 0.7
        else:  # SWING
            # Swing trades need balanced targets
            base_adjustment = 0.5
        
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
        momentum_factor = abs(current_momentum) * TARGET_MOMENTUM_FACTOR
        
        # Consider parent stock momentum (with lower weight)
        parent_factor = abs(parent_momentum) * 0.05
        
        # Calculate total adjustment factor
        adjustment_factor = base_adjustment + rsi_factor + momentum_factor + parent_factor
        
        # Limit adjustment within reasonable bounds
        adjustment_factor = max(MIN_TARGET_ADJUSTMENT, min(adjustment_factor, MAX_TARGET_ADJUSTMENT))
        
        # Calculate new target
        if option_type == "CE":
            # For call options, target is above entry price
            target_price_diff = (current_price - entry_price) * adjustment_factor
            new_target = current_price + target_price_diff
            
            # Only adjust target upward
            if new_target > current_target:
                trading_state.target[option_key] = new_target
                logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
        else:  # PE
            # For put options, target is below entry price
            target_price_diff = (entry_price - current_price) * adjustment_factor
            new_target = current_price - target_price_diff
            
            # Only adjust target downward
            if new_target < current_target:
                trading_state.target[option_key] = new_target
                logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
                
    except Exception as e:
        logger.error(f"Error updating dynamic target for {option_key}: {e}")

def check_time_based_exit(option_key):
    """Check if we should exit a trade based on time held."""
    if not trading_state.active_trades.get(option_key, False):
        return False
    
    entry_time = trading_state.entry_time[option_key]
    strategy_type = trading_state.strategy_type[option_key]
    current_time = datetime.now()
    
    # Calculate time held in minutes
    time_held = (current_time - entry_time).total_seconds() / 60 if entry_time else 0
    
    # Maximum time depends on strategy
    if strategy_type == "SCALP":
        max_time = MAX_POSITION_HOLDING_TIME_SCALP
    elif strategy_type == "MOMENTUM":
        max_time = MAX_POSITION_HOLDING_TIME_MOMENTUM
    else:  # SWING
        max_time = MAX_POSITION_HOLDING_TIME_SWING
    
    # Check if we've held for the maximum time
    if time_held >= max_time:
        logger.info(f"{option_key} maximum holding time reached ({max_time} min). Exiting.")
        exit_trade(option_key, reason="Time-based exit")
        return True
    
    return False

def exit_trade(option_key, reason="Manual"):
    """Exit a trade for the given option."""
    if not trading_state.active_trades.get(option_key, False):
        return False
    
    option_info = options_data.get(option_key)
    if not option_info:
        logger.warning(f"Cannot exit trade: Option {option_key} not found")
        return False
    
    current_price = option_info["ltp"]
    if current_price is None or current_price <= 0:
        logger.warning(f"Cannot exit trade: Invalid price {current_price}")
        return False
    
    # Calculate P&L
    entry_price = trading_state.entry_price[option_key]
    quantity = trading_state.quantity[option_key]
    pnl = (current_price - entry_price) * quantity
    pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    
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
    
    # Add to trade history
    trade_record = {
        'option_key': option_key,
        'strategy_type': trading_state.strategy_type[option_key],
        'entry_time': trading_state.entry_time[option_key],
        'exit_time': datetime.now(),
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
    
    logger.info(f"Exited {option_key} trade: Price={current_price}, P&L={pnl}, P&L%={pnl_pct:.2f}%, Reason={reason}")
    return True

def apply_trading_strategy():
    """Apply the trading strategy by checking for entry and exit conditions."""
    # Skip if all strategies are disabled or broker not connected
    if (not strategy_settings["SCALP_ENABLED"] and
        not strategy_settings["SWING_ENABLED"] and
        not strategy_settings["MOMENTUM_ENABLED"]) or not broker_connected:
        return
    
    # Check for trade exits first
    for option_key in list(trading_state.active_trades.keys()):
        if trading_state.active_trades.get(option_key, False):
            should_exit, reason = should_exit_trade(option_key)
            if should_exit:
                exit_trade(option_key, reason=reason)
            else:
                check_time_based_exit(option_key)
                update_dynamic_stop_loss(option_key)
                update_dynamic_target(option_key)
    
    # Check for trade entries
    for option_key in options_data.keys():
        if not trading_state.active_trades.get(option_key, False):
            should_enter, strategy_type = should_enter_trade(option_key)
            if should_enter and strategy_type:
                enter_trade(option_key, strategy_type)

# ============ Data Fetching ============
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

# Function to search for symbol tokens
@rate_limited
def search_symbols(search_text, exchange=None):
    """
    Enhanced symbol search with exact matching prioritized
    
    Args:
        search_text (str): Text to search for
        exchange (str, optional): Exchange to search in. Defaults to None.
    
    Returns:
        list: List of matching symbols with exact matches first
    """
    global smart_api, broker_connected, config
    
    if not search_text:
        logger.warning("Empty search text provided")
        return []
    
    # Make search text uppercase
    search_text = search_text.strip().upper()
    logger.info(f"Searching for symbol: '{search_text}'")
    
    # Connect to broker if not already connected
    if not broker_connected or smart_api is None:
        if not connect_to_broker():
            logger.warning("Cannot search symbols: Not connected to broker")
            return []
    
    # Use specified exchange or default from config
    target_exchange = exchange or config.exchange
    
    try:
        # Perform the search
        search_resp = smart_api.searchScrip(exchange=target_exchange, searchtext=search_text)
        
        if isinstance(search_resp, dict) and search_resp.get("status"):
            # Extract matching symbols
            matches = search_resp.get("data", [])
            
            if matches:
                logger.info(f"Found {len(matches)} potential matches for '{search_text}'")
                
                # Separate exact and partial matches
                exact_matches = []
                partial_matches = []
                
                for match in matches:
                    match_symbol = match.get("symbol", "").upper()
                    match_name = match.get("name", "").upper()
                    
                    # Check for exact match
                    if match_symbol == search_text or match_name == search_text:
                        exact_matches.append(match)
                    else:
                        partial_matches.append(match)
                
                # Return exact matches first, then partial matches
                return exact_matches + partial_matches
            else:
                logger.debug(f"No matches found for '{search_text}'")
        else:
            error_msg = search_resp.get("message", "Unknown error") if isinstance(search_resp, dict) else "Invalid response"
            logger.warning(f"Symbol search failed for '{search_text}': {error_msg}")
    
    except Exception as e:
        logger.error(f"Error searching for '{search_text}': {e}")
    
    # If all attempts failed
    logger.warning(f"All search attempts failed for '{search_text}'")
    return []

# Function to fetch data in bulk (up to 50 symbols at once)
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
            
            # Short delay between requests to avoid rate limiting
            time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    return results

def update_all_stocks():
    """Update all stocks using bulk API where possible."""
    global stocks_data
    
    # Skip if not connected to broker
    if not broker_connected:
        return
    
    # Get all symbols to update
    symbols_to_update = list(stocks_data.keys())
    
    # Process in batches of 50 to respect the new API limits
    for i in range(0, len(symbols_to_update), BULK_FETCH_SIZE):
        batch = symbols_to_update[i:i+BULK_FETCH_SIZE]
        
        # Fetch data for this batch
        bulk_results = fetch_bulk_stock_data(batch)
        
        # Update each stock with the fetched data
        for symbol, data in bulk_results.items():
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
                    'open': stock_info.get("open", data.get("ltp", 0)),
                    'high': stock_info.get("high", data.get("ltp", 0)),
                    'low': stock_info.get("low", data.get("ltp", 0))
                }
                
                # Append to price history DataFrame with proper index handling
                with price_history_lock:
                    stock_info["price_history"] = pd.concat([
                        stock_info["price_history"], 
                        pd.DataFrame([new_data])
                    ], ignore_index=True)
                    
                    # Limit price history size
                    if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                        stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
                
                # Update volatility
                if previous_ltp is not None and previous_ltp > 0:
                    pct_change = (data.get("ltp", 0) - previous_ltp) / previous_ltp * 100
                    update_volatility(symbol, pct_change)
                
                # Update support/resistance levels
                calculate_support_resistance(symbol)
                
                # Predict best strategy for this stock
                predict_strategy_for_stock(symbol)
                
                # Update last update time
                stock_info["last_updated"] = datetime.now()
                last_data_update["stocks"][symbol] = datetime.now()
                
                # Update UI data store
                ui_data_store['stocks'][symbol] = {
                    'price': data.get("ltp"),
                    'change': stock_info["change_percent"],
                    'ohlc': {
                        'open': stock_info["open"],
                        'high': stock_info["high"],
                        'low': stock_info["low"],
                        'previous': stock_info["previous"]
                    },
                    'last_updated': stock_info["last_updated"].strftime('%H:%M:%S')
                }
        
        # Short delay between batch requests to avoid overwhelming the API
        time.sleep(0.2)

def cleanup_old_data():
    """Clean up old data to prevent memory bloat."""
    global last_cleanup, stocks_data, options_data
    
    current_time = datetime.now()
    if (current_time - last_cleanup).total_seconds() < DATA_CLEANUP_INTERVAL:
        return
    
    logger.info("Running data cleanup...")
    
    # Keep only the latest MAX_PRICE_HISTORY_POINTS data points for stocks
    for symbol, stock_info in stocks_data.items():
        if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
            stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
    
    # Keep only the latest MAX_PRICE_HISTORY_POINTS data points for options
    for option_key, option_info in options_data.items():
        if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
            option_info["price_history"] = option_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
    
    last_cleanup = current_time
    logger.info("Data cleanup completed")

def check_day_rollover():
    """Check if trading day has changed and reset daily stats."""
    global trading_state
    
    current_date = datetime.now().date()
    if current_date != trading_state.trading_day:
        logger.info(f"New trading day detected: {current_date}. Resetting daily stats.")
        trading_state.trading_day = current_date
        trading_state.trades_today = 0
        trading_state.daily_pnl = 0

def should_remove_stock(symbol):
    """
    Determine if a stock should be automatically removed.
    """
    if symbol not in stocks_data:
        return True
        
    stock_info = stocks_data[symbol]
    current_time = datetime.now()
    
    # Skip default/index stocks from automatic removal
    if symbol in [s["symbol"] for s in DEFAULT_STOCKS]:
        return False
    
    # Check if the stock hasn't been updated in 15 minutes
    last_update = stock_info.get("last_updated")
    if last_update is None or (current_time - last_update).total_seconds() > 900:  # 15 minutes
        logger.warning(f"Stock {symbol} has not been updated for >15 minutes")
        return True
    
    # Only remove stocks with completely invalid price data
    if stock_info.get("ltp") is None:
        logger.warning(f"Stock {symbol} has no price data")
        return True
    
    return False

def cleanup_inactive_stocks():
    """
    Remove stocks that are inactive or have invalid data.
    """
    for symbol in list(stocks_data.keys()):
        if should_remove_stock(symbol):
            remove_stock(symbol)

def fetch_data_periodically():
    """Main function to fetch data periodically with smart retry logic."""
    global dashboard_initialized, data_thread_started, broker_error_message
    
    # Mark that the data thread has started
    data_thread_started = True
    logger.info("Data fetching thread started")
    
    # Load script master data file at startup
    load_script_master()
    
    # Initialize with default stocks
    for stock in DEFAULT_STOCKS:
        add_stock(stock["symbol"], stock["token"], stock["exchange"], stock["type"])
        
        # For NIFTY and BANKNIFTY, load historical data from CSV files
        if stock["symbol"] in ["NIFTY", "BANKNIFTY"]:
            load_historical_data(stock["symbol"])
    
    # Mark dashboard as initialized
    dashboard_initialized = True
    
    last_cleanup_check = datetime.now()
    cleanup_interval = 300  # Check every 5 minutes
    last_connection_attempt = datetime.now() - timedelta(minutes=10)  # Start with immediate connection attempt
    connection_retry_interval = 30  # Seconds between connection retries initially
    max_connection_retry_interval = 600  # Max 10 minutes between retries
    
    # Main data fetching loop
    while True:
        try:
            current_time = datetime.now()
            
            # Smart broker connection logic with increasing backoff
            if not broker_connected and (current_time - last_connection_attempt).total_seconds() >= connection_retry_interval:
                if broker_error_message and "Account blocked" in broker_error_message:
                    # For account blocks, use a much longer retry interval (12 hours)
                    connection_retry_interval = 43200  # 12 hours
                
                logger.info(f"Attempting broker connection after {connection_retry_interval} seconds...")
                connect_success = connect_to_broker()
                last_connection_attempt = current_time
                
                if connect_success:
                    # Reset retry interval upon success
                    connection_retry_interval = 30
                else:
                    # Increase retry interval up to the maximum
                    connection_retry_interval = min(connection_retry_interval * 2, max_connection_retry_interval)
            
            # Try to use broker data if connected
            if broker_connected:
               # Refresh session if needed
                refresh_session_if_needed()
                
                # Update all stocks using bulk API
                update_all_stocks()
                
                # Update option selection
                update_option_selection()
                
                # Update all options using bulk API
                update_all_options()
                
                # Update PCR data
                update_all_pcr_data()
                
                # Apply trading strategy
                apply_trading_strategy()
            
            # Always do these maintenance tasks, even without broker connection
            # Cleanup check
            if (current_time - last_cleanup_check).total_seconds() >= cleanup_interval:
                cleanup_inactive_stocks()
                cleanup_old_data()
                last_cleanup_check = current_time
            
            # Check for day rollover
            check_day_rollover()
            
            # Update UI data store with up-to-date broker connection status
            ui_data_store['connection'] = {
                'status': 'connected' if broker_connected else 'disconnected',
                'message': broker_error_message or '',
                'last_updated': datetime.now().strftime('%H:%M:%S'),
                'last_connection': last_connection_time.strftime('%H:%M:%S') if last_connection_time else 'Never'
            }
            
            # Wait before next update
            time.sleep(API_UPDATE_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in fetch_data_periodically: {e}", exc_info=True)
            time.sleep(1)

# ============ Dashboard UI ============
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

# Use a modern dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY], 
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Options Trading Dashboard"

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
                        dbc.Input(
                            id="add-stock-symbol", 
                            placeholder="Symbol (e.g., NIFTY, RELIANCE)", 
                            type="text",
                            className="border-0 bg-dark text-light"
                        ),
                    ], width=5),
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
                    ], width=4),
                ]),
                html.Div(id="add-stock-message", className="mt-2")
            ], className="px-4 py-3")
        ], 
        style=custom_css["card"],
        className="mb-3 border-warning"
    )

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
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H4(symbol, className="text-info d-inline me-1 mb-0"),
                html.Span(f"({stock_type})", className="text-muted small me-2"),
                html.Span(id={"type": "data-source-badge", "index": symbol}, className="badge ms-2"),
                strategy_badge,
                history_badge,  # ADD THIS HERE - Add the history badge
                dbc.Button("×", id={"type": "remove-stock-btn", "index": symbol}, 
                         color="danger", size="sm", className="float-end"),
                dbc.Button("Fetch History", id={"type": "fetch-history-btn", "index": symbol},  # ADD THIS HERE - Add the Fetch History button
                         color="primary", size="sm", className="float-end me-2")
            ])
        ], style=custom_css["header"]),

        dbc.CardBody([
            # Stock Information Row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Price: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-price", "index": symbol}, className="fs-4 text-light")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Change: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-change", "index": symbol}, className="text-light")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("OHLC: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-ohlc", "index": symbol}, className="text-light small")
                    ], className="mb-1"),
                ], width=4),
                
                dbc.Col([
                    html.Div([
                        html.Span("PCR: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-pcr", "index": symbol}, className="text-light"),
                        html.Span(id={"type": "pcr-strength", "index": symbol}, className="ms-2 small")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Sentiment: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-sentiment", "index": symbol}, className="badge")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("S/R Levels: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-sr-levels", "index": symbol}, className="text-light small")
                    ], className="mb-1"),
                ], width=4),
                
                dbc.Col([
                    html.Div(id={"type": "stock-last-update", "index": symbol}, className="text-muted small text-end"),
                    html.Div(id={"type": "strategy-prediction", "index": symbol}, className="text-end small mt-2")
                ], width=4)
            ], className="mb-3"),
            
            # Options Section - CE and PE side by side
            dbc.Row([
                # Call option
                dbc.Col([
                    html.H6("CALL OPTION", className="text-success mb-2 text-center"),
                    html.Div([
                        html.Span("Strike: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-strike", "index": symbol}, className="text-warning")
                    ], className="text-center mb-1"),
                    html.Div([
                        html.Span("LTP: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-price", "index": symbol}, className="fs-5 fw-bold")
                    ], className="text-center mb-2"),
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-signal", "index": symbol}, className="badge")
                    ], className="text-center mb-1"),
                    dbc.Progress(id={"type": "option-ce-strength", "index": symbol}, className="mb-2", style={"height": "6px"}),
                    html.Div(id={"type": "option-ce-trade-status", "index": symbol}, className="text-center small")
                ], width=6, className="border-end"),
                
                # Put option
                dbc.Col([
                    html.H6("PUT OPTION", className="text-danger mb-2 text-center"),
                    html.Div([
                        html.Span("Strike: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-strike", "index": symbol}, className="text-warning")
                    ], className="text-center mb-1"),
                    html.Div([
                        html.Span("LTP: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-price", "index": symbol}, className="fs-5 fw-bold")
                    ], className="text-center mb-2"),
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-signal", "index": symbol}, className="badge")
                    ], className="text-center mb-1"),
                    dbc.Progress(id={"type": "option-pe-strength", "index": symbol}, className="mb-2", style={"height": "6px"}),
                    html.Div(id={"type": "option-pe-trade-status", "index": symbol}, className="text-center small")
                ], width=6)
            ])
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-info"
    )

def create_strategy_settings_card():
    return dbc.Card([
        dbc.CardHeader(html.H4("Strategy Settings", className="text-warning mb-0"), 
                      style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="scalp-strategy-toggle",
                        label="SCALP Strategy",
                        value=strategy_settings["SCALP_ENABLED"],
                        className="mb-2"
                    ),
                    html.Div([
                        dbc.Progress(
                            id="scalp-progress", 
                            value=0, 
                            color="success", 
                            className="mb-1", 
                            style={"height": "5px"}
                        ),
                        html.Span(f"Signal Threshold: {MIN_SIGNAL_STRENGTH_SCALP}", className="text-light small")
                    ], className="mb-1")
                ], width=4),
                dbc.Col([
                    dbc.Switch(
                        id="swing-strategy-toggle",
                        label="SWING Strategy",
                        value=strategy_settings["SWING_ENABLED"],
                        className="mb-2"
                    ),
                    html.Div([
                        dbc.Progress(
                            id="swing-progress", 
                            value=0, 
                            color="primary", 
                            className="mb-1", 
                            style={"height": "5px"}
                        ),
                        html.Span(f"Signal Threshold: {MIN_SIGNAL_STRENGTH_SWING}", className="text-light small")
                    ], className="mb-1")
                ], width=4),
                dbc.Col([
                    dbc.Switch(
                        id="momentum-strategy-toggle",
                        label="MOMENTUM Strategy",
                        value=strategy_settings["MOMENTUM_ENABLED"],
                        className="mb-2"
                    ),
                    html.Div([
                        dbc.Progress(
                            id="momentum-progress", 
                            value=0, 
                            color="info", 
                            className="mb-1", 
                            style={"height": "5px"}
                        ),
                        html.Span(f"Signal Threshold: {MIN_SIGNAL_STRENGTH_MOMENTUM}", className="text-light small")
                    ], className="mb-1")
                ], width=4),
            ]),
            html.Div(id="strategy-status", className="text-center mt-2 text-info")
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-warning"
    )
   

def diagnose_sr_calculation(symbol):
    """Diagnostic function to check S/R calculation for a given symbol."""
    try:
        if symbol not in stocks_data:
            return f"Symbol {symbol} not found in stocks_data"
        
        stock_info = stocks_data[symbol]
        df = stock_info["price_history"]
        
        results = [
            f"Symbol: {symbol}",
            f"Data points: {len(df)}",
            f"Columns: {df.columns.tolist()}",
            f"Price column exists: {'price' in df.columns}",
            f"NaN values in price: {df['price'].isna().sum() if 'price' in df.columns else 'N/A'}",
            f"Current S/R values:",
            f"  Support: {stock_info.get('support_levels', [])}",
            f"  Resistance: {stock_info.get('resistance_levels', [])}"
        ]
        
        # Log this info for debugging
        logger.info("\n".join(results))
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error in diagnosis: {e}")
        return f"Error in diagnosis: {e}"
def create_market_sentiment_card():
    return dbc.Card([
        dbc.CardHeader(html.H4("Market Sentiment", className="text-warning mb-0"), 
                      style=custom_css["header"]),
        dbc.CardBody([
            html.Div([
                html.Span("Overall: ", className="text-muted me-2"),
                html.Span(id="overall-sentiment", className="badge fs-5 mb-1")
            ], className="mb-2 text-center"),
            html.Div([
                dbc.Progress(
                    id="bullish-strength", 
                    value=50, 
                    color="success", 
                    className="mb-2",
                    style={"height": "8px", "border-radius": "4px"}
                )
            ], className="mb-1"),
            html.Div([
                html.Span("Bulls ", className="text-success me-1"),
                html.Span(id="bull-bear-ratio", className="text-light me-1"),
                html.Span(" Bears", className="text-danger")
            ], className="text-center")
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-warning h-100"
    )

def create_performance_card():
    return dbc.Card([
        dbc.CardHeader(html.H4("Performance", className="text-warning mb-0"), 
                      style=custom_css["header"]),
        dbc.CardBody([
            html.Div([
                html.Span("Total P&L: ", className="text-muted me-2"),
                html.Span(id="total-pnl", className="fs-5")
            ], className="mb-2"),
            html.Div([
                html.Span("Today's P&L: ", className="text-muted me-2"),
                html.Span(id="daily-pnl", className="fs-5")
            ], className="mb-2"),
            html.Div([
                html.Span("Win Rate: ", className="text-muted me-2"),
                html.Span(id="win-rate", className="text-light")
            ], className="mb-2"),
            html.Div([
                html.Span("Trades Today: ", className="text-muted me-2"),
                html.Span(id="trades-today", className="text-light")
            ])
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-warning h-100"
    )

def create_active_trades_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Active Trades", className="text-success d-inline me-2 mb-0"),
            html.Span(id="active-trades-count", className="badge bg-success ms-2")
        ], style=custom_css["header"]),
        dbc.CardBody(
            id="active-trades-container", 
            className="px-4 py-3",
            style={"max-height": "320px", "overflow-y": "auto"}
        )
    ], 
    style=custom_css["card"],
    className="mb-3 border-success"
    )

def create_pnl_history_card():
    return dbc.Card([
        dbc.CardHeader(html.H4("P&L History", className="text-info mb-0"), 
                      style=custom_css["header"]),
        dbc.CardBody([
            dcc.Graph(
                id="pnl-graph", 
                config={'displayModeBar': False}, 
                style={"height": "240px"},
                className="mt-2"
            )
        ], className="px-2 py-2")
    ], 
    style=custom_css["card"],
    className="mb-3 border-info"
    )

def create_recent_trades_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Recent Trades", className="text-info d-inline me-2 mb-0"),
            html.Span(id="recent-trades-count", className="badge bg-info ms-2")
        ], style=custom_css["header"]),
        dbc.CardBody(
            id="recent-trades-container", 
            style={"max-height": "240px", "overflow-y": "auto"},
            className="px-4 py-3"
        )
    ], 
    style=custom_css["card"],
    className="mb-3 border-info"
    )

def create_connection_status_card():
    return dbc.Card([
        dbc.CardHeader(html.H4("Connection Status", className="text-primary mb-0"), 
                      style=custom_css["header"]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Broker: ", className="text-muted me-2"),
                        html.Span(id="broker-status", className="badge"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Last Connected: ", className="text-muted me-2"),
                        html.Span(id="last-connection-time", className="text-light"),
                    ], className="mb-2"),
                ], width=6),
                dbc.Col([
                    html.Div(id="connection-details", className="mt-1 small"),
                    html.Div([
                        dbc.Button(
                            "Retry Connection", 
                            id="retry-connection-button", 
                            color="primary", 
                            size="sm",
                            className="mt-2"
                        ),
                    ]),
                ], width=6),
            ]),
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-primary"
    )

# Create layout function
def create_layout():
    return dbc.Container(
        [
            create_notification_container(),
            
            # Header with logo and title
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1("Options Trading Dashboard", className="text-center mb-3 text-light"),
                        html.P("Real-time market data and automated trading signals", 
                            className="text-center text-muted mb-2")
                    ], className="py-3")
                ])
            ]),
            
            # Connection status card
            dbc.Row([
                dbc.Col([create_connection_status_card()], width=12)
            ]),
            
            # Controls Row - Strategy Settings and Add Stock
            dbc.Row([
                dbc.Col([create_strategy_settings_card()], width=8),
                dbc.Col([create_stock_control_card()], width=4)
            ]),
            
            # Stock cards container - Will be dynamically populated
            html.Div(id="stock-cards-container"),
            
            # Market Summary and Performance Row
            dbc.Row([
                dbc.Col([create_market_sentiment_card()], width=6),
                dbc.Col([create_performance_card()], width=6)
            ]),
            
            # Active trades
            dbc.Row([
                dbc.Col([create_active_trades_card()], width=12),
            ]),
            
            # Trade history graph and recents
            dbc.Row([
                dbc.Col([create_pnl_history_card()], width=8),
                dbc.Col([create_recent_trades_card()], width=4)
            ]),
            
            # Store for data that needs to be shared between callbacks
            dcc.Store(
                id='ui-data-store',
                storage_type='memory'
            ),
            
            # Refresh intervals - more efficient refresh rates
            dcc.Interval(
                id='fast-interval',
                interval=1000,  # 1 second for time-sensitive data
                n_intervals=0
            ),
            dcc.Interval(
                id='medium-interval',
                interval=5000,  # 5 seconds for regular updates
                n_intervals=0
            ),
            dcc.Interval(
                id='slow-interval',
                interval=30000,  # 30 seconds for non-critical updates
                n_intervals=0
            ),
        ],
        fluid=True,
        className="p-4",
        style={"background-color": COLOR_SCHEME["bg_dark"], "min-height": "100vh"}
    )

# Main application layout with improved UI
app.layout = create_layout()

# ============ Dashboard Callbacks ============
# Update UI Data Store to centralize data processing
@app.callback(
    Output('ui-data-store', 'data'),
    [Input('fast-interval', 'n_intervals')]
)
def update_ui_data_store(n_intervals):
    """Centralized data store updates to avoid redundant processing"""
    
    # Copy the current state for UI updates
    result = {
        'connection': {
            'status': 'connected' if broker_connected else 'disconnected',
            'message': broker_error_message or '',
            'last_connection': last_connection_time.strftime('%H:%M:%S') if last_connection_time else 'Never'
        },
        'stocks': {},
        'options': {},
        'pcr': {},
        'sentiment': market_sentiment.copy(),
        'predicted_strategies': ui_data_store.get('predicted_strategies', {}),
        'trading': {
            'active_trades': sum(1 for v in trading_state.active_trades.values() if v),
            'total_pnl': trading_state.total_pnl,
            'daily_pnl': trading_state.daily_pnl,
            'trades_today': trading_state.trades_today,
            'wins': trading_state.wins,
            'losses': trading_state.losses
        },
        'strategies': strategy_settings.copy()
    }
    
    # Add stock data
    for symbol, data in ui_data_store.get('stocks', {}).items():
        if symbol in stocks_data:
            result['stocks'][symbol] = data
    
    # Add options data
    for symbol, data in ui_data_store.get('options', {}).items():
        if symbol in stocks_data:
            result['options'][symbol] = data
    
    # Add PCR data
    for symbol, data in pcr_data.items():
        if symbol in stocks_data:
            result['pcr'][symbol] = {
                'current': data['current'],
                'trend': data['trend'],
                'strength': data['strength']
            }
    
    return result

# Connection status indicator callback
@app.callback(
    [Output("broker-status", "children"),
     Output("broker-status", "className"),
     Output("last-connection-time", "children"),
     Output("connection-details", "children")],
    [Input('ui-data-store', 'data')]
)
def update_connection_status(data):
    if not data or 'connection' not in data:
        return "Unknown", "badge bg-secondary", "Never", "No connection data"
    
    connection_data = data['connection']
    
    # Broker connection status
    if connection_data['status'] == 'connected':
        broker_status = "Connected"
        broker_class = "badge bg-success"
    else:
        if "blocked" in connection_data.get('message', '').lower():
            broker_status = "Blocked"
            broker_class = "badge bg-danger"
        else:
            broker_status = "Disconnected"
            broker_class = "badge bg-warning text-dark"
    
    # Connection details
    if connection_data['status'] == 'connected':
        details = ["Connected to broker API", html.Br(), "Using real-time market data"]
    else:
        details = []
        if connection_data.get('message'):
            details.extend([
                html.Span("Error: ", className="text-danger"),
                html.Span(connection_data['message']),
                html.Br()
            ])
        
        if broker_connection_retry_time:
            time_diff = (broker_connection_retry_time - datetime.now()).total_seconds()
            if time_diff > 0:
                minutes = int(time_diff // 60)
                seconds = int(time_diff % 60)
                details.extend([
                    html.Span("Next retry in: ", className="text-muted"),
                    html.Span(f"{minutes}m {seconds}s")
                ])
    
    return broker_status, broker_class, connection_data['last_connection'], details

# Retry connection button callback
@app.callback(
    [Output("retry-connection-button", "disabled"),
     Output("notification-toast", "is_open", allow_duplicate=True),
     Output("notification-toast", "header", allow_duplicate=True),
     Output("notification-toast", "children", allow_duplicate=True),
     Output("notification-toast", "icon", allow_duplicate=True)],
    [Input("retry-connection-button", "n_clicks")],
    prevent_initial_call=True
)
def handle_retry_connection(n_clicks):
    if not n_clicks:
        return False, False, "", "", "primary"
    
    # Reset retry time to force immediate connection attempt
    global broker_connection_retry_time
    broker_connection_retry_time = None
    
    # Attempt to connect
    success = connect_to_broker()
    
    if success:
        return False, True, "Connection Success", "Successfully connected to broker", "success"
    else:
        return False, True, "Connection Failed", broker_error_message or "Failed to connect to broker", "danger"

# Dynamic stock cards container callback
@app.callback(
    Output("stock-cards-container", "children"),
    [Input('medium-interval', 'n_intervals'),
     Input("add-stock-button", "n_clicks"),
     Input({"type": "remove-stock-btn", "index": ALL}, "n_clicks")]
)
def update_stock_cards(n_intervals, add_clicks, remove_clicks):
    # Create stock cards dynamically based on stocks_data
    stock_cards = []
    
    # Create a row of 2 cards per row
    symbols = list(stocks_data.keys())
    
    # Always ensure at least one placeholder row if all stocks are removed
    if not symbols:
        return [html.Div("No stocks added. Use the Add Stock form above to add stocks or indices.", 
                       className="text-center text-muted my-5")]
    
    # Ensure we have an even number of cards by adding a placeholder if needed
    if len(symbols) % 2 == 1:
        symbols_to_display = symbols.copy()
    else:
        symbols_to_display = symbols
    
    # Create rows of cards
    for i in range(0, len(symbols_to_display), 2):
        row_cards = []
        for j in range(2):
            idx = i + j
            if idx < len(symbols):
                symbol = symbols[idx]
                row_cards.append(dbc.Col([create_stock_option_card(symbol)], width=6))
            else:
                # Add an empty placeholder card when needed to maintain layout
                row_cards.append(dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Add a Stock", className="text-center text-muted mb-0")
                        ], style=custom_css["header"]),
                        dbc.CardBody([
                            html.Div("Use the Add Stock form above to add more stocks or indices.", 
                                   className="text-center text-muted")
                        ], className="px-4 py-3")
                    ], 
                    style=dict(custom_css["card"], **{"opacity": "0.5"}),
                    className="mb-3 border-secondary")
                ], width=6))
        
        # Add this row to the container
        stock_cards.append(dbc.Row(row_cards))
    
    return stock_cards

# Add Stock Callback
@app.callback(
    [Output("add-stock-message", "children"),
     Output("add-stock-message", "className"),
     Output("add-stock-symbol", "value")],
    [Input("add-stock-button", "n_clicks")],
    [State("add-stock-symbol", "value"),
     State("add-stock-type", "value")],
    prevent_initial_call=True
)
def add_stock_callback(n_clicks, symbol, stock_type):
    if not symbol:
        return "Please enter a symbol", "text-danger mt-2", ""
    
    # Standardize symbol to uppercase
    symbol = symbol.upper()
    
    # Check if stock already exists
    if symbol in stocks_data:
        return f"{symbol} is already being tracked", "text-warning mt-2", ""
    
    # Add stock
    success = add_stock(symbol, None, "NSE", stock_type)
    
    if success:
        return f"Added {symbol} successfully", "text-success mt-2", ""
    else:
        return f"Failed to add {symbol}", "text-danger mt-2", symbol

# Remove Stock Callback
@app.callback(
    [Output("notification-toast", "is_open"),
     Output("notification-toast", "header"),
     Output("notification-toast", "children"),
     Output("notification-toast", "icon")],
    [Input({"type": "remove-stock-btn", "index": ALL}, "n_clicks")],
    [State({"type": "remove-stock-btn", "index": ALL}, "id")],
    prevent_initial_call=True
)
def remove_stock_callback(n_clicks, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, "", "", "primary"
    
    # Find which button was clicked
    btn_idx = -1
    for i, clicks in enumerate(n_clicks):
        if clicks is not None and clicks > 0:
            btn_idx = i
            break
    
    if btn_idx == -1:
        return False, "", "", "primary"
            
    try:
        # Get the symbol to remove
        symbol = ids[btn_idx]["index"]
        
        # Remove the stock
        success = remove_stock(symbol)
        
        if success:
            # Force a refresh of the interval component to update the UI
            return True, "Stock Removed", f"Successfully removed {symbol}", "success"
        else:
            return True, "Error", f"Failed to remove {symbol}", "danger"
    except Exception as e:
        logger.error(f"Error removing stock: {str(e)}")
        return True, "Error", f"Failed to remove stock: {str(e)}", "danger"

# Strategy Settings Callback
@app.callback(
    [Output("strategy-status", "children"),
     Output("scalp-progress", "value"),
     Output("swing-progress", "value"),
     Output("momentum-progress", "value")],
    [Input("scalp-strategy-toggle", "value"),
     Input("swing-strategy-toggle", "value"),
     Input("momentum-strategy-toggle", "value"),
     Input('ui-data-store', 'data')]
)
def update_strategy_settings(scalp_enabled, swing_enabled, momentum_enabled, data):
    global strategy_settings
    strategy_settings["SCALP_ENABLED"] = scalp_enabled
    strategy_settings["SWING_ENABLED"] = swing_enabled
    strategy_settings["MOMENTUM_ENABLED"] = momentum_enabled
    
    enabled_strategies = []
    if scalp_enabled:
        enabled_strategies.append("SCALP")
    if swing_enabled:
        enabled_strategies.append("SWING")
    if momentum_enabled:
        enabled_strategies.append("MOMENTUM")
    
    # Calculate strategy execution percentages
    scalp_progress = 0
    swing_progress = 0
    momentum_progress = 0
    
    if data and 'predicted_strategies' in data:
        strategy_counts = {"SCALP": 0, "SWING": 0, "MOMENTUM": 0, "NONE": 0}
        total_stocks = 0
        
        # Count strategy predictions
        for symbol, strategy_data in data['predicted_strategies'].items():
            if strategy_data and 'strategy' in strategy_data:
                strategy = strategy_data['strategy']
                if strategy in strategy_counts:
                    strategy_counts[strategy] += 1
                total_stocks += 1
        
        # Calculate percentages
        if total_stocks > 0:
            scalp_progress = (strategy_counts["SCALP"] / total_stocks) * 100
            swing_progress = (strategy_counts["SWING"] / total_stocks) * 100
            momentum_progress = (strategy_counts["MOMENTUM"] / total_stocks) * 100
    
    if enabled_strategies:
        return [f"Enabled: {', '.join(enabled_strategies)}"], scalp_progress, swing_progress, momentum_progress
    else:
        return ["No trading strategies enabled - Trading paused"], 0, 0, 0

# Update Stock Display Callback
@app.callback(
    [
        Output({"type": "data-source-badge", "index": ALL}, "children"),
        Output({"type": "data-source-badge", "index": ALL}, "className"),
        Output({"type": "stock-price", "index": ALL}, "children"),
        Output({"type": "stock-change", "index": ALL}, "children"),
        Output({"type": "stock-change", "index": ALL}, "className"),
        Output({"type": "stock-ohlc", "index": ALL}, "children"),
        Output({"type": "stock-pcr", "index": ALL}, "children"),
        Output({"type": "pcr-strength", "index": ALL}, "children"),
        Output({"type": "stock-sentiment", "index": ALL}, "children"),
        Output({"type": "stock-sentiment", "index": ALL}, "className"),
        Output({"type": "stock-sr-levels", "index": ALL}, "children"),
        Output({"type": "stock-last-update", "index": ALL}, "children"),
        Output({"type": "strategy-prediction", "index": ALL}, "children")
    ],
    [Input('ui-data-store', 'data')],
    [State({"type": "data-source-badge", "index": ALL}, "id")]
)
def update_stocks_display(data, badge_ids):
    if not data:
        # Return placeholders if no data
        empty_result = [
            ["Live"] * len(badge_ids),  # data source badges
            ["badge bg-success ms-2 small"] * len(badge_ids),  # source badge classes
            ["Loading..."] * len(badge_ids),  # price outputs
            ["0.00%"] * len(badge_ids),  # change text outputs
            ["text-secondary"] * len(badge_ids),  # change class outputs
            ["OHLC data not available"] * len(badge_ids),  # ohlc outputs
            ["N/A"] * len(badge_ids),  # pcr outputs
            [""] * len(badge_ids),  # pcr strength outputs
            ["NEUTRAL"] * len(badge_ids),  # sentiment text outputs
            ["badge bg-secondary"] * len(badge_ids),  # sentiment class outputs
            ["S/R not available"] * len(badge_ids),  # sr levels outputs
            ["Not yet updated"] * len(badge_ids),  # last update outputs
            [""] * len(badge_ids)  # strategy predictions
        ]
        return empty_result

    symbols = [id_dict["index"] for id_dict in badge_ids]
    
    source_badges = []
    source_classes = []
    price_outputs = []
    change_text_outputs = []
    change_class_outputs = []
    ohlc_outputs = []
    pcr_outputs = []
    pcr_strength_outputs = []
    sentiment_text_outputs = []
    sentiment_class_outputs = []
    sr_levels_outputs = []
    last_update_outputs = []
    strategy_prediction_outputs = []
    
    for symbol in symbols:
        if symbol in stocks_data:
            stock_info = stocks_data[symbol]
            ltp = stock_info.get("ltp")
            
            # Set data source badge
            source_badges.append("Live" if broker_connected else "Offline")
            source_classes.append("badge bg-success ms-2 small" if broker_connected else "badge bg-warning text-dark ms-2 small")
            
            # Format price
            if ltp is not None:
                price_text = f"₹{ltp:.2f}"
                price_outputs.append(price_text)
            else:
                price_outputs.append("Waiting for data...")
            
            change_pct = stock_info.get("change_percent", 0)
            
            # Get OHLC values
            open_price = stock_info.get("open")
            high_price = stock_info.get("high")
            low_price = stock_info.get("low")
            previous_price = stock_info.get("previous")
            
            # Get PCR and sentiment
            pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
            pcr_strength = pcr_data.get(symbol, {}).get("strength", 0)
            sentiment = market_sentiment.get(symbol, "NEUTRAL")
            
            # Format PCR strength indicator
            if pcr_strength > 0.5:
                pcr_strength_text = html.Span("↑", className="text-success fw-bold", title="Strong Bullish Signal")
            elif pcr_strength > 0.2:
                pcr_strength_text = html.Span("↗", className="text-success", title="Bullish Signal")
            elif pcr_strength < -0.5:
                pcr_strength_text = html.Span("↓", className="text-danger fw-bold", title="Strong Bearish Signal")
            elif pcr_strength < -0.2:
                pcr_strength_text = html.Span("↘", className="text-danger", title="Bearish Signal")
            else:
                pcr_strength_text = html.Span("→", className="text-secondary", title="Neutral Signal")
            
            pcr_strength_outputs.append(pcr_strength_text)
            
            # Get support/resistance levels
            support_levels = stock_info.get("support_levels", [])
            resistance_levels = stock_info.get("resistance_levels", [])
            
            # Format change percentage
            if change_pct > 0:
                change_text_outputs.append(f"+{change_pct:.2f}%")
                change_class_outputs.append("text-success")
            elif change_pct < 0:
                change_text_outputs.append(f"{change_pct:.2f}%")
                change_class_outputs.append("text-danger")
            else:
                change_text_outputs.append("0.00%")
                change_class_outputs.append("text-warning")
            
            # Format OHLC
            if all(x is not None for x in [open_price, high_price, low_price, previous_price]):
                ohlc_text = f"O: {open_price:.2f} H: {high_price:.2f} L: {low_price:.2f} P: {previous_price:.2f}"
                ohlc_outputs.append(ohlc_text)
            else:
                ohlc_outputs.append("OHLC data not available")
            
            # Format PCR
            pcr_text = f"{pcr_value:.2f}"
            pcr_outputs.append(pcr_text)
            
            # Format sentiment
            sentiment_text_outputs.append(sentiment)
            if sentiment == "BULLISH":
                sentiment_class_outputs.append("badge bg-success")
            elif sentiment == "BEARISH":
                sentiment_class_outputs.append("badge bg-danger")
            else:
                sentiment_class_outputs.append("badge bg-secondary")
            
            # Format support/resistance levels
            if support_levels and resistance_levels:
                sr_text = f"S: {', '.join([str(s) for s in support_levels])} | R: {', '.join([str(r) for r in resistance_levels])}"
                sr_levels_outputs.append(sr_text)
            else:
                sr_levels_outputs.append("S/R not available")
            
            # Format last update time
            last_update_time = stock_info.get("last_updated")
            if last_update_time:
                update_text = f"Updated: {last_update_time.strftime('%H:%M:%S')}"
                last_update_outputs.append(update_text)
            else:
                last_update_outputs.append("Not yet updated")
            
            # Format strategy prediction
            predicted_strategy = stock_info.get("predicted_strategy")
            strategy_confidence = stock_info.get("strategy_confidence", 0)
            
            if predicted_strategy and strategy_confidence > 0.5:
                strategy_class = "text-success font-weight-bold" if strategy_confidence > 0.7 else "text-info"
                strategy_pred_text = html.Div([
                    html.Span("Predicted strategy: ", className="text-muted"),
                    html.Span(f"{predicted_strategy} ({strategy_confidence:.1%})", className=strategy_class)
                ])
                strategy_prediction_outputs.append(strategy_pred_text)
            else:
                strategy_prediction_outputs.append("")
                
        else:
            # Default values if stock not found
            source_badges.append("Live" if broker_connected else "Offline")
            source_classes.append("badge bg-success ms-2 small" if broker_connected else "badge bg-warning text-dark ms-2 small")
            price_outputs.append("N/A")
            change_text_outputs.append("0.00%")
            change_class_outputs.append("text-warning")
            ohlc_outputs.append("OHLC data not available")
            pcr_outputs.append("N/A")
            pcr_strength_outputs.append("")
            sentiment_text_outputs.append("NEUTRAL")
            sentiment_class_outputs.append("badge bg-secondary")
            sr_levels_outputs.append("S/R not available")
            last_update_outputs.append("Not found")
            strategy_prediction_outputs.append("")
    
    return (
        source_badges,
        source_classes,
        price_outputs,
        change_text_outputs,
        change_class_outputs,
        ohlc_outputs,
        pcr_outputs,
        pcr_strength_outputs,
        sentiment_text_outputs,
        sentiment_class_outputs,
        sr_levels_outputs,
        last_update_outputs,
        strategy_prediction_outputs
    )
# 2. Revert to a simpler fetch_history_button callback that doesn't modify the UI data store directly:

@app.callback(
    [
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True)
    ],
    [Input({"type": "fetch-history-btn", "index": ALL}, "n_clicks")],
    [State({"type": "fetch-history-btn", "index": ALL}, "id")],
    prevent_initial_call=True
)
def fetch_history_button(n_clicks, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, "", "", "primary"
    
    # Find which button was clicked
    btn_idx = None
    for i, clicks in enumerate(n_clicks):
        if clicks is not None and clicks > 0:
            btn_idx = i
            break
    
    if btn_idx is None:
        return False, "", "", "primary"
            
    try:
        # Get the symbol to fetch history for
        symbol = ids[btn_idx]["index"]
        
        # Fetch historical data with force refresh
        success = load_historical_data(symbol, period="3mo", force_refresh=True)
        
        # Force recalculation of S/R levels with extra safeguards
        if success:
            # Directly call calculate_support_resistance
            sr_success = calculate_support_resistance(symbol)
            
            # Manually trigger a refresh of the UI
            last_data_update["stocks"][symbol] = datetime.now()
            
            logger.info(f"S/R recalculation after history fetch for {symbol}: {'Success' if sr_success else 'Failed'}")
            logger.info(f"New S/R values - Support: {stocks_data[symbol].get('support_levels')}, Resistance: {stocks_data[symbol].get('resistance_levels')}")
            
            return True, "History Fetched", f"Successfully fetched historical data for {symbol} from Yahoo Finance", "success"
        else:
            return True, "Error", f"Failed to fetch historical data for {symbol}", "danger"
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}", exc_info=True)
        return True, "Error", f"Failed to fetch history: {str(e)}", "danger"
# Update Option Data Callback
@app.callback(
    [
        Output({"type": "option-ce-strike", "index": ALL}, "children"),
        Output({"type": "option-ce-price", "index": ALL}, "children"),
        Output({"type": "option-ce-signal", "index": ALL}, "children"),
        Output({"type": "option-ce-signal", "index": ALL}, "className"),
        Output({"type": "option-ce-strength", "index": ALL}, "value"),
        Output({"type": "option-ce-strength", "index": ALL}, "color"),
        Output({"type": "option-ce-trade-status", "index": ALL}, "children"),
        Output({"type": "option-pe-strike", "index": ALL}, "children"),
        Output({"type": "option-pe-price", "index": ALL}, "children"),
        Output({"type": "option-pe-signal", "index": ALL}, "children"),
        Output({"type": "option-pe-signal", "index": ALL}, "className"),
        Output({"type": "option-pe-strength", "index": ALL}, "value"),
        Output({"type": "option-pe-strength", "index": ALL}, "color"),
        Output({"type": "option-pe-trade-status", "index": ALL}, "children")
    ],
    [Input('ui-data-store', 'data')],
    [State({"type": "option-ce-strike", "index": ALL}, "id")]
)
def update_options_display(data, ce_strike_ids):
    if not data:
        # Return placeholders if no data
        n = len(ce_strike_ids)
        empty = ["N/A"] * n
        return empty, empty, ["NEUTRAL"] * n, ["badge bg-secondary"] * n, [0] * n, ["secondary"] * n, empty, empty, empty, ["NEUTRAL"] * n, ["badge bg-secondary"] * n, [0] * n, ["secondary"] * n, empty

    symbols = [id_dict["index"] for id_dict in ce_strike_ids]
    options_data_ui = data.get('options', {})
    
    ce_strike_outputs = []
    ce_price_outputs = []
    ce_signal_text_outputs = []
    ce_signal_class_outputs = []
    ce_strength_outputs = []
    ce_strength_color_outputs = []
    ce_trade_status_outputs = []
    
    pe_strike_outputs = []
    pe_price_outputs = []
    pe_signal_text_outputs = []
    pe_signal_class_outputs = []
    pe_strength_outputs = []
    pe_strength_color_outputs = []
    pe_trade_status_outputs = []
    
    for symbol in symbols:
        if symbol in stocks_data:
            stock_info = stocks_data[symbol]
            symbol_options = options_data_ui.get(symbol, {})
            
            # Process CE option
            ce_data = symbol_options.get('ce', {})
            if ce_data:
                # Strike
                ce_strike_outputs.append(ce_data.get('strike', 'N/A'))
                
                # Price
                ltp = ce_data.get('price')
                ce_price_outputs.append(f"₹{ltp:.2f}" if ltp is not None else "N/A")
                
                # Signal
                signal = ce_data.get('signal', 0)
                strength = ce_data.get('strength', 0)
                if signal > 2:
                    signal_text = "STRONG BUY"
                    signal_class = "badge bg-success"
                elif signal > 1:
                    signal_text = "BUY"
                    signal_class = "badge bg-success"
                elif signal > 0:
                    signal_text = "WEAK BUY"
                    signal_class = "badge bg-success opacity-75"
                elif signal < -2:
                    signal_text = "STRONG SELL"
                    signal_class = "badge bg-danger"
                elif signal < -1:
                    signal_text = "SELL"
                    signal_class = "badge bg-danger"
                elif signal < 0:
                    signal_text = "WEAK SELL"
                    signal_class = "badge bg-danger opacity-75"
                else:
                    signal_text = "NEUTRAL"
                    signal_class = "badge bg-secondary"
                
                ce_signal_text_outputs.append(signal_text)
                ce_signal_class_outputs.append(signal_class)
                
                # Format strength
                strength_val = min(abs(strength) * 10, 100)
                
                if signal > 0:
                    strength_color = "success"
                elif signal < 0:
                    strength_color = "danger"
                else:
                    strength_color = "secondary"
                
                ce_strength_outputs.append(strength_val)
                ce_strength_color_outputs.append(strength_color)
                
                # Check if we have an active trade
                option_key = stock_info.get("primary_ce")
                if option_key and trading_state.active_trades.get(option_key, False):
                    entry_price = trading_state.entry_price.get(option_key)
                    strategy = trading_state.strategy_type.get(option_key, "")
                    if entry_price and entry_price > 0 and ltp is not None:
                        pnl_pct = (ltp - entry_price) / entry_price * 100
                        trade_status = html.Div([
                            html.Span(f"{strategy} Trade: ", className="text-info small"),
                            html.Span(f"{pnl_pct:.2f}%", 
                                    className="text-success small" if pnl_pct >= 0 else "text-danger small")
                        ])
                    else:
                        trade_status = html.Span(f"In {strategy} Trade", className="text-info small")
                    ce_trade_status_outputs.append(trade_status)
                else:
                    ce_trade_status_outputs.append("")
            else:
                ce_strike_outputs.append("N/A")
                ce_price_outputs.append("N/A")
                ce_signal_text_outputs.append("NEUTRAL")
                ce_signal_class_outputs.append("badge bg-secondary")
                ce_strength_outputs.append(0)
                ce_strength_color_outputs.append("secondary")
                ce_trade_status_outputs.append("")
            
            # Process PE option
            pe_data = symbol_options.get('pe', {})
            if pe_data:
                # Strike
                pe_strike_outputs.append(pe_data.get('strike', 'N/A'))
                
                # Price
                ltp = pe_data.get('price')
                pe_price_outputs.append(f"₹{ltp:.2f}" if ltp is not None else "N/A")
                
                # Signal
                signal = pe_data.get('signal', 0)
                strength = pe_data.get('strength', 0)
                if signal > 2:
                    signal_text = "STRONG BUY"
                    signal_class = "badge bg-success"
                elif signal > 1:
                    signal_text = "BUY"
                    signal_class = "badge bg-success"
                elif signal > 0:
                    signal_text = "WEAK BUY"
                    signal_class = "badge bg-success opacity-75"
                elif signal < -2:
                    signal_text = "STRONG SELL"
                    signal_class = "badge bg-danger"
                elif signal < -1:
                    signal_text = "SELL"
                    signal_class = "badge bg-danger"
                elif signal < 0:
                    signal_text = "WEAK SELL"
                    signal_class = "badge bg-danger opacity-75"
                else:
                    signal_text = "NEUTRAL"
                    signal_class = "badge bg-secondary"
                
                pe_signal_text_outputs.append(signal_text)
                pe_signal_class_outputs.append(signal_class)
                
                # Format strength
                strength_val = min(abs(strength) * 10, 100)
                
                if signal > 0:
                    strength_color = "success"
                elif signal < 0:
                    strength_color = "danger"
                else:
                    strength_color = "secondary"
                
                pe_strength_outputs.append(strength_val)
                pe_strength_color_outputs.append(strength_color)
                
                # Check if we have an active trade
                option_key = stock_info.get("primary_pe")
                if option_key and trading_state.active_trades.get(option_key, False):
                    entry_price = trading_state.entry_price.get(option_key)
                    strategy = trading_state.strategy_type.get(option_key, "")
                    if entry_price and entry_price > 0 and ltp is not None:
                        pnl_pct = (ltp - entry_price) / entry_price * 100
                        trade_status = html.Div([
                            html.Span(f"{strategy} Trade: ", className="text-info small"),
                            html.Span(f"{pnl_pct:.2f}%", 
                                    className="text-success small" if pnl_pct >= 0 else "text-danger small")
                        ])
                    else:
                        trade_status = html.Span(f"In {strategy} Trade", className="text-info small")
                    pe_trade_status_outputs.append(trade_status)
                else:
                    pe_trade_status_outputs.append("")
            else:
                pe_strike_outputs.append("N/A")
                pe_price_outputs.append("N/A")
                pe_signal_text_outputs.append("NEUTRAL")
                pe_signal_class_outputs.append("badge bg-secondary")
                pe_strength_outputs.append(0)
                pe_strength_color_outputs.append("secondary")
                pe_trade_status_outputs.append("")
        else:
            # Default values if stock not found
            ce_strike_outputs.append("N/A")
            ce_price_outputs.append("N/A")
            ce_signal_text_outputs.append("NEUTRAL")
            ce_signal_class_outputs.append("badge bg-secondary")
            ce_strength_outputs.append(0)
            ce_strength_color_outputs.append("secondary")
            ce_trade_status_outputs.append("")
            
            pe_strike_outputs.append("N/A")
            pe_price_outputs.append("N/A")
            pe_signal_text_outputs.append("NEUTRAL")
            pe_signal_class_outputs.append("badge bg-secondary")
            pe_strength_outputs.append(0)
            pe_strength_color_outputs.append("secondary")
            pe_trade_status_outputs.append("")
    
    return (
        ce_strike_outputs,
        ce_price_outputs,
        ce_signal_text_outputs,
        ce_signal_class_outputs,
        ce_strength_outputs,
        ce_strength_color_outputs,
        ce_trade_status_outputs,
        pe_strike_outputs,
        pe_price_outputs,
        pe_signal_text_outputs,
        pe_signal_class_outputs,
        pe_strength_outputs,
        pe_strength_color_outputs,
        pe_trade_status_outputs
    )

# Update Market Sentiment Callback
@app.callback(
    [
        Output("overall-sentiment", "children"),
        Output("overall-sentiment", "className"),
        Output("bullish-strength", "value"),
        Output("bull-bear-ratio", "children")
    ],
    [Input('ui-data-store', 'data')]
)
def update_market_sentiment_ui(data):
    if not isinstance(data, dict):
        # Return some default or placeholder value
        return "NEUTRAL", "badge bg-secondary fs-5", 50, "N/A"
    
    if not data or 'sentiment' not in data:
        return "NEUTRAL", "badge bg-secondary fs-5", 50, "N/A"
    
    sentiment_data = data['sentiment']
    overall_sentiment = sentiment_data.get("overall", "NEUTRAL")
    
    # Count individual sentiments
    bullish_count = sum(1 for s in sentiment_data.values() if s == "BULLISH")
    bearish_count = sum(1 for s in sentiment_data.values() if s == "BEARISH")
    neutral_count = sum(1 for s in sentiment_data.values() if s == "NEUTRAL")
    
    # Calculate bull/bear ratio
    if bearish_count == 0:
        bull_bear_ratio = "∞"  # infinity symbol if no bears
    else:
        bull_bear_ratio = f"{bullish_count / bearish_count:.1f}"
    
    # Calculate bullish strength as percentage
    total_count = bullish_count + bearish_count + neutral_count
    if total_count > 0:
        bullish_strength = (bullish_count + (neutral_count * 0.5)) / total_count * 100
    else:
        bullish_strength = 50
    
    # Format overall sentiment
    if overall_sentiment == "BULLISH":
        overall_class = "badge bg-success fs-5"
    elif overall_sentiment == "BEARISH":
        overall_class = "badge bg-danger fs-5"
    else:
        overall_class = "badge bg-secondary fs-5"
    
    return overall_sentiment, overall_class, bullish_strength, bull_bear_ratio

# Update Performance Stats Callback
@app.callback(
    [
        Output("total-pnl", "children"),
        Output("total-pnl", "className"),
        Output("daily-pnl", "children"),
        Output("daily-pnl", "className"),
        Output("win-rate", "children"),
        Output("trades-today", "children"),
    ],
    [Input('ui-data-store', 'data')]
)
def update_performance_stats(data):
    if not data or 'trading' not in data:
        return "₹0.00", "text-secondary fs-5", "₹0.00", "text-secondary fs-5", "N/A (0/0)", "0"
        
    trading_data = data['trading']
    
    # Format total P&L
    total_pnl = trading_data.get('total_pnl', 0)
    if total_pnl > 0:
        total_pnl_text = f"₹{total_pnl:.2f}"
        total_pnl_class = "text-success fs-5"
    elif total_pnl < 0:
        total_pnl_text = f"-₹{abs(total_pnl):.2f}"
        total_pnl_class = "text-danger fs-5"
    else:
        total_pnl_text = "₹0.00"
        total_pnl_class = "text-secondary fs-5"
    
    # Format daily P&L
    daily_pnl = trading_data.get('daily_pnl', 0)
    if daily_pnl > 0:
        daily_pnl_text = f"₹{daily_pnl:.2f}"
        daily_pnl_class = "text-success fs-5"
    elif daily_pnl < 0:
        daily_pnl_text = f"-₹{abs(daily_pnl):.2f}"
        daily_pnl_class = "text-danger fs-5"
    else:
        daily_pnl_text = "₹0.00"
        daily_pnl_class = "text-secondary fs-5"
    
    # Calculate win rate
    wins = trading_data.get('wins', 0)
    losses = trading_data.get('losses', 0)
    total_trades = wins + losses
    if total_trades > 0:
        win_rate = wins / total_trades * 100
        win_rate_text = f"{win_rate:.1f}% ({wins}/{total_trades})"
    else:
        win_rate_text = "N/A (0/0)"
    
    # Format trades today
    trades_today_text = str(trading_data.get('trades_today', 0))
    
    return total_pnl_text, total_pnl_class, daily_pnl_text, daily_pnl_class, win_rate_text, trades_today_text

# Update Active Trades Callback
@app.callback(
    [
        Output("active-trades-container", "children"),
        Output("active-trades-count", "children")
    ],
    [Input('fast-interval', 'n_intervals')]
)
def update_active_trades(n_intervals):
    active_trades = []
    active_count = 0
    
    for option_key in list(trading_state.active_trades.keys()):
        if trading_state.active_trades.get(option_key, False):
            active_count += 1
            
            # Get trade details
            entry_price = trading_state.entry_price[option_key]
            entry_time = trading_state.entry_time[option_key]
            stop_loss = trading_state.stop_loss[option_key]
            target = trading_state.target[option_key]
            quantity = trading_state.quantity[option_key]
            strategy_type = trading_state.strategy_type[option_key]
            
            # Get option details
            option_info = options_data.get(option_key, {})
            symbol = option_info.get("symbol", option_key)
            parent_symbol = option_info.get("parent_symbol", "")
            strike = option_info.get("strike", "")
            option_type = option_info.get("option_type", "")
            
            # Calculate current P&L if price is available
            current_price = option_info.get("ltp")
            
            if current_price is not None and entry_price is not None:
                unrealized_pnl = (current_price - entry_price) * quantity
                pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                
                # Format P&L text and class
                if unrealized_pnl > 0:
                    pnl_text = f"₹{unrealized_pnl:.2f} ({pnl_pct:.2f}%)"
                    pnl_class = "text-success"
                elif unrealized_pnl < 0:
                    pnl_text = f"-₹{abs(unrealized_pnl):.2f} ({pnl_pct:.2f}%)"
                    pnl_class = "text-danger"
                else:
                    pnl_text = "₹0.00 (0.00%)"
                    pnl_class = "text-secondary"
            else:
                pnl_text = "N/A"
                pnl_class = "text-secondary"
            
            # Calculate time in trade
            if entry_time:
                time_in_trade = (datetime.now() - entry_time).total_seconds() / 60  # minutes
                time_text = f"{time_in_trade:.1f} min"
            else:
                time_text = "N/A"
            
            # Create strategy badge
            if strategy_type == "SCALP":
                strategy_badge = html.Span("SCALP", className="badge bg-success ms-2")
            elif strategy_type == "MOMENTUM":
                strategy_badge = html.Span("MOMENTUM", className="badge bg-info ms-2")
            elif strategy_type == "SWING":
                strategy_badge = html.Span("SWING", className="badge bg-primary ms-2")
            else:
                strategy_badge = ""
            
            # Create trade card
            trade_card = dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        f"{parent_symbol} {strike} {option_type}", 
                        strategy_badge
                    ], className="mb-0 d-inline")
                ], style=custom_css["header"]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Entry: ", className="text-muted me-1"),
                                html.Span(f"₹{entry_price:.2f}" if entry_price else "N/A", className="text-light")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Current: ", className="text-muted me-1"),
                                html.Span(f"₹{current_price:.2f}" if current_price is not None else "N/A", 
                                         className="text-info")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("P&L: ", className="text-muted me-1"),
                                html.Span(pnl_text, className=pnl_class)
                            ], className="mb-1")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Target: ", className="text-muted me-1"),
                                html.Span(f"₹{target:.2f}" if target else "N/A", className="text-success")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Stop Loss: ", className="text-muted me-1"),
                                html.Span(f"₹{stop_loss:.2f}" if stop_loss else "N/A", className="text-danger")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Quantity: ", className="text-muted me-1"),
                                html.Span(str(quantity), className="text-light")
                            ], className="mb-1")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Time in Trade: ", className="text-muted me-1"),
                                html.Span(time_text, className="text-light")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Entry Time: ", className="text-muted me-1"),
                                html.Span(entry_time.strftime("%H:%M:%S") if entry_time else "N/A", className="text-light")
                            ], className="mb-1"),
                            dbc.Button("Exit", id={"type": "exit-trade-btn", "index": option_key}, 
                                     color="danger", size="sm", className="mt-1")
                        ], width=4)
                    ])
                ], className="px-4 py-3")
            ], 
            style=custom_css["card"],
            className="mb-3 border-info"
            )
            
            active_trades.append(trade_card)
    
    # If no active trades, show a message
    if not active_trades:
        if not broker_connected:
            active_trades = [html.Div("Trading is disabled - Broker not connected", 
                                     className="text-center text-warning py-3")]
        else:
            active_trades = [html.Div("No active trades", 
                                     className="text-center text-muted py-3")]
    
    return active_trades, active_count

# Exit Trade Button Callback
@app.callback(
    [
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True)
    ],
    [Input({"type": "exit-trade-btn", "index": ALL}, "n_clicks")],
    [State({"type": "exit-trade-btn", "index": ALL}, "id")],
    prevent_initial_call=True
)
def exit_trade_button(n_clicks, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, "", "", "primary"
    
    # Find which button was clicked
    btn_idx = 0
    for i, clicks in enumerate(n_clicks):
        if clicks is not None and clicks > 0:
            btn_idx = i
            break
            
    try:
        # Parse the ID to get the option key
        option_key = ids[btn_idx]["index"]
        
        # Exit the trade
        success = exit_trade(option_key, reason="Manual Exit")
        
        if success:
            return True, "Trade Exited", f"Successfully exited trade for {option_key}", "success"
        else:
            return True, "Error", f"Failed to exit trade for {option_key}", "danger"
    except Exception as e:
        logger.error(f"Error exiting trade: {str(e)}")
        return True, "Error", f"Failed to exit trade: {str(e)}", "danger"

# Update Recent Trades Callback
@app.callback(
    [
        Output("recent-trades-container", "children"),
        Output("recent-trades-count", "children"),
        Output("pnl-graph", "figure")
    ],
    [Input('medium-interval', 'n_intervals')]
)
def update_trade_history(n_intervals):
    # Get the 10 most recent trades
    recent_trades = trading_state.trades_history[-10:] if trading_state.trades_history else []
    
    trade_cards = []
    for trade in reversed(recent_trades):  # Show most recent first
        option_key = trade['option_key']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl = trade['pnl']
        pnl_pct = trade['pnl_pct']
        reason = trade['reason']
        strategy_type = trade['strategy_type']
        
        # Get option info
        option_info = options_data.get(option_key, {})
        parent_symbol = option_info.get("parent_symbol", option_key.split('_')[0] if '_' in option_key else option_key)
        strike = option_info.get("strike", "")
        option_type = option_info.get("option_type", "")
        option_display = f"{parent_symbol} {strike} {option_type}"
        
        # Strategy badge
        if strategy_type == "SCALP":
            strategy_badge = html.Span("SCALP", className="badge bg-success ms-1")
        elif strategy_type == "MOMENTUM":
            strategy_badge = html.Span("MOMENTUM", className="badge bg-info ms-1")
        elif strategy_type == "SWING":
            strategy_badge = html.Span("SWING", className="badge bg-primary ms-1")
        else:
            strategy_badge = ""
        
        # Format trade card
        if pnl > 0:
            pnl_text = f"₹{pnl:.2f} ({pnl_pct:.2f}%)"
            pnl_class = "text-success"
            border_class = "border-success"
        elif pnl < 0:
            pnl_text = f"-₹{abs(pnl):.2f} ({pnl_pct:.2f}%)"
            pnl_class = "text-danger"
            border_class = "border-danger"
        else:
            pnl_text = "₹0.00 (0.00%)"
            pnl_class = "text-secondary"
            border_class = "border-secondary"
        
        trade_card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([option_display, strategy_badge], className="fw-bold d-flex align-items-center"),
                        html.Div(f"Entry: ₹{entry_price:.2f}", className="small"),
                        html.Div(f"Exit: ₹{exit_price:.2f}", className="small")
                    ], width=6),
                    dbc.Col([
                        html.Div(pnl_text, className=f"fw-bold {pnl_class}"),
                        html.Div(reason, className="small text-muted"),
                        html.Div(trade['exit_time'].strftime("%H:%M:%S"), className="small text-muted")
                    ], width=6)
                ])
            ], className="px-3 py-2")
        ], 
        style=dict(custom_css["card_alt"], **{"margin-bottom": "8px"}),
        className=f"mb-2 {border_class}"
        )
        
        trade_cards.append(trade_card)
    
    # If no trades, show a message
    if not trade_cards:
        trade_cards = [html.Div("No recent trades", className="text-center text-muted py-3")]
    
    # Create P&L graph
    if trading_state.trades_history:
        # Prepare data for graph
        dates = [trade['exit_time'] for trade in trading_state.trades_history]
        cumulative_pnl = np.cumsum([trade['pnl'] for trade in trading_state.trades_history])
        
        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_pnl,
            mode='lines+markers',
            line=dict(color='#00bcd4', width=3),
            marker=dict(size=6, color='#00bcd4'),
            name='P&L'
        ))
        
        # Customized modern layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                title=None,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)'
            ),
            hovermode='x unified'
        )
        
        # Add fill under the line - green above zero, red below zero
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_pnl,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(0,188,140,0.2)',  # Green with opacity
            hoverinfo='none',
            showlegend=False
        ))
    else:
        # Empty figure if no trades
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig.add_annotation(
            text="No trade history data available",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font_size=14,
            font_color="gray"
        )
    
    return trade_cards, len(trading_state.trades_history), fig

# Main entry point
def main():
    """Main entry point with error handling"""
    try:
        # Initialize logging
        logger.info("Starting Trading Dashboard...")
        
        # Start data fetch thread if not already started
        global data_thread_started
        if not data_thread_started:
            data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
            data_thread.start()
            logger.info("Data fetching thread started successfully")

        # Initialize and run the Dash app
        logger.info("Starting Dash server...")
        app.run_server(debug=False, host='0.0.0.0', port=8050, use_reloader=False)

    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()