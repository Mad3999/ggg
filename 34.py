import os
import time
import threading
import logging
import sys
import pyotp
import pandas as pd
import numpy as np
import dash
import queue
import concurrent.futures
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

# Check for feedparser (required for news functionality)
try:
    import feedparser
except ImportError:
    print("Warning: feedparser not installed. News functionality will be limited.")
    print("Install with: pip install feedparser")

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

# ============ API Helper Classes ============
class ApiRequestPool:
    """Enhanced request pool that manages concurrent API requests with prioritization"""
    def __init__(self, max_workers=10, rate_limit_per_sec=5):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.session = requests.Session()  # Reuse session for connection pooling
        self.rate_limiter = RateLimitHandler(max_requests=rate_limit_per_sec, time_window=1)
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_priority_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.cache_lock = threading.Lock()
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def submit(self, func, *args, priority=10, endpoint_key="default", cache_key=None, **kwargs):
        """Submit a task to the queue with priority (lower number = higher priority)"""
        if cache_key:
            with self.cache_lock:
                # Check if we have a cached result that's less than 2 seconds old
                if cache_key in self.results_cache:
                    result, timestamp = self.results_cache[cache_key]
                    if (datetime.now() - timestamp).total_seconds() < 2:
                        return result
        
        # Create a future to store the result
        future = concurrent.futures.Future()
        
        # Add request to appropriate queue based on priority
        if priority <= 5:  # High priority items (active trades, primary options)
            self.high_priority_queue.put((priority, (func, args, kwargs, future, endpoint_key, cache_key)))
        else:  # Normal priority items
            self.normal_priority_queue.put((priority, (func, args, kwargs, future, endpoint_key, cache_key)))
        
        return future
    
    def _process_queue(self):
        """Process queued requests respecting rate limits and priorities"""
        while self.running:
            try:
                # Process high priority queue first
                if not self.high_priority_queue.empty():
                    _, (func, args, kwargs, future, endpoint_key, cache_key) = self.high_priority_queue.get(block=False)
                # Then process normal priority queue
                elif not self.normal_priority_queue.empty():
                    _, (func, args, kwargs, future, endpoint_key, cache_key) = self.normal_priority_queue.get(block=False)
                else:
                    # Sleep briefly if no work to do
                    time.sleep(0.01)
                    continue
                
                # Apply rate limiting
                self.rate_limiter.wait_if_needed(endpoint_key)
                
                # Execute the function
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                    
                    # Cache result if requested
                    if cache_key:
                        with self.cache_lock:
                            self.results_cache[cache_key] = (result, datetime.now())
                except Exception as e:
                    future.set_exception(e)
            except queue.Empty:
                # No work to do, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
                time.sleep(0.1)
    
    def shutdown(self):
        """Shutdown the pool and executor"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)

# ============ Rate Limiting and Retry Logic ============
class RateLimitHandler:
    def __init__(self, max_requests=1, time_window=1, initial_backoff=1, max_backoff=60):
        # Updated to specified request limit per time window
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
            
            # Clean up old timestamps
            self.request_timestamps[endpoint_key] = [ts for ts in self.request_timestamps[endpoint_key] 
                                      if current_time - ts <= self.time_window]
            
            # Check rate limit hits
            if endpoint_key in self.rate_limit_hits:
                last_hit, hit_count = self.rate_limit_hits[endpoint_key]
                time_since_hit = current_time - last_hit
                if time_since_hit < 60:
                    # Calculate backoff with randomized jitter to prevent thundering herd
                    backoff_time = min(self.initial_backoff * (2 ** min(hit_count, 5)), self.max_backoff)
                    jitter = random.uniform(0, backoff_time * 0.1)  # 10% jitter
                    backoff_time += jitter
                    
                    remaining_wait = backoff_time - time_since_hit
                    if remaining_wait > 0:
                        logger.debug(f"Rate limit backoff for {endpoint_key}: {remaining_wait:.2f}s")
                        time.sleep(remaining_wait)
                        if time_since_hit + remaining_wait >= 60:
                            self.rate_limit_hits.pop(endpoint_key, None)
            
            # Apply rate limiting
            if len(self.request_timestamps[endpoint_key]) >= self.max_requests:
                oldest_timestamp = min(self.request_timestamps[endpoint_key])
                wait_time = self.time_window - (current_time - oldest_timestamp)
                if wait_time > 0:
                    jitter = random.uniform(0.01, 0.05)  # Small jitter for staggered requests
                    logger.debug(f"Rate limiting {endpoint_key}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time + jitter)
            
            # Record this request timestamp
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

# StopLoss Settings
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
TARGET_MOMENTUM_FACTOR = 0.4  # Factor for momentum-based target adjustment

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
    "NEWS_ENABLED": True
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

# API rate limiting parameters - Optimized for parallel processing
API_MAX_WORKERS = 20  # Maximum concurrent API workers
API_REQUESTS_PER_SECOND = 5  # Limit API requests per second for all endpoints
BULK_FETCH_SIZE = 20  # Maximum symbols per bulk request
INDEX_UPDATE_INTERVAL = 1  # Seconds between index data updates
OPTION_UPDATE_INTERVAL = 1  # Seconds between option data updates
PCR_UPDATE_INTERVAL = 30  # Seconds between PCR updates
DATA_CLEANUP_INTERVAL = 600  # Cleanup old data every 10 minutes
MAX_PRICE_HISTORY_POINTS = 1000  # Maximum number of price history points to store

# Data Update Priorities
UPDATE_PRIORITY = {
    "ACTIVE_TRADES": 1,  # Highest priority
    "PRIMARY_OPTIONS": 2,
    "INDEX": 3,
    "TRACKED_STOCKS": 4, 
    "SECONDARY_OPTIONS": 5,
    "PCR": 8,
    "NEWS": 9   # Lowest priority
}

# News Monitoring Parameters
NEWS_CHECK_INTERVAL = 30  # Check for news every 30 seconds
NEWS_SENTIMENT_THRESHOLD = 0.25  # Threshold for considering news as positive/negative
NEWS_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to trade on news
NEWS_MAX_AGE = 1800  # Max age of news in seconds (30 minutes) to consider for trading

# Historical Data Paths
NIFTY_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\NIFTY.csv"
BANK_HISTORY_PATH = r"C:\Users\madhu\Pictures\ubuntu\BANK.csv"

# Symbol mapping for broker
SYMBOL_MAPPING = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE"
}

# Default stocks to show if no stocks are manually added
DEFAULT_STOCKS = [
    {"symbol": "NIFTY", "token": "26000", "exchange": "NSE", "type": "INDEX"},
    {"symbol": "BANKNIFTY", "token": "26009", "exchange": "NSE", "type": "INDEX"},
    {"symbol": "RELIANCE", "token": "2885", "exchange": "NSE", "type": "STOCK"},
    {"symbol": "LTIM", "token": "17818", "exchange": "NSE", "type": "STOCK"}
]

# ============ Global Variables ============
smart_api = None
api_request_pool = None  # Will hold our request pool instance
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

# Timestamps to track last data updates
last_data_update = {
    "stocks": {},
    "options": {},
    "websocket": None
}

# Update queues for different data types
update_queues = {
    "stocks": queue.PriorityQueue(),
    "options": queue.PriorityQueue(),
    "news": queue.Queue(),
    "pcr": queue.Queue()
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
    'predicted_strategies': {},
    'news': {}
}

# ============ Trading State Class ============
class TradingState:
    def __init__(self):
        self.active_trades = {}
        self.entry_price = {}
        self.entry_time = {}
        self.stop_loss = {}
        self.last_sl_adjustment_price = {}
        self.initial_stop_loss = {}
        self.target = {}
        self.trailing_sl_activated = {}
        self.pnl = {}
        self.strategy_type = {}
        self.trade_source = {}
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
        self.lock = threading.Lock()  # Lock for thread-safe operations
        
    def add_option(self, option_key):
        """Initialize tracking for a new option with thread safety"""
        with self.lock:
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
        """Remove an option from trading state tracking with thread safety"""
        with self.lock:
            if option_key in self.active_trades and not self.active_trades[option_key]:
                if option_key in self.active_trades: del self.active_trades[option_key]
                if option_key in self.entry_price: del self.entry_price[option_key]
                if option_key in self.entry_time: del self.entry_time[option_key]
                if option_key in self.stop_loss: del self.stop_loss[option_key]
                if option_key in self.initial_stop_loss: del self.initial_stop_loss[option_key]
                if option_key in self.target: del self.target[option_key]
                if option_key in self.trailing_sl_activated: del self.trailing_sl_activated[option_key]
                if option_key in self.pnl: del self.pnl[option_key]
                if option_key in self.strategy_type: del self.strategy_type[option_key]
                if option_key in self.trade_source: del self.trade_source[option_key]
                if option_key in self.stock_entry_price: del self.stock_entry_price[option_key]
                if option_key in self.quantity: del self.quantity[option_key]
                return True
            return False
    
    def is_active(self, option_key):
        """Thread-safe check if a trade is active"""
        with self.lock:
            return self.active_trades.get(option_key, False)
    
    def get_active_trades_list(self):
        """Returns a list of option keys with active trades"""
        with self.lock:
            return [k for k, v in self.active_trades.items() if v]

trading_state = TradingState()

# ============ Script Master Data Management ============
# Global variables for script master data
SCRIPT_MASTER_PATH = r"C:\Users\madhu\Pictures\ubuntu\OpenAPIScripMaster.json"
script_master_data = None
script_master_loaded = False
option_token_cache = {}

# Script master index for faster lookups
script_master_index = {
    'by_symbol': {},
    'by_token': {}
}

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

# ============ Connection Management ============
def init_request_pool():
    """Initialize the request pool for parallel API calls"""
    global api_request_pool
    
    if api_request_pool is None:
        api_request_pool = ApiRequestPool(
            max_workers=API_MAX_WORKERS, 
            rate_limit_per_sec=API_REQUESTS_PER_SECOND
        )
        logger.info(f"Initialized API request pool with {API_MAX_WORKERS} workers")
    return api_request_pool

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
                    
                    # Initialize the API request pool
                    init_request_pool()
                    
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
        # Use 'searchscrip' instead of 'searchScrip'
        # Use 'exchange' and 'symbol' as arguments, not 'searchtext'
        search_resp = smart_api.searchscrip(exchange=target_exchange, symbol=search_text)
        
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

# ============ Stock Management Functions ============
def fetch_stock_data(symbol, use_cache=True):
    """Fetch data for a single stock with the API request pool"""
    global broker_connected, stocks_data, api_request_pool
    
    if not symbol or symbol not in stocks_data:
        return False
    
    if not broker_connected:
        return False
    
    # Initialize request pool if needed
    pool = init_request_pool()
    
    # Get stock details
    stock_info = stocks_data[symbol]
    token = stock_info.get("token")
    exchange = stock_info.get("exchange")
    
    if not token or not exchange:
        logger.warning(f"Missing token or exchange for {symbol}")
        return False
    
    # Create a cache key for this request
    cache_key = f"ltp_{exchange}_{symbol}_{token}"
    
    # Submit the request to the pool
    priority = UPDATE_PRIORITY["INDEX"] if stock_info.get("type") == "INDEX" else UPDATE_PRIORITY["TRACKED_STOCKS"]
    
    try:
        # Create a function to be executed in the thread pool
        def fetch_ltp_data():
            try:
                return smart_api.ltpData(exchange, symbol, token)
            except Exception as e:
                logger.error(f"Error in API call for {symbol}: {e}")
                return {"status": False, "message": str(e)}
        
        # Submit request to the pool with proper priority and caching
        future = pool.submit(
            fetch_ltp_data, 
            priority=priority, 
            endpoint_key=f"ltp_{exchange}", 
            cache_key=cache_key if use_cache else None
        )
        
        # Get the result (won't block for too long due to caching)
        result = future.result(timeout=2.0)
        
        # Process the result
        if isinstance(result, dict) and result.get("status"):
            data = result.get("data", {})
            
            # Process data and update stock_info
            process_stock_data(symbol, data)
            return True
        else:
            error_msg = result.get("message", "Unknown error") if isinstance(result, dict) else "Invalid response"
            logger.warning(f"Failed to fetch data for {symbol}: {error_msg}")
            return False
            
    except concurrent.futures.TimeoutError:
        logger.warning(f"Timeout waiting for {symbol} data")
        return False
    except Exception as e:
        logger.error(f"Error in fetch_stock_data for {symbol}: {e}")
        return False

def process_stock_data(symbol, data):
    """Process fetched stock data and update internal data structures"""
    global stocks_data
    
    if symbol not in stocks_data:
        return False
    
    stock_info = stocks_data[symbol]
    
    try:
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
        
        # Update support/resistance levels (periodically to avoid excessive calculations)
        if stock_info.get("last_sr_update") is None or \
          (datetime.now() - stock_info.get("last_sr_update")).total_seconds() > 300:  # Every 5 minutes
            calculate_support_resistance(symbol)
            stock_info["last_sr_update"] = datetime.now()
        
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
        
        # Predict the most suitable strategy for this stock (periodically)
        if stock_info.get("last_strategy_update") is None or \
           (datetime.now() - stock_info.get("last_strategy_update")).total_seconds() > 300:  # Every 5 minutes
            predict_strategy_for_stock(symbol)
            stock_info["last_strategy_update"] = datetime.now()
        
        return True
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}")
        return False

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

def fetch_history_from_yahoo(symbol, period="3mo"):
    """
    Fetch historical data for a symbol from Yahoo Finance with improved error handling
    """
    try:
        # Map symbols to Yahoo format
        if symbol.upper() in ["NIFTY"]:
            if symbol.upper() == "NIFTY":
                yahoo_symbol = "^NSEI"
            
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

def load_historical_data(symbol, period="1mo", force_refresh=False):
    """Load historical data with proper validation and error handling"""
    if symbol not in stocks_data:
        logger.warning(f"Cannot load history for unknown symbol: {symbol}")
        return False
    
    # Skip if data is already loaded and not forced to refresh
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
            logger.warning(f"Price column missing in historical data for {symbol}")
            return False
            
        valid_prices = history_df['price'].dropna()
        if len(valid_prices) < 30:  # Require at least 30 valid price points for better S/R
            logger.warning(f"Insufficient valid price data for {symbol}: {len(valid_prices)} points")
            return False
            
        # Store the data directly, overwriting any existing history
        with price_history_lock:
            stocks_data[symbol]["price_history"] = history_df
        
        # Update metadata
        stocks_data[symbol]["history_source"] = "yahoo"
        stocks_data[symbol]["history_fetched"] = True
        stocks_data[symbol]["last_history_fetch"] = datetime.now()
        
        # Force recalculation of support/resistance
        calculate_support_resistance(symbol)
        stocks_data[symbol]["last_sr_update"] = datetime.now()
        
        # Also recalculate strategy prediction with new historical data
        predict_strategy_for_stock(symbol)
        
        logger.info(f"Successfully loaded historical data for {symbol}: {len(history_df)} points")
        return True
    except Exception as e:
        logger.error(f"Error loading historical data for {symbol}: {e}", exc_info=True)
        return False

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

# ============ Option Management Functions ============
# Initialize symbol tracking when the module is first imported
def get_next_expiry_date(symbol=None):
    """
    Get the next expiry date for options with improved logic.
    
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
                
                # Using hardcoded value for now, this would be an API call in production
                logger.info(f"Using predefined expiry date for {api_symbol} (would fetch from API in production)")
                
                # For testing hardcoding to 27MAR25
                return "27MAR25"
                
            except Exception as e:
                logger.warning(f"Error fetching expiry date from broker API: {e}")
                # Fall through to default handling
        
        # Symbol-specific default expiry dates if API fails or broker not connected
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

def calculate_rounded_strike(current_price):
    """
    Calculate rounded strike price based on price magnitude:
    - Under 100: Round to nearest 5
    - 100-1000: Round to nearest 10
    - 1000-10000: Round to nearest 50
    - 10000-100000: Round to nearest 100
    - Above 100000: Round to nearest 100
    
    Maximum rounding adjustment is capped at 100
    
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

def add_option(symbol, strike, expiry, option_type, token=None, exchange="NFO"):
    """Add a new option with optimized token management"""
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
        return option_key
    
    # Retrieve or generate token
    if token is None:
        # Use our token lookup
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
    
    # Check token cache first
    cache_key = f"{symbol}_{expiry_date}_{strike_int}_{option_type}"
    if cache_key in option_token_cache:
        cache_entry = option_token_cache[cache_key]
        cache_age = datetime.now() - cache_entry.get("timestamp", datetime.min)
        
        # Use cache if it's not too old (less than 24 hours)
        if cache_age.total_seconds() < 86400:
            logger.info(f"Using cached token for {symbol} {strike_int} {option_type}: {cache_entry.get('token')}")
            return cache_entry
    
    # Generate a fallback token from a hash of the parameters
    fallback_token = str(abs(hash(f"{symbol}{expiry_date}{strike_int}{option_type}")) % 100000)
    
    # Prepare search pattern for CSV data matching
    # This is the primary method for quick, accurate token lookup
    option_symbols_df = None
    try:
        # Use the CSV token lookup database
        csv_path = r"C:\Users\madhu\Pictures\ubuntu\stocks_and_options.csv"
        option_symbols_df = pd.read_csv(csv_path)
        
        # Filter for options matching our criteria
        filtered_df = option_symbols_df[
            (option_symbols_df['exch_seg'] == 'NFO') & 
            (option_symbols_df['name'] == symbol)
        ]
        
        if not filtered_df.empty and 'strike' in filtered_df.columns and 'expiry' in filtered_df.columns:
            # Find closest strike
            if 'strike' in filtered_df.columns:
                filtered_df['strike_diff'] = abs(filtered_df['strike'] - strike_int)
                closest_strike_row = filtered_df.loc[filtered_df['strike_diff'].idxmin()]
                
                token = closest_strike_row.get('token')
                matched_symbol = closest_strike_row.get('symbol')
                matched_strike = closest_strike_row.get('strike')
                matched_expiry = closest_strike_row.get('expiry')
                
                logger.info(f"Found option in CSV: {matched_symbol} with strike {matched_strike} and expiry {matched_expiry}")
                
                if token:
                    result = {
                        "token": token,
                        "symbol": matched_symbol,
                        "strike": strike_int,
                        "expiry": expiry_date,
                        "timestamp": datetime.now()
                    }
                    
                    # Cache the result
                    option_token_cache[cache_key] = result
                    return result
    except Exception as e:
        logger.warning(f"Error searching CSV for option token: {e}")
    
    # Fall back to broker API search if connected
    if broker_connected and smart_api is not None:
        try:
            option_symbol = build_option_symbol(symbol, expiry_date, strike_int, option_type)
            matches = search_symbols(option_symbol)
            
            if matches:
                # Find the best match
                best_match = matches[0]  # Take the first match for simplicity
                match_token = best_match.get('token')
                
                if match_token:
                    result = {
                        "token": match_token,
                        "symbol": best_match.get('symbol', option_symbol),
                        "strike": strike_int,
                        "expiry": expiry_date,
                        "timestamp": datetime.now()
                    }
                    
                    # Cache the result
                    option_token_cache[cache_key] = result
                    return result
        except Exception as e:
            logger.warning(f"Error searching broker API for option token: {e}")
    
    # Use a fallback token if all else fails
    logger.warning(f"Using fallback token for {symbol} {strike_int} {option_type}: {fallback_token}")
    result = {
        "token": fallback_token,
        "symbol": build_option_symbol(symbol, expiry_date, strike_int, option_type),
        "strike": strike_int,
        "expiry": expiry_date,
        "timestamp": datetime.now(),
        "is_fallback": True
    }
    
    # Cache the fallback result
    option_token_cache[cache_key] = result
    return result

def fetch_bulk_option_data(option_keys, force_refresh=False):
    """
    Fetch data for multiple options in parallel using the request pool
    
    Args:
        option_keys (list): List of option keys to fetch
        force_refresh (bool): If True, ignore cache
        
    Returns:
        dict: Dictionary of fetched data by option key
    """
    global options_data, api_request_pool, broker_connected
    
    if not broker_connected or not option_keys:
        return {}
    
    # Initialize the request pool
    pool = init_request_pool()
    
    # Prepare futures for all options
    futures = {}
    results = {}
    
    # Check which options are in active trades first (for priority)
    active_trades = set(trading_state.get_active_trades_list())
    primary_options = set()
    
    # Find primary options
    for symbol in stocks_data:
        ce_key = stocks_data[symbol].get("primary_ce")
        pe_key = stocks_data[symbol].get("primary_pe")
        if ce_key:
            primary_options.add(ce_key)
        if pe_key:
            primary_options.add(pe_key)
    
    # Submit each option request to the pool with proper priority
    for option_key in option_keys:
        if option_key not in options_data:
            continue
            
        option_info = options_data[option_key]
        token = option_info.get("token")
        exchange = option_info.get("exchange", "NFO")
        symbol = option_info.get("symbol", option_key)
        
        if not token:
            continue
        
        # Set priority based on option type
        if option_key in active_trades:
            priority = UPDATE_PRIORITY["ACTIVE_TRADES"]
        elif option_key in primary_options:
            priority = UPDATE_PRIORITY["PRIMARY_OPTIONS"]
        else:
            priority = UPDATE_PRIORITY["SECONDARY_OPTIONS"]
        
        # Create cache key
        cache_key = f"ltp_{exchange}_{symbol}_{token}" if not force_refresh else None
        
        # Create the function to execute in the pool
        def fetch_option_ltp(option_key=option_key, exchange=exchange, symbol=symbol, token=token):
            try:
                # This is the actual API call that gets executed in the pool
                result = smart_api.ltpData(exchange, symbol, token)
                return (option_key, result)
            except Exception as e:
                logger.error(f"Error in API call for option {option_key}: {e}")
                return (option_key, {"status": False, "message": str(e)})
        
        # Submit to request pool
        future = pool.submit(
            fetch_option_ltp,
            priority=priority,
            endpoint_key=f"ltp_{exchange}",
            cache_key=cache_key
        )
        
        futures[option_key] = future
    
    # Process results without blocking the main thread for too long on any single request
    for option_key, future in futures.items():
        try:
            # Get the result with a short timeout
            result_key, result = future.result(timeout=0.5)
            
            # Process successful responses
            if isinstance(result, dict) and result.get("status"):
                data = result.get("data", {})
                if data:
                    results[option_key] = data
        except concurrent.futures.TimeoutError:
            # Just skip options that take too long, they'll get updated next time
            pass
        except Exception as e:
            logger.warning(f"Error processing option {option_key} result: {e}")
    
    # Now process all fetched results
    for option_key, data in results.items():
        process_option_data(option_key, data)
    
    return results

def process_option_data(option_key, data):
    """Process fetched option data and update internal data structures"""
    if option_key not in options_data:
        return False
    
    option_info = options_data[option_key]
    
    try:
        # Extract LTP with validation
        ltp = float(data.get("ltp", 0) or 0)
        
        # Skip invalid prices
        if ltp <= 0:
            # Try to get parent stock price and strike price for fallback
            parent_symbol = option_info.get("parent_symbol")
            strike_price = float(option_info.get("strike", 0) or 0)
            option_type = option_info.get("option_type", "").lower()
            
            # Calculate theoretical price if possible
            if parent_symbol in stocks_data and strike_price > 0:
                parent_price = stocks_data[parent_symbol].get("ltp")
                if parent_price:
                    # Simple intrinsic value calculation for fallback
                    if option_type == "ce":
                        ltp = max(parent_price - strike_price, 5) if parent_price > strike_price else 5
                    else:  # PE
                        ltp = max(strike_price - parent_price, 5) if strike_price > parent_price else 5
                else:
                    # Use previous price or default
                    ltp = option_info.get("ltp", 50) or 50
            else:
                # Use previous price or default
                ltp = option_info.get("ltp", 50) or 50
        
        # Ensure valid, non-zero LTP
        ltp = max(float(ltp), 0.01)
        
        # Extract other fields
        open_price = float(data.get("open", ltp) or ltp)
        high_price = float(data.get("high", ltp) or ltp)
        low_price = float(data.get("low", ltp) or ltp)
        previous_price = float(data.get("previous", option_info.get("ltp", ltp)) or ltp)
        
        # Ensure all prices are valid
        open_price = max(open_price, 0.01)
        high_price = max(high_price, ltp)
        low_price = max(min(low_price, ltp), 0.01)
        previous_price = max(previous_price, 0.01)
        
        # Update option info
        previous_ltp = option_info.get("ltp")
        option_info.update({
            "ltp": ltp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "previous": previous_price,
            "change_percent": ((ltp / (previous_ltp or ltp)) - 1) * 100 if previous_ltp and previous_ltp > 0 else 0,
            "last_updated": datetime.now(),
            "using_fallback": False
        })
        
        # Add to price history
        timestamp = pd.Timestamp.now()
        
        new_data = {
            'timestamp': timestamp,
            'price': ltp,
            'volume': 0,
            'open_interest': 0,
            'change': option_info.get("change_percent", 0),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'is_fallback': False
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
        
        # Generate signals
        generate_option_signals(option_key)
        
        # Update last data update timestamp
        last_data_update["options"][option_key] = datetime.now()
        
        # Update UI data store for primary options
        parent_symbol = option_info.get("parent_symbol")
        option_type = option_info.get("option_type", "").lower()
        
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
                    'using_fallback': False
                }
        
        return True
    except Exception as e:
        logger.error(f"Error processing option data for {option_key}: {e}")
        return False

def find_and_add_options(symbol):
    """
    Find and add appropriate options for a stock using CSV data.
    
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
    
    # Use the new function to update options from CSV
    return update_options_from_csv(symbol, current_price)

def update_options_from_csv(symbol, target_strike=None, csv_path=r"C:\Users\madhu\Pictures\ubuntu\stocks_and_options.csv"):
    """
    Update options data for a specific stock using CSV data with optimized processing.
    
    Args:
        symbol (str): Stock symbol
        target_strike (float, optional): Target strike price. If None, will use current price.
        csv_path (str): Path to the CSV file with option data
    
    Returns:
        dict: Dictionary with CE and PE option keys
    """
    global stocks_data, options_data
    
    if symbol not in stocks_data:
        logger.warning(f"Cannot update options: Stock {symbol} not found")
        return {"CE": None, "PE": None}
    
    # Get current price if target_strike is not provided
    if target_strike is None:
        target_strike = stocks_data[symbol].get("ltp")
        if not target_strike:
            logger.warning(f"No current price available for {symbol}")
            return {"CE": None, "PE": None}
    
    logger.info(f"Finding options for {symbol} near strike {target_strike}")
    
    try:
        # Read the CSV only once and cache the result
        # Use global CSV df as cache
        global tokens_and_symbols_df
        
        if tokens_and_symbols_df is None:
            # Load CSV with optimized memory usage - load only necessary columns
            tokens_and_symbols_df = pd.read_csv(
                csv_path, 
                usecols=['exch_seg', 'token', 'symbol', 'name', 'expiry', 'strike'],
                dtype={
                    'exch_seg': 'str',
                    'token': 'str',
                    'symbol': 'str',
                    'name': 'str',
                    'expiry': 'str',
                    'strike': 'float'
                }
            )
            logger.info(f"Loaded CSV with {len(tokens_and_symbols_df)} rows")
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data is empty or failed to load")
            return {"CE": None, "PE": None}
        
        # Filter options for the specified stock - index or name matching
        stock_options = tokens_and_symbols_df[
            (tokens_and_symbols_df['exch_seg'] == 'NFO') & 
            (tokens_and_symbols_df['name'] == symbol)
        ]
        
        if stock_options.empty:
            logger.warning(f"No options found for {symbol}")
            return {"CE": None, "PE": None}
        
        # Process expiry dates - look for nearest available
        if 'expiry' in stock_options.columns:
            # Convert to datetime for proper sorting
            try:
                # Find all available expiry dates
                stock_options['expiry_date'] = pd.to_datetime(stock_options['expiry'], format='%d-%b-%y')
                
                # Sort by expiry date (ascending)
                sorted_expirations = sorted(stock_options['expiry'].unique(), 
                                          key=lambda x: pd.to_datetime(x, format='%d-%b-%y'))
                
                if not sorted_expirations:
                    logger.warning(f"No expiry dates found for {symbol}")
                    return {"CE": None, "PE": None}
                
                # Select the closest expiry
                closest_expiry = sorted_expirations[0]
                logger.info(f"Selected expiry date: {closest_expiry}")
                
                # Filter options by closest expiry
                expiry_options = stock_options[stock_options['expiry'] == closest_expiry]
                
                # Separate CE and PE options
                ce_options = expiry_options[expiry_options['symbol'].str.endswith('CE')]
                pe_options = expiry_options[expiry_options['symbol'].str.endswith('PE')]
                
                # Find closest strikes to the target with efficient filtering
                ce_result = None
                pe_result = None
                
                if not ce_options.empty and 'strike' in ce_options.columns:
                    ce_options['strike_diff'] = abs(ce_options['strike'] - target_strike)
                    closest_ce = ce_options.loc[ce_options['strike_diff'].idxmin()]
                    ce_result = closest_ce.to_dict()
                    logger.info(f"Found CE option: {closest_ce['symbol']} with token {closest_ce['token']} at strike {closest_ce['strike']}")
                
                if not pe_options.empty and 'strike' in pe_options.columns:
                    pe_options['strike_diff'] = abs(pe_options['strike'] - target_strike)
                    closest_pe = pe_options.loc[pe_options['strike_diff'].idxmin()]
                    pe_result = closest_pe.to_dict()
                    logger.info(f"Found PE option: {closest_pe['symbol']} with token {closest_pe['token']} at strike {closest_pe['strike']}")
                
                # Process and add the options
                result = {"CE": None, "PE": None}
                
                # Initialize result
                result = {"CE": None, "PE": None}
                
                # Process CE option
                if ce_result:
                    ce_data = ce_result
                    # Create a unique option key
                    try:
                        expiry_date = ce_data.get('expiry', '').replace('-', '')
                        # Handle month format (convert Apr to APR if needed)
                        month_match = re.search(r'([A-Za-z]+)', expiry_date)
                        if month_match:
                            month = month_match.group(1).upper()
                            expiry_date = expiry_date.replace(month_match.group(1), month)
                    except:
                        # If regex fails, use expiry as is
                        expiry_date = ce_data.get('expiry', '')
                            
                    strike = str(int(float(ce_data.get('strike', 0))))
                    ce_key = f"{symbol}_{expiry_date}_{strike}_CE"
                    
                    # Add option to options_data if not already there
                    if ce_key not in options_data:
                        options_data[ce_key] = {
                            "symbol": ce_data.get("symbol"),
                            "token": ce_data.get("token"),
                            "exchange": "NFO",
                            "parent_symbol": symbol,
                            "expiry": ce_data.get("expiry"),
                            "strike": ce_data.get("strike"),
                            "option_type": "CE",
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
                        
                        # Initialize in trading state
                        trading_state.add_option(ce_key)
                    
                    # Set as primary CE option
                    stocks_data[symbol]["primary_ce"] = ce_key
                    
                    # Add to parent stock's options list if not already there
                    if "CE" not in stocks_data[symbol]["options"]:
                        stocks_data[symbol]["options"]["CE"] = []
                    
                    if ce_key not in stocks_data[symbol]["options"]["CE"]:
                        stocks_data[symbol]["options"]["CE"].append(ce_key)
                    
                    # Set result
                    result["CE"] = ce_key
                    logger.info(f"Updated CE option for {symbol}: {ce_key}")
                
                # Process PE option
                if pe_result:
                    pe_data = pe_result
                    # Create a unique option key
                    try:
                        expiry_date = pe_data.get('expiry', '').replace('-', '')
                        # Handle month format (convert Apr to APR if needed)
                        month_match = re.search(r'([A-Za-z]+)', expiry_date)
                        if month_match:
                            month = month_match.group(1).upper()
                            expiry_date = expiry_date.replace(month_match.group(1), month)
                    except:
                        # If regex fails, use expiry as is
                        expiry_date = pe_data.get('expiry', '')
                            
                    strike = str(int(float(pe_data.get('strike', 0))))
                    pe_key = f"{symbol}_{expiry_date}_{strike}_PE"
                    
                    # Add option to options_data if not already there
                    if pe_key not in options_data:
                        options_data[pe_key] = {
                            "symbol": pe_data.get("symbol"),
                            "token": pe_data.get("token"),
                            "exchange": "NFO",
                            "parent_symbol": symbol,
                            "expiry": pe_data.get("expiry"),
                            "strike": pe_data.get("strike"),
                            "option_type": "PE",
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
                        
                        # Initialize in trading state
                        trading_state.add_option(pe_key)
                    
                    # Set as primary PE option
                    stocks_data[symbol]["primary_pe"] = pe_key
                    
                    # Add to parent stock's options list if not already there
                    if "PE" not in stocks_data[symbol]["options"]:
                        stocks_data[symbol]["options"]["PE"] = []
                    
                    if pe_key not in stocks_data[symbol]["options"]["PE"]:
                        stocks_data[symbol]["options"]["PE"].append(pe_key)
                    
                    # Set result
                    result["PE"] = pe_key
                    logger.info(f"Updated PE option for {symbol}: {pe_key}")
                
                # Immediately queue option data updates
                if result["CE"] or result["PE"]:
                    option_keys = [k for k in [result["CE"], result["PE"]] if k]
                    # Fetch in the background via the request pool
                    schedule_option_updates(option_keys)
                
                return result
            except Exception as e:
                logger.error(f"Error processing expiry data: {e}")
                return {"CE": None, "PE": None}
        else:
            logger.error("CSV doesn't contain 'expiry' column")
            return {"CE": None, "PE": None}
    
    except Exception as e:
        logger.error(f"Error fetching options from CSV: {e}", exc_info=True)
        return {"CE": None, "PE": None}

# Add this global declaration for tokens and symbols DataFrame
tokens_and_symbols_df = None

def schedule_option_updates(option_keys, priority=None):
    """
    Schedule option updates with proper priority
    
    Args:
        option_keys (list): List of option keys to update
        priority (int, optional): Priority override (lower = higher priority)
    """
    if not option_keys:
        return
    
    # Group options by priority for more efficient processing
    active_trades = set(trading_state.get_active_trades_list())
    primary_options = set()
    
    # Find primary options
    for symbol in stocks_data:
        ce_key = stocks_data[symbol].get("primary_ce")
        pe_key = stocks_data[symbol].get("primary_pe")
        if ce_key:
            primary_options.add(ce_key)
        if pe_key:
            primary_options.add(pe_key)
    
    # Group by priority
    option_groups = {
        UPDATE_PRIORITY["ACTIVE_TRADES"]: [],
        UPDATE_PRIORITY["PRIMARY_OPTIONS"]: [],
        UPDATE_PRIORITY["SECONDARY_OPTIONS"]: []
    }
    
    # Assign each option to a priority group
    for option_key in option_keys:
        if priority is not None:
            # Override priority if specified
            if priority not in option_groups:
                option_groups[priority] = []
            option_groups[priority].append(option_key)
        elif option_key in active_trades:
            option_groups[UPDATE_PRIORITY["ACTIVE_TRADES"]].append(option_key)
        elif option_key in primary_options:
            option_groups[UPDATE_PRIORITY["PRIMARY_OPTIONS"]].append(option_key)
        else:
            option_groups[UPDATE_PRIORITY["SECONDARY_OPTIONS"]].append(option_key)
    
    # Process option updates in order of priority
    for priority, options in sorted(option_groups.items()):
        if options:
            # Process in batches of 10 options at a time
            for i in range(0, len(options), 10):
                batch = options[i:i+10]
                fetch_bulk_option_data(batch)

def update_option_selection(force_update=False):
    """
    Update option selection for all stocks based on current prices
    with enhanced expiry and strike selection and parallel processing
    """
    global last_option_selection_update, api_request_pool
    
    current_time = datetime.now()
    
    # Only update periodically unless forced
    if not force_update and (current_time - last_option_selection_update).total_seconds() < OPTION_AUTO_SELECT_INTERVAL:
        return
    
    logger.info("Updating option selection based on current stock prices")
    
    # Initialize pool
    pool = init_request_pool()
    
    # Process each stock with futures to parallelize
    futures = {}
    updates_needed = []
    
    for symbol, stock_info in stocks_data.items():
        current_price = stock_info.get("ltp")
        
        if current_price is None or current_price <= 0:
            logger.warning(f"Sk
