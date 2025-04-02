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
import concurrent.futures
import queue

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

# ============ Advanced Rate Limiting and Retry Logic ============
class EnhancedRateLimiter:
    """
    Enhanced rate limiter that implements token bucket algorithm with efficient
    request batching and prioritization
    """
    def __init__(self, tokens_per_second=1, bucket_size=10):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size  # Start with a full bucket
        self.last_refill = time.time()
        self.lock = threading.RLock()
        self.request_queue = queue.PriorityQueue()
        self.rate_limit_hits = {}
        self.processing_thread = None
        self.stop_event = threading.Event()
        
    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        time_passed = now - self.last_refill
        new_tokens = time_passed * self.tokens_per_second
        
        with self.lock:
            self.tokens = min(self.tokens + new_tokens, self.bucket_size)
            self.last_refill = now
    
    def _can_consume(self, tokens=1):
        """Check if we can consume tokens"""
        self._refill_tokens()
        with self.lock:
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def add_request(self, func, args=None, kwargs=None, priority=5, endpoint_key="default"):
        """Add a request to the queue with priority (lower is higher priority)"""
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
            
        with self.lock:
            # Check if we're already being rate limited for this endpoint
            if endpoint_key in self.rate_limit_hits:
                last_hit, hit_count = self.rate_limit_hits[endpoint_key]
                time_since_hit = time.time() - last_hit
                
                if time_since_hit < 60:
                    # Adjust priority based on rate limit history
                    adjusted_priority = priority + min(hit_count * 2, 10)
                else:
                    # Reset rate limit tracking after 60 seconds
                    self.rate_limit_hits.pop(endpoint_key, None)
                    adjusted_priority = priority
            else:
                adjusted_priority = priority
                
            # Add request to queue
            self.request_queue.put((adjusted_priority, (func, args, kwargs, endpoint_key)))
        
        # Ensure the processing thread is running
        self._ensure_processing_thread()
    
    def _ensure_processing_thread(self):
        """Ensure the request processing thread is running"""
        with self.lock:
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.stop_event.clear()
                self.processing_thread = threading.Thread(target=self._process_requests)
                self.processing_thread.daemon = True
                self.processing_thread.start()
    
    def _process_requests(self):
        """Process requests from the queue according to rate limits"""
        while not self.stop_event.is_set():
            try:
                # Wait until we can consume a token
                while not self._can_consume() and not self.stop_event.is_set():
                    time.sleep(0.01)
                
                if self.stop_event.is_set():
                    break
                
                try:
                    # Get the highest priority request
                    priority, (func, args, kwargs, endpoint_key) = self.request_queue.get(block=False)
                    
                    # Execute the request
                    try:
                        result = func(*args, **kwargs)
                        
                        # Check if we hit a rate limit
                        rate_limited = False
                        if isinstance(result, dict):
                            error_msg = str(result.get("message", "")).lower()
                            if not result.get("status") and ("access rate" in error_msg or 
                                                          "try after some time" in error_msg or 
                                                          "session expired" in error_msg):
                                rate_limited = True
                        
                        if rate_limited:
                            self._handle_rate_limit(endpoint_key, func, args, kwargs, priority)
                        
                    except Exception as e:
                        if "access rate" in str(e).lower() or "rate limit" in str(e).lower():
                            self._handle_rate_limit(endpoint_key, func, args, kwargs, priority)
                        logger.error(f"Error executing request to {endpoint_key}: {e}")
                    
                    finally:
                        # Mark task as done
                        self.request_queue.task_done()
                        
                except queue.Empty:
                    # No requests in queue, sleep for a short time
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in request processing thread: {e}")
                time.sleep(0.5)  # Sleep to avoid tight loop in case of persistent errors
    
    def _handle_rate_limit(self, endpoint_key, func, args, kwargs, original_priority):
        """Handle a rate limit hit by re-queuing the request with backoff"""
        with self.lock:
            current_time = time.time()
            
            if endpoint_key in self.rate_limit_hits:
                last_hit, hit_count = self.rate_limit_hits[endpoint_key]
                self.rate_limit_hits[endpoint_key] = (current_time, hit_count + 1)
                backoff = min(2 ** min(hit_count, 5), 60)  # Exponential backoff capped at 60 seconds
            else:
                self.rate_limit_hits[endpoint_key] = (current_time, 1)
                backoff = 1
            
            logger.warning(f"Rate limit hit for {endpoint_key}, implementing {backoff}s backoff")
            
            # Re-queue with higher priority after backoff
            backoff_priority = original_priority + 5  # Lower priority (higher number)
            
            # Use timer to re-queue after backoff
            timer = threading.Timer(
                backoff, 
                lambda: self.request_queue.put((backoff_priority, (func, args, kwargs, endpoint_key)))
            )
            timer.daemon = True
            timer.start()
    
    def stop(self):
        """Stop the processing thread"""
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def wait_until_complete(self, timeout=None):
        """Wait until all requests are complete"""
        self.request_queue.join()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()

# Initialize rate limiter with appropriate rate (1 request per second for SmartAPI)
rate_limiter = EnhancedRateLimiter(tokens_per_second=0.95, bucket_size=5)  # Slightly under 1 to be safe

# Batch manager for optimized data fetching
class BatchRequestManager:
    """
    Manages batch requests with intelligent batching, automatic retry,
    and efficient response processing
    """
    def __init__(self, max_batch_size=50, rate_limiter=None):
        self.max_batch_size = max_batch_size
        self.rate_limiter = rate_limiter or EnhancedRateLimiter()
        self.batch_queue = {}  # Endpoint -> queue of requests
        self.results = {}  # Request ID -> result
        self.lock = threading.RLock()
        self.batch_scheduler_thread = None
        self.stop_event = threading.Event()
        self.request_counter = 0
        self.batch_counter = 0
    
    def _get_request_id(self):
        """Generate a unique request ID"""
        with self.lock:
            self.request_counter += 1
            return f"req_{self.request_counter}"
    
    def _get_batch_id(self):
        """Generate a unique batch ID"""
        with self.lock:
            self.batch_counter += 1
            return f"batch_{self.batch_counter}"
    
    def add_request(self, endpoint_key, request_data, callback=None, priority=5):
        """
        Add a request to the batch queue
        
        Args:
            endpoint_key: The API endpoint to use
            request_data: The data for the request
            callback: A function to call with the result
            priority: Request priority (lower is higher priority)
            
        Returns:
            str: Request ID
        """
        request_id = self._get_request_id()
        
        with self.lock:
            if endpoint_key not in self.batch_queue:
                self.batch_queue[endpoint_key] = []
            
            self.batch_queue[endpoint_key].append({
                'id': request_id,
                'data': request_data,
                'callback': callback,
                'priority': priority
            })
        
        # Ensure the scheduler thread is running
        self._ensure_scheduler_thread()
        
        return request_id
    
    def wait_for_result(self, request_id, timeout=5.0):
        """
        Wait for a specific request result
        
        Args:
            request_id: The request ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The result or None if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
            time.sleep(0.05)
        
        return None  # Timed out
    
    def _ensure_scheduler_thread(self):
        """Ensure the batch scheduler thread is running"""
        with self.lock:
            if self.batch_scheduler_thread is None or not self.batch_scheduler_thread.is_alive():
                self.stop_event.clear()
                self.batch_scheduler_thread = threading.Thread(target=self._batch_scheduler)
                self.batch_scheduler_thread.daemon = True
                self.batch_scheduler_thread.start()
    
    def _batch_scheduler(self):
        """Process batches from the queue efficiently"""
        while not self.stop_event.is_set():
            try:
                batches_processed = False
                
                # Check each endpoint queue for enough requests to form a batch
                for endpoint_key, requests in list(self.batch_queue.items()):
                    with self.lock:
                        if not requests:
                            continue
                        
                        # Sort by priority
                        requests.sort(key=lambda x: x['priority'])
                        
                        # Process in batches of max_batch_size
                        if len(requests) >= self.max_batch_size:
                            batch = requests[:self.max_batch_size]
                            self.batch_queue[endpoint_key] = requests[self.max_batch_size:]
                        else:
                            # If we've accumulated some requests, process them anyway
                            # after a short delay to collect more
                            if len(requests) > 0 and time.time() % 2 < 0.1:  # Process ~every 2 seconds
                                batch = requests.copy()
                                self.batch_queue[endpoint_key] = []
                            else:
                                continue
                    
                    # If we have a batch to process
                    if batch:
                        batch_id = self._get_batch_id()
                        # Submit to rate limiter with appropriate priority
                        # We use the minimum priority from the batch as the batch priority
                        min_priority = min(req['priority'] for req in batch)
                        self.rate_limiter.add_request(
                            self._process_batch,
                            args=(endpoint_key, batch, batch_id),
                            priority=min_priority,
                            endpoint_key=endpoint_key
                        )
                        batches_processed = True
                
                # If no batches were processed, sleep briefly
                if not batches_processed:
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in batch scheduler: {e}")
                time.sleep(0.5)  # Sleep to avoid tight loop in case of persistent errors
    
    def _process_batch(self, endpoint_key, batch, batch_id):
        """Process a batch of requests"""
        try:
            if endpoint_key == "fetch_stock_data":
                self._process_stock_data_batch(batch)
            elif endpoint_key == "fetch_option_data":
                self._process_option_data_batch(batch)
            elif endpoint_key == "fetch_symbols":
                self._process_symbol_search_batch(batch)
            # Add more endpoint handlers as needed
            else:
                logger.warning(f"Unknown endpoint key: {endpoint_key}")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_id} for {endpoint_key}: {e}")
            
            # If batch processing fails, handle individual requests
            for request in batch:
                try:
                    if request['callback']:
                        request['callback'](None)  # Call with None to indicate error
                except Exception as callback_err:
                    logger.error(f"Error in callback for request {request['id']}: {callback_err}")
    
    def _process_stock_data_batch(self, batch):
        """Process a batch of stock data requests"""
        # Create a list of symbols to fetch
        symbols_to_fetch = []
        request_map = {}  # Symbol -> request
        
        for request in batch:
            symbol = request['data'].get('symbol')
            if symbol:
                symbols_to_fetch.append(symbol)
                request_map[symbol] = request
        
        # Fetch data in bulk
        if symbols_to_fetch:
            results = fetch_bulk_stock_data(symbols_to_fetch)
            
            # Process results
            for symbol, result in results.items():
                if symbol in request_map:
                    request = request_map[symbol]
                    
                    # Store result
                    with self.lock:
                        self.results[request['id']] = result
                    
                    # Call callback if provided
                    if request['callback']:
                        try:
                            request['callback'](result)
                        except Exception as e:
                            logger.error(f"Error in callback for request {request['id']}: {e}")
    
    def _process_option_data_batch(self, batch):
        """Process a batch of option data requests"""
        # Create a list of options to fetch
        options_to_fetch = []
        request_map = {}  # Option key -> request
        
        for request in batch:
            option_key = request['data'].get('option_key')
            if option_key:
                options_to_fetch.append(option_key)
                request_map[option_key] = request
        
        # Fetch data in bulk
        if options_to_fetch:
            results = fetch_bulk_option_data(options_to_fetch)
            
            # Process results
            for option_key, result in results.items():
                if option_key in request_map:
                    request = request_map[option_key]
                    
                    # Store result
                    with self.lock:
                        self.results[request['id']] = result
                    
                    # Call callback if provided
                    if request['callback']:
                        try:
                            request['callback'](result)
                        except Exception as e:
                            logger.error(f"Error in callback for request {request['id']}: {e}")
    
    def _process_symbol_search_batch(self, batch):
        """Process a batch of symbol search requests"""
        # In reality, symbol searches can't be batched easily
        # Process each request individually
        for request in batch:
            search_text = request['data'].get('search_text')
            exchange = request['data'].get('exchange')
            
            if search_text:
                try:
                    result = search_symbols(search_text, exchange)
                    
                    # Store result
                    with self.lock:
                        self.results[request['id']] = result
                    
                    # Call callback if provided
                    if request['callback']:
                        try:
                            request['callback'](result)
                        except Exception as e:
                            logger.error(f"Error in callback for request {request['id']}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error searching symbols for {search_text}: {e}")
                    
                    # Store error as result
                    with self.lock:
                        self.results[request['id']] = []
                    
                    # Call callback with empty result
                    if request['callback']:
                        try:
                            request['callback']([])
                        except Exception as e:
                            logger.error(f"Error in callback for request {request['id']}: {e}")
    
    def stop(self):
        """Stop the batch scheduler thread"""
        self.stop_event.set()
        if self.batch_scheduler_thread and self.batch_scheduler_thread.is_alive():
            self.batch_scheduler_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()

# Initialize batch manager
batch_manager = BatchRequestManager(max_batch_size=50, rate_limiter=rate_limiter)

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

# Symbol Caching
SYMBOL_CACHE_FILE = r"symbol_cache.json"
TOKEN_CACHE_FILE = r"token_cache.json"

# Worker Thread Pool Size
DATA_WORKER_THREADS = 10  # Number of worker threads for data processing

# ============ Global Variables ============
smart_api = None
broker_connected = False
broker_error_message = None
broker_connection_retry_time = None
dashboard_initialized = False
data_thread_started = False
lock = threading.RLock()
price_history_lock = threading.RLock()
last_historical_refresh = None
last_historical_date = None
last_option_selection_update = datetime.now() - timedelta(seconds=OPTION_AUTO_SELECT_INTERVAL)
last_pcr_update = datetime.now() - timedelta(seconds=PCR_UPDATE_INTERVAL)
last_news_update = datetime.now() - timedelta(seconds=NEWS_CHECK_INTERVAL)
last_cleanup = datetime.now()
last_connection_time = None

# Thread pool for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=DATA_WORKER_THREADS)

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
    'news': {}  # Added for news data
}

# Symbol caches for faster lookups
symbol_to_token_cache = {}  # Symbol -> Token mapping
token_to_symbol_cache = {}  # Token -> Symbol mapping

# Option token cache with expiry tracking
option_token_cache = {}

# Pending data requests
pending_requests = set()
pending_requests_lock = threading.RLock()

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
        self.lock = threading.RLock()  # Thread-safe operations
        
    def add_option(self, option_key):
        """Initialize tracking for a new option"""
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
        """Remove an option from trading state tracking"""
        with self.lock:
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

    def get_active_trades(self):
        """Get a copy of active trades for thread-safe access"""
        with self.lock:
            return {k: v for k, v in self.active_trades.items() if v}

    def get_trade_data(self, option_key):
        """Get all trade data for an option key in a thread-safe manner"""
        with self.lock:
            if option_key not in self.active_trades:
                return None
                
            return {
                'active': self.active_trades.get(option_key),
                'entry_price': self.entry_price.get(option_key),
                'entry_time': self.entry_time.get(option_key),
                'stop_loss': self.stop_loss.get(option_key),
                'initial_stop_loss': self.initial_stop_loss.get(option_key),
                'target': self.target.get(option_key),
                'trailing_activated': self.trailing_sl_activated.get(option_key),
                'pnl': self.pnl.get(option_key),
                'strategy_type': self.strategy_type.get(option_key),
                'trade_source': self.trade_source.get(option_key),
                'stock_entry_price': self.stock_entry_price.get(option_key),
                'quantity': self.quantity.get(option_key)
            }

trading_state = TradingState()

# ============ Utility Functions ============
def safe_float(value, default=0.0):
    """Safely convert a value to float with a default for None or errors"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide with default value for zero denominator"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def is_market_open():
    """Check if the market is currently open"""
    now = datetime.now()
    # Indian market hours: 9:15 AM to 3:30 PM, Monday to Friday
    is_weekday = now.weekday() < 5  # 0-4 for Monday to Friday
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return is_weekday and market_start <= now <= market_end

def load_symbol_cache():
    """Load symbol to token cache from file"""
    global symbol_to_token_cache, token_to_symbol_cache
    
    try:
        if os.path.exists(SYMBOL_CACHE_FILE):
            with open(SYMBOL_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                symbol_to_token_cache = cache_data.get('symbol_to_token', {})
                # Convert token keys from string back to int if needed
                token_to_symbol_cache = {
                    k: v for k, v in cache_data.get('token_to_symbol', {}).items()
                }
            logger.info(f"Loaded {len(symbol_to_token_cache)} symbols from cache")
            return True
    except Exception as e:
        logger.error(f"Error loading symbol cache: {e}")
    
    # Initialize empty caches if file doesn't exist or loading failed
    symbol_to_token_cache = {}
    token_to_symbol_cache = {}
    return False

def save_symbol_cache():
    """Save symbol to token cache to file"""
    try:
        with open(SYMBOL_CACHE_FILE, 'w') as f:
            cache_data = {
                'symbol_to_token': symbol_to_token_cache,
                'token_to_symbol': token_to_symbol_cache,
                'last_updated': datetime.now().isoformat()
            }
            json.dump(cache_data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving symbol cache: {e}")
        return False

def load_token_cache():
    """Load option token cache from file"""
    global option_token_cache
    
    try:
        if os.path.exists(TOKEN_CACHE_FILE):
            with open(TOKEN_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                # Convert timestamp strings back to datetime objects
                option_token_cache = {}
                for k, v in cache_data.items():
                    if 'timestamp' in v and isinstance(v['timestamp'], str):
                        try:
                            v['timestamp'] = datetime.fromisoformat(v['timestamp'])
                        except:
                            v['timestamp'] = datetime.now()
                    option_token_cache[k] = v
            logger.info(f"Loaded {len(option_token_cache)} option tokens from cache")
            return True
    except Exception as e:
        logger.error(f"Error loading token cache: {e}")
    
    # Initialize empty cache if file doesn't exist or loading failed
    option_token_cache = {}
    return False

def save_token_cache():
    """Save option token cache to file"""
    try:
        # Make a copy of the cache with datetime objects converted to strings
        cache_copy = {}
        for k, v in option_token_cache.items():
            v_copy = v.copy()
            if 'timestamp' in v_copy and isinstance(v_copy['timestamp'], datetime):
                v_copy['timestamp'] = v_copy['timestamp'].isoformat()
            cache_copy[k] = v_copy
        
        with open(TOKEN_CACHE_FILE, 'w') as f:
            json.dump(cache_copy, f)
        return True
    except Exception as e:
        logger.error(f"Error saving token cache: {e}")
        return False

# ============ Broker Connection ============
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
                    
                    # Load cached data on successful connection
                    load_symbol_cache()
                    load_token_cache()
                    
                    # Verify connection with a test API call
                    try:
                        # Get user profile
                        try:
                            # Try with refreshToken parameter first (newer API versions)
                            profile = smart_api.getProfile(refreshToken=config.refresh_token)
                        except:
                            # Fall back to older API version without refreshToken parameter
                            profile = smart_api.getProfile()
                            
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
    """Refresh broker session if it's been more than 6 hours since last refresh"""
    global smart_api, broker_connected, config
    
    if not broker_connected or not smart_api or not config.refresh_token:
        return False
    
    # Check if we need to refresh (every 6 hours)
    current_time = datetime.now()
    if config.last_refreshed and (current_time - config.last_refreshed).total_seconds() < 21600:  # 6 hours
        return True  # No need to refresh yet
    
    try:
        logger.info("Refreshing broker session...")
        
        try:
            # Try with refreshToken parameter (newer API versions)
            refresh_resp = smart_api.generateSession(
                config.username, 
                config.password, 
                refreshToken=config.refresh_token
            )
        except:
            # Fall back to older API version without refreshToken
            refresh_resp = smart_api.generateSession(
                config.username, 
                config.password
            )
        
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

# ============ Advanced Symbol Search ============
def search_symbols(search_text, exchange=None):
    """
    Enhanced symbol search with exact matching prioritized
    
    Args:
        search_text (str): Text to search for
        exchange (str, optional): Exchange to search in. Defaults to None.
    
    Returns:
        list: List of matching symbols with exact matches first
    """
    global smart_api, broker_connected, config, symbol_to_token_cache
    
    if not search_text:
        logger.warning("Empty search text provided")
        return []
    
    # First check cache for exact match
    search_text_upper = search_text.strip().upper()
    if search_text_upper in symbol_to_token_cache:
        cached_data = symbol_to_token_cache[search_text_upper]
        logger.info(f"Found exact match for '{search_text}' in cache")
        # Return a properly formatted result
        return [{
            "symbol": search_text_upper,
            "token": cached_data.get("token"),
            "exchange": cached_data.get("exchange", config.exchange),
            "name": cached_data.get("name", search_text_upper),
            "expiry": cached_data.get("expiry", ""),
            "strike": cached_data.get("strike", ""),
            "type": cached_data.get("type", "STOCK")
        }]
    
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
        # Use 'searchscrip' method with proper parameters
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
                    
                    # Standardize the match data
                    standardized_match = {
                        "symbol": match_symbol,
                        "token": match.get("token"),
                        "exchange": match.get("exch_seg", target_exchange),
                        "name": match_name,
                        "expiry": match.get("expiry", ""),
                        "strike": match.get("strike", ""),
                        "type": "STOCK"  # Default to STOCK, will be overridden if needed
                    }
                    
                    # Determine if it's an index
                    if "NIFTY" in match_symbol or "SENSEX" in match_symbol:
                        standardized_match["type"] = "INDEX"
                    
                    # Check for exact match
                    if match_symbol == search_text or match_name == search_text:
                        exact_matches.append(standardized_match)
                        
                        # Add to cache for future use
                        symbol_to_token_cache[match_symbol] = {
                            "token": match.get("token"),
                            "exchange": match.get("exch_seg", target_exchange),
                            "name": match_name,
                            "type": standardized_match["type"]
                        }
                        token_to_symbol_cache[match.get("token")] = match_symbol
                    else:
                        partial_matches.append(standardized_match)
                
                # Save updated cache
                save_symbol_cache()
                
                # Return exact matches first, then partial matches
                return exact_matches + partial_matches
            else:
                logger.debug(f"No matches found for '{search_text}'")
        else:
            error_msg = search_resp.get("message", "Unknown error") if isinstance(search_resp, dict) else "Invalid response"
            logger.warning(f"Symbol search failed for '{search_text}': {error_msg}")
    
    except Exception as e:
        logger.error(f"Error searching for '{search_text}': {e}")
    
    # If all attempts failed, try to find a match in our cache using fuzzy matching
    potential_matches = []
    for symbol, data in symbol_to_token_cache.items():
        # Simple fuzzy match - check if search text is contained in the symbol
        if search_text in symbol:
            potential_matches.append({
                "symbol": symbol,
                "token": data.get("token"),
                "exchange": data.get("exchange", config.exchange),
                "name": data.get("name", symbol),
                "type": data.get("type", "STOCK")
            })
    
    if potential_matches:
        logger.info(f"Found {len(potential_matches)} potential matches for '{search_text}' in cache")
        return potential_matches
    
    logger.warning(f"All search attempts failed for '{search_text}'")
    return []

# ============ Enhanced Data Fetching ============
def fetch_bulk_stock_data(symbols):
    """
    Fetch data for multiple stocks efficiently with request batching and
    parallel processing.
    
    Args:
        symbols (list): List of stock symbols to fetch data for
    
    Returns:
        dict: Dictionary of fetched stock data by symbol
    """
    global smart_api, broker_connected, stocks_data
    
    if not symbols:
        return {}
    
    # Ensure broker connection
    if not broker_connected or smart_api is None:
        if not connect_to_broker():
            logger.warning("Cannot fetch bulk data: Not connected to broker")
            return {}
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    # Process in smaller batches to avoid overwhelming the API
    results = {}
    max_batch_size = 10  # Process 10 symbols at a time
    
    # Create list of tasks
    tasks = []
    for symbol_batch in [symbols[i:i+max_batch_size] for i in range(0, len(symbols), max_batch_size)]:
        # Create a task for each symbol in the batch
        for symbol in symbol_batch:
            if symbol in stocks_data:
                tasks.append((symbol, stocks_data[symbol]))
    
    # Use a thread pool to fetch data in parallel
    def fetch_single_stock(args):
        symbol, stock_info = args
        try:
            # Get token and exchange from stock info
            token = stock_info.get("token")
            exchange = stock_info.get("exchange")
            
            # Skip invalid stocks
            if not token or not exchange:
                return symbol, None
            
            # Fetch LTP data with proper rate limiting
            with rate_limiter.lock:
                rate_limiter._refill_tokens()
                if not rate_limiter._can_consume():
                    time.sleep(0.1)  # Brief pause if we're out of tokens
            
            ltp_resp = smart_api.ltpData(exchange, symbol, token)
            
            # Process successful response
            if isinstance(ltp_resp, dict) and ltp_resp.get("status"):
                data = ltp_resp.get("data", {})
                
                # Safely extract LTP with default
                ltp = safe_float(data.get("ltp", 0), 0)
                
                # Use safe defaults for other fields
                open_price = safe_float(data.get("open", ltp), ltp)
                high_price = safe_float(data.get("high", ltp), ltp)
                low_price = safe_float(data.get("low", ltp), ltp)
                previous_price = safe_float(data.get("previous", ltp), ltp)
                
                # Ensure non-zero values
                ltp = max(ltp, 0.01)
                open_price = max(open_price, 0.01)
                high_price = max(high_price, 0.01)
                low_price = max(low_price, 0.01)
                previous_price = max(previous_price, 0.01)
                
                result = {
                    "ltp": ltp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "previous": previous_price,
                    "volume": data.get("tradingSymbol", 0),
                    "success": True
                }
                
                return symbol, result
            else:
                return symbol, None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return symbol, None
    
    # Submit all tasks to the thread pool and collect results
    futures = []
    for task in tasks:
        future = thread_pool.submit(fetch_single_stock, task)
        futures.append(future)
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            symbol, result = future.result()
            if result is not None:
                results[symbol] = result
        except Exception as e:
            logger.error(f"Error processing stock data result: {e}")
    
    logger.info(f"Fetched data for {len(results)}/{len(symbols)} stocks in bulk")
    return results

def fetch_bulk_option_data(option_keys):
    """
    Fetch data for multiple options with parallel processing
    
    Args:
        option_keys (list): List of option keys to fetch data for
        
    Returns:
        dict: Dictionary of fetched option data
    """
    global smart_api, broker_connected, options_data
    
    if not option_keys:
        return {}
    
    # Ensure broker connection
    if not broker_connected or smart_api is None:
        if not connect_to_broker():
            logger.warning("Cannot fetch bulk option data: Not connected to broker")
            return {}
    
    # Refresh session if needed
    refresh_session_if_needed()
    
    # Process in smaller batches to avoid overwhelming the API
    results = {}
    max_batch_size = 10  # Process 10 options at a time
    
    # Create list of tasks
    tasks = []
    for option_batch in [option_keys[i:i+max_batch_size] for i in range(0, len(option_keys), max_batch_size)]:
        # Create a task for each option in the batch
        for option_key in option_batch:
            if option_key in options_data:
                tasks.append((option_key, options_data[option_key]))
    
    # Use a thread pool to fetch data in parallel
    def fetch_single_option(args):
        option_key, option_info = args
        try:
            # Get token and exchange from option info
            token = option_info.get("token")
            exchange = option_info.get("exchange", "NFO")
            symbol = option_info.get("symbol", option_key)
            
            # Skip invalid options
            if not token:
                return option_key, None
            
            # Fetch LTP data with proper rate limiting
            with rate_limiter.lock:
                rate_limiter._refill_tokens()
                if not rate_limiter._can_consume():
                    time.sleep(0.1)  # Brief pause if we're out of tokens
            
            ltp_resp = smart_api.ltpData(exchange, symbol, token)
            
            # Process successful response
            if isinstance(ltp_resp, dict) and ltp_resp.get("status"):
                data = ltp_resp.get("data", {})
                
                # Safely extract LTP with default
                ltp = safe_float(data.get("ltp", 0), 0)
                
                # Use safe defaults for other fields
                open_price = safe_float(data.get("open", ltp), ltp)
                high_price = safe_float(data.get("high", ltp), ltp)
                low_price = safe_float(data.get("low", ltp), ltp)
                previous_price = safe_float(data.get("previous", ltp), ltp)
                
                # Ensure non-zero values
                ltp = max(ltp, 0.01)
                open_price = max(open_price, 0.01)
                high_price = max(high_price, 0.01)
                low_price = max(low_price, 0.01)
                previous_price = max(previous_price, 0.01)
                
                result = {
                    "ltp": ltp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "previous": previous_price,
                    "success": True,
                    "using_fallback": False
                }
                
                return option_key, result
            else:
                # Fallback to theoretical price if real data not available
                parent_symbol = option_info.get("parent_symbol")
                option_type = option_info.get("option_type")
                strike = safe_float(option_info.get("strike"), 0)
                
                if parent_symbol and parent_symbol in stocks_data and strike > 0:
                    stock_price = stocks_data[parent_symbol].get("ltp")
                    if stock_price:
                        # Simple theoretical option pricing as fallback
                        if option_type == "CE":
                            ltp = max(stock_price - strike, 5) if stock_price > strike else 5
                        else:  # PE
                            ltp = max(strike - stock_price, 5) if strike > stock_price else 5
                        
                        result = {
                            "ltp": ltp,
                            "open": ltp,
                            "high": ltp,
                            "low": ltp,
                            "previous": option_info.get("ltp", ltp),
                            "success": True,
                            "using_fallback": True,
                            "fallback_reason": "API returned error"
                        }
                        
                        return option_key, result
                
                # If all fallbacks fail
                return option_key, None
                
        except Exception as e:
            logger.error(f"Error fetching data for option {option_key}: {e}")
            return option_key, None
    
    # Submit all tasks to the thread pool and collect results
    futures = []
    for task in tasks:
        future = thread_pool.submit(fetch_single_option, task)
        futures.append(future)
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            option_key, result = future.result()
            if result is not None:
                results[option_key] = result
        except Exception as e:
            logger.error(f"Error processing option data result: {e}")
    
    logger.info(f"Fetched data for {len(results)}/{len(option_keys)} options in bulk")
    return results

def update_stock_with_data(symbol, data):
    """
    Update stock data with fetched information
    
    Args:
        symbol (str): Stock symbol
        data (dict): Fetched data
    
    Returns:
        bool: True if update was successful
    """
    global stocks_data, ui_data_store
    
    if symbol not in stocks_data or not data:
        return False
    
    try:
        stock_info = stocks_data[symbol]
        
        # Update with fetched data
        previous_ltp = stock_info.get("ltp")
        stock_info["ltp"] = data.get("ltp", previous_ltp)
        stock_info["open"] = data.get("open", stock_info.get("open"))
        stock_info["high"] = data.get("high", stock_info.get("high"))
        stock_info["low"] = data.get("low", stock_info.get("low"))
        stock_info["previous"] = data.get("previous", stock_info.get("previous"))
        
        # Calculate movement percentage
        if previous_ltp is not None and previous_ltp > 0 and stock_info["ltp"] is not None:
            stock_info["movement_pct"] = ((stock_info["ltp"] - previous_ltp) / previous_ltp) * 100
        
        # Calculate change percentage
        if stock_info["open"] and stock_info["open"] > 0 and stock_info["ltp"] is not None:
            stock_info["change_percent"] = ((stock_info["ltp"] - stock_info["open"]) / stock_info["open"]) * 100
        
        # Add to price history with proper locking
        timestamp = pd.Timestamp.now()
        new_data = {
            'timestamp': timestamp,
            'price': stock_info["ltp"] if stock_info["ltp"] is not None else previous_ltp,
            'volume': data.get("volume", 0),
            'open': stock_info.get("open"),
            'high': stock_info.get("high"),
            'low': stock_info.get("low")
        }
        
        # Thread-safe history update
        with price_history_lock:
            stock_info["price_history"] = pd.concat([
                stock_info["price_history"], 
                pd.DataFrame([new_data])
            ], ignore_index=True)
            
            # Limit history size
            if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
        
        # Update volatility
        if previous_ltp is not None and previous_ltp > 0 and stock_info["ltp"] is not None:
            pct_change = (stock_info["ltp"] - previous_ltp) / previous_ltp * 100
            update_volatility(symbol, pct_change)
        
        # Update support/resistance levels periodically
        if stock_info.get("last_sr_update") is None or \
           (datetime.now() - stock_info.get("last_sr_update")).total_seconds() > 300:  # Every 5 minutes
            calculate_support_resistance(symbol)
            stock_info["last_sr_update"] = datetime.now()
        
        # Update strategy prediction periodically
        if stock_info.get("last_strategy_update") is None or \
           (datetime.now() - stock_info.get("last_strategy_update")).total_seconds() > 300:  # Every 5 minutes
            predict_strategy_for_stock(symbol)
            stock_info["last_strategy_update"] = datetime.now()
        
        # Update last update time
        stock_info["last_updated"] = datetime.now()
        last_data_update["stocks"][symbol] = datetime.now()
        
        # Update UI data store
        ui_data_store['stocks'][symbol] = {
            'price': stock_info["ltp"],
            'change': stock_info.get("change_percent", 0),
            'ohlc': {
                'open': stock_info.get("open"),
                'high': stock_info.get("high"),
                'low': stock_info.get("low"),
                'previous': stock_info.get("previous")
            },
            'last_updated': stock_info["last_updated"].strftime('%H:%M:%S')
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating stock {symbol}: {e}")
        return False

def update_option_with_data(option_key, data):
    """
    Update option data with fetched information
    
    Args:
        option_key (str): Option key
        data (dict): Fetched data
    
    Returns:
        bool: True if update was successful
    """
    global options_data, ui_data_store
    
    if option_key not in options_data or not data:
        return False
    
    try:
        option_info = options_data[option_key]
        
        # Update with fetched data
        previous_ltp = option_info.get("ltp")
        ltp = data.get("ltp")
        
        if ltp is not None and ltp > 0:
            # Update option info
            option_info["ltp"] = ltp
            option_info["previous"] = previous_ltp
            option_info["open"] = data.get("open", option_info.get("open"))
            option_info["high"] = data.get("high", option_info.get("high", ltp))
            option_info["low"] = data.get("low", option_info.get("low", ltp))
            option_info["using_fallback"] = data.get("using_fallback", False)
            option_info["fallback_reason"] = data.get("fallback_reason")
            
            # Calculate change percentage
            if previous_ltp is not None and previous_ltp > 0:
                option_info["change_percent"] = ((ltp - previous_ltp) / previous_ltp) * 100
            
            # Add to price history with proper locking
            timestamp = pd.Timestamp.now()
            new_data = {
                'timestamp': timestamp,
                'price': ltp,
                'volume': 0,
                'open_interest': 0,
                'change': option_info.get("change_percent", 0),
                'open': option_info.get("open"),
                'high': option_info.get("high"),
                'low': option_info.get("low"),
                'is_fallback': option_info.get("using_fallback", False)
            }
            
            # Thread-safe history update
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
            try:
                generate_option_signals(option_key)
            except Exception as signal_err:
                logger.warning(f"Signal generation failed for {option_key}: {signal_err}")
            
            # Update last data update timestamp
            option_info["last_updated"] = datetime.now()
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
                        'using_fallback': option_info.get("using_fallback", False)
                    }
            
            return True
            
        else:
            # If we got a zero or None price, keep the previous price but update timestamp
            option_info["last_updated"] = datetime.now()
            last_data_update["options"][option_key] = datetime.now()
            return False
    
    except Exception as e:
        logger.error(f"Error updating option {option_key}: {e}")
        return False

def add_stock(symbol, token=None, exchange="NSE", stock_type="STOCK", with_options=True):
    """Add a new stock to track with improved token matching"""
    global stocks_data, volatility_data, market_sentiment, pcr_data

    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    # If already exists, just return
    if symbol in stocks_data:
        return True
    
    # If token is None, try to find the token
    if token is None:
        # First check our cache
        if symbol in symbol_to_token_cache:
            token = symbol_to_token_cache[symbol].get("token")
            exchange = symbol_to_token_cache[symbol].get("exchange", exchange)
            stock_type = symbol_to_token_cache[symbol].get("type", stock_type)
        else:
            # Search for symbol via API if connected
            if broker_connected:
                matches = search_symbols(symbol)
                
                # Find exact match
                for match in matches:
                    match_symbol = match.get("symbol", "").upper()
                    match_name = match.get("name", "").upper()
                    
                    if match_symbol == symbol or match_name == symbol:
                        token = match.get("token")
                        exchange = match.get("exchange", exchange)
                        stock_type = match.get("type", stock_type)
                        
                        # Add to cache
                        symbol_to_token_cache[symbol] = {
                            "token": token,
                            "exchange": exchange,
                            "name": match_name,
                            "type": stock_type
                        }
                        token_to_symbol_cache[token] = symbol
                        
                        # Save updated cache
                        save_symbol_cache()
                        break
        
        # If still no token, use placeholder
        if token is None:
            logger.warning(f"Could not find exact token for {symbol}, using placeholder")
            token = str(hash(symbol) % 100000)
    
    # Create new stock entry with thread-safe initialization
    with lock:
        if symbol in stocks_data:
            # If another thread already added this stock while we were processing
            return True
            
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
        # Schedule historical data loading in background
        thread_pool.submit(load_historical_data, symbol)
    
    # Schedule price data fetching in background
    if broker_connected:
        request_id = batch_manager.add_request(
            "fetch_stock_data",
            {"symbol": symbol},
            callback=lambda result: update_stock_with_data(symbol, result),
            priority=3  # Medium priority
        )
    
    # Add options if requested (after we get price data)
    if with_options:
        # Schedule options fetching for a slight delay to ensure price data is available
        def delayed_options_fetch():
            time.sleep(1)  # Short delay
            if symbol in stocks_data and stocks_data[symbol]["ltp"] is not None:
                find_and_add_options(symbol)
        
        thread_pool.submit(delayed_options_fetch)
    
    return True

def remove_stock(symbol):
    """Remove a stock and its options from tracking with thread safety"""
    global stocks_data, options_data, volatility_data, market_sentiment, pcr_data
    
    # Standardize symbol name to uppercase
    symbol = symbol.upper()
    
    if symbol not in stocks_data:
        return False
    
    # First remove all associated options with thread safety
    with lock:
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

# ============ Intelligent Data Fetching ============
def fetch_all_data_efficiently():
    """
    Efficiently fetch all required data in a coordinated batch
    with smart prioritization and resource management
    """
    global stocks_data, options_data, last_data_update, broker_connected
    
    if not broker_connected:
        return
    
    try:
        # Refresh session if needed
        refresh_session_if_needed()
        
        # Track which items need updating
        stocks_to_update = []
        options_to_update = []
        
        # Prioritize items for update
        current_time = datetime.now()
        
        # Step 1: Categorize stocks and options by priority
        high_priority_stocks = []   # Stocks with active trades or recent updates
        normal_stocks = []          # Regular stocks
        
        high_priority_options = []  # Options with active trades
        primary_options = []        # Primary options for stocks (used in display)
        normal_options = []         # Regular options
        
        # Identify stocks that need updates
        for symbol, stock_info in stocks_data.items():
            last_update = last_data_update["stocks"].get(symbol)
            
            # Check if update is needed
            needs_update = (
                last_update is None or
                (current_time - last_update).total_seconds() >= API_UPDATE_INTERVAL
            )
            
            if needs_update:
                # Check if this stock has any active option trades
                has_active_trades = False
                primary_ce = stock_info.get("primary_ce")
                primary_pe = stock_info.get("primary_pe")
                
                if primary_ce and trading_state.active_trades.get(primary_ce, False):
                    has_active_trades = True
                if primary_pe and trading_state.active_trades.get(primary_pe, False):
                    has_active_trades = True
                
                # Assign to appropriate category
                if has_active_trades or symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    high_priority_stocks.append(symbol)
                else:
                    normal_stocks.append(symbol)
        
        # Identify options that need updates
        active_trades = trading_state.get_active_trades()
        
        for option_key, option_info in options_data.items():
            last_update = last_data_update["options"].get(option_key)
            
            # Check if update is needed
            needs_update = (
                last_update is None or
                (current_time - last_update).total_seconds() >= API_UPDATE_INTERVAL
            )
            
            if needs_update:
                # Get parent symbol and check if this is a primary option
                parent_symbol = option_info.get("parent_symbol")
                option_type = option_info.get("option_type", "").lower()
                
                is_primary = (
                    parent_symbol in stocks_data and 
                    stocks_data[parent_symbol].get(f"primary_{option_type}") == option_key
                )
                
                is_active = option_key in active_trades
                
                # Assign to appropriate category
                if is_active:
                    high_priority_options.append(option_key)
                elif is_primary:
                    primary_options.append(option_key)
                else:
                    normal_options.append(option_key)
        
        # Step 2: Create update batches with proper priority
        
        # Schedule high priority stocks first
        for symbol in high_priority_stocks:
            request_id = batch_manager.add_request(
                "fetch_stock_data",
                {"symbol": symbol},
                callback=lambda result, sym=symbol: update_stock_with_data(sym, result),
                priority=1  # Highest priority
            )
        
        # Schedule high priority options next
        for option_key in high_priority_options:
            request_id = batch_manager.add_request(
                "fetch_option_data",
                {"option_key": option_key},
                callback=lambda result, key=option_key: update_option_with_data(key, result),
                priority=1  # Highest priority
            )
        
        # Schedule primary options
        for option_key in primary_options:
            request_id = batch_manager.add_request(
                "fetch_option_data",
                {"option_key": option_key},
                callback=lambda result, key=option_key: update_option_with_data(key, result),
                priority=2  # High priority
            )
        
        # Schedule normal stocks
        for symbol in normal_stocks:
            request_id = batch_manager.add_request(
                "fetch_stock_data",
                {"symbol": symbol},
                callback=lambda result, sym=symbol: update_stock_with_data(sym, result),
                priority=3  # Medium priority
            )
        
        # Schedule a subset of normal options (avoid overwhelming the API)
        # Take 20% of normal options each cycle, different ones each time
        if normal_options:
            subset_size = max(1, len(normal_options) // 5)
            # Use the timestamp to select a different subset each second
            offset = int(time.time()) % 5
            start_idx = offset * subset_size
            end_idx = min(start_idx + subset_size, len(normal_options))
            
            for option_key in normal_options[start_idx:end_idx]:
                request_id = batch_manager.add_request(
                    "fetch_option_data",
                    {"option_key": option_key},
                    callback=lambda result, key=option_key: update_option_with_data(key, result),
                    priority=4  # Low priority
                )
        
        # Step 3: Perform scheduled maintenance tasks with appropriate frequency
        
        # Update PCR data periodically
        if (current_time - last_pcr_update).total_seconds() >= PCR_UPDATE_INTERVAL:
            thread_pool.submit(update_all_pcr_data)
        
        # Update news data periodically if enabled
        if strategy_settings["NEWS_ENABLED"] and (current_time - last_news_update).total_seconds() >= NEWS_CHECK_INTERVAL:
            thread_pool.submit(update_news_data)
        
        # Update option selection periodically
        if (current_time - last_option_selection_update).total_seconds() >= OPTION_AUTO_SELECT_INTERVAL:
            thread_pool.submit(update_option_selection)
        
        # Cleanup old data periodically
        if (current_time - last_cleanup).total_seconds() >= DATA_CLEANUP_INTERVAL:
            thread_pool.submit(cleanup_old_data)
        
        # Step 4: Apply trading strategy checks
        thread_pool.submit(apply_trading_strategy)
        
        # Step 5: Update UI data store
        update_ui_data_store_efficiently()
        
    except Exception as e:
        logger.error(f"Error in fetch_all_data_efficiently: {e}", exc_info=True)

# ============ Technical Indicators ============
def calculate_rsi(data, period=RSI_PERIOD):
    """Calculate RSI technical indicator with improved error handling."""
    try:
        if len(data) < period + 1:
            return 50  # Default neutral RSI when not enough data
        
        # Get price differences using NumPy for performance
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
            
        deltas = np.zeros_like(prices)
        deltas[1:] = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = np.copy(deltas)
        losses = np.copy(deltas)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Use simple moving average for first calculation
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])
        
        # Initialize arrays for Wilder's smoothing
        smoothed_gains = np.zeros_like(prices)
        smoothed_losses = np.zeros_like(prices)
        smoothed_gains[period] = avg_gain
        smoothed_losses[period] = avg_loss
        
        # Calculate smoothed averages
        for i in range(period + 1, len(prices)):
            smoothed_gains[i] = (smoothed_gains[i-1] * (period-1) + gains[i]) / period
            smoothed_losses[i] = (smoothed_losses[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RS and RSI
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        # Handle division by zero
        for i in range(period, len(prices)):
            if smoothed_losses[i] == 0:
                rs[i] = 100.0
            else:
                rs[i] = smoothed_gains[i] / smoothed_losses[i]
            
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        # Get the latest value
        latest_rsi = rsi[-1]
        if np.isnan(latest_rsi):
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
        
        # Calculate EMAs using vectorized operations for better performance
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate EMA coefficients once for efficiency
        fast_alpha = 2 / (fast + 1)
        slow_alpha = 2 / (slow + 1)
        signal_alpha = 2 / (signal + 1)
        
        # Initialize arrays
        fast_ema = np.zeros_like(prices)
        slow_ema = np.zeros_like(prices)
        macd = np.zeros_like(prices)
        signal_line = np.zeros_like(prices)
        
        # Calculate initial values
        fast_ema[0] = prices[0]
        slow_ema[0] = prices[0]
        
        # Calculate EMAs
        for i in range(1, len(prices)):
            fast_ema[i] = prices[i] * fast_alpha + fast_ema[i-1] * (1 - fast_alpha)
            slow_ema[i] = prices[i] * slow_alpha + slow_ema[i-1] * (1 - slow_alpha)
            macd[i] = fast_ema[i] - slow_ema[i]
        
        # Calculate signal line
        signal_line[slow] = macd[slow]  # Initial value
        for i in range(slow + 1, len(prices)):
            signal_line[i] = macd[i] * signal_alpha + signal_line[i-1] * (1 - signal_alpha)
        
        # Calculate histogram
        histogram = macd - signal_line
        
        # Return the latest values
        return macd[-1], signal_line[-1], histogram[-1]
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return 0, 0, 0  # Return neutral values in case of error

def calculate_bollinger_bands(data, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
    """Calculate Bollinger Bands technical indicator with improved error handling."""
    try:
        if len(data) < period:
            return data.iloc[-1], data.iloc[-1], data.iloc[-1]  # Default when not enough data
        
        # Use NumPy for faster calculations
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate rolling window efficiently using NumPy
        rolling_mean = np.zeros_like(prices)
        rolling_std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            window = prices[i-(period-1):i+1]
            rolling_mean[i] = np.mean(window)
            rolling_std[i] = np.std(window)
        
        # Calculate bands
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        # Get latest values
        return upper_band[-1], rolling_mean[-1], lower_band[-1]
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
            # Convert to NumPy arrays for performance
            if isinstance(data, pd.Series):
                prices = data.values
                highs = high.values
                lows = low.values
            else:
                prices = np.array(data)
                highs = np.array(high)
                lows = np.array(low)
            
            # Calculate true ranges with vectorized operations
            tr1 = highs - lows  # Current high - current low
            tr2 = np.abs(np.zeros_like(prices))
            tr3 = np.abs(np.zeros_like(prices))
            
            # Handle price shifts efficiently
            tr2[1:] = np.abs(highs[1:] - prices[:-1])  # Current high - previous close
            tr3[1:] = np.abs(lows[1:] - prices[:-1])   # Current low - previous close
            
            # Find maximum of the three true ranges
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
        else:
            # Use price changes as a proxy for true range
            if isinstance(data, pd.Series):
                prices = data.values
            else:
                prices = np.array(data)
                
            # Calculate price differences
            tr = np.zeros_like(prices)
            tr[1:] = np.abs(prices[1:] - prices[:-1])
        
        # Calculate ATR using simple moving average
        atr = np.zeros_like(prices)
        atr[period] = np.mean(tr[1:period+1])
        
        # Use Wilder's smoothing method
        for i in range(period+1, len(prices)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Return latest value
        latest_atr = atr[-1]
        if np.isnan(latest_atr) or latest_atr <= 0:
            return 1.0
            
        return float(latest_atr)
    
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return 1.0  # Return default ATR in case of error

def calculate_ema(data, span):
    """Calculate Exponential Moving Average (EMA) with vectorized implementation."""
    try:
        if len(data) < span:
            return data.iloc[-1] if len(data) > 0 else 0  # Default when not enough data
        
        # Convert to NumPy array for performance
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        # Calculate EMA coefficient
        alpha = 2 / (span + 1)
        
        # Initialize EMA array
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        # Calculate EMA efficiently
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema[-1]
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return data.iloc[-1] if len(data) > 0 else 0  # Return last value in case of error

def calculate_momentum(data, period=10):
    """Calculate price momentum over a period with vectorized implementation."""
    try:
        if len(data) < period:
            return 0  # Default when not enough data
        
        # Use NumPy for performance
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data)
        
        if prices[-period] == 0:
            return 0  # Prevent division by zero
            
        return (prices[-1] - prices[-period]) / prices[-period] * 100
    except Exception as e:
        logger.error(f"Error calculating momentum: {e}")
        return 0  # Return neutral momentum in case of error

def calculate_trend_strength(data, period=20):
    """Calculate trend strength using optimized linear regression."""
    try:
        if len(data) < period:
            return 0, 0
        
        # Use NumPy for performance
        if isinstance(data, pd.Series):
            y = data.values[-period:]
        else:
            y = np.array(data)[-period:]
        
        # Create x values (time indices)
        x = np.arange(period)
        
        # Handle missing values
        valid_indices = ~np.isnan(y)
        if np.sum(valid_indices) < 5:  # Need at least 5 valid points
            return 0, 0
            
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        # Calculate linear regression efficiently
        n = len(x_valid)
        sum_x = np.sum(x_valid)
        sum_y = np.sum(y_valid)
        sum_xy = np.sum(x_valid * y_valid)
        sum_xx = np.sum(x_valid * x_valid)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate r-squared
        y_pred = slope * x_valid + intercept
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
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
    Calculate support and resistance levels with optimized algorithms
    """
    if symbol not in stocks_data:
        logger.warning(f"Cannot calculate S/R: Symbol {symbol} not found")
        return False
    
    stock_info = stocks_data[symbol]
    
    try:
        # Get the price history with proper validation
        df = stock_info["price_history"]
        
        if 'price' not in df.columns:
            logger.warning(f"Price column missing in historical data for {symbol}")
            return False
            
        # Get valid price data points (no NaN)
        prices = df['price'].dropna()
        
        if len(prices) < 30:  # Need at least 30 data points for reliable S/R
            logger.warning(f"Not enough valid price data for {symbol} to calculate S/R: {len(prices)} points")
            return False
        
        # ===== Improved S/R calculation algorithm with performance optimizations =====
        
        # Convert to NumPy array for faster processing
        price_values = prices.values
        n = len(price_values)
        
        # 1. Find local maxima and minima efficiently using gradient analysis
        # Calculate gradient (difference between adjacent points)
        gradient = np.zeros(n)
        gradient[1:] = price_values[1:] - price_values[:-1]
        
        # Find where gradient changes sign to identify potential peaks/troughs
        gradient_product = np.zeros(n-2)
        gradient_product = gradient[1:-1] * gradient[2:]
        
        # Points where gradient changes from positive to negative are peaks
        peak_indices = np.where((gradient[:-2] > 0) & (gradient[1:-1] <= 0))[0] + 1
        # Points where gradient changes from negative to positive are troughs
        trough_indices = np.where((gradient[:-2] < 0) & (gradient[1:-1] >= 0))[0] + 1
        
        # Extract the price values at these points
        peaks = price_values[peak_indices]
        troughs = price_values[trough_indices]
        
        # 2. Cluster peaks and troughs using fast K-means style algorithm
        
        # Helper function for clustering
        def cluster_points(points, threshold):
            if len(points) == 0:
                return []
                
            # Sort points for efficient clustering
            sorted_points = np.sort(points)
            clusters = []
            current_cluster = [sorted_points[0]]
            
            for i in range(1, len(sorted_points)):
                if sorted_points[i] - current_cluster[-1] <= threshold:
                    # Add to current cluster
                    current_cluster.append(sorted_points[i])
                else:
                    # Start a new cluster
                    if len(current_cluster) >= 2:  # Only keep clusters with at least 2 points
                        clusters.append(current_cluster)
                    current_cluster = [sorted_points[i]]
            
            # Add the last cluster if it has at least 2 points
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            
            # Calculate the average of each cluster
            return [np.mean(cluster) for cluster in clusters]
        
        # Determine clustering threshold based on price range
        price_range = np.max(price_values) - np.min(price_values)
        cluster_threshold = price_range * 0.01  # 1% of the price range
        
        # Cluster peaks and troughs
        resistance_levels = cluster_points(peaks, cluster_threshold)
        support_levels = cluster_points(troughs, cluster_threshold)
        
        # 3. Filter levels to focus on the most relevant ones
        current_price = stock_info.get("ltp", price_values[-1])
        
        # Sort levels and filter them
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)
        
        # Cap the number of levels
        resistance_levels = resistance_levels[:3]  # Keep only the 3 nearest resistance levels
        support_levels = support_levels[:3]  # Keep only the 3 nearest support levels
        
        # 4. Round levels to two decimal places for cleaner display
        resistance_levels = [round(r, 2) for r in resistance_levels]
        support_levels = [round(s, 2) for s in support_levels]
        
        # Update stock data with the calculated S/R levels
        stock_info["support_levels"] = support_levels
        stock_info["resistance_levels"] = resistance_levels
        
        # Update UI data store
        if 'stocks' not in ui_data_store:
            ui_data_store['stocks'] = {}
        if symbol not in ui_data_store['stocks']:
            ui_data_store['stocks'][symbol] = {}
        
        ui_data_store['stocks'][symbol]['support_levels'] = support_levels
        ui_data_store['stocks'][symbol]['resistance_levels'] = resistance_levels
        
        logger.info(f"Updated S/R for {symbol}: Support={support_levels}, Resistance={resistance_levels}")
        return True
    
    except Exception as e:
        logger.error(f"Error calculating S/R for {symbol}: {e}")
        
        # Set empty values to avoid cascading errors
        stock_info["support_levels"] = []
        stock_info["resistance_levels"] = []
        
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
    
    # Calculate current volatility efficiently
    if len(volatility_data[symbol]["window"]) >= 5:
        # Use numpy for faster standard deviation calculation
        volatility_data[symbol]["current"] = np.std(list(volatility_data[symbol]["window"]))

# ============ PCR Analysis ============
def update_all_pcr_data():
    """Update PCR data for all stocks efficiently."""
    global pcr_data, last_pcr_update
    
    current_time = datetime.now()
    if (current_time - last_pcr_update).total_seconds() < PCR_UPDATE_INTERVAL:
        return
    
    try:
        # Fetch PCR data for all symbols in a single efficient API call
        all_pcr_values = fetch_pcr_data()
        
        # Update PCR data for each symbol in a batch
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
                pcr_data[symbol]["last_updated"] = current_time
                
                # Determine PCR trend and strength
                determine_pcr_trend(symbol)
                calculate_pcr_strength(symbol)
            else:
                # Use simulated PCR if real data isn't available
                simulate_pcr_for_symbol(symbol)
        
        # Update market sentiment
        update_market_sentiment()
        
        last_pcr_update = current_time
        logger.info(f"Updated PCR data for {len(pcr_data)} symbols")
        
    except Exception as e:
        logger.error(f"Error updating PCR data: {e}")

def simulate_pcr_for_symbol(symbol):
    """Simulate realistic PCR data for a symbol based on its price action and other factors."""
    if symbol not in stocks_data or symbol not in pcr_data:
        return
    
    try:
        # Get stock data
        stock_info = stocks_data[symbol]
        price_history = stock_info.get("price_history")
        
        if price_history.empty or 'price' not in price_history.columns:
            return
            
        # Calculate factors that influence PCR
        prices = price_history['price'].dropna()
        if len(prices) < 10:
            return
            
        # Calculate recent price action
        recent_change = prices.iloc[-1] / prices.iloc[-10] - 1 if len(prices) >= 10 else 0
        
        # Invert relationship: Price going down typically increases PCR
        # (more puts relative to calls)
        pcr_adjustment = -recent_change * 0.5  # Scale the effect
        
        # Get current PCR
        current_pcr = pcr_data[symbol]["current"]
        
        # Calculate new PCR with some randomness
        new_pcr = current_pcr * (1 + pcr_adjustment) + random.uniform(-0.05, 0.05)
        
        # Keep PCR in realistic range
        new_pcr = max(0.5, min(2.0, new_pcr))
        
        # Update PCR data
        pcr_data[symbol]["current"] = new_pcr
        pcr_data[symbol]["history"].append(new_pcr)
        pcr_data[symbol]["last_updated"] = datetime.now()
        
        # Determine trend and strength
        determine_pcr_trend(symbol)
        calculate_pcr_strength(symbol)
        
    except Exception as e:
        logger.error(f"Error simulating PCR for {symbol}: {e}")

def determine_pcr_trend(symbol):
    """Determine the trend of PCR based on recent history with improved algorithm."""
    if symbol not in pcr_data:
        return
    
    # Only determine trend if we have enough history
    if len(pcr_data[symbol]["history"]) >= PCR_TREND_LOOKBACK:
        # Get recent PCR values
        recent_pcr = list(pcr_data[symbol]["history"])[-PCR_TREND_LOOKBACK:]
        
        # Calculate trend using linear regression for better accuracy
        x = np.arange(len(recent_pcr))
        y = np.array(recent_pcr)
        
        # Calculate slope efficiently
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        
        # Determine trend based on slope and significance
        if slope > 0.01:  # Significant upward trend
            pcr_data[symbol]["trend"] = "RISING"
        elif slope < -0.01:  # Significant downward trend
            pcr_data[symbol]["trend"] = "FALLING"
        else:
            pcr_data[symbol]["trend"] = "NEUTRAL"

def calculate_pcr_strength(symbol):
    """
    Calculate PCR strength based on current value and trend using an optimized algorithm
    """
    if symbol not in pcr_data:
        return
    
    pcr_val = pcr_data[symbol]["current"]
    pcr_trend = pcr_data[symbol]["trend"]
    
    # Base strength calculation with caching for performance
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

def fetch_pcr_data():
    """
    Fetch PCR data with efficient caching and fallback mechanisms
    
    Returns:
        dict: Dictionary with symbol as key and PCR value as value.
    """
    global smart_api, broker_connected
    
    pcr_dict = {}
    
    # Check if broker is connected
    if not broker_connected or smart_api is None:
        logger.warning("Cannot fetch PCR data: Not connected to broker")
        return pcr_dict
    
    try:
        # Try to fetch using AngelOne API
        logger.info("Fetching PCR data using broker API")
        
        # Get the session token from smart_api
        session_token = None
        
        if hasattr(smart_api, 'session_token'):
            session_token = smart_api.session_token
        elif hasattr(smart_api, '_SmartConnect__session_token'):
            session_token = smart_api._SmartConnect__session_token
        elif hasattr(smart_api, 'session'):
            session_token = smart_api.session
        
        if session_token:
            # Make API request with proper rate limiting
            with rate_limiter.lock:
                rate_limiter._refill_tokens()
                if not rate_limiter._can_consume():
                    time.sleep(0.1)  # Brief pause if we're out of tokens
            
            headers = {
                'Authorization': f'Bearer {session_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-PrivateKey': config.api_key
            }
            
            response = requests.get(
                'https://apiconnect.angelbroking.com/rest/secure/angelbroking/marketData/v1/putCallRatio',
                headers=headers,
                timeout=10  # Add timeout for responsiveness
            )
            
            if response.status_code == 200:
                pcr_resp = response.json()
                
                if pcr_resp.get("status"):
                    pcr_data_list = pcr_resp.get("data", [])
                    
                    for item in pcr_data_list:
                        trading_symbol = item.get("tradingSymbol", "")
                        pcr_value = item.get("pcr", 1.0)
                        
                        # Extract base symbol efficiently
                        base_symbol = re.match(r"([A-Z]+)(?:\d|\w)+", trading_symbol)
                        if base_symbol:
                            symbol = base_symbol.group(1)
                            pcr_dict[symbol] = pcr_value
                            logger.info(f"Live PCR for {symbol}: {pcr_value:.2f}")
                    
                    return pcr_dict
    
    except Exception as e:
        logger.warning(f"Error fetching PCR data: {e}")
    
    # If we reach here, API failed - calculate PCR from option data
    logger.info("Calculating PCR from option data")
    
    # Batch processing for better performance
    for symbol in list(stocks_data.keys()):
        try:
            # Find all CE and PE options for this symbol
            ce_keys = stocks_data[symbol].get("options", {}).get("CE", [])
            pe_keys = stocks_data[symbol].get("options", {}).get("PE", [])
            
            # Calculate total LTP as a proxy for open interest
            total_ce_ltp = 0
            total_pe_ltp = 0
            
            for key in ce_keys:
                if key in options_data:
                    ltp = options_data[key].get("ltp")
                    if ltp is not None and ltp > 0:
                        total_ce_ltp += ltp
            
            for key in pe_keys:
                if key in options_data:
                    ltp = options_data[key].get("ltp")
                    if ltp is not None and ltp > 0:
                        total_pe_ltp += ltp
            
            # Calculate PCR
            if total_ce_ltp > 0:
                pcr_value = total_pe_ltp / total_ce_ltp
                pcr_dict[symbol] = pcr_value
                logger.debug(f"Calculated PCR for {symbol}: {pcr_value:.2f}")
            else:
                pcr_dict[symbol] = 1.0  # Default neutral value
                
        except Exception as e:
            logger.error(f"Error calculating PCR for {symbol}: {e}")
            pcr_dict[symbol] = 1.0  # Default neutral value
    
    # Add defaults for major indices if missing
    defaults = {"NIFTY": 1.04, "BANKNIFTY": 0.95, "FINNIFTY": 1.02}
    for idx, val in defaults.items():
        if idx not in pcr_dict:
            pcr_dict[idx] = val
    
    return pcr_dict

# ============ Signal Generation ============
def generate_option_signals(option_key):
    """Generate trading signals for an option with optimized calculations."""
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
        
        # Calculate technical indicators efficiently
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
        
        # Get PCR and market sentiment
        pcr_value = 1.0
        pcr_strength = 0.0
        market_trend = "NEUTRAL"
        if parent_symbol and parent_symbol in pcr_data:
            pcr_value = pcr_data[parent_symbol].get("current", 1.0)
            pcr_strength = pcr_data[parent_symbol].get("strength", 0.0)
            pcr_trend = pcr_data[parent_symbol].get("trend", "NEUTRAL")
            
            if pcr_trend == "RISING":
                market_trend = "BEARISH"  # Rising PCR typically bearish
            elif pcr_trend == "FALLING":
                market_trend = "BULLISH"  # Falling PCR typically bullish
        
        # Initialize signal components
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
        option_info["signal"] = signal
        option_info["strength"] = signal_strength
        option_info["signal_components"] = signal_components
        
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
            
    except Exception as e:
        logger.error(f"Error generating option signals for {option_key}: {e}")
        
        # Set defaults in case of error
        option_info["signal"] = 0
        option_info["strength"] = 0
        option_info["trend"] = "NEUTRAL"

def update_market_sentiment():
    """Update market sentiment using parallel processing for better performance."""
    global market_sentiment, stocks_data
    
    # Process stocks in parallel using ThreadPoolExecutor
    def analyze_stock_sentiment(symbol):
        if symbol == "overall":
            return symbol, "NEUTRAL", 0  # Skip overall
            
        stock_info = stocks_data.get(symbol)
        if not stock_info:
            return symbol, "NEUTRAL", 0
        
        # Skip if not enough data
        if "price_history" not in stock_info or len(stock_info["price_history"]) < 20:
            return symbol, "NEUTRAL", 0
            
        price_series = stock_info["price_history"]['price']
        if 'price' not in stock_info["price_history"].columns or len(price_series) < 20:
            return symbol, "NEUTRAL", 0
        
        # Assign weight based on symbol importance
        if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            weight = 3.0  # Higher weight for major indices
        elif symbol in ["RELIANCE", "HDFCBANK", "ICICIBANK", "TCS", "INFY"]:
            weight = 2.0  # Higher weight for major stocks
        else:
            weight = 1.0  # Standard weight for other stocks
        
        try:
            # Calculate technical indicators
            rsi = calculate_rsi(price_series)
            macd_line, signal_line, histogram = calculate_macd(price_series)
            pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
            pcr_strength = pcr_data.get(symbol, {}).get("strength", 0)
            trend_slope, trend_r2 = calculate_trend_strength(price_series, 20)
            
            # Get EMA data for crossovers
            ema_short = calculate_ema(price_series, EMA_SHORT)
            ema_medium = calculate_ema(price_series, EMA_MEDIUM)
            ema_long = calculate_ema(price_series, EMA_LONG)
            
            # Initialize sentiment components
            sentiment_components = {
                "rsi": 0,
                "macd": 0,
                "pcr": 0,
                "trend": 0,
                "ema": 0,
                "price_action": 0
            }
            
            # 1. RSI component
            if rsi > 70:
                sentiment_components["rsi"] = -1.5  # Strongly overbought - bearish
            elif rsi > 60:
                sentiment_components["rsi"] = -0.8  # Mildly overbought - mildly bearish
            elif rsi < 30:
                sentiment_components["rsi"] = 1.5  # Strongly oversold - bullish
            elif rsi < 40:
                sentiment_components["rsi"] = 0.8  # Mildly oversold - mildly bullish
            else:
                # Normalized value between 40-60
                normalized_rsi = (rsi - 50) / 10
                sentiment_components["rsi"] = -normalized_rsi * 0.5  # Mild effect in middle range
            
            # 2. MACD component
            if histogram > 0:
                # Positive histogram
                if macd_line > 0 and macd_line > signal_line:
                    sentiment_components["macd"] = 1.5  # Strong bullish
                else:
                    sentiment_components["macd"] = 0.8  # Moderate bullish
            elif histogram < 0:
                # Negative histogram
                if macd_line < 0 and macd_line < signal_line:
                    sentiment_components["macd"] = -1.5  # Strong bearish
                else:
                    sentiment_components["macd"] = -0.8  # Moderate bearish
            
            # 3. PCR component
            sentiment_components["pcr"] = pcr_strength * 1.2
            
            # 4. Trend component
            if trend_r2 > 0.6:  # Only consider significant trends
                trend_factor = trend_slope * 0.8
                sentiment_components["trend"] = trend_factor
            
            # 5. EMA crossover component
            if ema_short > ema_medium and ema_medium > ema_long:
                sentiment_components["ema"] = 1.5  # Bullish alignment
            elif ema_short < ema_medium and ema_medium < ema_long:
                sentiment_components["ema"] = -1.5  # Bearish alignment
            elif ema_short > ema_medium:
                sentiment_components["ema"] = 0.7  # Partial bullish
            elif ema_short < ema_medium:
                sentiment_components["ema"] = -0.7  # Partial bearish
            
            # 6. Recent price action component
            try:
                if len(price_series) >= 5:
                    recent_change = (price_series.iloc[-1] - price_series.iloc[-5]) / price_series.iloc[-5] * 100
                    sentiment_components["price_action"] = (recent_change / 2.5) * 1.0
            except Exception:
                pass
            
            # Calculate total sentiment score
            sentiment_score = sum(sentiment_components.values())
            
            # Set sentiment based on score with improved thresholds
            sentiment = "NEUTRAL"  # Default
            if sentiment_score >= 3.0:
                sentiment = "STRONGLY BULLISH"
            elif sentiment_score >= 1.5:
                sentiment = "BULLISH"
            elif sentiment_score > 0.5:
                sentiment = "MODERATELY BULLISH"
            elif sentiment_score <= -3.0:
                sentiment = "STRONGLY BEARISH"
            elif sentiment_score <= -1.5:
                sentiment = "BEARISH"
            elif sentiment_score < -0.5:
                sentiment = "MODERATELY BEARISH"
            
            # Return symbol, sentiment, and weight
            return symbol, sentiment, weight
            
        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return symbol, "NEUTRAL", weight
    
    # Submit sentiment calculation tasks in parallel
    results = {}
    futures = []
    
    for symbol in stocks_data.keys():
        future = thread_pool.submit(analyze_stock_sentiment, symbol)
        futures.append(future)
    
    # Reset counters for overall calculation
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    total_weight = 0
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            symbol, sentiment, weight = future.result()
            
            # Track overall metrics
            if "BULLISH" in sentiment:
                if "STRONGLY" in sentiment:
                    bullish_count += weight * 2
                elif "MODERATELY" in sentiment:
                    bullish_count += weight * 0.5
                else:
                    bullish_count += weight
            elif "BEARISH" in sentiment:
                if "STRONGLY" in sentiment:
                    bearish_count += weight * 2
                elif "MODERATELY" in sentiment:
                    bearish_count += weight * 0.5
                else:
                    bearish_count += weight
            else:
                neutral_count += weight
                
            total_weight += weight
            
            # Update individual sentiment
            market_sentiment[symbol] = sentiment
            
        except Exception as e:
            logger.error(f"Error processing sentiment result: {e}")
    
    # Calculate overall market sentiment
    if total_weight > 0:
        # Adjust neutral weight for overall calculation
        effective_neutral = neutral_count * 0.5  # Count neutrals as half
        
        # Calculate percentages
        bullish_percentage = bullish_count / total_weight
        bearish_percentage = bearish_count / total_weight
        
        # Determine overall sentiment with more balanced thresholds
        if bullish_percentage > 0.5:
            market_sentiment["overall"] = "BULLISH"
        elif bearish_percentage > 0.5:
            market_sentiment["overall"] = "BEARISH"
        elif bullish_percentage > 0.3 and bullish_percentage > bearish_percentage + 0.1:
            market_sentiment["overall"] = "MODERATELY BULLISH"
        elif bearish_percentage > 0.3 and bearish_percentage > bullish_percentage + 0.1:
            market_sentiment["overall"] = "MODERATELY BEARISH"
        else:
            market_sentiment["overall"] = "NEUTRAL"
    else:
        market_sentiment["overall"] = "NEUTRAL"
        
    logger.info(f"Updated market sentiment. Overall: {market_sentiment['overall']} (Bullish: {bullish_count:.1f}, Bearish: {bearish_count:.1f}, Neutral: {neutral_count:.1f})")

# ============ Strategy Prediction ============
def predict_strategy_for_stock(symbol):
    """
    Predict the most suitable trading strategy for a stock with optimized calculations
    """
    if symbol not in stocks_data:
        return None
    
    stock_info = stocks_data[symbol]
    
    try:
        # Skip if we don't have enough price history
        if len(stock_info["price_history"]) < 30:
            return None
        
        if 'price' not in stock_info["price_history"].columns:
            return None
            
        price_series = stock_info["price_history"]['price'].dropna()
        
        if len(price_series) < 30:
            return None
        
        # Calculate key technical metrics efficiently
        rsi = calculate_rsi(price_series)
        rsi_7 = calculate_rsi(price_series, period=7)  # Short-term RSI
        rsi_21 = calculate_rsi(price_series, period=21)  # Long-term RSI
        volatility = calculate_volatility(symbol)
        momentum_short = calculate_momentum(price_series, 5)  # 5-period momentum
        momentum_medium = calculate_momentum(price_series, 10)  # 10-period momentum
        atr = calculate_atr(price_series)
        trend_slope, trend_r2 = calculate_trend_strength(price_series, 20)
        trend_slope_short, trend_r2_short = calculate_trend_strength(price_series, 10)
        
        # Calculate EMAs
        ema_short = calculate_ema(price_series, EMA_SHORT)
        ema_medium = calculate_ema(price_series, EMA_MEDIUM)
        ema_long = calculate_ema(price_series, EMA_LONG)
        
        # MACD indicator
        macd_line, signal_line, histogram = calculate_macd(price_series)
        
        # Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(price_series)
        
        # Calculate band width (volatility indicator)
        band_width = (upper_band - lower_band) / middle_band if middle_band > 0 else 0
        
        # Get PCR data
        pcr_value = pcr_data.get(symbol, {}).get("current", 1.0)
        pcr_trend = pcr_data.get(symbol, {}).get("trend", "NEUTRAL")
        pcr_strength = pcr_data.get(symbol, {}).get("strength", 0.0)
        
        # Get market sentiment
        sentiment = market_sentiment.get(symbol, "NEUTRAL")
        overall_sentiment = market_sentiment.get("overall", "NEUTRAL")
        
        # Check news mentions for this stock
        news_mentions = news_data.get("mentions", {}).get(symbol, [])
        has_recent_news = len(news_mentions) > 0
        
        # Initialize strategy scores
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
        if isinstance(price_series, pd.Series):
            macd_values = pd.Series([0, 0])  # Initialize with 2 values
            if len(price_series) >= MACD_SLOW + MACD_SIGNAL + 2:
                # Calculate MACD for the last two points
                macd_prev, _, _ = calculate_macd(price_series.iloc[:-1])
                macd_curr, _, _ = calculate_macd(price_series)
                macd_direction = macd_curr - macd_prev
                
                if (macd_curr > 0 and macd_direction > 0) or (macd_curr < 0 and macd_direction < 0):
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
            avg_news_sentiment = sum(mention.get('sentiment', 0) for mention in news_mentions) / len(news_mentions)
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
        next_best_strategy = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[1][0]
        next_best_score = strategy_scores[next_best_strategy]
        
        if best_score > 0:
            score_difference = best_score - next_best_score
            relative_difference = score_difference / best_score
            confidence = min(0.5 + relative_difference * 0.5, 1.0)  # 0.5-1.0 range
        else:
            confidence = 0  # No good strategy
        
        # Look for strong signals
        threshold = 5.0  # Minimum score to consider the strategy strong enough
        
        if best_score > threshold and confidence > 0.6:
            # Store the prediction
            stock_info["predicted_strategy"] = best_strategy
            stock_info["strategy_confidence"] = confidence
            stock_info["strategy_score"] = best_score
            
            # Update UI data store
            ui_data_store['predicted_strategies'][symbol] = {
                'strategy': best_strategy,
                'confidence': confidence,
                'score': strategy_scores
            }
            
            return best_strategy
        else:
            # Not enough confidence for any strategy
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
        logger.error(f"Error predicting strategy for {symbol}: {e}")
        return None

# ============ News Monitoring ============
def fetch_news_from_sources():
    """
    Fetch financial news from various free sources with parallel
    processing for improved performance
    """
    news_items = []
    current_time = datetime.now()
    
    try:
        # Check if feedparser is available
        if 'feedparser' in sys.modules:
            # Use thread pool to fetch from multiple sources in parallel
            sources = [
                {
                    'url': 'https://finance.yahoo.com/news/rssindex',
                    'name': 'Yahoo Finance',
                    'max_items': 15
                },
                {
                    'url': 'https://www.moneycontrol.com/rss/latestnews.xml',
                    'name': 'Moneycontrol',
                    'max_items': 15
                },
                {
                    'url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                    'name': 'Economic Times',
                    'max_items': 15
                },
                {
                    'url': 'https://www.business-standard.com/rss/markets-106.rss',
                    'name': 'Business Standard',
                    'max_items': 15
                }
            ]
            
            # Function to fetch from a single source
            def fetch_from_source(source):
                source_items = []
                try:
                    feed = feedparser.parse(source['url'])
                    
                    if feed.entries:
                        for entry in feed.entries[:source['max_items']]:
                            # Try to parse the published date from the feed
                            try:
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    item_time = datetime(*entry.published_parsed[:6])
                                else:
                                    item_time = current_time
                            except:
                                item_time = current_time
                                
                            source_items.append({
                                'title': entry.title,
                                'description': entry.get('description', ''),
                                'source': source['name'],
                                'timestamp': item_time,
                                'url': entry.link
                            })
                        
                        logger.info(f"Successfully fetched {len(source_items)} news items from {source['name']}")
                    else:
                        logger.warning(f"{source['name']} feed returned no entries")
                except Exception as e:
                    logger.warning(f"Error fetching {source['name']} news: {e}")
                    
                return source_items
            
            # Submit tasks to thread pool
            futures = []
            for source in sources:
                future = thread_pool.submit(fetch_from_source, source)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                news_items.extend(future.result())
                
            if news_items:
                logger.info(f"Fetched {len(news_items)} news items from all sources")
            else:
                logger.warning("Failed to fetch news from any source")
        else:
            # Fallback to a simplified approach if feedparser is not available
            logger.warning("feedparser not available, using placeholder news. Install with: pip install feedparser")
            news_items.append({
                'title': 'Market Update: NIFTY shows strong momentum',
                'description': 'Market analysis indicates bullish trend for NIFTY',
                'source': 'Dashboard',
                'timestamp': current_time,
                'url': '#'
            })
        
        # Sort news by timestamp, most recent first
        news_items.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return news_items
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

def analyze_news_for_stocks(news_items, stock_universe):
    """
    Analyze news items to find mentions of stocks and determine sentiment
    using an optimized algorithm
    """
    stock_mentions = {}
    
    try:
        # Import regex module
        import re
        
        # Pre-compile regex patterns for performance
        stock_patterns = {stock: re.compile(r'\b' + re.escape(stock) + r'\b', re.IGNORECASE) for stock in stock_universe}
        
        # Prepare variations for common stocks
        stock_variations = {}
        for stock in stock_universe:
            variations = [stock]
            
            # Add variations for common Indian stocks
            if stock == "RELIANCE":
                variations.extend(["Reliance Industries", "RIL"])
            elif stock == "INFY":
                variations.extend(["Infosys", "Infosys Technologies"])
            elif stock == "TCS":
                variations.extend(["Tata Consultancy", "Tata Consultancy Services"])
            elif stock == "HDFCBANK":
                variations.extend(["HDFC Bank", "Housing Development Finance Corporation"])
            elif stock == "SBIN":
                variations.extend(["State Bank of India", "SBI"])
            
            # Compile patterns for variations
            stock_variations[stock] = [re.compile(r'\b' + re.escape(var) + r'\b', re.IGNORECASE) for var in variations]
        
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
        
        # Process news items in batches for better performance
        batch_size = 10
        for i in range(0, len(news_items), batch_size):
            batch = news_items[i:min(i+batch_size, len(news_items))]
            
            for item in batch:
                title = item.get('title', '')
                description = item.get('description', '')
                full_text = f"{title} {description}"
                
                # Calculate sentiment score for this news item
                pos_score = 0
                neg_score = 0
                
                # Check for positive words
                for word, weight in positive_words.items():
                    if word.lower() in full_text.lower():
                        # Count occurrences
                        count = full_text.lower().count(word.lower())
                        # Add to score with weight
                        pos_score += count * weight
                        
                        # Title bonus - words in title have more impact
                        if word.lower() in title.lower():
                            pos_score += 0.5 * weight
                
                # Check for negative words
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
                sentiment = 0  # Default neutral
                if pos_score + neg_score > 0:
                    sentiment = (pos_score - neg_score) / (pos_score + neg_score)
                
                # Look for stock mentions
                for stock in stock_universe:
                    # Check all variations of the stock name
                    found = False
                    for pattern in stock_variations.get(stock, [stock_patterns[stock]]):
                        if pattern.search(full_text):
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
        
        logger.info(f"Found mentions of {len(stock_mentions)} stocks in news")
        return stock_mentions
    except Exception as e:
        logger.error(f"Error analyzing news: {e}")
        return {}

def generate_news_trading_signals(stock_mentions):
    """
    Generate trading signals based on news mentions and sentiment
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
    
    # Sort signals by confidence (highest first)
    trading_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return trading_signals

def update_news_data():
    """
    Update news data and generate trading signals with optimized
    processing and reduced latency
    """
    global news_data, last_news_update
    
    current_time = datetime.now()
    if (current_time - last_news_update).total_seconds() < NEWS_CHECK_INTERVAL:
        return
    
    try:
        # Get all tracked stocks as the universe
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
        
        # Fetch news - this is optimized inside the function using parallel fetching
        news_items = fetch_news_from_sources()
        
        # Only update if we got new news items
        if news_items:
            # Check if we have new items compared to previously fetched news
            existing_titles = set(item.get('title', '') for item in news_data.get("items", []))
            new_items = [item for item in news_items if item.get('title', '') not in existing_titles]
            
            if new_items:
                logger.info(f"Found {len(new_items)} new news items")
                
                # Analyze news for stock mentions efficiently
                stock_mentions = analyze_news_for_stocks(news_items, stock_universe)
                
                # Generate trading signals
                trading_signals = generate_news_trading_signals(stock_mentions)
                
                # Merge new items with existing items, keeping the most recent
                all_news = news_items + [
                    item for item in news_data.get("items", []) 
                    if item.get('title', '') not in set(ni.get('title', '') for ni in news_items)
                ]
                
                # Sort by timestamp and keep the most recent items (limit to 100)
                all_news.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
                all_news = all_news[:100]  # Keep only the 100 most recent items
                
                # Update news data
                news_data["items"] = all_news
                news_data["mentions"] = stock_mentions
                news_data["trading_signals"] = trading_signals
                news_data["last_updated"] = current_time
                
                # Update UI data store
                ui_data_store['news'] = {
                    'items': all_news[:10],  # Store the 10 most recent items
                    'mentions': stock_mentions,
                    'signals': trading_signals,
                    'last_updated': current_time.strftime('%H:%M:%S')
                }
                
                # Add stocks mentioned in news
                add_news_mentioned_stocks()
                
                # Try to execute trades if any signals were generated
                if trading_signals and broker_connected:
                    execute_news_based_trades()
            else:
                # Update the last_updated timestamp even if no new items
                news_data["last_updated"] = current_time
                ui_data_store['news']['last_updated'] = current_time.strftime('%H:%M:%S')
        
        last_news_update = current_time
        
    except Exception as e:
        logger.error(f"Error updating news data: {e}")
        last_news_update = current_time + timedelta(seconds=30)  # Delay next attempt

def add_news_mentioned_stocks():
    """Add stocks that are mentioned in news but not currently tracked"""
    if not news_data.get("mentions"):
        return
    
    # Get all mentioned stocks
    mentioned_stocks = set(news_data["mentions"].keys())
    
    # Get currently tracked stocks
    tracked_stocks = set(stocks_data.keys())
    
    # Find stocks to add (mentioned but not tracked)
    stocks_to_add = mentioned_stocks - tracked_stocks
    
    # Add each stock in parallel
    def add_stock_task(symbol):
        # Skip if it doesn't look like a valid stock symbol
        if not symbol.isalpha() or len(symbol) < 2:
            return
            
        logger.info(f"Adding stock {symbol} based on news mentions")
        add_stock(symbol, None, "NSE", "STOCK")
        
        # Try to fetch data immediately
        if broker_connected:
            request_id = batch_manager.add_request(
                "fetch_stock_data",
                {"symbol": symbol},
                callback=lambda result: update_stock_with_data(symbol, result),
                priority=3  # Medium priority
            )
    
    # Process in parallel
    for symbol in stocks_to_add:
        thread_pool.submit(add_stock_task, symbol)

def execute_news_based_trades():
    """Execute trades based on news analysis with refined execution strategy"""
    if not strategy_settings["NEWS_ENABLED"]:
        logger.info("News-based trading is disabled")
        return
    
    if not broker_connected:
        logger.warning("Cannot execute news-based trades: Not connected to broker")
        return
    
    # Check if we've hit the maximum trades for the day
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
        if stock not in stocks_data:
            logger.info(f"Adding stock {stock} based on news")
            add_stock(stock, None, "NSE", "STOCK")
            
            # Wait briefly for stock data to be loaded
            time.sleep(0.5)
        
        # Check if stock data is available
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

# ============ Trading Strategy ============
def should_enter_trade(option_key):
    """Determine if we should enter a trade for the given option with enhanced decision making."""
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
    signal = option_info.get("signal", 0)
    strength = option_info.get("strength", 0)
    
    # Get signal components for more detailed analysis
    signal_components = option_info.get("signal_components", {})
    
    # Check the market sentiment for the parent symbol
    sentiment = market_sentiment.get(parent_symbol, "NEUTRAL")
    overall_sentiment = market_sentiment.get("overall", "NEUTRAL")
    
    # Check PCR data
    pcr_value = pcr_data.get(parent_symbol, {}).get("current", 1.0)
    pcr_strength = pcr_data.get(parent_symbol, {}).get("strength", 0.0)
    
    # Check if there's a predicted strategy for this stock
    predicted_strategy = stocks_data.get(parent_symbol, {}).get("predicted_strategy")
    strategy_confidence = stocks_data.get(parent_symbol, {}).get("strategy_confidence", 0)
    
    # For a call option, we want positive signal; for a put, we want negative signal
    signal_aligned = (option_type == "CE" and signal > 0) or (option_type == "PE" and signal < 0)
    
    # For a call option, we want bullish sentiment; for a put, we want bearish sentiment
    sentiment_aligned = False
    if option_type == "CE":
        sentiment_aligned = "BULLISH" in sentiment
    else:  # PE
        sentiment_aligned = "BEARISH" in sentiment
    
    # For a call option, we want positive PCR strength; for a put, we want negative PCR strength
    pcr_aligned = (option_type == "CE" and pcr_strength > 0) or (option_type == "PE" and pcr_strength < 0)
    
    # Check news signals first
    if strategy_settings["NEWS_ENABLED"]:
        # Check for fresh news signals for this stock
        news_signals = []
        for s in news_data.get("trading_signals", []):
            if s.get("stock") == parent_symbol and not s.get("executed", False):
                signal_time = s.get("timestamp")
                if isinstance(signal_time, datetime) and (datetime.now() - signal_time).total_seconds() < 1800:  # 30 minutes
                    news_signals.append(s)
        
        for news_signal in news_signals:
            action = news_signal.get("action", "")
            confidence = news_signal.get("confidence", 0)
            
            if ((action == "BUY_CE" and option_type == "CE") or (action == "BUY_PE" and option_type == "PE")) and confidence > NEWS_CONFIDENCE_THRESHOLD:
                strategy_type = "NEWS"
                return True, strategy_type, confidence
    
    # If we have a predicted strategy with high confidence, use it
    if predicted_strategy and strategy_confidence > 0.7 and strategy_settings.get(f"{predicted_strategy}_ENABLED", False):
        # For CE options, we need a bullish signal; for PE options, we need a bearish signal
        strategy_signal_aligned = False
        
        # Check alignment based on option type and predicted strategy
        if option_type == "CE" and (
            predicted_strategy == "MOMENTUM" and signal > 1.5 or
            predicted_strategy == "SWING" and sentiment_aligned and signal > 1.0 or
            predicted_strategy == "SCALP" and signal > 2.0
        ):
            strategy_signal_aligned = True
        elif option_type == "PE" and (
            predicted_strategy == "MOMENTUM" and signal < -1.5 or
            predicted_strategy == "SWING" and sentiment_aligned and signal < -1.0 or
            predicted_strategy == "SCALP" and signal < -2.0
        ):
            strategy_signal_aligned = True
        
        if strategy_signal_aligned:
            min_strength = globals().get(f"MIN_SIGNAL_STRENGTH_{predicted_strategy}", 3.0)
            
            if strength >= min_strength:
                logger.info(f"Using predicted strategy {predicted_strategy} for {option_key} with {strategy_confidence:.2f} confidence")
                return True, predicted_strategy, strategy_confidence
    
    # If no predicted strategy or not qualified, fall back to normal strategy selection
    # Check each strategy with more detailed criteria
    
    # Check scalping conditions
    if strategy_settings["SCALP_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_SCALP:
        # Additional scalping-specific criteria
        rsi_component = signal_components.get("rsi", 0)
        bollinger_component = signal_components.get("bollinger", 0)
        
        # Scalping works better with RSI and Bollinger band signals
        if abs(rsi_component) + abs(bollinger_component) > 1.5:
            return True, "SCALP", strength / 10
    
    # Check momentum conditions
    if strategy_settings["MOMENTUM_ENABLED"] and signal_aligned and strength >= MIN_SIGNAL_STRENGTH_MOMENTUM:
        # Additional momentum-specific criteria
        momentum_component = signal_components.get("momentum", 0)
        trend_component = signal_components.get("trend", 0)
        
        # Momentum strategies need strong trend and momentum signals
        if abs(momentum_component) + abs(trend_component) > 2.0:
            return True, "MOMENTUM", strength / 10
    
    # Check swing trading conditions
    if strategy_settings["SWING_ENABLED"] and signal_aligned and sentiment_aligned and strength >= MIN_SIGNAL_STRENGTH_SWING:
        # Additional swing-specific criteria
        ema_component = signal_components.get("ema", 0)
        pcr_component = signal_components.get("pcr", 0)
        
        # Swing trades work better with aligned indicators
        swing_alignment = signal_aligned + sentiment_aligned + pcr_aligned
        
        # Need at least 2 aligned indicators for swing
        if swing_alignment >= 2 and abs(ema_component) > 0.8:
            return True, "SWING", strength / 10
    
    # No valid strategy found
    return False, None, 0

def enter_trade(option_key, strategy_type, trade_source="TECHNICAL"):
    """Enter a trade for the given option with optimized calculations."""
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
    option_type = option_info["option_type"]
    
    # Calculate position size based on risk management
    total_capital = trading_state.capital
    risk_amount = total_capital * (RISK_PER_TRADE / 100)
    
    # Adjust strategy-based parameters for better risk management
    if strategy_type == "SCALP":
        stop_loss_pct = 0.02   # 2.0% for scalping
        target_pct = 0.04      # 4.0% for scalping
    elif strategy_type == "MOMENTUM":
        stop_loss_pct = 0.025  # 2.5% for momentum
        target_pct = 0.07      # 7.0% for momentum
    elif strategy_type == "NEWS":
        stop_loss_pct = 0.03   # 3.0% for news
        target_pct = 0.12      # 12% for news
    else:  # SWING
        stop_loss_pct = 0.035  # 3.5% for swing
        target_pct = 0.10      # 10% for swing
    
    # Calculate stop loss amount - absolute value
    stop_loss_amount = current_price * stop_loss_pct
    
    # Calculate quantity based on risk per trade - ensure at least 1
    quantity = max(int(risk_amount / stop_loss_amount), 1)
    
    # Cap quantity to avoid too large positions - max 15% of capital
    max_quantity = int(total_capital * 0.15 / current_price)
    quantity = min(quantity, max_quantity)
    
    # Calculate actual stop loss and target with stock volatility adjustment
    volatility_factor = 1.0
    if parent_symbol in volatility_data:
        current_volatility = volatility_data[parent_symbol].get("current", 0)
        # Adjust SL and target based on volatility
        if current_volatility > 0.8:  # High volatility
            volatility_factor = 1.2    # Wider SL and target
        elif current_volatility < 0.3:  # Low volatility
            volatility_factor = 0.9    # Tighter SL and target
    
    # Apply volatility adjustment
    adjusted_sl_pct = stop_loss_pct * volatility_factor
    adjusted_target_pct = target_pct * volatility_factor
    
    # Calculate final stop loss and target prices
    if option_type == "CE":
        stop_loss = current_price * (1 - adjusted_sl_pct)
        target = current_price * (1 + adjusted_target_pct)
    else:  # PE
        stop_loss = current_price * (1 + adjusted_sl_pct)
        target = current_price * (1 - adjusted_target_pct)
    
    # Update trading state with thread safety
    with trading_state.lock:
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
        trading_state.last_sl_adjustment_price[option_key] = current_price
        
        # Store the parent stock price
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
    """Determine if we should exit a trade with optimized decision criteria."""
    if option_key not in options_data or not trading_state.active_trades.get(option_key, False):
        return False, None
    
    option_info = options_data[option_key]
    current_price = option_info["ltp"]
    
    if current_price is None or current_price <= 0:
        return False, None
    
    # Get trade data with thread safety using the new helper method
    trade_data = trading_state.get_trade_data(option_key)
    if not trade_data or not trade_data['active']:
        return False, None
        
    entry_price = trade_data['entry_price']
    stop_loss = trade_data['stop_loss']
    target = trade_data['target']
    option_type = option_info["option_type"]
    strategy_type = trade_data['strategy_type']
    entry_time = trade_data['entry_time']
    
    # Calculate time since entry
    current_time = datetime.now()
    time_since_entry = (current_time - entry_time).total_seconds() if entry_time else 0
    
    # Add minimum holding time - 30 seconds
    MINIMUM_HOLDING_TIME = 30  # seconds
    if time_since_entry < MINIMUM_HOLDING_TIME:
        # Don't exit trades that have been held for less than the minimum time
        # EXCEPT if it's a stop loss hit (to protect capital)
        if not ((option_type == "CE" and current_price <= stop_loss) or 
                (option_type == "PE" and current_price >= stop_loss)):
            return False, None
    
    # Calculate current P&L percentage
    if option_type == "CE":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:  # PE
        pnl_pct = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
    
    # Check for stop loss hit - highest priority
    if (option_type == "CE" and current_price <= stop_loss) or (option_type == "PE" and current_price >= stop_loss):
        return True, "Stop Loss"
    
    # Check for target hit
    if (option_type == "CE" and current_price >= target) or (option_type == "PE" and current_price <= target):
        return True, "Target"
    
    # Check for signal reversal
    signal = option_info["signal"]
    strength = option_info["strength"]
    
    # Check for strong reversal signals - require higher thresholds to prevent premature exits
    if ((option_type == "CE" and signal < -3 and strength > 7) or 
        (option_type == "PE" and signal > 3 and strength > 7)):
        # Only exit on reversal if we've held for at least 2 minutes
        if time_since_entry > 120:
            return True, "Strong Signal Reversal"
    
    # Check time-based exit
    holding_time_minutes = time_since_entry / 60
    
    # Different max holding times based on strategy
    if strategy_type == "SCALP" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_SCALP:
        return True, "Max Holding Time (Scalp)"
    elif strategy_type == "MOMENTUM" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_MOMENTUM:
        return True, "Max Holding Time (Momentum)"
    elif strategy_type == "NEWS" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_NEWS:
        return True, "Max Holding Time (News)"
    elif strategy_type == "SWING" and holding_time_minutes > MAX_POSITION_HOLDING_TIME_SWING:
        return True, "Max Holding Time (Swing)"
    
    # Check for deteriorating profits - only after minimum 3 minutes
    if time_since_entry > 180 and pnl_pct > 0 and trade_data['trailing_activated']:
        # Get maximum potential target
        max_target_pct = 10 if strategy_type == "SWING" else 6  # Higher for swing
        
        # If we've reached 75% of the way to max target and are still holding
        if pnl_pct > max_target_pct * 0.75:
            return True, "Taking Profits (Near Max)"
    
    # Check for extended unprofitable trade - only check after at least 3 minutes
    if pnl_pct < -1 and holding_time_minutes > 3:
        # Calculate time-weighted expectation
        time_weight = min(holding_time_minutes / 60, 1.0)  # Scale up to 1 hour
        
        # Time-weighted stop-loss - the longer we're underwater, the tighter the exit
        if pnl_pct < -2 * time_weight:
            # Only exit if we've held for at least 5 minutes
            if holding_time_minutes > 5:
                return True, "Time-weighted Exit (Not Performing)"
    
    return False, None

def update_dynamic_stop_loss(option_key):
    """
    Update stop loss dynamically with optimized algorithm that triggers
    after 10 points movement
    """
    if not trading_state.active_trades.get(option_key, False):
        return
    
    option_info = options_data.get(option_key)
    if not option_info:
        return
    
    current_price = option_info["ltp"]
    if current_price is None or current_price <= 0:
        return
    
    # Get trade data with thread safety using the helper method
    trade_data = trading_state.get_trade_data(option_key)
    if not trade_data or not trade_data['active']:
        return
    
    entry_price = trade_data['entry_price']
    current_stop_loss = trade_data['stop_loss']
    initial_stop_loss = trade_data['initial_stop_loss']
    option_type = option_info["option_type"]
    strategy_type = trade_data['strategy_type']
    
    # Define movement threshold (10 points)
    MOVEMENT_THRESHOLD = 10
    
    # Get last adjustment price with thread safety
    last_adjustment_price = trading_state.last_sl_adjustment_price.get(option_key, entry_price)
    
    # Calculate price movement since last adjustment
    price_movement = abs(current_price - last_adjustment_price)
    
    # Only proceed if price has moved by at least 10 points since last adjustment
    if price_movement >= MOVEMENT_THRESHOLD:
        logger.info(f"Price moved {price_movement:.2f} points for {option_key} since last SL adjustment")
        
        if option_type == "CE":
            # For Call options
            if current_price > last_adjustment_price:
                # Price moved up - adjust SL upward
                # Move SL to protect 50% of the new gains
                new_stop_loss = entry_price + (current_price - entry_price) * 0.5
                
                # Only update if new SL is higher than current SL
                if new_stop_loss > current_stop_loss:
                    old_sl = current_stop_loss
                    
                    # Update with thread safety
                    with trading_state.lock:
                        trading_state.stop_loss[option_key] = new_stop_loss
                        trading_state.last_sl_adjustment_price[option_key] = current_price
                        trading_state.trailing_sl_activated[option_key] = True
                    
                    logger.info(f"{option_key} stop loss adjusted from {old_sl:.2f} to {new_stop_loss:.2f} after upward move of {price_movement:.2f} points")
            
            # Don't move SL down for CE options when price moves down
            # This would increase risk, not decrease it
        
        else:  # PE option
            # For Put options
            if current_price < last_adjustment_price:
                # Price moved down - adjust SL downward (higher price for PE)
                # Move SL to protect 50% of the new gains
                new_stop_loss = entry_price - (entry_price - current_price) * 0.5
                
                # Only update if new SL is lower than current SL
                if new_stop_loss < current_stop_loss:
                    old_sl = current_stop_loss
                    
                    # Update with thread safety
                    with trading_state.lock:
                        trading_state.stop_loss[option_key] = new_stop_loss
                        trading_state.last_sl_adjustment_price[option_key] = current_price
                        trading_state.trailing_sl_activated[option_key] = True
                    
                    logger.info(f"{option_key} stop loss adjusted from {old_sl:.2f} to {new_stop_loss:.2f} after downward move of {price_movement:.2f} points")
            
            # Don't move SL up for PE options when price moves up
            # This would increase risk, not decrease it

def update_dynamic_target(option_key):
    """
    Update target price dynamically with optimized calculations
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
    
    # Get trade data with thread safety
    trade_data = trading_state.get_trade_data(option_key)
    if not trade_data or not trade_data['active']:
        return
    
    entry_price = trade_data['entry_price']
    current_target = trade_data['target']
    option_type = option_info["option_type"]
    parent_symbol = option_info["parent_symbol"]
    strategy_type = trade_data['strategy_type']
    
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
        
        # Get current signal strength and trend
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
            
            # Only adjust target upward with thread safety
            if new_target > current_target:
                with trading_state.lock:
                    trading_state.target[option_key] = new_target
                logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
        else:  # PE
            # For put options, target is below entry price
            target_price_diff = (entry_price - current_price) * adjustment_factor
            new_target = current_price - target_price_diff
            
            # Only adjust target downward with thread safety
            if new_target < current_target:
                with trading_state.lock:
                    trading_state.target[option_key] = new_target
                logger.info(f"Adjusted target for {option_key} to {new_target:.2f} (factor: {adjustment_factor:.2f})")
                
    except Exception as e:
        logger.error(f"Error updating dynamic target for {option_key}: {e}")

def exit_trade(option_key, reason="Manual"):
    """Exit a trade with thread safety and comprehensive trade record."""
    # Check with thread safety if the trade is active
    with trading_state.lock:
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
    
    # Get trade data with thread safety
    trade_data = trading_state.get_trade_data(option_key)
    if not trade_data:
        return False
    
    # Extract necessary values
    entry_price = trade_data['entry_price']
    entry_time = trade_data['entry_time']
    quantity = trade_data['quantity']
    strategy_type = trade_data['strategy_type']
    trade_source = trade_data['trade_source']
    
    # Calculate P&L
    option_type = option_info["option_type"]
    parent_symbol = option_info.get("parent_symbol", "")
    strike = option_info.get("strike", "")
    
    # Calculate P&L in both absolute and percentage terms
    if option_type == "CE":
        pnl = (current_price - entry_price) * quantity
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    else:  # PE
        pnl = (entry_price - current_price) * quantity
        pnl_pct = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0
    
    # Calculate trade duration
    exit_time = datetime.now()
    duration_seconds = (exit_time - entry_time).total_seconds() if entry_time else 0
    duration_minutes = duration_seconds / 60
    
    # Create trade record
    trade_record = {
        'option_key': option_key,
        'parent_symbol': parent_symbol,
        'strategy_type': strategy_type,
        'trade_source': trade_source,
        'option_type': option_type,
        'strike': strike,
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
    
    # Update trading state with thread safety
    with trading_state.lock:
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
        trading_state.trades_history.append(trade_record)
        
        # Reset state values
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
    """Apply trading strategy with optimized batch processing."""
    # Skip if all strategies are disabled or broker not connected
    if (not any(strategy_settings.values())) or not broker_connected:
        return
    
    try:
        # Update news data and try to execute news-based trades first (if enabled)
        if strategy_settings["NEWS_ENABLED"]:
            if (datetime.now() - last_news_update).total_seconds() >= NEWS_CHECK_INTERVAL:
                thread_pool.submit(update_news_data)
        
        # Step 1: Check for trade exits first (to free up capital)
        active_trades = trading_state.get_active_trades()
        for option_key in active_trades:
            should_exit, reason = should_exit_trade(option_key)
            if should_exit:
                exit_trade(option_key, reason=reason)
            else:
                update_dynamic_stop_loss(option_key)
                update_dynamic_target(option_key)
        
        # Step 2: Check for trade entries with improved prioritization
        if trading_state.trades_today < MAX_TRADES_PER_DAY:
            # Get all options with their entry signals in a single pass
            potential_entries = []
            
            # First, check primary options for all stocks
            for symbol, stock_info in stocks_data.items():
                primary_ce = stock_info.get("primary_ce")
                primary_pe = stock_info.get("primary_pe")
                
                for option_key in [primary_ce, primary_pe]:
                    if option_key and option_key in options_data and not trading_state.active_trades.get(option_key, False):
                        should_enter, strategy_type, confidence = should_enter_trade(option_key)
                        if should_enter and strategy_type:
                            potential_entries.append((option_key, strategy_type, confidence))
            
            # Sort by confidence score, highest first
            potential_entries.sort(key=lambda x: x[2], reverse=True)
            
            # Take the top N entries, where N is limited by max trades per day
            remaining_trades = MAX_TRADES_PER_DAY - trading_state.trades_today
            
            for i, (option_key, strategy_type, confidence) in enumerate(potential_entries):
                if i < remaining_trades and confidence > 0.5:  # Only take high confidence trades
                    # Determine the source of the trade
                    trade_source = "TECHNICAL"
                    
                    # Execute the trade
                    enter_trade(option_key, strategy_type, trade_source)
                    
                    # Only process one trade at a time to avoid overwhelming the system
                    break
        
    except Exception as e:
        logger.error(f"Error in apply_trading_strategy: {e}")

# ============ Option Selection and Management ============
def find_and_add_options(symbol):
    """
    Find and add options for a stock using CSV-based lookup with caching
    for improved performance
    """
    # Check if the stock exists
    if symbol not in stocks_data:
        logger.warning(f"Cannot find options: Stock {symbol} not found")
        return {"CE": None, "PE": None}

    # Get the current price
    current_price = stocks_data[symbol].get("ltp")
    if current_price is None or current_price <= 0:
        logger.warning(f"Cannot find options: Invalid price for {symbol}")
        return {"CE": None, "PE": None}
    
    # Use the optimized function to update options from CSV
    return update_options_from_csv(symbol, current_price)

def update_options_from_csv(symbol, target_strike=None, csv_path=r"C:\Users\madhu\Pictures\ubuntu\stocks_and_options.csv"):
    """
    Update options data using CSV lookup with optimized matching algorithm
    """
    global tokens_and_symbols_df, stocks_data, options_data
    
    logger.info(f"Fetching options for {symbol} near strike {target_strike}")
    
    try:
        # Use cached CSV data if available
        if 'tokens_and_symbols_df' not in globals() or tokens_and_symbols_df is None:
            tokens_and_symbols_df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(tokens_and_symbols_df)} rows")
        
        if tokens_and_symbols_df.empty:
            logger.error("CSV data is empty or failed to load")
            return {"CE": None, "PE": None}
        
        # Get current price if target_strike is not provided
        if target_strike is None and symbol in stocks_data:
            target_strike = stocks_data[symbol].get("ltp")
            if not target_strike:
                logger.warning(f"No current price available for {symbol}")
                return {"CE": None, "PE": None}
        elif target_strike is None:
            logger.warning(f"No strike price provided and {symbol} not found in stocks_data")
            return {"CE": None, "PE": None}
        
        # Filter options for the specified stock
        stock_options = tokens_and_symbols_df[
            (tokens_and_symbols_df['exch_seg'] == 'NFO') & 
            (tokens_and_symbols_df['name'] == symbol)
        ]
        
        if stock_options.empty:
            logger.warning(f"No options found for {symbol}")
            return {"CE": None, "PE": None}
        
        # Extract unique expiry dates
        if 'expiry' in stock_options.columns:
            # Convert to datetime for proper sorting
            stock_options['expiry_date'] = pd.to_datetime(stock_options['expiry'], format='%d-%b-%y')
            
            # Sort by expiry date (ascending)
            expirations = stock_options['expiry'].unique()
            sorted_expirations = sorted(expirations, key=lambda x: pd.to_datetime(x, format='%d-%b-%y'))
            
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
            
            # Find closest strikes to the target
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
            
            # Process option results
            result = {"CE": None, "PE": None}
            
            # Process CE option
            if ce_result:
                # Create a unique option key
                try:
                    expiry_date = ce_result.get('expiry', '').replace('-', '')
                    # Handle month format (convert Apr to APR if needed)
                    month_match = re.search(r'([A-Za-z]+)', expiry_date)
                    if month_match:
                        month = month_match.group(1).upper()
                        expiry_date = expiry_date.replace(month_match.group(1), month)
                except:
                    # If regex fails, use expiry as is
                    expiry_date = ce_result.get('expiry', '')
                    
                strike = str(int(float(ce_result.get('strike', 0))))
                ce_key = f"{symbol}_{expiry_date}_{strike}_CE"
                
                # Add option to options_data if not already there
                if ce_key not in options_data:
                    options_data[ce_key] = {
                        "symbol": ce_result.get("symbol"),
                        "token": ce_result.get("token"),
                        "exchange": "NFO",
                        "parent_symbol": symbol,
                        "expiry": ce_result.get("expiry"),
                        "strike": ce_result.get("strike"),
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
                # Create a unique option key
                try:
                    expiry_date = pe_result.get('expiry', '').replace('-', '')
                    # Handle month format (convert Apr to APR if needed)
                    month_match = re.search(r'([A-Za-z]+)', expiry_date)
                    if month_match:
                        month = month_match.group(1).upper()
                        expiry_date = expiry_date.replace(month_match.group(1), month)
                except:
                    # If regex fails, use expiry as is
                    expiry_date = pe_result.get('expiry', '')
                    
                strike = str(int(float(pe_result.get('strike', 0))))
                pe_key = f"{symbol}_{expiry_date}_{strike}_PE"
                
                # Add option to options_data if not already there
                if pe_key not in options_data:
                    options_data[pe_key] = {
                        "symbol": pe_result.get("symbol"),
                        "token": pe_result.get("token"),
                        "exchange": "NFO",
                        "parent_symbol": symbol,
                        "expiry": pe_result.get("expiry"),
                        "strike": pe_result.get("strike"),
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
            
            # Schedule option data fetching
            if broker_connected:
                if result["CE"]:
                    request_id = batch_manager.add_request(
                        "fetch_option_data",
                        {"option_key": result["CE"]},
                        callback=lambda result, key=result["CE"]: update_option_with_data(key, result),
                        priority=2  # High priority
                    )
                
                if result["PE"]:
                    request_id = batch_manager.add_request(
                        "fetch_option_data",
                        {"option_key": result["PE"]},
                        callback=lambda result, key=result["PE"]: update_option_with_data(key, result),
                        priority=2  # High priority
                    )
            
            return result
        else:
            logger.error("CSV doesn't contain 'expiry' column")
            return {"CE": None, "PE": None}
    
    except Exception as e:
        logger.error(f"Error fetching options from CSV: {e}", exc_info=True)
        return {"CE": None, "PE": None}

def update_option_selection(force_update=False):
    """
    Update option selection for all stocks with efficient batching
    """
    global last_option_selection_update
    
    current_time = datetime.now()
    
    # Only update periodically unless forced
    if not force_update and (current_time - last_option_selection_update).total_seconds() < OPTION_AUTO_SELECT_INTERVAL:
        return
    
    logger.info("Updating option selection based on current stock prices")
    
    # Collect stocks that need option updates
    stocks_to_update = []
    
    for symbol, stock_info in stocks_data.items():
        current_price = stock_info.get("ltp")
        
        if current_price is None or current_price <= 0:
            continue
        
        # Check if options need to be updated
        update_needed = (
            stock_info.get("primary_ce") is None or 
            stock_info.get("primary_pe") is None or
            _should_update_options(symbol, current_price)
        )
        
        if update_needed:
            stocks_to_update.append((symbol, current_price))
    
    # Define a function to update options for a single stock
    def update_stock_options(args):
        symbol, price = args
        find_and_add_options(symbol)
    
    # Submit option update tasks to thread pool
    if stocks_to_update:
        # Process in batches for better control
        batch_size = 5  # Update 5 stocks at a time
        for i in range(0, len(stocks_to_update), batch_size):
            batch = stocks_to_update[i:min(i+batch_size, len(stocks_to_update))]
            
            # Submit tasks for this batch
            futures = []
            for stock in batch:
                future = thread_pool.submit(update_stock_options, stock)
                futures.append(future)
            
            # Wait for batch to complete before processing next batch
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Get result to catch any exceptions
                except Exception as e:
                    logger.error(f"Error updating options: {e}")
    
    # Update timestamp
    last_option_selection_update = current_time

def _should_update_options(symbol, current_price):
    """
    Determine if options should be updated with optimized calculations
    """
    stock_info = stocks_data.get(symbol, {})
    
    # Check for primary options
    primary_ce = stock_info.get("primary_ce")
    primary_pe = stock_info.get("primary_pe")
    
    if not primary_ce or not primary_pe:
        return True
    
    # Check if options exist in options_data
    if (primary_ce not in options_data or 
        primary_pe not in options_data):
        return True
    
    # Check strike proximity
    ce_details = options_data.get(primary_ce, {})
    pe_details = options_data.get(primary_pe, {})
    
    ce_strike = float(ce_details.get('strike', 0))
    pe_strike = float(pe_details.get('strike', 0))
    
    # Determine appropriate strike interval
    if current_price < 100:
        strike_interval = 5
    elif current_price < 1000:
        strike_interval = 10
    elif current_price < 10000:
        strike_interval = 50
    else:
        strike_interval = 100
    
    # Calculate ATM strike
    atm_strike = round(current_price / strike_interval) * strike_interval
    
    # Check if current options are too far from ATM
    ce_distance = abs(ce_strike - atm_strike)
    pe_distance = abs(pe_strike - atm_strike)
    
    if ce_distance > strike_interval * 2 or pe_distance > strike_interval * 2:
        return True
    
    # Check if options are using fallback data
    if (ce_details.get("using_fallback", False) or 
        pe_details.get("using_fallback", False)):
        return True
    
    return False

def load_historical_data(symbol, period="1mo", force_refresh=False):
    """Load historical data with improved caching"""
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
            logger.warning(f"No price column in fetched data for {symbol}")
            return False
            
        valid_prices = history_df['price'].dropna()
        if len(valid_prices) < 30:  # Require at least 30 valid price points for better S/R
            logger.warning(f"Insufficient valid price data for {symbol}: {len(valid_prices)} points")
            return False
            
        # Store the data with thread safety
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
        logger.error(f"Error loading historical data for {symbol}: {e}")
        return False

def fetch_history_from_yahoo(symbol, period="3mo"):
    """
    Fetch historical data from Yahoo Finance with improved error handling
    """
    try:
        # Map symbols to Yahoo format
        if symbol.upper() in ["NIFTY"]:
            if symbol.upper() == "NIFTY":
                yahoo_symbol = "^NSEI"
            
        else:
            yahoo_symbol = f"{symbol}.NS"
        
        logger.info(f"Fetching history for {symbol} (Yahoo: {yahoo_symbol}) for period {period}")
        
        # Fetch data from Yahoo Finance with a maximum of 3 retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                history = yf.download(yahoo_symbol, period=period, progress=False, timeout=10)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} fetching history for {symbol}: {e}")
                    time.sleep(1)  # Short delay before retry
                else:
                    raise  # Re-raise on final attempt
        
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
        logger.error(f"Error fetching history for {symbol} from Yahoo Finance: {e}")
        return None

# ============ Maintenance ============
def cleanup_old_data():
    """
    Clean up old data to prevent memory bloat with improved efficiency
    """
    global last_cleanup, stocks_data, options_data, option_token_cache
    
    current_time = datetime.now()
    if (current_time - last_cleanup).total_seconds() < DATA_CLEANUP_INTERVAL:
        return
    
    logger.info("Running data cleanup...")
    
    try:
        # Use thread-safe operations with locks
        with price_history_lock:
            # Keep only the latest MAX_PRICE_HISTORY_POINTS data points for stocks
            for symbol, stock_info in stocks_data.items():
                if len(stock_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                    stock_info["price_history"] = stock_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
            
            # Keep only the latest MAX_PRICE_HISTORY_POINTS data points for options
            for option_key, option_info in options_data.items():
                if len(option_info["price_history"]) > MAX_PRICE_HISTORY_POINTS:
                    option_info["price_history"] = option_info["price_history"].tail(MAX_PRICE_HISTORY_POINTS)
        
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
        
        # Clean up old cached tokens (older than 24 hours)
        stale_tokens = []
        for key, data in option_token_cache.items():
            if 'timestamp' in data:
                age = (current_time - data['timestamp']).total_seconds()
                if age > 86400:  # 24 hours
                    stale_tokens.append(key)
        
        # Remove stale tokens
        for key in stale_tokens:
            del option_token_cache[key]
        
        if stale_tokens:
            logger.info(f"Removed {len(stale_tokens)} stale option tokens from cache")
            # Save updated token cache
            save_token_cache()
            
        # Save updated symbol cache
        save_symbol_cache()
    
    except Exception as e:
        logger.error(f"Error during data cleanup: {e}")
    
    last_cleanup = current_time
    logger.info("Data cleanup completed")

def check_day_rollover():
    """Check if trading day has changed and reset daily stats."""
    global trading_state
    
    current_date = datetime.now().date()
    if current_date != trading_state.trading_day:
        logger.info(f"New trading day detected: {current_date}. Resetting daily stats.")
        
        # Update with thread safety
        with trading_state.lock:
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
    Remove inactive stocks with thread safety
    """
    global stocks_data
    
    try:
        # Identify stocks to remove
        inactive_stocks = []
        
        for symbol in list(stocks_data.keys()):
            if should_remove_stock(symbol):
                inactive_stocks.append(symbol)
        
        # Remove each inactive stock
        for symbol in inactive_stocks:
            remove_stock(symbol)
            
        if inactive_stocks:
            logger.info(f"Removed {len(inactive_stocks)} inactive stocks")
    
    except Exception as e:
        logger.error(f"Error cleaning up inactive stocks: {e}")

def update_ui_data_store_efficiently():
    """Update UI data store with minimal processing for faster updates"""
    global ui_data_store
    
    try:
        # Create a centralized data store for UI updates in a format optimized for the Dash callbacks
        updated_ui_data = {
            'connection': {
                'status': 'connected' if broker_connected else 'disconnected',
                'message': broker_error_message or '',
                'last_connection': last_connection_time.strftime('%H:%M:%S') if last_connection_time else 'Never'
            },
            'stocks': ui_data_store.get('stocks', {}),
            'options': ui_data_store.get('options', {}),
            'pcr': ui_data_store.get('pcr', {}),
            'sentiment': market_sentiment.copy(),
            'predicted_strategies': ui_data_store.get('predicted_strategies', {}),
            'news': ui_data_store.get('news', {}),
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
        
        # Update PCR data
        for symbol, data in pcr_data.items():
            if symbol in stocks_data:
                updated_ui_data['pcr'][symbol] = {
                    'current': data.get('current', 1.0),
                    'trend': data.get('trend', 'NEUTRAL'),
                    'strength': data.get('strength', 0)
                }
        
        # Update the ui_data_store with the refreshed data
        ui_data_store.update(updated_ui_data)
        
    except Exception as e:
        logger.error(f"Error updating UI data store: {e}")

# ============ Main Data Thread Function ============
def fetch_data_periodically():
    """Main function to fetch data periodically with intelligent batching"""
    global dashboard_initialized, data_thread_started, broker_error_message, broker_connection_retry_time
    
    # Mark that the data thread has started
    data_thread_started = True
    logger.info("Data fetching thread started")
    
    # Load cached data at startup
    load_symbol_cache()
    load_token_cache()
    
    # Initialize with default stocks
    for stock in DEFAULT_STOCKS:
        add_stock(stock["symbol"], stock["token"], stock["exchange"], stock["type"])
    
    # Mark dashboard as initialized
    dashboard_initialized = True
    
    # Initialize timestamps for periodic tasks
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
            
            # Use the optimized data fetching function if connected
            if broker_connected:
                # This is the key function that efficiently fetches all data
                fetch_all_data_efficiently()
            
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
            
            # Wait before next update - maintain an exact 1-second cycle
            cycle_end_time = datetime.now()
            elapsed = (cycle_end_time - current_time).total_seconds()
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)  # Sleep to maintain exactly 1-second cycle
            
        except Exception as e:
            logger.error(f"Error in fetch_data_periodically: {e}", exc_info=True)
            time.sleep(1)  # Sleep before retry on error

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

# Define custom CSS for smooth transitions and animations
app_css = '''
/* Base transitions for all updating elements */
.smooth-transition {
    transition: all 0.7s ease-in-out;
}

/* Price value transitions with flash animations */
.price-value {
    position: relative;
    transition: color 0.5s ease-out, background-color 0.5s ease-out;
    border-radius: 4px;
    will-change: color, background-color;
}

.price-change {
    transition: color 0.6s ease-out;
    will-change: color;
}

/* Flash animations for price changes */
@keyframes flash-green {
    0% { background-color: rgba(40, 167, 69, 0); }
    20% { background-color: rgba(40, 167, 69, 0.3); }
    100% { background-color: rgba(40, 167, 69, 0); }
}

@keyframes flash-red {
    0% { background-color: rgba(220, 53, 69, 0); }
    20% { background-color: rgba(220, 53, 69, 0.3); }
    100% { background-color: rgba(220, 53, 69, 0); }
}

.highlight-positive {
    animation: flash-green 2s ease-out;
}

.highlight-negative {
    animation: flash-red 2s ease-out;
}

/* Badge transitions */
.badge {
    transition: background-color 0.5s ease-out, opacity 0.5s ease-out;
}

/* Card and container transitions */
.card, .card-body {
    transition: border-color 0.5s ease-out, box-shadow 0.5s ease-out;
}

/* Progress bar transitions */
.progress-bar {
    transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Stock card transition when data changes */
.stock-data-changed {
    box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
    transition: box-shadow 0.5s ease-out;
}

/* Reduce animation for users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    .smooth-transition, .price-value, .price-change, .badge, .card, .card-body, .progress-bar {
        transition: none;
    }
    
    .highlight-positive, .highlight-negative {
        animation: none;
    }
}
/* For numeric values that change frequently */
.numeric-value {
    transition: color 0.7s ease-in-out, 
                background-color 0.7s ease-in-out;
}

/* For badges that change state */
.badge {
    transition: background-color 0.5s ease-in-out, 
                color 0.5s ease-in-out;
}

/* For progress bars */
.progress-bar {
    transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}

/* For cards and containers */
.card, .card-body {
    transition: border-color 0.5s ease-in-out;
}

/* Brief highlight effect for significant changes */
@keyframes highlight-green {
    0% { background-color: rgba(40, 167, 69, 0.3); }
    100% { background-color: transparent; }
}

@keyframes highlight-red {
    0% { background-color: rgba(220, 53, 69, 0.3); }
    100% { background-color: transparent; }
}

.highlight-positive {
    animation: highlight-green 1.5s ease-out;
}

.highlight-negative {
    animation: highlight-red 1.5s ease-out;
}
'''

# Initialize Dash app with custom settings and CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY], 
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Options Trading Dashboard"

# Override the default index string to include custom CSS
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

def create_news_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Latest Market News", className="text-primary d-inline me-2 mb-0"),
            html.Span(id="news-last-updated", className="badge bg-info ms-2 small")
        ], style=custom_css["header"]),
        dbc.CardBody(
            id="news-container", 
            className="px-4 py-3",
            style={"max-height": "300px", "overflow-y": "auto"}
        )
    ], 
    style=custom_css["card"],
    className="mb-3 border-primary"
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
                        html.Span(id={"type": "stock-price", "index": symbol}, className="fs-4 text-light smooth-transition price-change")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Change: ", className="text-muted me-2"),
                        html.Span(id={"type": "stock-change", "index": symbol}, className="text-light smooth-transition price-change")
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
                    ], className="mb-1"),
                ], width=4),
                
                dbc.Col([
                    html.Div(id={"type": "stock-last-update", "index": symbol}, className="text-muted small text-end smooth-transition"),
                    html.Div(id={"type": "strategy-prediction", "index": symbol}, className="text-end small mt-2 smooth-transition")
                ], width=4)
            ], className="mb-3"),
            
            # Options Section - CE and PE side by side
            dbc.Row([
                # Call option
                dbc.Col([
                    html.H6("CALL OPTION", className="text-success mb-2 text-center"),
                    html.Div([
                        html.Span("Strike: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-strike", "index": symbol}, className="text-warning smooth-transition")
                    ], className="text-center mb-1"),
                    html.Div([
                        html.Span("LTP: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-price", "index": symbol}, className="fs-5 fw-bold smooth-transition price-change")
                    ], className="text-center mb-2"),
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-ce-signal", "index": symbol}, className="badge smooth-transition")
                    ], className="text-center mb-1"),
                    dbc.Progress(id={"type": "option-ce-strength", "index": symbol}, className="mb-2 smooth-transition", style={"height": "6px"}),
                    html.Div(id={"type": "option-ce-trade-status", "index": symbol}, className="text-center small smooth-transition")
                ], width=6, className="border-end"),
                
                # Put option
                dbc.Col([
                    html.H6("PUT OPTION", className="text-danger mb-2 text-center"),
                    html.Div([
                        html.Span("Strike: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-strike", "index": symbol}, className="text-warning smooth-transition")
                    ], className="text-center mb-1"),
                    html.Div([
                        html.Span("LTP: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-price", "index": symbol}, className="fs-5 fw-bold smooth-transition price-change")
                    ], className="text-center mb-2"),
                    html.Div([
                        html.Span("Signal: ", className="text-muted me-1"),
                        html.Span(id={"type": "option-pe-signal", "index": symbol}, className="badge smooth-transition")
                    ], className="text-center mb-1"),
                    dbc.Progress(id={"type": "option-pe-strength", "index": symbol}, className="mb-2 smooth-transition", style={"height": "6px"}),
                    html.Div(id={"type": "option-pe-trade-status", "index": symbol}, className="text-center small smooth-transition")
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
                ], width=3),
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
                ], width=3),
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
                ], width=3),
                dbc.Col([
                    dbc.Switch(
                        id="news-strategy-toggle",
                        label="NEWS Strategy",
                        value=strategy_settings["NEWS_ENABLED"],
                        className="mb-2"
                    ),
                    html.Div([
                        dbc.Progress(
                            id="news-progress", 
                            value=0, 
                            color="warning", 
                            className="mb-1", 
                            style={"height": "5px"}
                        ),
                        html.Span(f"Signal Threshold: {MIN_SIGNAL_STRENGTH_NEWS}", className="text-light small")
                    ], className="mb-1")
                ], width=3),
            ]),
            html.Div(id="strategy-status", className="text-center mt-2 text-info")
        ], className="px-4 py-3")
    ], 
    style=custom_css["card"],
    className="mb-3 border-warning"
    )

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
            
            # News section
            dbc.Row([
                dbc.Col([create_news_card()], width=12)
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

# Set the layout
app.layout = create_layout()

# ============ Dashboard Callbacks ============
@app.callback(
    Output('ui-data-store', 'data'),
    [Input('fast-interval', 'n_intervals')]
)
def update_ui_data_store(n_intervals):
    """Provide data to UI components with efficient updates"""
    return ui_data_store

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

# News display callback
@app.callback(
    [Output("news-container", "children"),
     Output("news-last-updated", "children")],
    [Input('medium-interval', 'n_intervals'),
     Input('ui-data-store', 'data')]
)
def update_news_display(n_intervals, data):
    if not data or 'news' not in data:
        return "No market news available.", "Not updated"
    
    news_data = data['news']
    news_items = news_data.get('items', [])
    last_updated = news_data.get('last_updated', "Not updated")
    
    if not news_items:
        return "No market news available.", last_updated
    
    # Create news item cards
    news_cards = []
    
    for item in news_items[:10]:  # Display the 10 most recent items
        title = item.get('title', 'No title')
        source = item.get('source', 'Unknown')
        timestamp = item.get('timestamp')
        
        # Format timestamp
        if isinstance(timestamp, str):
            time_str = timestamp
        elif isinstance(timestamp, datetime):
            # Calculate how recent the news is
            time_diff = datetime.now() - timestamp
            if time_diff.total_seconds() < 3600:  # Less than an hour
                time_str = f"{int(time_diff.total_seconds() / 60)} minutes ago"
            elif time_diff.total_seconds() < 86400:  # Less than a day
                time_str = f"{int(time_diff.total_seconds() / 3600)} hours ago"
            else:
                time_str = timestamp.strftime("%d-%b %H:%M")
        else:
            time_str = "Unknown time"
        
        # Create card with badge for recency
        is_new = isinstance(timestamp, datetime) and (datetime.now() - timestamp).total_seconds() < 600  # 10 minutes
        new_badge = html.Span("NEW", className="badge bg-danger ms-2") if is_new else ""
        
        card = dbc.Card([
            dbc.CardBody([
                html.H5([title, new_badge], className="mb-1"),
                html.Div([
                    html.Span(source, className="text-muted me-2"),
                    html.Span(time_str, className="text-muted small")
                ], className="d-flex justify-content-between")
            ], className="p-3")
        ], className="mb-2", style=custom_css["card_alt"])
        
        news_cards.append(card)
    
    # Add news trading signals if available
    signals = news_data.get('signals', [])
    
    if signals:
        signals_section = html.Div([
            html.H5("Trading Signals from News", className="mt-3 mb-2 text-warning"),
            html.Div([
                html.Div([
                    html.Strong(f"{s['stock']}: ", className="me-1"),
                    html.Span(f"{s['action']} - {s['latest_news'][:50]}...", 
                             className="text-success" if "BUY_CE" in s['action'] else "text-danger")
                ], className="mb-1") for s in signals[:3]  # Show only the top 3 signals
            ])
        ])
        
        news_cards.append(signals_section)
    
    # Format the last updated text
    if isinstance(last_updated, str):
        last_updated_text = last_updated
    else:
        last_updated_text = "Updated every 30 seconds"
    
    return news_cards, last_updated_text

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

