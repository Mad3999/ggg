import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import asyncio
from dhanhq import DhanContext, MarketFeed

# Hardcoded credentials
DHAN_CLIENT_ID = "1101812495"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ1MDQzNzI5LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMTgxMjQ5NSJ9.VlLATxaAJxKUH6GGHHZfDw8wjxQufHzqTvyU6UckYCcRbj6dXpF0TehhmwMndIPWEf4B00N7GP7zGFzQ9XeFvQ"

# Debug flag - set to True for detailed logging
DEBUG = True

class MarketFeedApp:
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title("Dhan Market Feed")
        self.root.geometry("1000x700")
        self.csv_path = csv_path
        
        # Load data - fix for mixed types warning
        self.df = pd.read_csv(csv_path, low_memory=False)
        
        # Variables
        self.search_var = tk.StringVar()
        self.instrument_type_var = tk.StringVar(value="EQUITY")
        self.exchange_var = tk.StringVar(value="NSE")
        self.selected_instruments = []
        self.feed_running = False
        self.feed_thread = None
        self.stop_event = threading.Event()
        
        # Create and layout widgets
        self.create_widgets()
        self.layout_widgets()
        
        # Initialize market feed data storage
        self.market_data = {}
        
        # Security ID to Symbol/Exchange mapping
        self.security_map = {}
        
        # Previous close values for calculating percent change
        self.prev_close_map = {}
        
        # Debug log
        self.log("Application started")
    
    def log(self, message):
        """Log messages to debug console"""
        if DEBUG:
            timestamp = time.strftime('%H:%M:%S')
            self.debug_text.configure(state="normal")
            self.debug_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.debug_text.see(tk.END)
            self.debug_text.configure(state="disabled")
            print(f"[{timestamp}] {message}")
    
    def create_widgets(self):
        # Create a paned window to allow resizing sections
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        
        # Frame for search controls
        self.search_frame = ttk.LabelFrame(self.paned_window, text="Search Instruments")
        
        # Search input
        ttk.Label(self.search_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.search_symbol)
        self.search_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Instrument type selection
        ttk.Label(self.search_frame, text="Type:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.type_combo = ttk.Combobox(self.search_frame, textvariable=self.instrument_type_var, 
                                       values=["EQUITY", "INDEX"], width=10)
        self.type_combo.grid(row=0, column=4, padx=5, pady=5)
        
        # Exchange selection
        ttk.Label(self.search_frame, text="Exchange:").grid(row=0, column=5, padx=5, pady=5, sticky="w")
        self.exchange_combo = ttk.Combobox(self.search_frame, textvariable=self.exchange_var, 
                                          values=["NSE", "BSE"], width=10)
        self.exchange_combo.grid(row=0, column=6, padx=5, pady=5)
        
        # Results and selection area in tabs
        self.tab_control = ttk.Notebook(self.paned_window)
        
        # Tab for search results
        self.results_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.results_tab, text="Search Results")
        
        # Search results tree view
        self.results_tree = ttk.Treeview(self.results_tab, columns=("Exchange", "Security ID", "Symbol", "Type"), 
                                        show="headings", selectmode="extended")
        self.results_tree.heading("Exchange", text="Exchange")
        self.results_tree.heading("Security ID", text="Security ID")
        self.results_tree.heading("Symbol", text="Symbol")
        self.results_tree.heading("Type", text="Type")
        
        self.results_tree.column("Exchange", width=80)
        self.results_tree.column("Security ID", width=100)
        self.results_tree.column("Symbol", width=150)
        self.results_tree.column("Type", width=80)
        
        # Scrollbar for results
        self.results_scroll = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=self.results_scroll.set)
        
        # Button frame for search results
        self.results_button_frame = ttk.Frame(self.results_tab)
        self.add_button = ttk.Button(self.results_button_frame, text="Add Selected", command=self.add_selected)
        
        # Tab for selected instruments
        self.selected_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.selected_tab, text="Selected Instruments")
        
        # Selected instruments tree view
        self.selected_tree = ttk.Treeview(self.selected_tab, columns=("Exchange", "Security ID", "Symbol", "Type"), 
                                         show="headings", selectmode="extended")
        self.selected_tree.heading("Exchange", text="Exchange")
        self.selected_tree.heading("Security ID", text="Security ID")
        self.selected_tree.heading("Symbol", text="Symbol")
        self.selected_tree.heading("Type", text="Type")
        
        self.selected_tree.column("Exchange", width=80)
        self.selected_tree.column("Security ID", width=100)
        self.selected_tree.column("Symbol", width=150)
        self.selected_tree.column("Type", width=80)
        
        # Scrollbar for selected instruments
        self.selected_scroll = ttk.Scrollbar(self.selected_tab, orient="vertical", command=self.selected_tree.yview)
        self.selected_tree.configure(yscrollcommand=self.selected_scroll.set)
        
        # Button frame for selected instruments
        self.selected_button_frame = ttk.Frame(self.selected_tab)
        self.remove_button = ttk.Button(self.selected_button_frame, text="Remove Selected", command=self.remove_selected)
        self.clear_button = ttk.Button(self.selected_button_frame, text="Clear All", command=self.clear_selected)
        
        # Market Data frame
        self.market_frame = ttk.LabelFrame(self.paned_window, text="Market Data")
        
        # Market data tree view
        self.market_tree = ttk.Treeview(self.market_frame, 
                                      columns=("Symbol", "Exchange", "LTP", "Change", "Updated"), 
                                      show="headings")
        self.market_tree.heading("Symbol", text="Symbol")
        self.market_tree.heading("Exchange", text="Exchange")
        self.market_tree.heading("LTP", text="LTP")
        self.market_tree.heading("Change", text="Change %")
        self.market_tree.heading("Updated", text="Last Updated")
        
        self.market_tree.column("Symbol", width=150)
        self.market_tree.column("Exchange", width=80)
        self.market_tree.column("LTP", width=100)
        self.market_tree.column("Change", width=100)
        self.market_tree.column("Updated", width=150)
        
        # Scrollbar for market data
        self.market_scroll = ttk.Scrollbar(self.market_frame, orient="vertical", command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=self.market_scroll.set)
        
        # Feed control buttons
        self.feed_frame = ttk.Frame(self.market_frame)
        self.start_button = ttk.Button(self.feed_frame, text="Start Feed", command=self.start_feed)
        self.stop_button = ttk.Button(self.feed_frame, text="Stop Feed", command=self.stop_feed, state="disabled")
        
        # Debug console
        self.debug_frame = ttk.LabelFrame(self.paned_window, text="Debug Console")
        self.debug_text = scrolledtext.ScrolledText(self.debug_frame, height=8, wrap=tk.WORD)
        self.debug_text.configure(state="disabled")
        self.clear_debug_button = ttk.Button(self.debug_frame, text="Clear Log", 
                                            command=lambda: self.debug_text.delete(1.0, tk.END))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
    
    def layout_widgets(self):
        # Configure root grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        
        # Add paned window to root
        self.paned_window.grid(row=0, column=0, sticky="nsew")
        
        # Add frames to paned window
        self.paned_window.add(self.search_frame, weight=0)
        self.paned_window.add(self.tab_control, weight=1)
        self.paned_window.add(self.market_frame, weight=2)
        self.paned_window.add(self.debug_frame, weight=1)
        
        # Configure search frame
        self.search_frame.columnconfigure(1, weight=1)
        
        # Configure results tab
        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=0)
        
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_scroll.grid(row=0, column=1, sticky="ns")
        
        self.results_button_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.add_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Configure selected tab
        self.selected_tab.columnconfigure(0, weight=1)
        self.selected_tab.rowconfigure(0, weight=1)
        self.selected_tab.rowconfigure(1, weight=0)
        
        self.selected_tree.grid(row=0, column=0, sticky="nsew")
        self.selected_scroll.grid(row=0, column=1, sticky="ns")
        
        self.selected_button_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Configure market frame
        self.market_frame.columnconfigure(0, weight=1)
        self.market_frame.rowconfigure(0, weight=1)
        self.market_frame.rowconfigure(1, weight=0)
        
        self.market_tree.grid(row=0, column=0, sticky="nsew")
        self.market_scroll.grid(row=0, column=1, sticky="ns")
        
        self.feed_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Configure debug frame
        self.debug_frame.columnconfigure(0, weight=1)
        self.debug_frame.rowconfigure(0, weight=1)
        self.debug_frame.rowconfigure(1, weight=0)
        
        self.debug_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.clear_debug_button.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        
        # Place status bar
        self.status_bar.grid(row=1, column=0, sticky="ew")
    
    def search_symbol(self):
        # Get search parameters
        symbol = self.search_var.get().strip().upper()
        instrument_type = self.instrument_type_var.get()
        exchange = self.exchange_var.get()
        
        if not symbol:
            messagebox.showwarning("Warning", "Please enter a symbol to search")
            return
        
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Filter dataframe
        try:
            # For partial matching
            filtered_df = self.df[
                (self.df['SEM_TRADING_SYMBOL'].str.contains(symbol)) & 
                (self.df['SEM_INSTRUMENT_NAME'] == instrument_type)
            ]
            
            if exchange != "ALL":
                filtered_df = filtered_df[filtered_df['SEM_EXM_EXCH_ID'] == exchange]
            
            # Display results
            if filtered_df.empty:
                self.status_var.set(f"No results found for {symbol}")
                self.log(f"No results found for {symbol}")
            else:
                count = 0
                for index, row in filtered_df.iterrows():
                    self.results_tree.insert("", "end", values=(
                        row['SEM_EXM_EXCH_ID'],
                        row['SEM_SMST_SECURITY_ID'],
                        row['SEM_TRADING_SYMBOL'],
                        row['SEM_INSTRUMENT_NAME']
                    ))
                    count += 1
                
                self.status_var.set(f"Found {count} results for {symbol}")
                self.log(f"Found {count} results for {symbol}")
                
                # Switch to results tab
                self.tab_control.select(0)
                
        except Exception as e:
            self.log(f"Error searching: {str(e)}")
            messagebox.showerror("Error", f"Search error: {str(e)}")
    
    def add_selected(self):
        # Get selected items from results
        selected_items = self.results_tree.selection()
        
        if not selected_items:
            messagebox.showwarning("Warning", "Please select instruments to add")
            return
        
        # Add selected items to the selected list
        added_count = 0
        for item in selected_items:
            values = self.results_tree.item(item, "values")
            
            # Check if already in selected list
            already_selected = False
            for selected in self.selected_tree.get_children():
                if self.selected_tree.item(selected, "values")[1] == values[1]:
                    already_selected = True
                    break
            
            if not already_selected:
                self.selected_tree.insert("", "end", values=values)
                added_count += 1
                self.log(f"Added: {values[2]} ({values[0]}:{values[1]})")
        
        self.status_var.set(f"Added {added_count} instruments")
        
        if added_count > 0:
            # Switch to selected tab
            self.tab_control.select(1)
    
    def remove_selected(self):
        # Remove selected items from selected list
        selected_items = self.selected_tree.selection()
        
        if not selected_items:
            messagebox.showwarning("Warning", "Please select instruments to remove")
            return
        
        removed_count = 0
        for item in selected_items:
            values = self.selected_tree.item(item, "values")
            self.selected_tree.delete(item)
            removed_count += 1
            self.log(f"Removed: {values[2]} ({values[0]}:{values[1]})")
        
        self.status_var.set(f"Removed {removed_count} instruments")
    
    def clear_selected(self):
        # Count items
        count = len(self.selected_tree.get_children())
        
        # Clear all selected instruments
        for item in self.selected_tree.get_children():
            self.selected_tree.delete(item)
        
        self.log(f"Cleared all {count} selected instruments")
        self.status_var.set(f"Cleared all {count} selected instruments")
    
    def start_feed(self):
        # Get all selected instruments
        instruments = []
        
        # Clear existing market data display
        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        
        self.market_data = {}
        self.security_map = {}  # Reset the security map
        self.prev_close_map = {}  # Reset previous close map
        
        for item in self.selected_tree.get_children():
            values = self.selected_tree.item(item, "values")
            exchange = values[0]
            security_id = values[1]
            symbol = values[2]
            instr_type = values[3]
            
            # Add to instruments list - use proper codes for the API
            if exchange == "NSE":
                exchange_code = MarketFeed.NSE
            else:
                exchange_code = MarketFeed.BSE
            
            # Build a map of security_id to symbol and exchange
            self.security_map[security_id] = {
                'symbol': symbol,
                'exchange': exchange,
                'type': instr_type
            }
            
            instruments.append((exchange_code, security_id, MarketFeed.Ticker))
            
            # Initialize market data for this instrument
            key = f"{exchange}:{security_id}"
            self.market_data[key] = {
                'symbol': symbol,
                'exchange': exchange,
                'ltp': 'N/A',
                'change': 'N/A',
                'updated': 'Never'
            }
            
            # Add to market data tree
            self.market_tree.insert("", "end", iid=key, values=(
                symbol, exchange, 'N/A', 'N/A', 'Never'
            ))
            
            self.log(f"Added to market watch: {symbol} ({exchange}:{security_id})")
        
        if not instruments:
            messagebox.showwarning("Warning", "Please select instruments to monitor")
            return
        
        # Reset stop event
        self.stop_event.clear()
        
        # Update UI state
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting market feed...")
        
        # Start market feed in a separate thread
        self.feed_thread = threading.Thread(target=self.run_market_feed, args=(instruments,))
        self.feed_thread.daemon = True
        self.feed_thread.start()
        
        self.log(f"Started market feed thread for {len(instruments)} instruments")
    
    def stop_feed(self):
        if self.feed_thread and self.feed_thread.is_alive():
            self.status_var.set("Stopping market feed...")
            self.log("Stopping market feed...")
            self.stop_event.set()
            
            # Update UI state
            self.stop_button.configure(state="disabled")
            
            # Wait for thread to finish
            self.feed_thread.join(timeout=5.0)
            
            self.status_var.set("Market feed stopped")
            self.start_button.configure(state="normal")
            self.log("Market feed stopped")
    
    def run_market_feed(self, instruments):
        try:
            # Set up asyncio event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.log("Created new asyncio event loop for thread")
            
            # Initialize DhanContext
            dhan_context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
            
            # Set up market feed
            version = "v2"
            
            # Update status
            self.root.after(0, lambda: self.status_var.set(f"Connected to market feed for {len(instruments)} instruments"))
            self.log(f"Initialized market feed for {len(instruments)} instruments")
            
            # Log the instruments being subscribed to
            for instrument in instruments:
                exchange, security_id, feed_type = instrument
                # Convert the exchange code back to a string for logging
                exchange_name = "NSE" if exchange == MarketFeed.NSE else "BSE" if exchange == MarketFeed.BSE else "Unknown"
                self.log(f"Subscribing to: {exchange_name}:{security_id}")
            
            # Start the market feed
            try:
                data = MarketFeed(dhan_context, instruments, version)
                self.log("MarketFeed instance created successfully")
            except Exception as e:
                self.log(f"Error creating MarketFeed: {str(e)}")
                raise
            
            # Process data loop
            update_count = 0
            while not self.stop_event.is_set():
                try:
                    # Handle websocket connection properly with asyncio
                    if hasattr(data, 'run_forever_async'):
                        # If there's an async version, use it
                        loop.run_until_complete(data.run_forever_async())
                    else:
                        # Regular version but in the context of an event loop
                        loop.run_until_complete(asyncio.sleep(0.1))  # Small pause to let asyncio process events
                        data.run_forever()
                    
                    response = data.get_data()
                    
                    if response:
                        update_count += 1
                        # Log raw response periodically to avoid flooding
                        self.log(f"Received data: {json.dumps(response)}")
                        
                        # PROCESS THE RESPONSE BASED ON ACTUAL FORMAT
                        try:
                            # Fields from the actual response format
                            message_type = response.get('type', 'Unknown')
                            exchange_segment = response.get('exchange_segment', None)
                            security_id = str(response.get('security_id', 'Unknown'))
                            
                            # Convert exchange segment to string name
                            if exchange_segment == 1 or exchange_segment == MarketFeed.NSE:
                                exchange = "NSE"
                            elif exchange_segment == 2 or exchange_segment == MarketFeed.BSE:
                                exchange = "BSE"
                            else:
                                exchange = f"Unknown({exchange_segment})"
                            
                            # Use our security map to get symbol
                            symbol = "Unknown"
                            if security_id in self.security_map:
                                symbol = self.security_map[security_id]['symbol']
                            
                            # Different processing based on message type
                            if message_type == "Ticker Data":
                                # LTP data
                                ltp = response.get('LTP', 'N/A')
                                timestamp = response.get('LTT', time.strftime('%H:%M:%S'))
                                
                                # Calculate change if we have previous close
                                change = 'N/A'
                                if security_id in self.prev_close_map and ltp != 'N/A':
                                    prev_close = self.prev_close_map[security_id]
                                    try:
                                        ltp_float = float(ltp)
                                        prev_close_float = float(prev_close)
                                        if prev_close_float > 0:
                                            change = ((ltp_float - prev_close_float) / prev_close_float) * 100
                                        else:
                                            change = 0
                                    except (ValueError, TypeError):
                                        change = 'N/A'
                                
                                # Update market data
                                key = f"{exchange}:{security_id}"
                                self.market_data[key] = {
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'ltp': ltp,
                                    'change': change,
                                    'updated': timestamp
                                }
                                
                                # Update UI in main thread
                                self.root.after(0, lambda k=key, s=symbol, e=exchange, l=ltp, c=change, t=timestamp: 
                                               self.update_market_item(k, s, e, l, c, t))
                                
                                self.log(f"Processed LTP update for {symbol} ({key}): LTP={ltp}, Change={change}")
                                
                            elif message_type == "Previous Close":
                                # Store previous close for percent change calculation
                                prev_close = response.get('prev_close', 'N/A')
                                
                                # Store in our map
                                self.prev_close_map[security_id] = prev_close
                                
                                self.log(f"Stored previous close for {symbol} ({security_id}): {prev_close}")
                                
                                # Update any existing market data if we have LTP
                                key = f"{exchange}:{security_id}"
                                if key in self.market_data and self.market_data[key]['ltp'] != 'N/A':
                                    try:
                                        ltp = self.market_data[key]['ltp']
                                        ltp_float = float(ltp)
                                        prev_close_float = float(prev_close)
                                        if prev_close_float > 0:
                                            change = ((ltp_float - prev_close_float) / prev_close_float) * 100
                                        else:
                                            change = 0
                                            
                                        # Update stored data
                                        self.market_data[key]['change'] = change
                                        
                                        # Update UI
                                        self.root.after(0, lambda k=key, s=symbol, e=exchange, l=ltp, c=change, t=self.market_data[key]['updated']: 
                                                       self.update_market_item(k, s, e, l, c, t))
                                                       
                                        self.log(f"Updated percent change for {symbol} based on previous close")
                                    except (ValueError, TypeError):
                                        self.log(f"Could not calculate percent change: invalid values")
                            else:
                                self.log(f"Ignoring message type: {message_type}")
                                
                        except Exception as e:
                            self.log(f"Error processing response: {str(e)}")
                    else:
                        # No data received, log occasionally
                        if update_count == 0 and time.time() % 5 < 0.1:  # Log every ~5 seconds if no updates
                            self.log("No data received yet from feed")
                
                except Exception as e:
                    self.log(f"Error in feed loop: {str(e)}")
                
                # Add a short delay
                time.sleep(0.1)
        
        except Exception as e:
            self.log(f"Fatal error in market feed: {str(e)}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.start_button.configure(state="normal"))
            self.root.after(0, lambda: self.stop_button.configure(state="disabled"))
    
    def update_market_item(self, key, symbol, exchange, ltp, change, timestamp):
        try:
            # Format the change value if numeric
            if isinstance(change, (int, float)):
                formatted_change = f"{change:.2f}%"
                # Set color based on positive/negative change
                if change > 0:
                    color = "green"
                elif change < 0:
                    color = "red"
                else:
                    color = "black"
            else:
                formatted_change = str(change)
                color = "black"
            
            # Ensure LTP is formatted properly
            if isinstance(ltp, (int, float)):
                formatted_ltp = f"{ltp:.2f}"
            else:
                formatted_ltp = str(ltp)
            
            # Update item if it exists in the tree
            if self.market_tree.exists(key):
                self.market_tree.item(key, values=(symbol, exchange, formatted_ltp, formatted_change, timestamp))
                
                # Set color for the item based on change
                tag_name = f"change_{key}"
                self.market_tree.tag_configure(tag_name, foreground=color)
                self.market_tree.item(key, tags=(tag_name,))
                
                self.log(f"Updated UI for {symbol} ({key}): LTP={formatted_ltp}, Change={formatted_change}")
            else:
                self.log(f"Creating new market item for {symbol} ({key})")
                self.market_tree.insert("", "end", iid=key, values=(
                    symbol, exchange, formatted_ltp, formatted_change, timestamp
                ))
                
        except Exception as e:
            self.log(f"Error updating UI for {key}: {str(e)}")

if __name__ == "__main__":
    # CSV file path
    csv_path = r"C:\Users\madhu\Downloads\api-scrip-master.csv"
    
    # Create the main window
    root = tk.Tk()
    app = MarketFeedApp(root, csv_path)
    
    # Start the Tkinter event loop
    root.mainloop()
