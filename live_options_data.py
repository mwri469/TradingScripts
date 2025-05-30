import asyncio
import sqlite3
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
import logging
import sys
import argparse
from ib_async import *
from sklearn.preprocessing import StandardScaler

DB_PATH = 'market_data.db'

def setup_logger():
    """Configure logging for the application."""
    # Configure root logger to see all IB logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create our module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler('hmm_atr_strategy.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger if they don't exist
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# Create the logger instance
logger = setup_logger()

def log_message(**kwargs):
    """Legacy logging function for backward compatibility"""
    timestamp = datetime.now().strftime("%Y-%m-%d::%H.%M.%S")
    message = f">>> {timestamp} "
    
    for k, v in kwargs.items():
        message += f"{k}: {v} "
    
    logger.info(message)

def load_tickers(file_path='tickers.txt'):
    """Load ticker symbols from a text file."""
    tickers = []
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
    except FileNotFoundError:
        logger.error(f"Error: File {file_path} not found.")
        # Return default ticker if file not found
        tickers = ["SPY"]
    return tickers

class AsyncIBOptionChainSaver:
    def __init__(self, tickers_file, db_path="option_data.db", host="127.0.0.1", port=7497, client_id=12, timeout=30):
        """
        Initialize the asynchronous IB option chain saver.
        
        Args:
            tickers_file (str): Path to a file containing ticker symbols, one per line
            db_path (str): Path to SQLite database for storing option data
            host (str): IB TWS/Gateway host address
            port (int): IB TWS/Gateway port
            client_id (int): Client ID for IB connection
            timeout (int): Connection timeout in seconds
        """
        self.ib = None  # Initialize later in connect method
        self.db_path = db_path
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.tickers = load_tickers(tickers_file)
        self.logger = logger
        self.init_database()
        
    def init_database(self):
        """Initialize the SQLite database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='options'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                # Create options table if it doesn't exist
                cursor.execute('''
                CREATE TABLE options (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    strike REAL NOT NULL,
                    right TEXT NOT NULL,
                    bid REAL,
                    ask REAL,
                    last REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_vol REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    timestamp TEXT NOT NULL
                )
                ''')
                
                # Create indexes after table is confirmed to exist
                cursor.execute('CREATE INDEX idx_options_symbol ON options (symbol)')
                cursor.execute('CREATE INDEX idx_options_expiry ON options (expiry)')
                
                self.logger.info("Database table and indexes created successfully")
            else:
                self.logger.info("Database table already exists")
                
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    async def connect(self):
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            # Create a new IB instance each time we connect
            self.ib = IB()
            
            # Set longer timeout values for the client connection
            self.ib.client.RaiseRequestErrors = True
            self.ib.client.MaxRequests = 100
            print('--- connecting now ---')

            # Set a longer connection timeout
            self.ib.RequestTimeout = max(self.timeout, 60)  # At least 60 seconds
            self.logger.info(f"Attempting to connect to IB at {self.host}:{self.port} with clientId {self.client_id}")
            self.logger.info(f"Connection timeout set to {self.timeout} seconds")
            
            # Connect without additional wait_for timeout
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, readonly=True)
            self.ib.reqMarketDataType(3)  # Use delayed market data
            
            self.logger.info(f"Successfully connected to IB at {self.host}:{self.port}")
            self.logger.info(f"TWS/Gateway version: {self.ib.client.serverVersion()}")
            
            # Test if we can get server time
            server_time = await self.ib.reqCurrentTimeAsync()
            self.logger.info(f"Server time: {server_time}")
            
            return True
            
        except asyncio.TimeoutError:
            self.logger.error(f"Connection timed out after {self.timeout} seconds")
            self.logger.error("Please check that TWS/Gateway is running and that API connections are enabled")
            self.logger.error("Also verify your port number and client ID")
            return False
        except ConnectionRefusedError:
            self.logger.error(f"Connection refused at {self.host}:{self.port}")
            self.logger.error("Please verify TWS/Gateway is running and accepting connections")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            return False
        
    async def get_options_chain(self, symbol: str):
        """Retrieve option chain data for a given symbol."""
        try:
            self.logger.info(f"Requesting option chain for: {symbol}")

            # Create underlying stock contract
            underlying = Stock(symbol, "SMART", "USD")
            
            # Qualify the contract
            qualified_contracts = await self.ib.qualifyContractsAsync(underlying)
            if not qualified_contracts:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return []
            
            underlying = qualified_contracts[0]
            self.logger.info(f"Qualified underlying contract: {underlying}")

            # Get current market price
            ticker = await self.ib.reqTickersAsync(underlying)
            if not ticker:
                self.logger.error(f"Could not get ticker data for {symbol}")
                return []
            
            ticker = ticker[0]
            market_price = ticker.marketPrice()
            if not market_price or market_price <= 0:
                market_price = ticker.close if ticker.close > 0 else ticker.last
            
            self.logger.info(f"Current market price for {symbol}: {market_price}")

            # Get option chain parameters
            chains = await self.ib.reqSecDefOptParamsAsync(
                underlyingSymbol=underlying.symbol, 
                futFopExchange='', 
                underlyingSecType=underlying.secType, 
                underlyingConId=underlying.conId
            )
            
            if not chains:
                self.logger.error(f"No option chains found for {symbol}")
                return []

            self.logger.info(f"Found {len(chains)} option chain(s) for {symbol}")
            
            result = []
            
            # Process each chain (different exchanges/trading classes)
            for chain in chains:
                self.logger.info(f"Processing chain: exchange={chain.exchange}, tradingClass={chain.tradingClass}")
                
                # Filter strikes around current price (±20% range)
                price_range = market_price * 0.2 if market_price else 50
                strikes = [strike for strike in chain.strikes 
                          if market_price - price_range <= strike <= market_price + price_range
                          and strike % 5 == 0]  # Only strikes divisible by 5
                
                # Get first 3 expirations
                expirations = sorted(chain.expirations)[:3]
                rights = ['P', 'C']
                
                self.logger.info(f"Selected {len(strikes)} strikes and {len(expirations)} expirations")
                
                # Create option contracts
                contracts = []
                for right in rights:
                    for expiration in expirations:
                        for strike in strikes:
                            contract = Option(
                                symbol=symbol, 
                                lastTradeDateOrContractMonth=expiration, 
                                strike=strike, 
                                right=right, 
                                exchange=chain.exchange, 
                                tradingClass=chain.tradingClass
                            )
                            contracts.append(contract)
                
                if contracts:
                    # Qualify option contracts
                    self.logger.info(f"Qualifying {len(contracts)} option contracts")
                    qualified_options = await self.ib.qualifyContractsAsync(*contracts)
                    self.logger.info(f"Successfully qualified {len(qualified_options)} option contracts")
                    
                    # Get market data for options
                    if qualified_options:
                        tickers = await self.ib.reqTickersAsync(*qualified_options)
                        self.logger.info(f"Retrieved market data for {len(tickers)} options")
                        
                        # Store option data
                        for ticker in tickers:
                            await self.store_option_ticker_data(ticker)
                        
                        result.extend(tickers)
                
                # Brief pause between chains
                await asyncio.sleep(1)

            return result
            
        except Exception as e:
            self.logger.error(f"Error getting options chain for {symbol}: {e}")
            return []

    async def store_option_ticker_data(self, ticker):
        """Store option ticker data to database."""
        try:
            contract = ticker.contract
            timestamp = datetime.now().isoformat()
            
            # Extract Greeks from model or bid/ask/last Greeks
            greeks = ticker.modelGreeks or ticker.bidGreeks or ticker.askGreeks or ticker.lastGreeks
            
            option_data = {
                'symbol': contract.symbol,
                'expiry': contract.lastTradeDateOrContractMonth,
                'strike': contract.strike,
                'right': contract.right,
                'bid': ticker.bid if ticker.bid and ticker.bid > 0 else None,
                'ask': ticker.ask if ticker.ask and ticker.ask > 0 else None,
                'last': ticker.last if ticker.last and ticker.last > 0 else None,
                'volume': ticker.volume if hasattr(ticker, 'volume') else None,
                'open_interest': ticker.openInterest if hasattr(ticker, 'openInterest') else None,
                'implied_vol': greeks.impliedVol if greeks and hasattr(greeks, 'impliedVol') else None,
                'delta': greeks.delta if greeks and hasattr(greeks, 'delta') else None,
                'gamma': greeks.gamma if greeks and hasattr(greeks, 'gamma') else None,
                'theta': greeks.theta if greeks and hasattr(greeks, 'theta') else None,
                'vega': greeks.vega if greeks and hasattr(greeks, 'vega') else None,
                'timestamp': timestamp
            }
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO options (
                symbol, expiry, strike, right, bid, ask, last, volume, 
                open_interest, implied_vol, delta, gamma, theta, vega, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                option_data['symbol'], option_data['expiry'], option_data['strike'], option_data['right'],
                option_data['bid'], option_data['ask'], option_data['last'], option_data['volume'],
                option_data['open_interest'], option_data['implied_vol'], option_data['delta'],
                option_data['gamma'], option_data['theta'], option_data['vega'], option_data['timestamp']
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Stored option data: {contract.symbol} {contract.right} {contract.strike} exp:{contract.lastTradeDateOrContractMonth}")
            
        except Exception as e:
            self.logger.error(f"Error storing option data: {e}")

    async def run(self):
        """Main method to run the option chain saver."""
        connection_success = await self.connect()
        if not connection_success:
            self.logger.error("Failed to connect. Exiting.")
            return
        
        try:
            # Process each ticker in the list
            for symbol in self.tickers:
                self.logger.info(f"Processing {symbol}")
                options_data = await self.get_options_chain(symbol)
                self.logger.info(f"Retrieved {len(options_data)} option contracts for {symbol}")
                
                # Brief pause between tickers
                await asyncio.sleep(2)
                
            self.logger.info("Completed processing all tickers")
            
        except Exception as e:
            self.logger.error(f"Run error: {e}")
        finally:
            # Disconnect from IB
            if self.ib:
                self.ib.disconnect()
                self.logger.info("Disconnected from IB")

async def main():
    """Main function to run the AsyncIBOptionChainSaver."""
    parser = argparse.ArgumentParser(description='Download option chain data from Interactive Brokers')
    parser.add_argument('--host', default='127.0.0.1', help='TWS/Gateway host address')
    parser.add_argument('--port', type=int, default=7497, help='TWS/Gateway port')
    parser.add_argument('--clientid', type=int, default=12, help='Client ID for IB connection')
    parser.add_argument('--tickers', default='./tickers.txt', help='Path to file with ticker symbols')
    parser.add_argument('--db', default='option_data.db', help='Path to SQLite database file')
    parser.add_argument('--timeout', type=int, default=30, help='Connection timeout in seconds')
    
    args = parser.parse_args()
    
    print(f"Connecting to IB at {args.host}:{args.port} with client ID {args.clientid}")
    print(f"Using tickers from {args.tickers} and database {args.db}")
    
    saver = AsyncIBOptionChainSaver(
        tickers_file=args.tickers,
        db_path=args.db,
        host=args.host,
        port=args.port,
        client_id=args.clientid,
        timeout=args.timeout
    )
    
    await saver.run()

if __name__ == '__main__':
    asyncio.run(main())
