import asyncio
import pandas as pd
import datetime
import sqlite3
from ib_async import IB, Contract, util
import logging
import nest_asyncio
import sys

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class AsyncIBOptionChainSaver:
    def __init__(self, tickers_file, db_path="option_data.db", host="127.0.0.1", port=7497, client_id=0, timeout=30):
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
        self.tickers = self.load_tickers(tickers_file)
        self.logger = self.setup_logger()
        self.init_database()
        
    def setup_logger(self):
        """Configure logging for the application."""
        # Configure root logger to see all IB logs
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Create our module logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose output
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler('ib_options.log')
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
        
    def load_tickers(self, path):
        """Load ticker symbols from a file."""
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: Ticker file '{path}' not found!")
            return ["SPY"]  # Default to SPY if file not found
    
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
            self.ib.reqMarketDataType(3)
            
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
    
    async def get_option_chains(self, symbol):
        """Retrieve option chain data for a given symbol."""
        self.logger.info(f"Requesting option chain for: {symbol}")
        
        # Create underlying contract
        contract = Contract(symbol=symbol, secType='STK', exchange='SMART', currency='USD')
        
        try:
            # Qualify the contract to get conId
            self.logger.debug(f"Qualifying contract for {symbol}")
            qualified = await self.ib.qualifyContractsAsync(contract)
            if not qualified:
                self.logger.error(f"Could not qualify contract for {symbol}")
                return
            
            contract = qualified[0]
            self.logger.debug(f"Contract qualified: {contract}")
            
            # Request option chains
            self.logger.debug(f"Requesting option chain parameters for {symbol} with conId {contract.conId}")
            chains = await self.ib.reqSecDefOptParamsAsync(
                underlyingSymbol=symbol,
                futFopExchange='',
                underlyingSecType='STK',
                underlyingConId=contract.conId
            )
            
            if not chains:
                self.logger.warning(f"No option chains found for {symbol}")
                return
            
            self.logger.debug(f"Received {len(chains)} option chain definition(s)")
            
            for chain in chains:
                # Process each expiration and strike
                expirations = sorted(chain.expirations)
                strikes = sorted(chain.strikes)
                
                self.logger.info(f"Found {len(strikes)} strikes and {len(expirations)} expirations for {symbol}")
                
                # Get underlying price for ATM strike filtering if needed
                underlying_price = await self.get_underlying_price(contract)
                
                # Include all expirations for flexibility
                # You mentioned wanting a range of expiries
                for expiry in expirations:
                    # If there are too many strikes, filter them to focus around ATM
                    strike_subset = strikes
                    if len(strikes) > 30 and underlying_price:
                        strike_subset = self.filter_strikes(strikes, underlying_price, 15)
                        self.logger.info(f"Filtered from {len(strikes)} to {len(strike_subset)} strikes around price {underlying_price}")
                    
                    await self.process_options_for_expiry(symbol, expiry, strike_subset, chain.exchange, chain.multiplier)
                    # Brief pause between expirations
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            self.logger.error(f"Error processing options for {symbol}: {e}")
    
    async def get_underlying_price(self, contract):
        """Get the current price of the underlying asset."""
        try:
            self.logger.debug(f"Requesting market data for {contract.symbol}")
            ticker = await self.ib.reqMarketData(contract, '')
            await asyncio.sleep(0.5)  # Give time for market data to arrive
            
            self.logger.debug(f"Received market data for {contract.symbol}: marketPrice={ticker.marketPrice()}, last={ticker.last}, close={ticker.close}")
            
            if ticker.marketPrice() > 0:
                return ticker.marketPrice()
            elif ticker.last > 0:
                return ticker.last
            elif ticker.close > 0:
                return ticker.close
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {contract.symbol}: {e}")
            return None
    
    def filter_strikes(self, strikes, current_price, num_strikes):
        """Filter strikes to get a reasonable number around current price."""
        strikes = sorted(strikes)
        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
        
        start_idx = max(0, closest_idx - num_strikes)
        end_idx = min(len(strikes), closest_idx + num_strikes + 1)
        
        return strikes[start_idx:end_idx]
    
    async def process_options_for_expiry(self, symbol, expiry, strikes, exchange, multiplier):
        """Process all options for a specific expiration date."""
        self.logger.info(f"Processing options for {symbol} expiring {expiry}")
        
        option_contracts = []
        
        # Create contract for each strike and right (call/put)
        for strike in strikes:
            for right in ['C', 'P']:
                contract = Contract(
                    symbol=symbol,
                    secType='OPT',
                    exchange=exchange,
                    currency='USD',
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    multiplier=multiplier
                )
                option_contracts.append(contract)
        
        # Request market data for each option
        self.logger.debug(f"Qualifying {len(option_contracts)} option contracts")
        qualified_contracts = await self.ib.qualifyContractsAsync(*option_contracts)
        self.logger.debug(f"Successfully qualified {len(qualified_contracts)} contracts")
        
        # Process in chunks to avoid overwhelming the TWS/Gateway
        chunk_size = 50
        for i in range(0, len(qualified_contracts), chunk_size):
            chunk = qualified_contracts[i:i+chunk_size]
            self.logger.debug(f"Processing chunk {i//chunk_size + 1} with {len(chunk)} contracts")
            await self.request_option_data(chunk)
            # Allow some time between chunks
            await asyncio.sleep(2)
    
    async def request_option_data(self, contracts):
        """Request market data for a list of option contracts."""
        if not contracts:
            return
            
        tickers = {}
        
        # Request market data for each contract
        for contract in contracts:
            try:
                self.logger.debug(f"Requesting market data for {contract.symbol} {contract.right} {contract.strike} {contract.lastTradeDateOrContractMonth}")
                # Legal ones for (OPT) are: 100(Option Volume),101(Option Open Interest),105(Average Opt Volume),106(impvolat),165(Misc. Stats),221/220(Creditman Mark Price)
                #                           ,225(Auction),232/221(Pl Price),233(RTVolume),236(inventory),258/47(Fundamentals),292(Wide_news),293(TradeCount),294(TradeRate),
                #                           295(VolumeRate),318(LastRTHTrade),375(RTTrdVolume),411(rthistvol),456/59(IBDividends),460(Bond Factor Multiplier),
                #                           577(EtfNavLast(navlast)),586(IPOHLMPRC),587(Pl Price Delayed),588(Futures Open Interest),595(Short-Term Volume X Mins),
                #                           614(EtfNavMisc(high/low)),619(Creditman Slow Mark Price),623(EtfFrozenNavLast(fznavlast))
                ticker = self.ib.reqMktData(contract, genericTickList='100,101,105,106,165,221,225,232,233,236,258,293,294,295,318,375,411,577,586,587,588,595,614,619,623')
                tickers[ticker] = contract
            except Exception as e:
                self.logger.error(f"Error requesting market data: {e}")
        
        # Wait for data to arrive
        await asyncio.sleep(2)
        
        # Process received data
        for ticker, contract in tickers.items():
            try:
                timestamp = pd.Timestamp.utcnow().isoformat()
                
                # Extract option data
                option_data = {
                    'symbol': contract.symbol,
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': contract.right,
                    'bid': ticker.bid if hasattr(ticker, 'bid') else None,
                    'ask': ticker.ask if hasattr(ticker, 'ask') else None,
                    'last': ticker.last if hasattr(ticker, 'last') else None,
                    'volume': ticker.volume if hasattr(ticker, 'volume') else None,
                    'open_interest': ticker.openInterest if hasattr(ticker, 'openInterest') else None,
                    'implied_vol': ticker.impliedVol if hasattr(ticker, 'impliedVol') else None,
                    'delta': ticker.delta if hasattr(ticker, 'delta') else None,
                    'gamma': ticker.gamma if hasattr(ticker, 'gamma') else None,
                    'theta': ticker.theta if hasattr(ticker, 'theta') else None,
                    'vega': ticker.vega if hasattr(ticker, 'vega') else None,
                    'timestamp': timestamp
                }
                
                self.logger.debug(f"Collected data for {contract.symbol} {contract.right} {contract.strike}: bid={option_data['bid']}, ask={option_data['ask']}")
                
                # Store data in database
                self.store_option_data(option_data)
                
                # Cancel the market data subscription
                self.ib.cancelMktData(contract)
                
            except Exception as e:
                self.logger.error(f"Error processing data for {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike} {contract.right}: {e}")
    
    def store_option_data(self, data):
        """Store option data in the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO options (
                symbol, expiry, strike, right, bid, ask, last, volume, 
                open_interest, implied_vol, delta, gamma, theta, vega, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['symbol'], data['expiry'], data['strike'], data['right'],
                data['bid'], data['ask'], data['last'], data['volume'],
                data['open_interest'], data['implied_vol'], data['delta'],
                data['gamma'], data['theta'], data['vega'], data['timestamp']
            ))
            
            conn.commit()
            conn.close()
            self.logger.debug(f"Stored data for {data['symbol']} {data['right']} {data['strike']} expiring {data['expiry']}")
            
        except Exception as e:
            self.logger.error(f"Database error: {e}")
    
    async def run(self):
        """Main method to run the option chain saver."""
        connection_success = await self.connect()
        if not connection_success:
            self.logger.error("Failed to connect. Exiting.")
            return
        
        try:
            # Process each ticker in the list
            for ticker in self.tickers:
                await self.get_option_chains(ticker)
                # Brief pause between tickers
                await asyncio.sleep(1)
                
            self.logger.info("Completed processing all tickers")
            
        except Exception as e:
            self.logger.error(f"Run error: {e}")
        finally:
            # Disconnect from IB
            if self.ib:
                self.ib.disconnect()
                self.logger.info("Disconnected from IB")

def main():
    """Main function to run the AsyncIBOptionChainSaver."""
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      handlers=[
                          logging.FileHandler("ib_options.log"),
                          logging.StreamHandler(sys.stdout)
                      ])
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get command line arguments for custom settings
        import argparse
        parser = argparse.ArgumentParser(description='Download option chain data from Interactive Brokers')
        parser.add_argument('--host', default='127.0.0.1', help='TWS/Gateway host address')
        parser.add_argument('--port', type=int, default=7497, help='TWS/Gateway port (7496/7497 for TWS, 4001/4002 for Gateway)')
        parser.add_argument('--clientid', type=int, default=0, help='Client ID for IB connection')
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
        
        loop.run_until_complete(saver.run())
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Unhandled exception: {e}")
    finally:
        # Clean shutdown
        loop.close()

if __name__ == "__main__":
    main()
