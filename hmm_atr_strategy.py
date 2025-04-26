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
from hmmlearn import hmm
from ib_async import IB, Stock, MarketOrder, util
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
    return tickers

class DataHandler:
    """Fetches and stores market data in a SQLite database."""
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT,
                    datetime TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY(symbol, datetime)
                )''')
            
            # Add a table to track collected symbols and their last update
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    last_update TEXT,
                    status TEXT
                )''')

    def store_bars(self, symbol: str, df: pd.DataFrame):
        """Insert new bars into the database."""
        if df.empty:
            logger.warning(f"No data to store for {symbol}")
            return 0
            
        records = []
        for dt, row in df.iterrows():
            # Ensure dt is converted to string format
            dt_str = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
            records.append((
                symbol,
                dt_str,
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume
            ))
            
        if not records:
            return 0
            
        with self.conn:
            self.conn.executemany(
                'INSERT OR IGNORE INTO bars VALUES (?, ?, ?, ?, ?, ?, ?)',
                records
            )
            # Update the symbol's last update time
            self.conn.execute(
                '''
                INSERT OR REPLACE INTO symbols (symbol, last_update, status)
                VALUES (?, ?, ?)
                ''',
                (symbol, datetime.now().isoformat(), 'updated')
            )
        return len(records)

    def load_bars(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Load bars from database into a DataFrame."""
        query = 'SELECT * FROM bars WHERE symbol = ?'
        params = [symbol]
        if start:
            query += ' AND datetime >= ?'
            params.append(start)
        if end:
            query += ' AND datetime <= ?'
            params.append(end)
        query += ' ORDER BY datetime'
        df = pd.read_sql(query, self.conn, params=params, parse_dates=['datetime'], index_col='datetime')
        return df
        
    def get_symbols(self):
        """Get list of symbols in the database."""
        query = 'SELECT symbol FROM symbols'
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]
        
    def get_symbol_info(self, symbol):
        """Get information about a symbol."""
        query = 'SELECT * FROM symbols WHERE symbol = ?'
        cursor = self.conn.execute(query, (symbol,))
        row = cursor.fetchone()
        if row:
            return {
                'symbol': row[0],
                'last_update': row[1],
                'status': row[2]
            }
        return None

class DataCollector:
    """Collects historical data for multiple tickers."""
    def __init__(self, db_path=DB_PATH):
        self.ib = IB()
        self.data_handler = DataHandler(db_path)
        
    async def connect(self, host='127.0.0.1', port=7497, client_id=1, timeout=30):
        """Connect to Interactive Brokers."""
        if not self.ib.isConnected():
            logger.info(f"Connecting to Interactive Brokers at {host}:{port} with client ID {client_id}")
            
            try:
                self.ib.RequestTimeout = max(timeout, 60)  # At least 60 seconds
                await self.ib.connectAsync(host, port, clientId=client_id)
                logger.info("Connected to Interactive Brokers")
                return True
            except asyncio.TimeoutError:
                logger.error(f"Connection timed out after {timeout} seconds")
                logger.error("Please check that TWS/Gateway is running and that API connections are enabled")
                return False
            except ConnectionRefusedError:
                logger.error(f"Connection refused at {host}:{port}")
                logger.error("Please verify TWS/Gateway is running and accepting connections")
                return False
            except Exception as e:
                logger.error(f"Failed to connect to IB: {e}")
                return False
        return True
        
    async def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from Interactive Brokers")
    
    async def collect_data(self, symbols, lookback='365 D', bar_size='1 day', exchange='SMART', currency='USD'):
        """Collect historical data for multiple symbols."""
        connection_success = await self.connect()
        if not connection_success:
            logger.error("Failed to connect. Cannot collect data.")
            return []
        
        results = []
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}...")
            contract = Stock(symbol, exchange, currency)
            
            try:
                await self.ib.qualifyContractsAsync(contract)
                logger.debug(f"Contract qualified: {contract}")
                
                # Request historical data
                bars = await self.ib.reqHistoricalDataAsync(
                    contract, 
                    endDateTime='', 
                    durationStr=lookback,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True
                )
                
                if bars:
                    # Convert to DataFrame
                    df = util.df(bars)
                    if not df.empty:
                        # Convert date to datetime and set as index
                        if 'date' in df.columns:
                            df['datetime'] = pd.to_datetime(df['date'])
                            df = df.set_index('datetime')
                        
                        # Store in database
                        stored = self.data_handler.store_bars(symbol, df)
                        logger.info(f"Stored {stored} bars for {symbol}")
                        results.append({
                            'symbol': symbol,
                            'status': 'success',
                            'bars': len(df)
                        })
                    else:
                        logger.warning(f"No data received for {symbol}")
                        results.append({
                            'symbol': symbol,
                            'status': 'no_data',
                            'bars': 0
                        })
                else:
                    logger.warning(f"No bars received for {symbol}")
                    results.append({
                        'symbol': symbol,
                        'status': 'no_bars',
                        'bars': 0
                    })
                
                # Add a small delay to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e),
                    'bars': 0
                })
        
        return results
    
    def show_database_stats(self):
        """Show statistics about the database."""
        symbols = self.data_handler.get_symbols()
        logger.info(f"Database contains {len(symbols)} symbols")
        
        for symbol in symbols:
            df = self.data_handler.load_bars(symbol)
            info = self.data_handler.get_symbol_info(symbol)
            logger.info(df.index)
            df = df[df.index.notnull()]
            date_range = f"{df.index.min().date()} to {df.index.max().date()}" if not df.empty else "No data"
            
            logger.info(f"  {symbol}: {len(df)} bars, {date_range}, last updated: {info['last_update'] if info else 'Unknown'}")

class HMMModel:
    """Wraps HMM training and prediction on ATR series."""
    def __init__(self, n_states=2, n_iter=500):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=n_iter)
        self.scaler = StandardScaler()

    @staticmethod
    def compute_atr(df, period=14):
        """Compute Average True Range."""
        result = df.copy()
        result['previous_close'] = result['close'].shift(1)
        result['high-low'] = result['high'] - result['low']
        result['high-pc'] = abs(result['high'] - result['previous_close'])
        result['low-pc'] = abs(result['low'] - result['previous_close'])
        result['tr'] = result[['high-low', 'high-pc', 'low-pc']].max(axis=1)
        result['atr'] = result['tr'].rolling(period).mean()
        result.dropna(subset=['atr'], inplace=True)  # Only drop rows with missing ATR
        return result

    def fit(self, atr_series: np.ndarray):
        """Train the HMM model on ATR series."""
        # Check if atr_series is a pandas Series or numpy array
        if isinstance(atr_series, pd.Series):
            X = atr_series.values.reshape(-1, 1)
        else:
            X = atr_series.reshape(-1, 1)
            
        if X.size == 0:
            raise ValueError("ATR series is empty. Cannot train HMM.")
            
        X = self.scaler.fit_transform(X)
        self.model.fit(X)

    def predict(self, atr_value: float) -> int:
        """Predict the hidden state for a given ATR value."""
        x = self.scaler.transform([[atr_value]])
        return int(self.model.predict(x)[0])

class Backtester:
    """Simple backtest engine for HMM-ATR strategy."""
    def __init__(self, df: pd.DataFrame, strategy, initial_cash=100_000):
        self.df = df
        self.strategy = strategy
        self.cash = initial_cash
        self.position = 0
        self.trades = []

    def run(self):
        for ts, row in self.df.iterrows():
            signal = self.strategy.signal(row)
            price = row.close
            # Execute simple quantity: 100 shares
            qty = 100
            if signal == 'BUY' and self.position <= 0:
                self.cash -= qty * price
                self.position += qty
                self.trades.append((ts, 'BUY', price, qty))
            elif signal == 'SELL' and self.position >= 0:
                self.cash += qty * price
                self.position -= qty
                self.trades.append((ts, 'SELL', price, qty))
        # Close any open position at last price
        if self.position != 0:
            last_price = self.df.close.iloc[-1]
            self.cash += self.position * last_price
            self.trades.append((self.df.index[-1], 'EXIT', last_price, self.position))
            self.position = 0
        return self._performance()

    def _performance(self):
        pnl = self.cash - 100_000
        return {
            'final_cash': self.cash,
            'pnl': pnl,
            'trades': pd.DataFrame(self.trades, columns=['datetime', 'action', 'price', 'qty'])
        }

class HMMATRStrategy:
    """Generates buy/sell signals based on HMM-predicted ATR state."""
    def __init__(self, period=14, n_states=2):
        self.period = period
        self.model = HMMModel(n_states)
        self.history = deque(maxlen=period)
        self.logging = 1

    def prepare(self, df: pd.DataFrame):
        log_message(function="prepare", df_shape=df.shape)
        # Compute ATR and train model
        df2 = self.model.compute_atr(df, self.period)
        self.model.fit(df2.atr.values)
        # Preload history
        for val in df2.tr.values[-self.period:]:
            self.history.append(val)
        return df2  # Return the processed dataframe

    def signal(self, tick):
        # tick: pd.Series with fields close, high, low
        prev_close = getattr(self, 'prev_close', None)
        # compute TR
        high_low = tick.high - tick.low
        if prev_close is None:
            high_pc = low_pc = 0
        else:
            high_pc = abs(tick.high - prev_close)
            low_pc = abs(tick.low - prev_close)
        tr = max(high_low, high_pc, low_pc)
        self.history.append(tr)
        self.prev_close = tick.close
        if len(self.history) < self.period:
            return None
        atr = np.mean(self.history)
        state = self.model.predict(atr)
        return 'BUY' if state == 0 else 'SELL'

class LiveTrader(HMMATRStrategy):
    """Runs live trading using ib_async and SQLite caching."""
    def __init__(self, symbol, period=14, n_states=2, exchange='SMART', currency='USD'):
        super().__init__(period, n_states)
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.ib = IB()
        self.data = DataHandler()
        self.contract = None
        self.last_dt = None

    async def connect(self, host='127.0.0.1', port=7497, client_id=1, timeout=30):
        logger.info(f"Connecting to IB at {host}:{port} with client ID {client_id}")
        try:
            self.ib.RequestTimeout = max(timeout, 60)
            await self.ib.connectAsync(host, port, clientId=client_id)
            logger.info("Connected to Interactive Brokers")
            self.contract = Stock(self.symbol, self.exchange, self.currency)
            await self.ib.qualifyContractsAsync(self.contract)
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def initialize(self, lookback='14 D', bar_size='5 mins'):
        # Load from DB or fetch
        df = self.data.load_bars(self.symbol)
        if df.empty:
            logger.info("DB empty, fetching history...")
            bars = await self.ib.reqHistoricalDataAsync(
                self.contract, '', lookback, bar_size, 'TRADES', False, 1
            )
            df = util.df(bars).set_index('date')
            self.data.store_bars(self.symbol, df)
        # prepare strategy
        self.prepare(df)
        self.last_dt = df.index[-1]

    async def run(self, host='127.0.0.1', port=7497, client_id=1, timeout=30):
        connection_success = await self.connect(host, port, client_id, timeout)
        if not connection_success:
            logger.error("Failed to connect. Cannot start live trading.")
            return
            
        await self.initialize()
        logger.info(f"Starting live trading for {self.symbol}")
        
        try:
            while True:
                bars = await self.ib.reqHistoricalDataAsync(
                    self.contract, '', '1 D', '5 mins', 'TRADES', False, 1
                )
                df_live = util.df(bars).set_index('date')
                # filter for new bars
                new_df = df_live[df_live.index > self.last_dt]
                for dt, row in new_df.iterrows():
                    sig = self.signal(row)
                    if sig:
                        logger.info(f"{dt} signal: {sig}")
                        await self._execute(sig)
                    # store bar
                    self.data.store_bars(self.symbol, pd.DataFrame([row], index=[dt]))
                    self.last_dt = dt
                await asyncio.sleep(300)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error in live trading loop: {e}")
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IB")

    async def _execute(self, signal):
        # simple fixed size
        qty = 100
        # check positions
        positions = self.ib.positions()
        pos = next((p for p in positions if p.contract == self.contract), None)
        current = pos.position if pos else 0
        if signal == 'BUY' and current <= 0:
            order = MarketOrder('BUY', qty)
            await self.ib.placeOrderAsync(self.contract, order)
            logger.info(f"Placed BUY order for {qty} shares of {self.symbol}")
        elif signal == 'SELL' and current >= 0:
            order = MarketOrder('SELL', qty)
            await self.ib.placeOrderAsync(self.contract, order)
            logger.info(f"Placed SELL order for {qty} shares of {self.symbol}")

async def collect_data_mode(args):
    """Special mode to just collect data for all tickers."""
    tickers = load_tickers(args.tickers)
    if not tickers:
        logger.error("No tickers found in the file.")
        return
        
    collector = DataCollector(args.db)
    logger.info(f"Collecting data for {len(tickers)} tickers with lookback {args.lookback} and bar size {args.barsize}")
    results = await collector.collect_data(tickers, args.lookback, args.barsize, args.exchange, args.currency)
    
    # Print summary
    logger.info("\nData collection complete!")
    logger.info(f"Processed {len(results)} tickers")
    
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] in ['error', 'no_data', 'no_bars'])
    
    logger.info(f"Success: {success}")
    logger.info(f"Errors: {errors}")
    
    # Show database stats
    logger.info("\nDatabase Statistics:")
    collector.show_database_stats()
    
    await collector.disconnect()

async def train_model_mode(args):
    """Train and test a model for a single symbol."""
    handler = DataHandler(args.db)
    df = handler.load_bars(args.symbol)
    
    if df.empty:
        logger.error(f"No data found for {args.symbol}. Please collect data first.")
        return
        
    strat = HMMATRStrategy(args.period, args.states)
    processed_df = strat.prepare(df)
    
    # Print some statistics
    logger.info(f"\nTrained HMM model for {args.symbol}")
    logger.info(f"Data period: {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"Number of bars: {len(df)}")
    logger.info(f"ATR statistics:")
    if 'atr' in processed_df.columns:
        logger.info(f"  Min: {processed_df.atr.min():.4f}")
        logger.info(f"  Max: {processed_df.atr.max():.4f}")
        logger.info(f"  Mean: {processed_df.atr.mean():.4f}")
        logger.info(f"  Std: {processed_df.atr.std():.4f}")
    
    # Run a quick backtest
    bt = Backtester(processed_df, strat)
    perf = bt.run()
    logger.info(f"\nBacktest Results:")
    logger.info(f"PNL: ${perf['pnl']:.2f}")
    logger.info(f"Number of trades: {len(perf['trades'])}")
    
    # Print trades summary
    trades_df = perf['trades']
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        logger.info(f"Buy trades: {len(buy_trades)}")
        logger.info(f"Sell trades: {len(sell_trades)}")

async def backtest_mode(args):
    """Run a backtest for a strategy on a symbol."""
    handler = DataHandler(args.db)
    df = handler.load_bars(args.symbol)
    if df.empty:
        logger.error(f"No data found for {args.symbol}. Please collect data first.")
        return
    
    logger.info(f"Running backtest for {args.symbol} with period={args.period}, states={args.states}")
    strat = HMMATRStrategy(args.period, args.states)
    processed_df = strat.prepare(df)
    bt = Backtester(processed_df, strat)
    perf = bt.run()
    
    logger.info(f"Backtest Results for {args.symbol}:")
    logger.info(f"PNL: ${perf['pnl']:.2f}")
    logger.info(f"Final cash: ${perf['final_cash']:.2f}")
    logger.info(f"Number of trades: {len(perf['trades'])}")
    
    # Print trades summary
    if not perf['trades'].empty:
        logger.info("\nTrade Summary:")
        logger.info(perf['trades'].head())

async def live_mode(args):
    """Run the strategy in live trading mode."""
    logger.info(f"Starting live trading for {args.symbol}")
    logger.info(f"Connecting to IB at {args.host}:{args.port} with client ID {args.clientid}")
    
    trader = LiveTrader(args.symbol, args.period, args.states, args.exchange, args.currency)
    await trader.run(args.host, args.port, args.clientid, args.timeout)

async def info_mode(args):
    """Show information about the database."""
    collector = DataCollector(args.db)
    collector.show_database_stats()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HMM-ATR Trading Strategy')
    
    # Common arguments
    parser.add_argument('--db', default='market_data.db', help='Path to SQLite database file')
    parser.add_argument('--host', default='127.0.0.1', help='TWS/Gateway host address')
    parser.add_argument('--port', type=int, default=7497, help='TWS/Gateway port (7496/7497 for TWS, 4001/4002 for Gateway)')
    parser.add_argument('--clientid', type=int, default=1, help='Client ID for IB connection')
    parser.add_argument('--timeout', type=int, default=30, help='Connection timeout in seconds')
    parser.add_argument('--exchange', default='SMART', help='Exchange to use')
    parser.add_argument('--currency', default='USD', help='Currency to use')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                      help='Logging level')
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Collect data mode
    collect_parser = subparsers.add_parser('collect', help='Collect historical data')
    collect_parser.add_argument('--tickers', default='tickers.txt', help='Path to file with ticker symbols')
    collect_parser.add_argument('--lookback', default='2 Y', help='Lookback period for historical data')
    collect_parser.add_argument('--barsize', default='1 day', help='Bar size for historical data')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train and test a model')
    train_parser.add_argument('symbol', help='Symbol to train the model on')
    train_parser.add_argument('--period', type=int, default=14, help='ATR period')
    train_parser.add_argument('--states', type=int, default=2, help='Number of HMM states')
    
    # Backtest mode
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('symbol', help='Symbol to backtest')
    backtest_parser.add_argument('--period', type=int, default=14, help='ATR period')
    backtest_parser.add_argument('--states', type=int, default=2, help='Number of HMM states')
    
    # Live mode
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('symbol', help='Symbol to trade')
    live_parser.add_argument('--period', type=int, default=14, help='ATR period')
    live_parser.add_argument('--states', type=int, default=2, help='Number of HMM states')
    
    # Info mode
    info_parser = subparsers.add_parser('info', help='Show database information')
    
    args = parser.parse_args()
    
    # Set default mode if none specified
    if not args.mode:
        args.mode = 'info'
        
    return args

async def main():
    """Main entry point with different modes."""
    args = parse_args()
    
    # Configure logging level based on arguments
    logging_level = getattr(logging, args.loglevel.upper())
    logger.setLevel(logging_level)
    for handler in logger.handlers:
        handler.setLevel(logging_level)
    
    logger.info(f"Starting HMM-ATR Strategy in {args.mode} mode")
    
    try:
        if args.mode == 'collect':
            await collect_data_mode(args)
        elif args.mode == 'train':
            await train_model_mode(args)
        elif args.mode == 'backtest':
            await backtest_mode(args)
        elif args.mode == 'live':
            await live_mode(args)
        elif args.mode == 'info':
            await info_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            logger.info("Available modes: collect, train, backtest, live, info")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    finally:
        # Clean shutdown
        loop.close()