import asyncio
import sqlite3
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
from hmmlearn import hmm
from ib_async import IB, Stock, MarketOrder, util
from sklearn.preprocessing import StandardScaler

DB_PATH = 'market_data.db'

globals = {
    'logging': 2  # 0: errors, 1: warnings, 2: verbose 
}

def logging(**kwargs):
    """Prints log messages with timestamp and any provided key-value pairs"""
    timestamp = datetime.now().strftime("%Y-%m-%d::%H.%M.%S")
    print(f">>> {timestamp}", end=" ")
    
    for k, v in kwargs.items():
        print(f"{k}: {v}", end=" ")
    print()  # New line after the log entry

def load_tickers(file_path='tickers.txt'):
    """Load ticker symbols from a text file."""
    tickers = []
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
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
            print(f"No data to store for {symbol}")
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
        
    async def connect(self, host='127.0.0.1', port=7497, client_id=1):
        """Connect to Interactive Brokers."""
        if not self.ib.isConnected():
            await self.ib.connectAsync(host, port, clientId=client_id)
            print("Connected to Interactive Brokers")
        
    async def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from Interactive Brokers")
    
    async def collect_data(self, symbols, lookback='365 D', bar_size='1 day', exchange='SMART', currency='USD'):
        """Collect historical data for multiple symbols."""
        await self.connect()
        
        results = []
        for symbol in symbols:
            print(f"Collecting data for {symbol}...")
            contract = Stock(symbol, exchange, currency)
            
            try:
                await self.ib.qualifyContractsAsync(contract)
                print(f"Contract qualified: {contract}")
                
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
                        print(f"Stored {stored} bars for {symbol}")
                        results.append({
                            'symbol': symbol,
                            'status': 'success',
                            'bars': len(df)
                        })
                    else:
                        print(f"No data received for {symbol}")
                        results.append({
                            'symbol': symbol,
                            'status': 'no_data',
                            'bars': 0
                        })
                else:
                    print(f"No bars received for {symbol}")
                    results.append({
                        'symbol': symbol,
                        'status': 'no_bars',
                        'bars': 0
                    })
                
                # Add a small delay to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
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
        print(f"Database contains {len(symbols)} symbols")
        
        for symbol in symbols:
            df = self.data_handler.load_bars(symbol)
            info = self.data_handler.get_symbol_info(symbol)
            date_range = f"{df.index.min().date()} to {df.index.max().date()}" if not df.empty else "No data"
            
            print(f"  {symbol}: {len(df)} bars, {date_range}, last updated: {info['last_update'] if info else 'Unknown'}")

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
        logging(function="prepare", df_shape=df.shape)
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

    async def connect(self):
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        self.contract = Stock(self.symbol, self.exchange, self.currency)
        await self.ib.qualifyContractsAsync(self.contract)

    async def initialize(self, lookback='14 D', bar_size='5 mins'):
        # Load from DB or fetch
        df = self.data.load_bars(self.symbol)
        if df.empty:
            print("DB empty, fetching history...")
            bars = await self.ib.reqHistoricalDataAsync(
                self.contract, '', lookback, bar_size, 'TRADES', False, 1
            )
            df = util.df(bars).set_index('date')
            self.data.store_bars(self.symbol, df)
        # prepare strategy
        self.prepare(df)
        self.last_dt = df.index[-1]

    async def run(self):
        await self.connect()
        await self.initialize()
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
                    print(f"{dt} signal: {sig}")
                    await self._execute(sig)
                # store bar
                self.data.store_bars(self.symbol, pd.DataFrame([row], index=[dt]))
                self.last_dt = dt
            await asyncio.sleep(300)

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
        elif signal == 'SELL' and current >= 0:
            order = MarketOrder('SELL', qty)
            await self.ib.placeOrderAsync(self.contract, order)

async def collect_data_mode(tickers_file='tickers.txt', lookback='365 D', bar_size='1 day'):
    """Special mode to just collect data for all tickers."""
    tickers = load_tickers(tickers_file)
    if not tickers:
        print("No tickers found in the file.")
        return
        
    collector = DataCollector()
    results = await collector.collect_data(tickers, lookback, bar_size)
    
    # Print summary
    print("\nData collection complete!")
    print(f"Processed {len(results)} tickers")
    
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] in ['error', 'no_data', 'no_bars'])
    
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    
    # Show database stats
    print("\nDatabase Statistics:")
    collector.show_database_stats()
    
    await collector.disconnect()

async def train_model_mode(symbol, period=14, n_states=2):
    """Train and test a model for a single symbol."""
    handler = DataHandler()
    df = handler.load_bars(symbol)
    
    if df.empty:
        print(f"No data found for {symbol}. Please collect data first.")
        return
        
    strat = HMMATRStrategy(period, n_states)
    processed_df = strat.prepare(df)
    
    # Print some statistics
    print(f"\nTrained HMM model for {symbol}")
    print(f"Data period: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Number of bars: {len(df)}")
    print(f"ATR statistics:")
    if 'atr' in processed_df.columns:
        print(f"  Min: {processed_df.atr.min():.4f}")
        print(f"  Max: {processed_df.atr.max():.4f}")
        print(f"  Mean: {processed_df.atr.mean():.4f}")
        print(f"  Std: {processed_df.atr.std():.4f}")
    
    # Run a quick backtest
    bt = Backtester(processed_df, strat)
    perf = bt.run()
    print(f"\nBacktest Results:")
    print(f"PNL: ${perf['pnl']:.2f}")
    print(f"Number of trades: {len(perf['trades'])}")
    
    # Print trades summary
    trades_df = perf['trades']
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        print(f"Buy trades: {len(buy_trades)}")
        print(f"Sell trades: {len(sell_trades)}")

async def main(mode='backtest', symbol=None, tickers_file='tickers.txt'):
    """Main entry point with different modes."""
    if mode == 'collect':
        await collect_data_mode(tickers_file)
    elif mode == 'train':
        if not symbol:
            print("Error: Symbol is required for training mode.")
            return
        await train_model_mode(symbol)
    elif mode == 'backtest':
        if not symbol:
            print("Error: Symbol is required for backtest mode.")
            return
        handler = DataHandler()
        df = handler.load_bars(symbol)
        if df.empty:
            print(f"No data found for {symbol}. Please collect data first.")
            return
        strat = HMMATRStrategy()
        strat.prepare(df)
        bt = Backtester(df, strat)
        perf = bt.run()
        print(f"PNL: {perf['pnl']}, Trades:\n", perf['trades'])
    elif mode == 'live':
        if not symbol:
            print("Error: Symbol is required for live mode.")
            return
        trader = LiveTrader(symbol)
        await trader.run()
    elif mode == 'info':
        collector = DataCollector()
        collector.show_database_stats()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: collect, train, backtest, live, info")

if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Default values
    mode = 'backtest'
    symbol = None
    tickers_file = 'tickers.txt'
    
    # Basic command line argument parsing
    if args:
        mode = args[0]
        
    if len(args) > 1:
        if mode in ['train', 'backtest', 'live']:
            symbol = args[1]
        elif mode == 'collect':
            tickers_file = args[1]
    
    asyncio.run(main(mode, symbol, tickers_file))