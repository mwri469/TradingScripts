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
		
	async def get_options_chain(self, symbol: str):
		"""Retrieve option chain data for a given symbol."""
		logger.info(f"Requesting option chain for: {symbol}")

		self.ib.reqMarketDataType(3)
		underlying = Stock(symbol, "SMART")
		ticker = self.ib.reqTickers(underlying)

		MP = ticker.marketPrice()

def main():
	saver = AsyncIBOptionChainSaver('./tickers.txt')

if __name__ =='__main__':
	main()