# TradingScripts

This is mostly a collection of my trading scripts that I've been using for learning.

Note a lot of these require the IB TWS/Gateway API to work and I've used the ib_async library.

Most of my data management is through SQLite.

## HMM ATR

This script intends to train, execute or backtest hidden markov models (HMM's) using average true range (ATR)

## Get options

This intends to save live option data from IB and save to a sqlite3 .db.

It can succesfully connect to the IB API, but has issues with the queries (may be lack of data permissions).

Note the backtesting/execution portion is not tested yet.
