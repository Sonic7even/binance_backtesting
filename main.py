import ta
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from binance.error import ClientError
from binance.um_futures import UMFutures
from time import time
import logging
import webbrowser
import numpy as np
import io
import ta.momentum
import ta.trend
import ta.volatility
from scipy.stats import normaltest  # type: ignore
from skopt import gp_minimize  # type: ignore
from skopt.space import Integer, Real  # type: ignore
from skopt.utils import use_named_args  # type: ignore
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

symbol = 'ENSUSDT'
tframe = '15m'
days = 30

class Cross(Strategy):
    """
    A trading strategy that uses various technical indicators such as SMA, RSI, MACD, and Bollinger Bands
    to generate buy and sell signals.

    Attributes:
        n1 (int): Period for the first Simple Moving Average (SMA).
        n2 (int): Period for the second Simple Moving Average (SMA).
        rsi_period (int): Period for the Relative Strength Index (RSI).
        rsi_upper (float): Upper threshold for the RSI.
        rsi_lower (float): Lower threshold for the RSI.
        bb_window (int): Window period for the Bollinger Bands.
        bb_stddev (int): Standard deviation for the Bollinger Bands.
    """
    n1 = 10
    n2 = 200
    rsi_period = 23
    rsi_upper = 54
    rsi_lower = 21
    bb_window = 15
    bb_stddev = 2

    def init(self):
        """
        Initialize the indicators for the strategy.
        """
        close = self.data.Close
        self.sma1 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n1)
        self.sma2 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n2)
        self.rsi = self.I(ta.momentum.rsi, pd.Series(close), self.rsi_period)
        self.macd = self.I(ta.trend.macd, pd.Series(close), 12, 26)
        self.macd_signal = self.I(ta.trend.macd_signal, pd.Series(close), 12, 26, 9)

        bb = ta.volatility.BollingerBands(pd.Series(close), window=self.bb_window, window_dev=self.bb_stddev)
        self.bb_hband = self.I(bb.bollinger_hband)
        self.bb_lband = self.I(bb.bollinger_lband)

    def next(self):
        """
        Define the trading logic to execute at each step of the backtest.

        Buy Signal:
            - If the short SMA crosses above the long SMA (bullish crossover).
            - RSI is below the upper threshold, indicating that the market is not overbought.
            - MACD is above its signal line, indicating bullish momentum.
            - Entry price is set just below the lower Bollinger Band.
            - Take profit (TP) is set at the upper Bollinger Band.
            - Stop loss (SL) is set 5% below the entry price.

        Sell Signal:
            - If the short SMA crosses below the long SMA (bearish crossover).
            - RSI is above the lower threshold, indicating that the market is not oversold.
            - MACD is below its signal line, indicating bearish momentum.
            - Entry price is set just above the lower Bollinger Band.
            - Take profit (TP) is set at the lower Bollinger Band.
            - Stop loss (SL) is set 5% above the entry price.
        """
        macd_diff = self.macd - self.macd_signal
        if crossover(self.sma1, self.sma2) and (self.rsi[-1] < self.rsi_upper) and (macd_diff > 0):
            entry_price = self.bb_lband[-1] * 0.995
            tp_price = self.bb_hband[-1] * 1.02
            self.buy(limit=entry_price, tp=tp_price, sl=entry_price * 0.95)
        elif crossover(self.sma2, self.sma1) and (self.rsi[-1] > self.rsi_lower) and (macd_diff < 0):
            entry_price = self.bb_lband[-1] * 1.012
            tp_price = self.bb_lband[-1] * 0.98
            self.sell(limit=entry_price, tp=tp_price, sl=entry_price * 1.05)

client = UMFutures()

def get_tickers_usdt():
    """
    Fetch all trading pairs that include 'USDT' from Binance Futures.

    Returns:
        list: A list of trading pair symbols.
    """
    try:
        tickers = []
        resp = client.ticker_price()
        for elem in resp:
            if 'USDT' in elem['symbol']:
                tickers.append(elem['symbol'])
        return tickers
    except ClientError as error:
        print(f"Found error. status: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}")

intervals = {'1m': 1,
             '3m': 3,
             '5m': 5,
             '15m': 15,
             '30m': 30,
             '1h': 60,
             '2h': 120,
             '4h': 240,
             '6h': 360,
             '8h': 480,
             '12h': 720,
             '1d': 1440,
             '3d': 4320,
             '1w': 10080,
             }

def klines(symbol, timeframe=tframe, limit=1500, start=None, end=None):
    """
    Fetch historical klines (OHLCV data) for a given symbol and timeframe from Binance Futures.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        timeframe (str): The timeframe for each kline (e.g., '5m', '1h').
        limit (int): The maximum number of klines to fetch.
        start (int, optional): The start time in milliseconds.
        end (int, optional): The end time in milliseconds.

    Returns:
        pandas.DataFrame: A DataFrame containing the OHLCV data.
    """
    try:
        resp = pd.DataFrame(client.klines(symbol, timeframe, limit=limit, startTime=start, endTime=end))
        resp = resp.iloc[:, :6]
        resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        resp = resp.set_index('Time')
        resp.index = pd.to_datetime(resp.index, unit='ms')
        resp = resp.astype(float)
        return resp
    except ClientError as error:
        print(f"Found error. status: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}")

def klines_extended(symbol, timeframe=tframe, interval_days=days):
    """
    Fetch extended historical klines data by making multiple API requests if necessary.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        timeframe (str): The timeframe for each kline (e.g., '5m', '1h').
        interval_days (int): The total number of days to fetch data for.

    Returns:
        pandas.DataFrame: A DataFrame containing the OHLCV data for the specified period.
    """
    ms_interval = interval_days * 24 * 3600 * 1000
    limit = ms_interval / (intervals[timeframe] * 60 * 1000)
    steps = limit / 1500
    first_limit = int(steps)
    last_step = steps - int(steps)
    last_limit = round(1500 * last_step)
    current_time = time() * 1000
    p = pd.DataFrame()
    for i in range(first_limit):
        start = int(current_time - (ms_interval - i * 1500 * intervals[timeframe] * 60 * 1000))
        end = start + 1500 * intervals[timeframe] * 60 * 1000
        res = klines(symbol, timeframe=timeframe, limit=1500, start=start, end=end)
        p = pd.concat([p, res])
    p = pd.concat([p, klines(symbol, timeframe=timeframe, limit=last_limit)])
    return p

# Fetch historical data for the specified symbol and timeframe
df = klines_extended(symbol, timeframe=tframe, interval_days=days)
logger.info("\n" + df.head().to_string())

# Capture df.info() output
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
logger.info("\n" + info_str)

# Run the backtest with the optimized strategy
bt = Backtest(df, Cross, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)
"""
Run the backtest with the optimized strategy and log the results.

This block of code performs the following steps:

1. **Initialize Backtest**: 
   - Creates an instance of the `Backtest` class with the provided historical data (`df`), the trading strategy (`Cross`), and various backtesting parameters.
   - Parameters:
     - `df`: The historical OHLCV data for the specified symbol and timeframe.
     - `Cross`: The trading strategy class that implements the trading logic using technical indicators.
     - `cash`: Initial cash available for trading, set to $100.
     - `margin`: Margin ratio for leveraged trading, set to 20% (2/10).
     - `commission`: Commission fee per trade, set to 0.2% (0.002).
     - `exclusive_orders`: Boolean indicating whether orders should be exclusive.

2. **Optimize Strategy Parameters**:
   - Calls the `optimize` method on the `Backtest` instance to find the optimal parameters for the trading strategy.
   - Parameters:
     - `n1`: Range for the first Simple Moving Average (SMA) period, from 10 to 200 with a step of 10.
     - `n2`: Range for the second Simple Moving Average (SMA) period, from 10 to 200 with a step of 10.
     - `rsi_upper`: Range for the upper threshold of the Relative Strength Index (RSI), from 50 to 90 with a step of 5.
     - `rsi_lower`: Range for the lower threshold of the Relative Strength Index (RSI), from 10 to 50 with a step of 5.
     - `bb_window`: Range for the Bollinger Bands window period, from 7 to 24 with a step of 2.
     - `maximize`: The performance metric to maximize during optimization, set to 'SQN' (System Quality Number).
     - `method`: The optimization method to use, set to 'skopt' (Scikit-Optimize).

3. **Log and Print Results**:
   - Logs the results of the backtest, specifically the trades executed during the backtest.
   - Parameters:
     - `bt_results['_trades']`: DataFrame containing the details of each trade executed during the backtest.
   - Prints the results of the backtest, including the optimized parameters and performance metrics.

### Detailed Breakdown of Parameters:

- **n1**: This parameter represents the period for the first Simple Moving Average (SMA). The optimizer will search for the best period within the range from 10 to 200 in steps of 10.
- **n2**: This parameter represents the period for the second Simple Moving Average (SMA). The optimizer will search for the best period within the range from 10 to 200 in steps of 10.
- **rsi_upper**: This parameter represents the upper threshold for the Relative Strength Index (RSI). The optimizer will search for the best threshold within the range from 50 to 90 in steps of 5.
- **rsi_lower**: This parameter represents the lower threshold for the Relative Strength Index (RSI). The optimizer will search for the best threshold within the range from 10 to 50 in steps of 5.
- **bb_window**: This parameter represents the window period for the Bollinger Bands. The optimizer will search for the best window period within the range from 7 to 24 in steps of 2.
- **maximize**: This parameter specifies the performance metric to maximize during optimization. 'SQN' (System Quality Number) is used to evaluate the overall quality of the trading system.
- **method**: This parameter specifies the optimization method to use. 'skopt' (Scikit-Optimize) is used for Bayesian optimization of the strategy parameters.

### Example Usage:

The optimization process will search for the best combination of `n1`, `n2`, `rsi_upper`, `rsi_lower`, and `bb_window` that maximizes the 'SQN' metric. The results of the backtest, including the optimized parameters and performance metrics, will be logged and printed.

"""
bt_results = bt.optimize(
    n1=range(10, 200, 10),
    n2=range(10, 200, 10),
    rsi_upper=range(50, 90, 5),
    rsi_lower=range(10, 50, 5),
    bb_window=range(7, 24, 2),
    maximize='SQN',
    method='skopt'
)
logger.info("Backtest results trades:")
logger.info("\n" + bt_results['_trades'].to_string())
print(bt_results)


# Plot the results
plot_filename = None
try:
    logger.info("Plotting optimized strategy")
    bt.plot(open_browser=True, filename=plot_filename, resample=False)
    webbrowser.open(plot_filename)
except Exception as err:
    logger.error(f"Plot failed {err}")

