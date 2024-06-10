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
from scipy.stats import normaltest  # type: ignore
from skopt import gp_minimize  # type: ignore
from skopt.space import Integer, Real  # type: ignore
import ta.momentum
import ta.trend
import ta.volatility
from skopt.utils import use_named_args  # type: ignore
import io
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

symbol = 'ENSUSDT'
days = 30
timeframe = '1m'

class Cross(Strategy):
    n1 = 28
    n2 = 102
    rsi_period = 14
    rsi_upper = 57
    rsi_lower = 27
    bb_window = 22
    bb_stddev = 2
    
    def init(self):
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
    try:
        tickers = []
        resp = client.ticker_price()
        for elem in resp:
            if 'USDT' in elem['symbol']:
                tickers.append(elem['symbol'])
        return tickers
    except ClientError as error:
        print(f"Found error. status: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}")

intervals = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080}

def klines(symbol, timeframe=timeframe, limit=1500, start=None, end=None):
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

def klines_extended(symbol, timeframe=timeframe, interval_days=days):
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

def calculate_transaction_costs(price, shares, commission_per_share=0.01, slippage=0.001):
    commission = shares * commission_per_share
    slippage_cost = price * shares * slippage
    return commission + slippage_cost

def apply_transaction_costs(trades):
    total_cost = 0
    for trade in trades:
        price, shares = trade['price'], trade['shares']
        total_cost += calculate_transaction_costs(price, shares)
    return total_cost

def monte_carlo_simulation(strategy_returns, num_simulations=1000):
    simulations = []
    for _ in range(num_simulations):
        simulated_returns = np.random.choice(strategy_returns, len(strategy_returns), replace=True)
        simulations.append(np.sum(simulated_returns))
    return simulations

def walk_forward_analysis(data, strategy, window_size=365):
    results = []
    for start in range(0, len(data) - window_size, window_size):
        train_data = data[start:start+window_size]
        test_data = data[start+window_size:start+2*window_size]
        strategy_instance = strategy()
        bt = Backtest(train_data, strategy_instance, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)
        bt.run()
        results.append(bt._results)
    return results

def optimize_parameters(train_data, strategy, search_space):
    def objective(n1, n2, rsi_period, rsi_upper, rsi_lower, bb_window, bb_stddev):
        class CrossWithParams(Cross):
            def init(self):
                self.n1 = n1
                self.n2 = n2
                self.rsi_period = rsi_period
                self.rsi_upper = rsi_upper
                self.rsi_lower = rsi_lower
                self.bb_window = bb_window
                self.bb_stddev = bb_stddev
                super().init()
                
        bt = Backtest(train_data, CrossWithParams, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)
        bt_results = bt.run()
        return -bt_results['Sharpe Ratio']
    
    res = gp_minimize(objective, search_space, n_calls=50, random_state=0)
    best_params = dict(zip(['n1', 'n2', 'rsi_period', 'bb_window', 'bb_stddev', 'rsi_upper', 'rsi_lower'], res.x))
    
    return best_params

def walk_forward_optimization(data, strategy, search_space, window_size=days, step_size=30):
    results = []
    for start in range(0, len(data) - window_size, step_size):
        train_data = data[start:start+window_size]
        test_data = data[start+window_size:start+window_size+step_size]
        # Perform optimization on train_data
        best_params = optimize_parameters(train_data, strategy, search_space)
        # Apply the best parameters on test_data
        optimized_strategy = strategy(**best_params)
        bt = Backtest(test_data, optimized_strategy, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)
        test_results = bt.run()
        results.append(test_results)
    return results

class RiskManagement:
    def __init__(self, max_drawdown, stop_loss, position_sizing):
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.position_sizing = position_sizing
        self.initial_cash = None
        self.current_cash = None
        self.current_drawdown = 0
    
    def apply(self, bt):
        self.initial_cash = bt.cash
        self.current_cash = self.initial_cash

        def _risk_management_logic():
            equity = bt.equity
            drawdown = (self.initial_cash - equity) / self.initial_cash

            # Update current drawdown
            self.current_drawdown = max(self.current_drawdown, drawdown)

            # Implement max drawdown logic
            if self.current_drawdown >= self.max_drawdown:
                bt.position.close()
                return

            # Implement stop loss logic
            if bt.position:
                for trade in bt.trades:
                    if trade.pl < -self.stop_loss * equity:
                        trade.close()

            # Implement position sizing logic
            if bt.position_size > self.position_sizing:
                bt.position.close()
                bt.buy(size=self.position_sizing)

        bt.add_ticker(_risk_management_logic)

        return bt

def scenario_analysis(data, scenarios):
    results = {}
    for scenario_name, scenario_data in scenarios.items():
        modified_data = data.copy()
        modified_data.update(scenario_data)
        result = Strategy(modified_data)
        results[scenario_name] = result
    return results


def calculate_sortino_ratio(returns):
    # Calculate the downside standard deviation
    downside_returns = np.clip(returns, a_min=0, a_max=None)
    downside_std = np.std(downside_returns)

    # Calculate the mean return
    mean_return = np.mean(returns)

    # Calculate the Sortino ratio
    sortino_ratio = mean_return / downside_std
 
    return sortino_ratio

def statistical_analysis(strategy_returns):
    stats = {
        'mean': np.mean(strategy_returns),
        'std': np.std(strategy_returns),
        'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns),
        'sortino_ratio': calculate_sortino_ratio(strategy_returns),
        'normality_test': normaltest(strategy_returns),
    }
    return stats


class PortfolioBacktest:
    def __init__(self, strategies):
        self.strategies = strategies

    def backtest(self, data):
        portfolio_returns = np.zeros(len(data))
        # Your remaining code here...

        for strategy in self.strategies:
            strategy_returns = strategy.backtest(data)
            portfolio_returns += strategy_returns / len(self.strategies)
        return portfolio_returns

def parameter_sensitivity_analysis(strategy, parameters):
    results = {}
    for param, values in parameters.items():
        for value in values:
            strategy.set_parameter(param, value)
            result = strategy.backtest()
            results[(param, value)] = result
    return results

def measure_execution_time(strategy, data):
    start_time = time()
    strategy.backtest(data)
    end_time = time()
    return end_time - start_time

# Fetch historical data for the specified symbol and timeframe
df = klines_extended(symbol, timeframe=timeframe, interval_days=days)
logger.info("\n" + df.head().to_string())

# Capture df.info() output
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
logger.info("\n" + info_str)

bt = Backtest(df, Cross, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)

bt_results = bt.run()
logger.info("Backtest results trades:")
logger.info("\n" + bt_results['_trades'].to_string())

optims = bt.optimize(
    n1=range(10, 200, 10),
    n2=range(10, 200, 10),
    rsi_upper = range(50,90,5),
    rsi_lower = range(10,50,5),
    bb_window = range(7,24,2),
    maximize='SQN',
    method='skopt'
)

plot_filename = None
try:
    logger.info("Plotting")
    bt.plot(open_browser=True, filename=plot_filename, resample=False)
    webbrowser.open(plot_filename)
except Exception as err:
    logger.error(f"Plot failed {err}")

print(optims)

# Adding Monte Carlo simulation and walk-forward analysis
strategy_returns = bt_results['Equity Final [$]'].pct_change().dropna()
monte_carlo_results = monte_carlo_simulation(strategy_returns)
walk_forward_results = walk_forward_analysis(df, Cross, window_size=1000)

# Print and log results
logger.info("Monte Carlo results:")
logger.info(monte_carlo_results)
logger.info("Walk-forward results:")
logger.info(walk_forward_results)
