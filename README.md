# Binance_Futures_Scipy_Backtesting.py

## Description

Binance_Futures_Scikit_Backtesting is a comprehensive Python project designed to provide a robust framework for backtesting trading strategies on Binance futures data. Utilizing the `backtesting.py` library, `scikit-optimize` for Bayesian optimization, and various technical analysis tools from the `ta` library, this project enables users to develop, optimize, and analyze trading strategies efficiently.

## Features

- **Historical Data Retrieval**: Seamlessly fetch historical market data from Binance Futures for specified symbols and timeframes.
- **Strategy Implementation**: Define and implement custom trading strategies using various technical indicators such as SMA, RSI, MACD, and Bollinger Bands.
- **Optimization**: Optimize strategy parameters using Bayesian optimization with `scikit-optimize` to maximize performance metrics like the Sharpe Ratio.
- **Backtesting**: Perform backtesting on historical data to evaluate strategy performance, including key metrics like return, drawdown, and trade statistics.
- **Risk Management**: Implement risk management techniques such as stop-loss, position sizing, and max drawdown limits.
- **Monte Carlo Simulation**: Conduct Monte Carlo simulations to assess the robustness of strategies under different market conditions.
- **Walk-Forward Analysis**: Perform walk-forward analysis to validate the strategy's performance over different time periods.
- **Scenario Analysis**: Analyze strategy performance under various market scenarios (e.g., market crashes, bull markets).
- **Visualization**: Generate detailed visualizations of backtesting results, including equity curves, trade markers, and performance metrics.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python libraries: `pandas`, `numpy`, `scipy`, `ta`, `backtesting.py`, `scikit-optimize`, `binance-futures`, `matplotlib`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Binance_Futures_Scikit_Backtesting.git
   cd Binance_Futures_Scikit_Backtesting
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Fetch historical data and configure the backtest:
   ```python
   df = klines_extended(symbol='ENSUSDT', timeframe='5m', interval_days=60)
   bt = Backtest(df, Cross, cash=100, margin=2/10, commission=0.002, exclusive_orders=True)
   ```

2. Run the backtest:
   ```python
   bt_results = bt.run()
   ```

3. Optimize strategy parameters:
   ```python
   result = gp_minimize(objective, search_space, n_calls=50, random_state=0)
   ```

4. Perform analysis and visualize results:
   ```python
   bt.plot(open_browser=True, filename=plot_filename, resample=False)
   ```

### Example Strategy

The included `Cross` strategy uses a combination of Simple Moving Averages (SMA), Relative Strength Index (RSI), MACD, and Bollinger Bands for trade signals. Parameters are optimized using Bayesian optimization to maximize the Sharpe Ratio.

```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the robust trading strategies and optimization techniques provided by `backtesting.py` and `scikit-optimize`.
- Data sourced from Binance Futures API.
