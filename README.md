# Backtesting Binance Futures

## Description

This is a comprehensive Python project designed to provide a robust framework for backtesting trading strategies on Binance futures data. Utilizing the `backtesting.py` library, `scikit-optimize` for Bayesian optimization, and various technical analysis tools from the `ta` library, this project enables users to develop, optimize, and analyze trading strategies efficiently.

## Features

- **Historical Data Retrieval**: Seamlessly fetch historical market data from Binance Futures for specified symbols and timeframes.
- **Strategy Implementation**: Define and implement custom trading strategies using various technical indicators such as SMA, RSI, MACD, and Bollinger Bands.
- **Optimization**: Optimize strategy parameters using Bayesian optimization with `scikit-optimize` to maximize performance metrics like the Sharpe Ratio.
- **Backtesting**: Perform backtesting on historical data to evaluate strategy performance, including key metrics like return, drawdown, and trade statistics.
- **Risk Management**: Implement risk management techniques such as stop-loss, position sizing, and max drawdown limits.
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

## Performance Analysis

### Sharpe Ratio

The Sharpe Ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. It's a way to understand the return of an investment per unit of risk taken.

**Formula:**
\[ \text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p} \]

Where:
- \( R_p \) is the expected portfolio return.
- \( R_f \) is the risk-free rate (e.g., returns of a U.S. Treasury bond).
- \( \sigma_p \) is the standard deviation of the portfolio return (a measure of risk).

**Interpretation:**
- A higher Sharpe Ratio indicates better risk-adjusted performance.
- A Sharpe Ratio greater than 1 is generally considered good, greater than 2 is very good, and greater than 3 is excellent.

### Sortino Ratio

The Sortino Ratio is similar to the Sharpe Ratio but differentiates harmful volatility from total overall volatility by using the downside risk (standard deviation of negative returns) instead of the standard deviation of all returns.

**Formula:**
\[ \text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d} \]

Where:
- \( R_p \) is the expected portfolio return.
- \( R_f \) is the risk-free rate.
- \( \sigma_d \) is the downside deviation (standard deviation of negative returns).

**Interpretation:**
- The Sortino Ratio provides a more accurate measure of risk-adjusted return for investments that are not normally distributed or have large negative returns.
- A higher Sortino Ratio indicates better risk-adjusted performance with a focus on downside risk.

### Calmar Ratio

The Calmar Ratio measures the return of an investment relative to its maximum drawdown. It is used to evaluate the risk-adjusted performance of an investment by focusing on the potential for significant losses.

**Formula:**
\[ \text{Calmar Ratio} = \frac{R_p}{\text{Max Drawdown}} \]

Where:
- \( R_p \) is the annualized return of the portfolio.
- Max Drawdown is the maximum observed loss from a peak to a trough of a portfolio before a new peak is attained.

**Interpretation:**
- A higher Calmar Ratio indicates better performance, as it suggests that the investment is providing good returns relative to the risk of significant losses.
- Investors typically seek a high Calmar Ratio to ensure that the potential for drawdowns is justified by higher returns.

### Summary

- **Sharpe Ratio**: Measures risk-adjusted returns using the standard deviation of all returns. Higher is better.
- **Sortino Ratio**: Measures risk-adjusted returns using the standard deviation of negative returns (downside risk). Higher is better and more sensitive to negative returns.
- **Calmar Ratio**: Measures risk-adjusted returns using the maximum drawdown. Higher is better, focusing on the risk of significant losses.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the robust trading strategies and optimization techniques provided by `backtesting.py` and `scikit-optimize`.
- Data sourced from Binance Futures API.

## Buy me a coffee :) :
   ### uni swap: sonic7even
   ### XRP(XRP): r9KEE8zj6jxYvArq3H6yReANUYMXX7WhLN
   ### MATIC(Polygon): 0xfE67a6d0b18FE3f910d69d69D31F9B9E24E17951
