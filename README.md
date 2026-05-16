# Algorithmic Trading Bot - Bitcoin Price Prediction

## Overview

This project implements a machine learning-based algorithmic trading system for Bitcoin price prediction using Long Short-Term Memory (LSTM) neural networks. The system analyzes historical Bitcoin trading data (2-hour candlesticks) and predicts future opening, closing, high, and low prices. Technical indicators such as Bollinger Bands, Simple Moving Averages, MACD, RSI, and Parabolic SAR are utilized for comprehensive market analysis.

## Project Context

This project was developed as part of a technical competition at IIT Kharagpur Technical Fest, focusing on practical applications of deep learning in quantitative finance.

## Dataset

- **Source**: Bitcoin 2-hour candlestick data (btc_2h.csv)
- **Time Period**: January 1, 2018 - January 31, 2022
- **Total Data Points**: 17,842 records
- **Features**: Open, High, Low, Close, Volume, DateTime

## Architecture

### Neural Network Model

The system employs a dual-layer LSTM architecture for sequence prediction:

```
Input Layer (1, 3) → LSTM(256, return_sequences=True) → LSTM(256) → Dense(1) → Output
```

**Model Specifications:**
- Input Shape: 1 timestep, 3 lookback periods
- First LSTM Layer: 256 units with return sequences enabled
- Second LSTM Layer: 256 units
- Output Layer: Dense layer with single neuron for scalar prediction
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Epochs: 20
- Batch Size: 50
- Lookback Period: 3 timesteps

## Technical Indicators

### 1. Simple Moving Averages (SMA)
- SMA-7: Short-term trend identification
- SMA-25: Intermediate trend analysis
- SMA-99: Long-term trend analysis

**Implementation**: ta.trend.SMAIndicator with window parameters 7, 25, and 99.

### 2. Bollinger Bands
- Upper Band: Mean + (2 × Standard Deviation)
- Middle Band: 20-period SMA
- Lower Band: Mean - (2 × Standard Deviation)

**Parameters**: Window=20, Standard Deviations=2

**Purpose**: Identifies overbought/oversold conditions and volatility extremes.

### 3. Moving Average Convergence Divergence (MACD)
- Fast EMA (12-period)
- Slow EMA (26-period)
- Signal Line: 9-period EMA of MACD

**Purpose**: Identifies trend changes and momentum shifts.

### 4. Relative Strength Index (RSI)
- Calculation Period: 14 candles
- Range: 0-100
- Overbought: > 70
- Oversold: < 30

**Purpose**: Measures momentum and identifies potential reversal points.

### 5. Parabolic SAR (Stop and Reverse)
- Acceleration Factor Step: 0.02
- Maximum Step: 2
- Reset: Yes (fillna=True)

**Purpose**: Identifies trend reversals and provides trailing stop levels.

## Model Performance

### Prediction Results

| Metric | Close Price | Open Price | High Price | Low Price |
|--------|------------|-----------|-----------|----------|
| Train RMSE | $1,746.08 | $1,435.32 | $2,116.59 | TBD |
| Test RMSE | $2,747.46 | $2,822.90 | $2,691.37 | TBD |

**Data Split**: 75% training, 25% testing

### Performance Analysis

- The model demonstrates reasonable accuracy on training data with RMSE values between $1,400-$2,100
- Test set performance shows expected generalization with RMSE values between $2,600-$2,800
- The model captures general price trends but exhibits prediction variance in volatile market conditions

## Dependencies

```
ta==0.11.0                          # Technical Analysis library
mplfinance==0.12.10b0               # Candlestick charting
matplotlib==3.7.1                   # Visualization
pandas==1.5.3                       # Data manipulation
numpy==1.23.5                       # Numerical computing
keras                               # Deep learning framework
scikit-learn==latest                # Preprocessing and metrics
tensorboardX==2.6.2.2               # TensorBoard visualization
python-dateutil>=2.8.1              # Date/time utilities
pytz>=2020.1                        # Timezone support
```

### Installation

```bash
pip install ta mplfinance matplotlib pandas numpy keras scikit-learn tensorboardX
```

## Usage

### 1. Data Loading and Preparation

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('btc_2h.csv')
df = df.sort_values('datetime')

# Normalize data
scaler = MinMaxScaler()
dataset = scaler.fit_transform(df[['close']].values)
```

### 2. Adding Technical Indicators

```python
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi

def AddIndicators(df):
    df["sma7"] = SMAIndicator(close=df["close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["close"], window=99, fillna=True).sma_indicator()
    
    indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    indicator_psar = PSARIndicator(high=df["high"], low=df["low"], close=df["close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()
    
    df["MACD"] = macd(close=df["close"], window_slow=26, window_fast=12, fillna=True)
    df["RSI"] = rsi(close=df["close"], window=14, fillna=True)
    
    return df
```

### 3. Visualization

```python
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates

# Plot OHLC with indicators
Plot_OHCL(df, ax1_indicators=["bb_bbm", "bb_bbh", "bb_bbl"], ax2_indicators=["volume"])
```

## File Structure

```
Algo-Trading-Bot/
├── Algo_trading_bot.ipynb          # Main Jupyter notebook with complete workflow
├── btc_2h.csv                      # Bitcoin OHLCV data (2-hour candles)
└── README.md                       # This file
```

## Key Features

- Multi-target prediction (OHLC prices)
- Dual-layer LSTM architecture for improved sequence learning
- Comprehensive technical indicator suite
- Data normalization using MinMaxScaler
- Train-test split validation (75-25)
- RMSE evaluation metrics
- Visualization of predictions vs. actual prices
- Candlestick chart generation with technical overlays

## Trading Strategy Implications

### Signal Generation

The combination of technical indicators enables the following trading signals:

1. **Bollinger Band Breakouts**: When price moves beyond bands, potential trend continuation
2. **SMA Crossovers**: Golden cross (SMA-7 > SMA-25) signals uptrend; Death cross signals downtrend
3. **RSI Extremes**: RSI > 70 (potential sell), RSI < 30 (potential buy)
4. **MACD Divergence**: Momentum shifts indicated by MACD crossovers
5. **Parabolic SAR**: Trailing stops and trend reversals

### Risk Management

- Position sizing based on volatility (Bollinger Band width)
- Stop-loss placement using Parabolic SAR levels
- Take-profit targets based on resistance/support levels

## Limitations

1. **Historical Data Bias**: Model trained exclusively on 2018-2022 data; performance may vary in different market regimes
2. **Prediction Variance**: LSTM predictions show increasing error for longer forecast horizons
3. **Black Swan Events**: Model cannot predict unprecedented market events
4. **Market Microstructure**: Does not account for order book dynamics, liquidity, or execution slippage
5. **Non-Stationary Data**: Cryptocurrency markets exhibit changing statistical properties over time
6. **Overfitting Risk**: LSTM models may overfit to training period patterns
7. **Real-Time Constraints**: Latency in data feeds and model execution not addressed

## Future Enhancements

1. Implement ensemble methods combining multiple LSTM architectures
2. Add attention mechanisms for feature importance weighting
3. Incorporate sentiment analysis from social media and news sources
4. Implement reinforcement learning for optimal position sizing
5. Add model interpretability through LIME or SHAP analysis
6. Deploy real-time prediction pipeline with live market data feeds
7. Integrate with trading exchange APIs for automated execution
8. Implement hierarchical time series forecasting for multiple timeframes

## Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance is not indicative of future results. The predictions generated by this model should not be considered financial advice. Users are responsible for their own investment decisions and should conduct thorough due diligence before deploying any automated trading system in production environments. The authors assume no liability for financial losses incurred through use of this system.

## References

- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Technical Analysis Library (TA): https://github.com/bukosabino/ta
- Keras/TensorFlow Documentation: https://keras.io/
- scikit-learn Preprocessing: https://scikit-learn.org/

## Author

Developed for IIT Kharagpur Technical Fest

## License

This project is provided as-is for educational purposes.
