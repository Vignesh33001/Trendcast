import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fetch stock data and compute indicators
def fetch_data(ticker, start="2015-01-01", end="2025-01-01"):
    data = yf.download(ticker, start=start, end=end)

    # Ensure "Close" column is a Pandas Series (1D)
    close_series = data["Close"].squeeze()

    # Compute RSI
    rsi = RSIIndicator(close=close_series, window=14).rsi()
    data["RSI"] = rsi.squeeze()

    # Compute MACD
    macd = MACD(close_series, window_slow=26, window_fast=12, window_sign=9)
    data["MACD"] = macd.macd().squeeze()
    data["MACD_Signal"] = macd.macd_signal().squeeze()

    # Debugging prints
    print("Close column shape:", close_series.shape)
    print("RSI shape:", data["RSI"].shape)
    print("MACD shape:", data["MACD"].shape)
    print("MACD_Signal shape:", data["MACD_Signal"].shape)

    data.dropna(inplace=True)
    return data[["Close", "RSI", "MACD", "MACD_Signal"]]

# Apply FFT to extract dominant cycles
def apply_fft(data):
    close_prices = data["Close"].values
    fft_result = np.fft.fft(close_prices)
    fft_magnitudes = np.abs(fft_result)
    
    # Keep only the most significant frequencies
    top_k = 5
    dominant_frequencies = np.argsort(-fft_magnitudes)[:top_k]

    return dominant_frequencies, fft_magnitudes

# Train ARIMA model for time series prediction
def train_arima(data):
    close_prices = data["Close"].values
    model = ARIMA(close_prices, order=(5, 1, 0))  # ARIMA(p=5, d=1, q=0)
    model_fit = model.fit()
    return model_fit

# Run for NIFTY 50 and SENSEX
tickers = ["^NSEI", "^BSESN"]  # NIFTY 50 & SENSEX
for ticker in tickers:
    print(f"\nProcessing {ticker}...")

    # Fetch data
    data = fetch_data(ticker)

    # Apply FFT analysis
    freqs, magnitudes = apply_fft(data)
    print(f"Dominant frequencies for {ticker}: {freqs}")

    # Train ARIMA model
    model_fit = train_arima(data)

    # Make predictions
    forecast = model_fit.forecast(steps=30)
    print(f"Next 30-day predictions for {ticker}:\n", forecast)

    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-100:], data["Close"].values[-100:], label="Actual Prices")
    plt.plot(pd.date_range(start=data.index[-1], periods=30, freq="D"), forecast, label="Predicted Prices", linestyle="dashed")
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.legend()
    plt.show()
