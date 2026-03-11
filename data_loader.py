import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker, start_date, end_date):
    """Fetch historical data for a given ticker from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}' with provided dates.")
    return data

def preprocess_data(df):
    """Calculate log returns and realized volatility."""
    df['Returns'] = 100 * np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    # Yearly rolling volatility (252 trading days)
    df['Realized_Vol'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
    return df
