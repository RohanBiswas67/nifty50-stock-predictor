"""
WISH ME LUCK !
Day 1: Fetch historical stock data for Nifty50 stocks
Author: Rohan Biswas
Date: December 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# Stock Indices of Choice Ofcourse
STOCKS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'SBIN': 'SBIN.NS',
    'ITC': 'ITC.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'KOTAKBANK': 'KOTAKBANK.NS'
}

# Date range: 4 years of data
END_DATE = datetime.now() # I am taking current date alright
START_DATE = END_DATE - timedelta(days=4*365)

def fetch_stock_data(ticker, name, start, end):
    """
    Fetch historical stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS')
        name: Stock name for display
        start: Start date
        end: End date
    
    Returns:
        DataFrame with stock data
    """
    print(f"Fetching {name} ({ticker})...")
    
    try:
        stock = yf.Ticker(ticker) #So here I am just downloading the data
        df = stock.history(start=start, end=end)
        
        if df.empty:
            print(f" No data found for {name}")
            return None
        
        # Add stock identifier
        df['Stock'] = name
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        print(f"Downloaded {len(df)} days of data")
        return df
        
    except Exception as e:
        print(f"Error fetching {name}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def save_data(df, filename):
    filepath = os.path.join('data', 'raw', filename)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

def main():
    print("=" * 60)
    print("NIFTY50 STOCK DATA FETCHER")
    print("=" * 60)
    print(f"Fetching data from {START_DATE.date()} to {END_DATE.date()}")
    print(f"Total stocks: {len(STOCKS)}")
    print("=" * 60)
    print()
    
    os.makedirs('data/raw', exist_ok=True)
    
    all_data = []
    
    # Fetch data for each stock
    for name, ticker in STOCKS.items():
        df = fetch_stock_data(ticker, name, START_DATE, END_DATE)
        
        if df is not None:
            # Add technical indicators
            df = calculate_technical_indicators(df)
            
            # Save individual stock data
            save_data(df, f"{name}_raw.csv")
            
            all_data.append(df)
        
        # Be nice to Yahoo Finance API
        time.sleep(1)
        print()
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        save_data(combined_df, 'all_stocks_combined.csv')
        
        print("=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total stocks fetched: {len(all_data)}")
        print(f"Total data points: {len(combined_df)}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        print(f"Columns: {', '.join(combined_df.columns)}")
        print("=" * 60)
        
        # Basic statistics
        print("\nBASIC STATISTICS:")
        print(combined_df.groupby('Stock')['Close'].agg(['mean', 'std', 'min', 'max']))
        
        return combined_df
    else:
        print("No data was fetched successfully")
        return None

if __name__ == "__main__":
    df = main()
    
    if df is not None:
        print("\nData is ready for analysis.")
    else:
        print("\nData fetching failed.")
