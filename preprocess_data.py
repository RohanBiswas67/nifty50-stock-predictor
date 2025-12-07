"""
WISH ME LUCK AGAIN

Day 2: Data Preprocessing & Feature Engineering
Author: Rohan Biswas
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)

print("\nLoading raw data...")
df = pd.read_csv('data/raw/all_stocks_combined.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"Loaded {len(df)} rows, {len(df['Stock'].unique())} stocks")

print("\nHandling missing values...")
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

df_clean = df.groupby('Stock').apply(lambda x: x.iloc[50:]).reset_index(drop=True)

print(f"\nRemoved first 50 days per stock for clean indicators")
print(f"Remaining rows: {len(df_clean)}")
remaining_nans = df_clean.isnull().sum().sum()
if remaining_nans > 0:
    print(f"Still have {remaining_nans} NaNs - filling with forward fill")
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
else:
    print("No NaNs remaining, we're good to go yeyyy!")


print("\nEngineering new features...")

def add_features(df):
    """Add additional predictive features"""
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['Price_Lag_1'] = df['Close'].shift(1)  # Last day
    df['Price_Lag_5'] = df['Close'].shift(5)  # Last week
    df['Price_Lag_10'] = df['Close'].shift(10)  # 2 weeks ago
    
    df['Momentum_5'] = (df['Close'] - df['Price_Lag_5']) / df['Price_Lag_5']
    df['Momentum_10'] = (df['Close'] - df['Price_Lag_10']) / df['Price_Lag_10']
    
    df['Volatility_Ratio'] = df['Volatility'] / df['Volatility'].rolling(30).mean()
    
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    df['SMA_Distance'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    return df

df_engineered = df_clean.groupby('Stock').apply(add_features).reset_index(drop=True)

print(f"Added {len(df_engineered.columns) - len(df_clean.columns)} new features")
print(f"  New features: Price lags, momentum, volatility ratios, volume changes")


print("\nCreating sector indices...")

# Domestic economy cluster: Banking + Telecom 
#This is from our last finding of high correlation between Bharti Airtel and Banks
DOMESTIC_CLUSTER = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'BHARTIARTL']
IT_CLUSTER = ['TCS', 'INFY']
DIVERSIFIED = ['RELIANCE', 'ITC', 'HINDUNILVR']

def add_sector_indices(df):
    """Add sector average as feature"""
    
    dates = df['Date'].unique()
    
    # Calculating the domestic economy index (average of cluster stocks)
    domestic_index = []
    it_index = []
    
    for date in dates:
        date_data = df[df['Date'] == date]
        domestic_stocks = date_data[date_data['Stock'].isin(DOMESTIC_CLUSTER)]
        if len(domestic_stocks) > 0:
            domestic_avg = domestic_stocks['Daily_Return'].mean()
        else:
            domestic_avg = 0
        domestic_index.append({'Date': date, 'Domestic_Index': domestic_avg})
        
        # This one is the IT cluster part
        it_stocks = date_data[date_data['Stock'].isin(IT_CLUSTER)]
        if len(it_stocks) > 0:
            it_avg = it_stocks['Daily_Return'].mean()
        else:
            it_avg = 0
        it_index.append({'Date': date, 'IT_Index': it_avg})

    domestic_df = pd.DataFrame(domestic_index)
    it_df = pd.DataFrame(it_index)
    
    df = df.merge(domestic_df, on='Date', how='left')
    df = df.merge(it_df, on='Date', how='left')
    
    return df

df_with_sectors = add_sector_indices(df_engineered)
print(" Added sector indices: Domestic Economy Index, IT Index")

print("\nFinal cleaning...")

before_drop = len(df_with_sectors)
df_final = df_with_sectors.dropna()
after_drop = len(df_final)

print(f"Dropped {before_drop - after_drop} rows with NaN from new features")
print(f"Final dataset: {after_drop} rows")

# Lets now split the dataset into train and test as we'll anyways do ML later 
print("\nCreating train/validation/test splits...")

# We want to split by Time (not random) - use oldest data for training ( Old is gold )
# 70% train, 15% validation, 15% test

train_data = {}
val_data = {}
test_data = {}

for stock in df_final['Stock'].unique():
    stock_data = df_final[df_final['Stock'] == stock].sort_values('Date').reset_index(drop=True)
    
    n = len(stock_data)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    
    train = stock_data[:train_size]
    val = stock_data[train_size:train_size + val_size]
    test = stock_data[train_size + val_size:]
    
    train_data[stock] = train
    val_data[stock] = val
    test_data[stock] = test
    
    print(f"{stock:12} - Train: {len(train):3d} | Val: {len(val):2d} | Test: {len(test):2d}")

train_df = pd.concat(train_data.values(), ignore_index=True)
val_df = pd.concat(val_data.values(), ignore_index=True)
test_df = pd.concat(test_data.values(), ignore_index=True)

print(f"\nTotal splits:")
print(f"  Training:   {len(train_df)} samples ({len(train_df)/len(df_final)*100:.1f}%)")
print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df_final)*100:.1f}%)")
print(f"  Test:       {len(test_df)} samples ({len(test_df)/len(df_final)*100:.1f}%)")

# Data saving 
print("\nSaving processed data...")

os.makedirs('data/processed', exist_ok=True)

train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)
df_final.to_csv('data/processed/all_data_processed.csv', index=False)

print("Saved to data/processed/")
print("  - train.csv")
print("  - val.csv")
print("  - test.csv")
print("  - all_data_processed.csv")