"""
Day 3 task:LSTM Training
Author: Rohan Biswas
Date: December 2025
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

print("="*60)

SEQUENCE_LENGTH = 30  # Using last 30 days to predict next day
TARGET_COLUMN = 'Close'

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_20', 'SMA_50',
    'RSI', 'MACD',
    'Volatility', 'Volatility_Ratio',
    'Momentum_5', 'Momentum_10',
    'Volume_MA_Ratio', 'Price_Position',
    'SMA_Distance', 'Domestic_Index', 'IT_Index'
]

print(f"\nConfiguration:")
print(f"  Sequence length: {SEQUENCE_LENGTH} days")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Target: {TARGET_COLUMN}")

print("\nLoading processed data...")

train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

print(f"Train: {len(train)} samples")
print(f"Val:   {len(val)} samples")
print(f"Test:  {len(test)} samples")

print("\nNormalizing features...")

scaler = MinMaxScaler()
scaler.fit(train[FEATURE_COLS])

train_scaled = train.copy()
val_scaled = val.copy()
test_scaled = test.copy()

train_scaled[FEATURE_COLS] = scaler.transform(train[FEATURE_COLS])
val_scaled[FEATURE_COLS] = scaler.transform(val[FEATURE_COLS])
test_scaled[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

print(f"Features normalized to [0, 1] range")
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to models/scaler.pkl")

print(f"\nCreating sequences (last {SEQUENCE_LENGTH} days ---> next day)...")

def create_sequences(data, stock_name, seq_length=30):
    """
    Create sequences for LSTM training
    
    Args:
        data: DataFrame with features
        stock_name: Name of stock (for filtering)
        seq_length: Number of days to look back
    
    Returns:
        X: Input sequences (samples, seq_length, features)
        y: Target values (samples,)
    """
    stock_data = data[data['Stock'] == stock_name].sort_values('Date').reset_index(drop=True)
    
    X = []
    y = []
    
    for i in range(seq_length, len(stock_data)):
        X.append(stock_data[FEATURE_COLS].iloc[i-seq_length:i].values)
        y.append(stock_data[TARGET_COLUMN].iloc[i])
    return np.array(X), np.array(y)

stocks = train['Stock'].unique()

X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []

print("\nCreating sequences per stock:")
for stock in stocks:
    X_tr, y_tr = create_sequences(train_scaled, stock, SEQUENCE_LENGTH)
    X_train_list.append(X_tr)
    y_train_list.append(y_tr)
    
    X_v, y_v = create_sequences(val_scaled, stock, SEQUENCE_LENGTH)
    X_val_list.append(X_v)
    y_val_list.append(y_v)
    
    X_te, y_te = create_sequences(test_scaled, stock, SEQUENCE_LENGTH)
    X_test_list.append(X_te)
    y_test_list.append(y_te)
    
    print(f"  {stock:12} - Train: {len(X_tr):3d} | Val: {len(X_v):2d} | Test: {len(X_te):2d}")

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_val = np.concatenate(X_val_list, axis=0)
y_val = np.concatenate(y_val_list, axis=0)

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"\nSequences created:")
print(f"  X_train shape: {X_train.shape} (samples, seq_length, features)")
print(f"  y_train shape: {y_train.shape}")
print(f"  X_val shape:   {X_val.shape}")
print(f"  y_val shape:   {y_val.shape}")
print(f"  X_test shape:  {X_test.shape}")
print(f"  y_test shape:  {y_test.shape}")

print("\nSaving sequences...")

np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)

print("Saved sequences to data/processed/")
print("  - X_train.npy, y_train.npy")
print("  - X_val.npy, y_val.npy")
print("  - X_test.npy, y_test.npy")