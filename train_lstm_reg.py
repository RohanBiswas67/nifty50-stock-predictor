"""
Day 3: LSTM Model Training
Author: Rohan Biswas
Date: December 2025
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import os

print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

print("\nLoading data and creating return targets...")

train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

def add_return_target(df):
    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
    df['Next_Return'] = df.groupby('Stock')['Close'].pct_change().shift(-1)
    return df

train = add_return_target(train)
val = add_return_target(val)
test = add_return_target(test)

train = train.dropna(subset=['Next_Return'])
val = val.dropna(subset=['Next_Return'])
test = test.dropna(subset=['Next_Return'])

print(f"Train: {len(train)} samples")
print(f"Val:   {len(val)} samples")
print(f"Test:  {len(test)} samples")

print("\nNormalizing features...")

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_20', 'SMA_50',
    'RSI', 'MACD',
    'Volatility', 'Volatility_Ratio',
    'Momentum_5', 'Momentum_10',
    'Volume_MA_Ratio', 'Price_Position',
    'SMA_Distance', 'Domestic_Index', 'IT_Index'
]

scaler = MinMaxScaler()
scaler.fit(train[FEATURE_COLS])

train[FEATURE_COLS] = scaler.transform(train[FEATURE_COLS])
val[FEATURE_COLS] = scaler.transform(val[FEATURE_COLS])
test[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

print(f"Normalized {len(FEATURE_COLS)} features")

os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nCreating sequences...")

SEQUENCE_LENGTH = 30

def create_sequences_with_returns(data, seq_length=30):
    X_list, y_list = [], []
    
    for stock in data['Stock'].unique():
        stock_data = data[data['Stock'] == stock].sort_values('Date').reset_index(drop=True)
        
        for i in range(seq_length, len(stock_data)):
            X_list.append(stock_data[FEATURE_COLS].iloc[i-seq_length:i].values)
            y_list.append(stock_data['Next_Return'].iloc[i])
    
    return np.array(X_list), np.array(y_list)

X_train, y_train = create_sequences_with_returns(train, SEQUENCE_LENGTH)
X_val, y_val = create_sequences_with_returns(val, SEQUENCE_LENGTH)
X_test, y_test = create_sequences_with_returns(test, SEQUENCE_LENGTH)

print(f"X_train shape: {X_train.shape} (samples, 30 days, 19 features)")
print(f"y_train shape: {y_train.shape} (next-day returns)")

print("\nCreating DataLoaders...")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")

print("\nBuilding LSTM model...")

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

input_size = X_train.shape[2]  # 19 features
model = StockLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3)
model = model.to(device)

print(f"Model built:")
print(f"Input: {input_size} features")
print(f"Hidden: 128 units")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTraining...")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

NUM_EPOCHS = 100
print(f"Epochs: {NUM_EPOCHS}")
print("-" * 60)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_lstm_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

print("-" * 60)
print(f"Training complete! Best val loss: {best_val_loss:.6f}")

print("\nEvaluating on test set...")

model.load_state_dict(torch.load('models/best_lstm_model.pth'))
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy().flatten()

test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_predictions)

actual_direction = y_test > 0
pred_direction = test_predictions > 0
directional_accuracy = np.mean(actual_direction == pred_direction) * 100

print(f"\nTEST SET RESULTS:")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE:  {test_mae:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

print("\nGenerating visualizations...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(val_losses, label='Val Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(y_test, test_predictions, alpha=0.3, s=10)
plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect')
plt.xlabel('Actual Return')
plt.ylabel('Predicted Return')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(y_test[:200], label='Actual', alpha=0.7)
plt.plot(test_predictions[:200], label='Predicted', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Return')
plt.title('First 200 Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_results.png', dpi=150)
print("Saved: models/training_results.png")