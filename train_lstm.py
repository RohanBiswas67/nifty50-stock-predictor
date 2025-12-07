"""
Day 3: LSTM Classification Model
Author: Rohan Biswas
Date: December 2025
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import pickle
import os
import seaborn as sns

print("DAY 3: LSTM CLASSIFICATION - UP/DOWN PREDICTION")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\nLoading data...")

train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

def add_direction_target(df):
    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
    df['Next_Return'] = df.groupby('Stock')['Close'].pct_change().shift(-1)
    df['Direction'] = (df['Next_Return'] > 0).astype(int)
    return df

train = add_direction_target(train)
val = add_direction_target(val)
test = add_direction_target(test)

train = train.dropna(subset=['Next_Return'])
val = val.dropna(subset=['Next_Return'])
test = test.dropna(subset=['Next_Return'])

print(f"Train: {len(train)} samples")
print(f"Val:   {len(val)} samples")
print(f"Test:  {len(test)} samples")

print(f"\nClass distribution:")
print(f"UP (1):   {(train['Direction']==1).sum()} ({(train['Direction']==1).mean()*100:.1f}%)")
print(f"DOWN (0): {(train['Direction']==0).sum()} ({(train['Direction']==0).mean()*100:.1f}%)")

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

print("\nCreating sequences...")

SEQUENCE_LENGTH = 30

def create_sequences_classification(data, seq_length=30):
    X_list, y_list = [], []
    
    for stock in data['Stock'].unique():
        stock_data = data[data['Stock'] == stock].sort_values('Date').reset_index(drop=True)
        
        for i in range(seq_length, len(stock_data)):
            X_list.append(stock_data[FEATURE_COLS].iloc[i-seq_length:i].values)
            y_list.append(stock_data['Direction'].iloc[i])
    
    return np.array(X_list), np.array(y_list)

X_train, y_train = create_sequences_classification(train, SEQUENCE_LENGTH)
X_val, y_val = create_sequences_classification(val, SEQUENCE_LENGTH)
X_test, y_test = create_sequences_classification(test, SEQUENCE_LENGTH)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nCreating DataLoaders...")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\nBuilding LSTM Classifier...")

class StockLSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(StockLSTM_Classifier, self).__init__()
        
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
        self.fc3 = nn.Linear(32, 2)
        
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

input_size = X_train.shape[2]
model = StockLSTM_Classifier(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3)
model = model.to(device)

print(f"Output: 2 classes (DOWN=0, UP=1)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTraining...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

NUM_EPOCHS = 80
print(f"Epochs: {NUM_EPOCHS}")
print("-" * 60)

train_losses = []
val_accs = []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_y.numpy())
    
    val_acc = accuracy_score(val_true, val_preds) * 100
    val_accs.append(val_acc)
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/lstm_classifier_best.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

print("-" * 60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")

print("\nEvaluating on test set...")

model.load_state_dict(torch.load('models/lstm_classifier_best.pth'))
model.eval()

test_preds = []
test_probs = []

with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    probs = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    
    test_preds = predicted.cpu().numpy()
    test_probs = probs.cpu().numpy()

test_acc = accuracy_score(y_test, test_preds) * 100

print(f"\nTEST SET RESULTS:")
print(f"Accuracy: {test_acc:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=['DOWN', 'UP']))

cm = confusion_matrix(y_test, test_preds)
print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              DOWN  UP")
print(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       UP     {cm[1,0]:4d}  {cm[1,1]:4d}")

print("\nGenerating visualizations...")

fig = plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(val_accs)
plt.axhline(y=50, color='r', linestyle='--', label='Random (50%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 3, 4)
up_probs = test_probs[:, 1]
plt.hist(up_probs[y_test == 0], bins=50, alpha=0.5, label='Actually DOWN', color='red')
plt.hist(up_probs[y_test == 1], bins=50, alpha=0.5, label='Actually UP', color='green')
plt.xlabel('Predicted Probability of UP')
plt.ylabel('Count')
plt.title('Prediction Confidence')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
thresholds = np.linspace(0.5, 0.9, 20)
accs = []
for thresh in thresholds:
    confident_mask = (np.max(test_probs, axis=1) > thresh)
    if confident_mask.sum() > 0:
        acc = accuracy_score(y_test[confident_mask], test_preds[confident_mask]) * 100
        accs.append(acc)
    else:
        accs.append(0)

plt.plot(thresholds, accs, marker='o')
plt.xlabel('Confidence Threshold')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Confidence')
plt.grid(True)

plt.subplot(2, 3, 6)
sample_size = 200
plt.scatter(range(sample_size), y_test[:sample_size], alpha=0.5, label='Actual', s=20)
plt.scatter(range(sample_size), test_preds[:sample_size], alpha=0.5, label='Predicted', s=20, marker='x')
plt.xlabel('Time Step')
plt.ylabel('Direction (0=DOWN, 1=UP)')
plt.title('Sample Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/classifier_results.png', dpi=150)
print("Saved: models/classifier_results.png")

print("\n" + "="*60)
print(f"\nTest Accuracy: {test_acc:.2f}%")
print("="*60)