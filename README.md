# Nifty50 Stock Price Predictor

**[LIVE DASHBOARD](https://nifty50-stock-predictor-dashboard.streamlit.app/)** - Try it now!

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An LSTM-based stock price prediction system for Indian equity markets that:
- Predicts next-day price movements for top Nifty50 stocks
- Uses deep learning to identify patterns in market data
- Provides directional predictions (UP/DOWN) with confidence scores
- Demonstrates end-to-end ML pipeline from data collection to model deployment

## Tech Stack

**Data & Processing:**
- Python 3.10+
- yfinance (NSE data)
- pandas, numpy (data processing)
- scikit-learn (preprocessing, metrics)

**Machine Learning:**
- PyTorch (LSTM model)
- Custom feature engineering pipeline

**Visualization:**
- matplotlib, seaborn, plotly (charts)
- streamlit

**Development:**
- Git/GitHub (version control)
- Jupyter notebooks (exploration)

## Technical Features

**Indicators Calculated:**
- Simple Moving Averages (10, 20, 50-day)
- Exponential Moving Averages (12, 26-day)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands (20-day, 2 std dev)
- Daily Returns & 20-day Rolling Volatility

**Engineered Features:**
- Price momentum (5-day, 10-day)
- Volatility ratios
- Volume change patterns
- Sector correlation indices
- Price position relative to ranges

## Quick Start
```bash
# Clone the repository
git clone https://github.com/RohanBiswas67/nifty50-stock-predictor.git
cd nifty50-stock-predictor

# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch latest data
python fetch_data.py

# Step 2: Preprocess data and engineer features
python preprocess_data.py

# Step 3: Create sequences for LSTM
python create_sequences.py

# Step 4: Train the model
python train_lstm.py

# Explore the data and results
jupyter notebook notebooks/exploration.ipynb
jupyter notebook notebooks/feature_analysis.ipynb

# Running the dashboard

### Option 1 : Run Locally
streamlit run app/dashboard.py

### Option 2 : Use Live Dashboard ( Recommended )
Visit the deployed dashboard: **[https://nifty50-stock-predictor-dashboard.streamlit.app/](https://nifty50-stock-predictor-dashboard.streamlit.app/)**

```
The dashboard will open at 'http://localhost:8501'

## Project Structure
```
nifty50-stock-predictor/
├── data/
│   ├── raw/              # Historical stock data (CSV files)
│   └── processed/        # Cleaned data, engineered features, sequences
├── models/               # Trained model weights and scaler
├── notebooks/            
│   ├── day1_exploration.ipynb      # Data exploration & visualization
│   └── day2_feature_analysis.ipynb # Feature engineering analysis
├── fetch_data.py         # Data collection script
├── preprocess_data.py    # Feature engineering pipeline
├── create_sequences.py   # Create LSTM input sequences
├── train_lstm.py         # Model training script
├── requirements.txt      # Python dependencies
└── README.md
```

## Model Performance

**Test Set Results:**
- Directional Accuracy: 51%
- Approach: Binary classification (UP/DOWN)
- Model: 2-layer LSTM with dropout
- Training: 80 epochs with early stopping


## Follow the Build
Building in public over 7 days:
- **LinkedIn**: Daily updates at [linkedin.com/in/rohan-biswas-0rb](https://www.linkedin.com/in/rohan-biswas-0rb)
- **GitHub**: Check commits for progress

## Contact

**Rohan Biswas**
- Email: rohanbiswas031@gmail.com
- LinkedIn: [linkedin.com/in/rohan-biswas-0rb](https://www.linkedin.com/in/rohan-biswas-0rb)
- Portfolio: [rohan-biswas-portfolio.vercel.app](https://rohan-biswas-portfolio.vercel.app)

---

:) Star this repo to follow the journey!

---

**Disclaimer:** This is an educational project demonstrating machine learning techniques. Not financial advice. Do not make trading decisions based on predictions from this model.