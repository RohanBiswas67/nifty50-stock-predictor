# Nifty50 Stock Price Predictor

**Work in Progress** - Building in public over 7 days

## What I'm Building

An LSTM-based stock price prediction system for Indian equity markets that:
- Predicts next-day price movements for top Nifty50 stocks
- Uses deep learning to identify patterns in market data
- Provides directional predictions (UP/DOWN) with confidence scores
- Demonstrates end-to-end ML pipeline from data collection to model deployment

## Current Status: Day 3 Complete

### Day 1: Data Collection
- Built data collection pipeline for 10 Nifty50 stocks
- Fetched 4 years of historical data (2021-2025)
- Calculated technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Created exploratory data analysis with visualizations
- 9,000+ clean data points collected

**Stocks Tracked:** Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Bharti Airtel, SBI, ITC, HUL, Kotak Bank

### Day 2: Data Preprocessing & Feature Engineering
- Cleaned dataset: 8,500+ samples ready for ML
- Created 15+ predictive features:
  - Momentum indicators (5-day, 10-day)
  - Volatility ratios
  - Volume change patterns
  - Sector indices (Domestic Economy, IT)
  - Price position indicators
- Time-based train/val/test splits (70/15/15)
- Feature correlation analysis

**Key Discovery:** Indian banking stocks and Bharti Airtel show 0.97+ correlation - they move together as a "Domestic Economy Index" based on RBI policy and GDP growth.

### Day 3: LSTM Model Training
- Built 2-layer LSTM neural network (218,000+ parameters)
- Implemented multiple approaches:
  - Regression model (predicting returns)
  - Classification model (predicting UP/DOWN)
- Achieved 51% directional accuracy on test set
- Created comprehensive evaluation metrics and visualizations
- Proper training methodology with early stopping and validation

**Key Learning:** Stock market prediction with public data is extremely challenging. Market efficiency and high noise levels make prediction inherently difficult.

### Coming Next:
- Day 4: Streamlit dashboard for interactive visualization
- Day 5: Dashboard enhancements and user experience
- Day 6: Deployment and documentation
- Day 7: Final polish and launch

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
streamlit run app/dashboard.py

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