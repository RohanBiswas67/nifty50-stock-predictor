# Nifty50 Stock Price Predictor

**Work in Progress** - Building in public over 7 days

## What I'm Building

An LSTM-based stock price prediction system for Indian equity markets that will:
- Predict next-day closing prices for top Nifty50 stocks
- Provide buy/hold/sell recommendations based on predictions
- Display confidence intervals and risk metrics
- Target: 70%+ directional accuracy

## Current Status: Day 1 

**Completed:**
- Data collection pipeline for 10 Nifty50 stocks
- 4 years of historical data (2021-2025)
- Technical indicators calculated (RSI, MACD, Bollinger Bands, Moving Averages)
- Exploratory data analysis with visualizations
- 9,000+ clean data points ready for modeling

**Stocks Tracked:**
Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Bharti Airtel, SBI, ITC, HUL, Kotak Bank

**Coming Next:**
- Day 2: Data preprocessing & feature engineering
- Day 3: LSTM model implementation
- Day 4: Model optimization & validation
- Day 5: Streamlit dashboard development
- Day 6: Deployment
- Day 7: Documentation & launch

## Tech Stack

**Currently Using:**
- Python 3.10+
- yfinance (NSE data)
- pandas, numpy (data processing)
- matplotlib, seaborn, plotly (visualization)

**Coming Soon:**
- PyTorch (LSTM model)
- Streamlit (web dashboard)
- Deployment platform TBD

## Technical Indicators Implemented

- Simple Moving Averages (10, 20, 50-day)
- Exponential Moving Averages (12, 26-day)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands (20-day, 2 std dev)
- Daily Returns & 20-day Rolling Volatility

## Quick Start
```bash
# Clone the repository
git clone https://github.com/RohanBiswas67/nifty50-stock-predictor.git
cd nifty50-stock-predictor

# Install dependencies
pip install -r requirements.txt

# Fetch latest data
python fetch_data.py

# Explore data
jupyter notebook notebooks/day1_exploration.ipynb
```

## Project Structure
```
nifty50-stock-predictor/
├── data/
│   └── raw/              # Historical stock data (CSV files)
├── notebooks/            
│   └── day1_exploration.ipynb  # Data exploration & visualization
├── fetch_data.py         # Data fetching script
├── requirements.txt      # Python dependencies
└── README.md
```

## Follow the Build

I'm building this in public over 7 days. Follow along:
- **LinkedIn**: Daily progress updates at [linkedin.com/in/rohan-biswas-0rb](https://www.linkedin.com/in/rohan-biswas-0rb)
- **This Repo**: Check commits for daily progress

## Contact

**Rohan Biswas**
- Email: rohanbiswas031@gmail.com
- LinkedIn: [linkedin.com/in/rohan-biswas-0rb](https://www.linkedin.com/in/rohan-biswas-0rb)
- Portfolio: [rohan-biswas-portfolio.vercel.app](https://rohan-biswas-portfolio.vercel.app)

---

:) Star this repo to follow the journey!

---

<<<<<<< HEAD
**Disclaimer:** This is an educational project. Not financial advice. Do not make trading decisions based on predictions from this model.
=======
**Disclaimer:** This is an educational project. Not financial advice. Do not make trading decisions based on predictions from this model.
>>>>>>> bd34013 (STUFFS ADDED)
