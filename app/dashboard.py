import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Nifty50 Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_data():
    try:
        train = pd.read_csv('data/processed/train.csv')
        val = pd.read_csv('data/processed/val.csv')
        test = pd.read_csv('data/processed/test.csv')
        all_data = pd.concat([train, val, test], ignore_index=True)
        all_data['Date'] = pd.to_datetime(all_data['Date'])
        return all_data
    except:
        return None

@st.cache_resource
def load_model():
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        model = StockLSTM_Classifier(input_size=19, hidden_size=128, num_layers=2, dropout=0.3)
        model.load_state_dict(torch.load('models/lstm_classifier_best.pth', map_location='cpu'))
        model.eval()
        
        return model, scaler
    except:
        return None, None

data = load_data()
model, scaler = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Model Performance", "Live Predictions"])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Nifty50 Stock Predictor**

An LSTM-based deep learning system for Indian stock market prediction.

Built by: Rohan Biswas  
Date: December 2025
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Disclaimer")
st.sidebar.warning("Educational project only. Not financial advice.")

# Main content
if page == "Home":
    st.markdown("<h1 class='main-header'>Nifty50 Stock Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>LSTM-Based Deep Learning for Indian Stock Market Prediction</p>", unsafe_allow_html=True)
    
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Tracked", len(data['Stock'].unique()))
        with col2:
            st.metric("Data Points", f"{len(data):,}")
        with col3:
            st.metric("Date Range", f"{(data['Date'].max() - data['Date'].min()).days} days")
        with col4:
            st.metric("Model Accuracy", "51.2%")
        
        st.markdown("---")
        
        st.subheader("Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What This System Does
            
            - **Collects** 4 years of historical data for 10 Nifty50 stocks
            - **Engineers** 15+ predictive features from raw price data
            - **Trains** an LSTM neural network on 8,500+ samples
            - **Predicts** next-day price movements (UP/DOWN)
            - **Visualizes** results through interactive dashboard
            
            ### Tech Stack
            
            - **Data:** yfinance, pandas, numpy
            - **ML:** PyTorch LSTM (218K parameters)
            - **Features:** Technical indicators + custom engineering
            - **UI:** Streamlit with Plotly visualizations
            """)
        
        with col2:
            st.markdown("""
            ### Stocks Tracked
            
            1. **Reliance Industries** (RELIANCE.NS)
            2. **TCS** (TCS.NS)
            3. **HDFC Bank** (HDFCBANK.NS)
            4. **Infosys** (INFY.NS)
            5. **ICICI Bank** (ICICIBANK.NS)
            6. **Bharti Airtel** (BHARTIARTL.NS)
            7. **State Bank of India** (SBIN.NS)
            8. **ITC** (ITC.NS)
            9. **Hindustan Unilever** (HINDUNILVR.NS)
            10. **Kotak Mahindra Bank** (KOTAKBANK.NS)
            """)
        
        st.markdown("---")
        
        st.subheader("Stock Performance Summary")
        
        returns_data = []
        for stock in data['Stock'].unique():
            stock_data = data[data['Stock'] == stock].sort_values('Date')
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            total_return = ((end_price - start_price) / start_price) * 100
            returns_data.append({'Stock': stock, 'Return (%)': total_return})
        
        returns_df = pd.DataFrame(returns_data).sort_values('Return (%)', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=returns_df['Stock'],
                y=returns_df['Return (%)'],
                marker_color=['green' if x > 0 else 'red' for x in returns_df['Return (%)']],
                text=returns_df['Return (%)'].round(2),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Total Returns Over Period",
            xaxis_title="Stock",
            yaxis_title="Return (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Data not found. Please run the data collection scripts first.")

elif page == "Data Explorer":
    st.title("Data Explorer")
    
    if data is not None:
        tab1, tab2, tab3 = st.tabs(["Price Charts", "Technical Indicators", "Correlation Analysis"])
        
        with tab1:
            st.subheader("Stock Price Trends")
            
            selected_stocks = st.multiselect(
                "Select stocks to compare",
                options=data['Stock'].unique(),
                default=list(data['Stock'].unique())[:3]
            )
            
            if selected_stocks:
                fig = go.Figure()
                
                for stock in selected_stocks:
                    stock_data = data[data['Stock'] == stock].sort_values('Date')
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['Close'],
                        name=stock,
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title="Stock Price Comparison",
                    xaxis_title="Date",
                    yaxis_title="Close Price (₹)",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Normalized Performance (Base = 100)")
                
                fig2 = go.Figure()
                
                for stock in selected_stocks:
                    stock_data = data[data['Stock'] == stock].sort_values('Date')
                    normalized = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
                    
                    fig2.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=normalized,
                        name=stock,
                        mode='lines'
                    ))
                
                fig2.update_layout(
                    title="Normalized Stock Performance",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            selected_stock = st.selectbox("Select stock", data['Stock'].unique())
            stock_data = data[data['Stock'] == selected_stock].sort_values('Date')
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume'),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_50'], name='SMA 50', line=dict(color='red', dash='dash')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MACD'], name='MACD', line=dict(color='green')), row=3, col=1)
            
            fig.add_trace(go.Bar(x=stock_data['Date'], y=stock_data['Volume'], name='Volume', marker_color='lightblue'), row=4, col=1)
            
            fig.update_layout(height=1000, showlegend=True)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Stock Correlation Matrix")
            
            pivot_df = data.pivot_table(index='Date', columns='Stock', values='Close')
            correlation = pivot_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Stock Price Correlation Matrix',
                height=600,
                width=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Key Finding: ICICI Bank and Bharti Airtel show 0.97+ correlation - they form a 'Domestic Economy Index' influenced by RBI policy and Indian GDP growth.")

elif page == "Model Performance": 
    st.title("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", "51.2%", help="Directional accuracy on unseen test data")
    with col2:
        st.metric("Model Parameters", "218,786", help="Total trainable parameters in LSTM")
    with col3:
        st.metric("Training Epochs", "80", help="Number of training iterations")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Training History", "Confusion Matrix", "Performance Analysis"])
    
    with tab1:
        st.subheader("Training Progress")
        st.info("Training curves show model learning over 80 epochs")
        
        import os
        if os.path.exists('models/classifier_results.png'):
            st.image('models/classifier_results.png')
        else:
            st.warning("Training results image not found. Please run train_lstm.py first.")
            
            # Show alternative visualization
            st.markdown("""
            **Expected visualizations:**
            - Training loss curve over 80 epochs
            - Validation accuracy progression
            - Confusion matrix heatmap
            - Prediction confidence distribution
            - Accuracy vs confidence threshold
            - Sample predictions timeline
            """)
    
    with tab2:
        st.subheader("Classification Performance")
        
        st.markdown("""
        **Test Set Results:**
        - Accuracy: 51.21%
        - Precision (DOWN): 0.55
        - Precision (UP): 0.49
        - Recall (DOWN): 0.38
        - Recall (UP): 0.65
        
        Confusion Matrix:
```
                    Predicted
                    DOWN    UP
        Actual DOWN  217   347
               UP    175   331
```
        
        Interpretation:
        - Model predicts UP more frequently (tendency to be optimistic)
        - Better at identifying UP movements (65% recall) than DOWN (38% recall)
        - Overall accuracy near random (50%) indicates market unpredictability
        """)
    
    with tab3:
        st.subheader("Why 51% Accuracy?")
        
        st.markdown("""
        ### The Reality of Stock Market Prediction
        
        **51% accuracy is close to random (50%) - but this is expected.**
        
        
        
        **The goal wasn't 90% accuracy - it was building a real ML system and understanding the domain.**
        """)

elif page == "Live Predictions":
    st.title("Live Predictions")
    
    if model is not None and scaler is not None and data is not None:
        st.info("Select a stock to see the model's prediction for next-day movement")
        
        selected_stock = st.selectbox("Choose Stock", data['Stock'].unique())
        
        stock_data = data[data['Stock'] == selected_stock].sort_values('Date')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{selected_stock} - Recent Performance")
            
            recent_data = stock_data.tail(90)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=recent_data['Date'],
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name='Price'
            ))
            
            fig.update_layout(
                title=f"{selected_stock} - Last 90 Days",
                yaxis_title="Price (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Latest Data")
            
            latest = stock_data.iloc[-1]
            
            st.metric("Close Price", f"₹{latest['Close']:.2f}")
            st.metric("Daily Change", f"{latest['Daily_Return']*100:.2f}%", 
                     delta=f"{latest['Daily_Return']*100:.2f}%")
            st.metric("RSI", f"{latest['RSI']:.2f}")
            st.metric("Volume", f"{latest['Volume']:,.0f}")
        
        st.markdown("---")
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Running model..."):
                FEATURE_COLS = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_10', 'SMA_20', 'SMA_50',
                    'RSI', 'MACD',
                    'Volatility', 'Volatility_Ratio',
                    'Momentum_5', 'Momentum_10',
                    'Volume_MA_Ratio', 'Price_Position',
                    'SMA_Distance', 'Domestic_Index', 'IT_Index'
                ]
                
                last_30_days = stock_data[FEATURE_COLS].tail(30).values
                last_30_days_scaled = scaler.transform(last_30_days)
                
                input_tensor = torch.FloatTensor(last_30_days_scaled).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = probs[0][prediction].item() * 100
                
                st.markdown("Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.success("UP")
                        st.write("Model predicts price will **increase** tomorrow")
                    else:
                        st.error("DOWN")
                        st.write("Model predicts price will **decrease** tomorrow")
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    if confidence > 60:
                        st.success("High confidence")
                    elif confidence > 55:
                        st.warning("Moderate confidence")
                    else:
                        st.info("Low confidence")
                
                with col3:
                    st.write("**Probability Distribution:**")
                    st.write(f"DOWN: {probs[0][0].item()*100:.1f}%")
                    st.write(f"UP: {probs[0][1].item()*100:.1f}%")
                
                st.markdown("---")
                
                st.warning("""
                 **Important Disclaimer:**
                
                - This is an **educational project**, not professional trading advice
                - Model accuracy is ~51% (barely better than random)
                - Stock markets are unpredictable and influenced by many factors
                - **Do not make trading decisions based on this prediction**
                - Past performance does not guarantee future results
                - Always consult with a licensed financial advisor
                """)
    
    else:
        st.error("Model not loaded. Please ensure training is complete.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built by <strong>Rohan Biswas</strong> | December 2025</p>
    <p>GitHub: <a href='https://github.com/RohanBiswas67/nifty50-stock-predictor'>nifty50-stock-predictor</a></p>
</div>
""", unsafe_allow_html=True)