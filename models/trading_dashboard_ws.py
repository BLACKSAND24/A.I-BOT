# demo_ready_ai_ml_dashboard.py

import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import time

# -----------------------------
# 1️⃣ Dashboard Setup & Authentication
# -----------------------------
st.set_page_config(page_title="Demo-Ready AI/ML Crypto Dashboard", layout="wide")
st.title("Demo-Ready AI/ML Crypto Trading Dashboard")

USERNAME = "admin"
PASSWORD = "admin123"
login_user = st.text_input("Username")
login_pass = st.text_input("Password", type="password")
if login_user != USERNAME or login_pass != PASSWORD:
    st.warning("Enter correct credentials")
    st.stop()

# -----------------------------
# 2️⃣ Binance API Keys
# -----------------------------
API_KEY = "YOUR_API_KEY"         
API_SECRET = "YOUR_API_SECRET"  
client = Client(API_KEY, API_SECRET)

# -----------------------------
# 3️⃣ Dashboard Options
# -----------------------------
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
timeframes = ["1m","5m","15m"]

selected_symbol = st.selectbox("Select Trading Pair:", symbols)
selected_timeframe = st.selectbox("Select Timeframe:", timeframes)
ml_model_option = st.radio("Select ML Model:", ["LSTM", "XGBoost"])
portfolio_start = st.number_input("Simulated Starting Capital ($):", value=10000)

# Containers
price_container = st.empty()
signal_container = st.empty()
confidence_container = st.empty()
chart_container = st.empty()
portfolio_container = st.empty()
metrics_container = st.empty()
backtest_container = st.empty()
heatmap_container = st.empty()

# -----------------------------
# 4️⃣ Fetch Historical Data for Backtesting
# -----------------------------
def get_historical_klines(symbol, interval, limit=500):
    df = pd.DataFrame(client.get_klines(symbol=symbol, interval=interval, limit=limit))
    df = df.iloc[:,0:6]
    df.columns = ['timestamp','open','high','low','close','volume']
    df = df.astype({'open':'float','high':'float','low':'float','close':'float','volume':'float'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

ohlcv = get_historical_klines(selected_symbol, selected_timeframe)

# -----------------------------
# 5️⃣ Technical Indicators
# -----------------------------
ohlcv['rsi'] = ta.momentum.RSIIndicator(ohlcv['close'], window=14).rsi()
ohlcv['macd'] = ta.trend.MACD(ohlcv['close']).macd()
ohlcv['bb_high'] = ta.volatility.BollingerBands(ohlcv['close']).bollinger_hband()
ohlcv['bb_low'] = ta.volatility.BollingerBands(ohlcv['close']).bollinger_lband()

# -----------------------------
# 6️⃣ ML Model Functions
# -----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
n_steps = 10

def train_lstm(df):
    if len(df) < n_steps+5:
        return None
    data = df['close'].values.reshape(-1,1)
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(n_steps, len(scaled)-5):
        X.append(scaled[i-n_steps:i,0])
        y.append(scaled[i:i+5,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1],1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model

def train_xgboost(df):
    if len(df) < n_steps+5:
        return None
    X, y = [], []
    data = df['close'].values
    for i in range(n_steps, len(data)-5):
        X.append(data[i-n_steps:i])
        y.append(data[i:i+5])
    X, y = np.array(X), np.array(y)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model

def predict_next_steps(model, df):
    if ml_model_option=="LSTM":
        last_data = df['close'].values[-n_steps:].reshape(-1,1)
        scaled = scaler.transform(last_data)
        scaled = scaled.reshape((1,n_steps,1))
        pred_scaled = model.predict(scaled, verbose=0)
        pred = scaler.inverse_transform(pred_scaled.T).flatten()
    else:
        last_data = df['close'].values[-n_steps:]
        pred = model.predict(last_data.reshape(1,-1)).flatten()
    return pred

# -----------------------------
# 7️⃣ Backtesting
# -----------------------------
def backtest(df, model):
    portfolio = portfolio_start
    trades = []
    for i in range(n_steps, len(df)-5):
        current_price = df['close'].iloc[i]
        if ml_model_option=="LSTM":
            last_data = df['close'].iloc[i-n_steps:i].values.reshape(-1,1)
            scaled = scaler.transform(last_data)
            scaled = scaled.reshape((1,n_steps,1))
            pred = scaler.inverse_transform(model.predict(scaled).T).flatten()
        else:
            last_data = df['close'].iloc[i-n_steps:i].values
            pred = model.predict(last_data.reshape(1,-1)).flatten()
        avg_pred = np.mean(pred)
        if avg_pred > current_price:
            portfolio += (avg_pred-current_price)
            trades.append((df['timestamp'].iloc[i], 'BUY', current_price))
        else:
            portfolio += (current_price-avg_pred)
            trades.append((df['timestamp'].iloc[i], 'SELL', current_price))
    return portfolio, trades

# -----------------------------
# 8️⃣ Train Model and Predict
# -----------------------------
if ml_model_option=="LSTM":
    model = train_lstm(ohlcv)
else:
    model = train_xgboost(ohlcv)

predicted_prices = predict_next_steps(model, ohlcv)

# -----------------------------
# 9️⃣ Generate Signal
# -----------------------------
current_price = ohlcv['close'].iloc[-1]
avg_pred = np.mean(predicted_prices)
signal = "BUY" if avg_pred>current_price else "SELL"
confidence = abs(avg_pred-current_price)/current_price*100

price_container.metric(f"{selected_symbol} Price", f"${current_price:.2f}")
signal_container.markdown(f"**Trading Signal:** `{signal}`")
confidence_container.markdown(f"**Signal Confidence:** {confidence:.2f}%")

# -----------------------------
# 10️⃣ Portfolio Simulation
# -----------------------------
portfolio_value, trades = backtest(ohlcv, model)
portfolio_container.markdown(f"**Simulated Portfolio Value:** ${portfolio_value:.2f}")

# -----------------------------
# 11️⃣ Chart + Trend
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ohlcv['timestamp'], open=ohlcv['open'], high=ohlcv['high'], low=ohlcv['low'], close=ohlcv['close'], name="Price"
))
fig.add_trace(go.Bar(
    x=ohlcv['timestamp'], y=ohlcv['volume'], name="Volume", opacity=0.3, yaxis='y2'
))
pred_times = pd.date_range(start=ohlcv['timestamp'].iloc[-1], periods=5, freq='T')
fig.add_trace(go.Scatter(x=pred_times, y=predicted_prices, mode='lines+markers', name='Predicted Trend', line=dict(color='orange', dash='dot')))
fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title='Price', yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False), height=600)
chart_container.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 12️⃣ Profit/Loss Heatmap
# -----------------------------
if trades:
    trade_df = pd.DataFrame(trades, columns=['timestamp','type','price'])
    trade_df['pnl'] = trade_df['price'].diff().fillna(0)
    pivot = trade_df.pivot_table(index=trade_df['timestamp'].dt.date, columns='type', values='pnl', aggfunc='sum').fillna(0)
    fig2, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Profit/Loss Heatmap")
    heatmap_container.pyplot(fig2)
