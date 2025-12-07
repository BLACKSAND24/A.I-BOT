import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import threading

# -----------------------------
# 1️⃣ Dashboard Setup
# -----------------------------
st.set_page_config(page_title="AI/ML Trading Dashboard", layout="wide")
st.title("Professional AI/ML Crypto Trading Terminal")

# -----------------------------
# 2️⃣ Binance API Setup
# -----------------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(API_KEY, API_SECRET)

# -----------------------------
# 3️⃣ User Settings
# -----------------------------
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
timeframes = ["1m","5m","15m"]
ml_model_option = st.radio("Select ML Model:", ["LSTM", "XGBoost"])
starting_capital = st.number_input("Starting Capital ($):", value=10000)

# -----------------------------
# 4️⃣ Helper Functions
# -----------------------------
n_steps = 10
scaler = MinMaxScaler(feature_range=(0,1))

def get_historical_klines(symbol, interval, limit=300):
    try:
        df = pd.DataFrame(client.get_klines(symbol=symbol, interval=interval, limit=limit))
        df = df.iloc[:,0:6]
        df.columns = ['timestamp','open','high','low','close','volume']
        df = df.astype({'open':'float','high':'float','low':'float','close':'float','volume':'float'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

def add_indicators(df):
    if df.empty: return df
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

def train_lstm(df):
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
    import xgboost as xgb
    data = df['close'].values
    X, y = [], []
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
# 5️⃣ Tabs Setup
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Live Trading", "Paper Trading", "Backtesting"])

# -----------------------------
# 6️⃣ Live Trading Tab
# -----------------------------
with tab1:
    symbol_live = st.selectbox("Symbol", symbols, key="live_symbol")
    tf_live = st.selectbox("Timeframe", timeframes, key="live_tf")
    ohlcv_live = add_indicators(get_historical_klines(symbol_live, tf_live))
    model_live = train_lstm(ohlcv_live) if ml_model_option=="LSTM" else train_xgboost(ohlcv_live)

    if 'portfolio_live' not in st.session_state:
        st.session_state.portfolio_live = starting_capital
        st.session_state.trades_live = []
        st.session_state.trade_log_live = []

    # Live price fetch
    try:
        live_price = float(client.get_symbol_ticker(symbol=symbol_live)['price'])
    except:
        live_price = ohlcv_live['close'].iloc[-1]

    pred_live = predict_next_steps(model_live, ohlcv_live)
    avg_pred = np.mean(pred_live) if pred_live is not None else live_price

    # AI Signal
    signal = "BUY" if avg_pred>live_price else "SELL" if avg_pred<live_price else "HOLD"

    # Update portfolio simulation
    if signal=="BUY":
        st.session_state.portfolio_live += (avg_pred-live_price)
        st.session_state.trades_live.append((datetime.now(), 'BUY', live_price, avg_pred))
    elif signal=="SELL":
        st.session_state.portfolio_live += (live_price-avg_pred)
        st.session_state.trades_live.append((datetime.now(), 'SELL', live_price, avg_pred))

    st.metric(f"{symbol_live} Live Price", f"${live_price:.2f}")
    st.metric(f"Portfolio Value", f"${st.session_state.portfolio_live:.2f}")

    # Candlestick chart + prediction
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=ohlcv_live['timestamp'], open=ohlcv_live['open'], high=ohlcv_live['high'],
                                 low=ohlcv_live['low'], close=ohlcv_live['close'], name="Price"))
    if pred_live is not None:
        pred_times = pd.date_range(start=ohlcv_live['timestamp'].iloc[-1], periods=5, freq='T')
        fig.add_trace(go.Scatter(x=pred_times, y=pred_live, mode='lines+markers', name='Predicted Trend',
                                 line=dict(color='orange', dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Live Trade Chat")
    for t in st.session_state.trades_live[-20:]:
        st.write(f"{t[0]} | {t[1]} | Price: {t[2]:.2f} | Pred: {t[3]:.2f}")

# -----------------------------
# 7️⃣ Paper Trading Tab
# -----------------------------
with tab2:
    symbol_paper = st.selectbox("Symbol", symbols, key="paper_symbol")
    tf_paper = st.selectbox("Timeframe", timeframes, key="paper_tf")
    ohlcv_paper = add_indicators(get_historical_klines(symbol_paper, tf_paper))
    model_paper = train_lstm(ohlcv_paper) if ml_model_option=="LSTM" else train_xgboost(ohlcv_paper)

    portfolio_paper = starting_capital
    trades_paper = []

    st.subheader("Paper Trading Simulation")
    for i in range(n_steps, len(ohlcv_paper)-5, 10):  # sample every 10 candles
        price = ohlcv_paper['close'].iloc[i]
        pred = predict_next_steps(model_paper, ohlcv_paper.iloc[:i+1])
        avg_pred = np.mean(pred)
        signal = "BUY" if avg_pred>price else "SELL" if avg_pred<price else "HOLD"
        if signal=="BUY":
            portfolio_paper += (avg_pred-price)
        elif signal=="SELL":
            portfolio_paper += (price-avg_pred)
        trades_paper.append((ohlcv_paper['timestamp'].iloc[i], signal, price, avg_pred))

    st.metric("Paper Portfolio Value", f"${portfolio_paper:.2f}")
    st.dataframe(pd.DataFrame(trades_paper, columns=['Timestamp','Signal','Price','Predicted']))

# -----------------------------
# 8️⃣ Backtesting Tab
# -----------------------------
with tab3:
    symbol_bt = st.selectbox("Symbol", symbols, key="bt_symbol")
    tf_bt = st.selectbox("Timeframe", timeframes, key="bt_tf")
    ohlcv_bt = add_indicators(get_historical_klines(symbol_bt, tf_bt))
    model_bt = train_lstm(ohlcv_bt) if ml_model_option=="LSTM" else train_xgboost(ohlcv_bt)

    portfolio_bt = starting_capital
    trades_bt = []

    for i in range(n_steps, len(ohlcv_bt)-5):
        price = ohlcv_bt['close'].iloc[i]
        pred = predict_next_steps(model_bt, ohlcv_bt.iloc[:i+1])
        avg_pred = np.mean(pred)
        signal = "BUY" if avg_pred>price else "SELL" if avg_pred<price else "HOLD"
        if signal=="BUY":
            portfolio_bt += (avg_pred-price)
        elif signal=="SELL":
            portfolio_bt += (price-avg_pred)
        trades_bt.append((ohlcv_bt['timestamp'].iloc[i], signal, price, avg_pred))

    st.metric("Backtest Portfolio Value", f"${portfolio_bt:.2f}")
    st.subheader("Trade History")
    st.dataframe(pd.DataFrame(trades_bt, columns=['Timestamp','Signal','Price','Predicted']))

    # Equity curve
    equity = [starting_capital]
    for t in trades_bt:
        equity.append(equity[-1]+(t[3]-t[2] if t[1]=="BUY" else t[2]-t[3] if t[1]=="SELL" else 0))
    st.line_chart(equity)
