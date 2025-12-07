import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_dataset(series, look_back=20):
    X, y = [], []
    for i in range(len(series)-look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def train_lstm(close_prices, save_path='models/lstm_model.keras', scaler_path='models/scaler.save'):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.reshape(-1,1))
    X, y = create_dataset(scaled.flatten(), look_back=20)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    model.save('models/lstm_model.keras')
    print("Model saved to models/lstm_model.keras")
    joblib.dump(scaler, scaler_path)
    return model, scaler

if __name__ == '__main__':
    # quick demo using historic data CSV if available
    import sys
    data = pd.read_csv(sys.argv[1])  # expects CSV with close column
    closes = data['close'].values
    train_lstm(closes)
