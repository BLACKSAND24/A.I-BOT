import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, model_path='models/lstm_model.h5', scaler_path='models/scaler.save'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.model = None
            self.scaler = None

    def predict_from_df(self, df):
        if self.model is None:
            # fallback: naive next-step prediction = last close
            return df['close'].iloc[-1]
        close = df['close'].values
        scaled = self.scaler.transform(close.reshape(-1,1)).flatten()
        look_back = 20
        x = scaled[-look_back:]
        x = x.reshape((1, look_back, 1))
        pred_scaled = self.model.predict(x)
        pred = self.scaler.inverse_transform(pred_scaled)[0,0]
        return pred
