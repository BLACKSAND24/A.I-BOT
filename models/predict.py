from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self):
        self.model = load_model('models/lstm_model.keras')
    
    def predict_from_df(self, df):
        # Normalize close prices and prepare input for LSTM
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[['close']].values)
        
        # Reshape for LSTM (samples, timesteps, features)
        if len(data) >= 60:
            X = data[-60:].reshape(1, 60, 1)
            pred = self.model.predict(X, verbose=0)
            return float(pred[0][0])
        return float(df['close'].iloc[-1])