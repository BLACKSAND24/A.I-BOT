import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)
n = 500
ts = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='min')
price = 100 + np.cumsum(np.random.randn(n) * 0.2)
open_ = price + np.random.randn(n) * 0.05
high = np.maximum(open_, price) + abs(np.random.randn(n) * 0.1)
low = np.minimum(open_, price) - abs(np.random.randn(n) * 0.1)
close = price
volume = np.random.randint(1, 1000, size=n)

df = pd.DataFrame({
    'timestamp': ts,
    'open': open_,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
})
df.to_csv('data/sample_ohlcv.csv', index=False)
print('Wrote data/sample_ohlcv.csv')