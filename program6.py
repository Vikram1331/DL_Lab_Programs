import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import urllib.request 
from tensorflow.keras.models import Sequen al 
from tensorflow.keras.layers import LSTM, GRU, Dense 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error 
url = "h ps://raw.githubusercontent.com/jbrownlee/Datasets/master/airline
passengers.csv" 
urllib.request.urlretrieve(url, "airline.csv") 
df   = pd.read_csv("airline.csv") 
data = df['Passengers'].values.astype(float) 
print(f"Dataset loaded — {len(data)} monthly records") 
print(df.head()) 
scaler      = MinMaxScaler() 
data_scaled = scaler.fit_transform(data.reshape(-1, 1)) 
def create_sequences(data, seq_len=10): 
    X, y = [], [] 
    for i in range(len(data) - seq_len): 
        X.append(data[i : i + seq_len]) 
        y.append(data[i + seq_len]) 
    return np.array(X), np.array(y) 
X, y = create_sequences(data_scaled, seq_len=10) 
split           = int(len(X) * 0.8) 
X_train, X_test = X[:split], X[split:] 
y_train, y_test = y[:split], y[split:] 
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}") 
lstm_model = Sequen al([LSTM(32, input_shape=(10, 1)), Dense(1)]) 
lstm_model.compile(opmizer='adam', loss='mse') 
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0) 
print("LSTM training done!") 
gru_model = Sequen al([GRU(32, input_shape=(10, 1)), Dense(1)]) 
gru_model.compile(opmizer='adam', loss='mse') 
gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0) 
print("GRU  training done!") 
lstm_preds = lstm_model.predict(X_test) 
gru_preds  = gru_model.predict(X_test) 
lstm_actual  = scaler.inverse_transform(lstm_preds) 
gru_actual   = scaler.inverse_transform(gru_preds) 
real_actual  = scaler.inverse_transform(y_test.reshape(-1, 1)) 
plt.figure(figsize=(8, 4)) 
plt.plot(lstm_history.history['loss'], label='LSTM Loss', color='steelblue') 
plt.plot(gru_history.history['loss'],  label='GRU Loss',  color='seagreen') 
plt. tle("Training Loss — LSTM vs GRU") 
plt.xlabel("Epoch") 
plt.ylabel("MSE Loss") 
plt.legend() 
plt. ght_layout() 
plt.show() 
plt.figure(figsize=(10, 4)) 
plt.plot(real_actual,  label='Actual', color='gray',      linewidth=1.5) 
plt.plot(lstm_actual,  label='LSTM',   color='steelblue', linewidth=1.5) 
plt.plot(gru_actual,   label='GRU',    color='seagreen',  linewidth=1.5) 
plt. tle("LSTM vs GRU — Airline Passenger Predic ons") 
plt.xlabel("Month") 
plt.ylabel("Passengers") 
plt.legend() 
plt. ght_layout() 
plt.show() 
print("\nMonth | Actual | LSTM Pred | GRU Pred") 
print("-" * 42) 
for i in range(len(real_actual)): 
    print(f"{i+1:>5} | {real_actual[i][0]:>6.0f} | {lstm_actual[i][0]:>9.0f} | 
{gru_actual[i][0]:>8.0f}") 
lstm_mse = mean_squared_error(real_actual, lstm_actual) 
gru_mse  = mean_squared_error(real_actual, gru_actual) 
print("\n" + "=" * 42) 
print(f"{'Model':<10} {'MSE':>10} {'Parameters':>15}") 
print("-" * 42) 
print(f"{'LSTM':<10} {lstm_mse:>10.2f} {lstm_model.count_params():>15}") 