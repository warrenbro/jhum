import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# Step 1: Load Data and Compute Lt, Ut
# -----------------------------
df = pd.read_excel("HDFC.xlsx")

df['xt'] = np.minimum(df['Open'], df['Close'].shift(1))
df['yt'] = np.maximum(df['Open'], df['Close'].shift(1))
df['Lt'] = (df['Low'] - df['yt']) / df['yt']
df['Ut'] = (df['High'] - df['xt']) / df['xt']
df = df.dropna().reset_index(drop=True)
print(df.tail(3))
'''# -----------------------------
# Step 2: Define lagged feature function
# -----------------------------
def create_lagged_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# -----------------------------
# Step 3: Forecast Lt with lag=24
# -----------------------------
lag_Lt = 24
X_Lt, y_Lt = create_lagged_features(df['Lt'].values, lag_Lt)

model_Lt = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
model_Lt.fit(X_Lt, y_Lt)

# RMSE (Last 12 points)
split_Lt = -12
y_pred_Lt = model_Lt.predict(X_Lt[split_Lt:])
rmse_Lt = np.sqrt(mean_squared_error(y_Lt[split_Lt:], y_pred_Lt))

# Forecast Lt
forecast_input_Lt = df['Lt'].values[-lag_Lt:]
forecast_Lt = []
for _ in range(12):
    pred = model_Lt.predict(forecast_input_Lt.reshape(1, -1))[0]
    forecast_Lt.append(pred)
    forecast_input_Lt = np.append(forecast_input_Lt[1:], pred)

# -----------------------------
# Step 4: Forecast Ut with lag=18
# -----------------------------
lag_Ut = 18
X_Ut, y_Ut = create_lagged_features(df['Ut'].values, lag_Ut)

model_Ut = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
model_Ut.fit(X_Ut, y_Ut)

# RMSE (Last 12 points)
split_Ut = -12
y_pred_Ut = model_Ut.predict(X_Ut[split_Ut:])
rmse_Ut = np.sqrt(mean_squared_error(y_Ut[split_Ut:], y_pred_Ut))

# Forecast Ut
forecast_input_Ut = df['Ut'].values[-lag_Ut:]
forecast_Ut = []
for _ in range(12):
    pred = model_Ut.predict(forecast_input_Ut.reshape(1, -1))[0]
    forecast_Ut.append(pred)
    forecast_input_Ut = np.append(forecast_input_Ut[1:], pred)

# -----------------------------
# Step 5: Print RMSE and Forecasts
# -----------------------------
print(f"RMSE (Lt - Last 12): {rmse_Lt:.5f}")
print(f"RMSE (Ut - Last 12): {rmse_Ut:.5f}\n")

print("Forecasted Lt (Next 12 Months):")
print(np.round(forecast_Lt, 5))

print("\nForecasted Ut (Next 12 Months):")
print(np.round(forecast_Ut, 5))

# -----------------------------
# Step 6: Plot Results
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(df['Lt'].values, label="Actual Lt")
plt.plot(range(len(df), len(df) + 12), forecast_Lt, label="Forecasted Lt", color='orange')
plt.title("Lt Forecast (Lower Return Interval)")
plt.xlabel("Time")
plt.ylabel("Lt")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df['Ut'].values, label="Actual Ut")
plt.plot(range(len(df), len(df) + 12), forecast_Ut, label="Forecasted Ut", color='green')
plt.title("Ut Forecast (Upper Return Interval)")
plt.xlabel("Time")
plt.ylabel("Ut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Backtesting Validation (Last 12 Known Values)
# -----------------------------
# This will compare model prediction vs actual for last 12 known points

# Re-create lagged features for Lt and Ut
X_Lt_all, y_Lt_all = create_lagged_features(df['Lt'].values, lag_Lt)
X_Ut_all, y_Ut_all = create_lagged_features(df['Ut'].values, lag_Ut)

# Get predictions for last 12 known Lt values
y_true_Lt = y_Lt_all[-12:]
y_pred_Lt_known = model_Lt.predict(X_Lt_all[-12:])
rmse_backtest_Lt = np.sqrt(mean_squared_error(y_true_Lt, y_pred_Lt_known))

# Get predictions for last 12 known Ut values
y_true_Ut = y_Ut_all[-12:]
y_pred_Ut_known = model_Ut.predict(X_Ut_all[-12:])
rmse_backtest_Ut = np.sqrt(mean_squared_error(y_true_Ut, y_pred_Ut_known))

print("\n📉 Backtest Validation (Last 12 Known Points):")
print(f"Backtest RMSE (Lt): {rmse_backtest_Lt:.5f}")
print(f"Backtest RMSE (Ut): {rmse_backtest_Ut:.5f}")

# Optional: Plot actual vs predicted for backtest
plt.figure(figsize=(10, 4))
plt.plot(range(1, 13), y_true_Lt, marker='o', label='Actual Lt')
plt.plot(range(1, 13), y_pred_Lt_known, marker='x', label='Predicted Lt')
plt.title("Backtest - Lt (Last 12 Known Points)")
plt.xlabel("Month")
plt.ylabel("Lt")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, 13), y_true_Ut, marker='o', label='Actual Ut')
plt.plot(range(1, 13), y_pred_Ut_known, marker='x', label='Predicted Ut')
plt.title("Backtest - Ut (Last 12 Known Points)")
plt.xlabel("Month")
plt.ylabel("Ut")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''
