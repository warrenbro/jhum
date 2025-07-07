import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import sys

sector_map = {
    "FMCG": {
        "HUL":    {"file": "HUL.xlsx",    "lag_Lt": 12, "lag_Ut": 18},
        "ITC":    {"file": "ITC.xlsx",    "lag_Lt": 18, "lag_Ut": 12},
        "Nestle": {"file": "Nestle.xlsx", "lag_Lt": 18, "lag_Ut": 18},
    },
    "Banking": {
        "HDFC":  {"file": "HDFC.xlsx", "lag_Lt": 24, "lag_Ut": 18},
        "Kotak": {"file": "Kotak.xlsx", "lag_Lt": 18, "lag_Ut": 6},
        "SBI":   {"file": "SBI.xlsx", "lag_Lt": 18, "lag_Ut": 12},
    },
    "Oil & Energy": {
        "ONGC": {"file": "ONGC.xlsx", "lag_Lt": 18, "lag_Ut": 6},
        "IOC":  {"file": "IOC.xlsx", "lag_Lt": 24, "lag_Ut": 6},
        "RIL":  {"file": "RIL.xlsx", "lag_Lt": 24, "lag_Ut": 6},
    },
    "IT": {
        "Infosys": {"file": "Infosys.xlsx", "lag_Lt": 6, "lag_Ut": 12},
        "TCS":     {"file": "TCS.xlsx", "lag_Lt": 6, "lag_Ut": 12},
        "Wipro":   {"file": "Wipro.xlsx", "lag_Lt": 24, "lag_Ut": 24},
    },
    "Auto": {
        "M&M":    {"file": "M&M.xlsx",    "lag_Lt": 12, "lag_Ut": 12},
        "Maruti": {"file": "Maruti.xlsx", "lag_Lt": 6,  "lag_Ut": 24},
        "Tata":   {"file": "Tata.xlsx",   "lag_Lt": 18, "lag_Ut": 24},
    }
}

def select_sector_and_company():
    print("Available Sectors:")
    for idx, sector in enumerate(sector_map.keys()):
        print(f"{idx+1}. {sector}")
    sector_choice = int(input("Select sector number: ")) - 1
    sector = list(sector_map.keys())[sector_choice]
    print(f"\nAvailable Companies in {sector}:")
    companies = list(sector_map[sector].keys())
    for idx, comp in enumerate(companies):
        print(f"{idx+1}. {comp}")
    comp_choice = int(input("Select company number: ")) - 1
    company = companies[comp_choice]
    return sector, company

def create_lagged_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def run_forecast(config, company_name):
    df = pd.read_excel(config["file"])
    df['xt'] = np.minimum(df['Open'], df['Close'].shift(1))
    df['yt'] = np.maximum(df['Open'], df['Close'].shift(1))
    df['Lt'] = (df['Low'] - df['yt']) / df['yt']
    df['Ut'] = (df['High'] - df['xt']) / df['xt']
    df = df.dropna().reset_index(drop=True)
    dff=df.tail(3)
    print(dff.iloc[:, -2:])

'''    lag_Lt, lag_Ut = config["lag_Lt"], config["lag_Ut"]

    # Lt modeling
    X_Lt, y_Lt = create_lagged_features(df['Lt'].values, lag_Lt)
    model_Lt = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    model_Lt.fit(X_Lt, y_Lt)
    split_Lt = -12
    y_pred_Lt = model_Lt.predict(X_Lt[split_Lt:])
    rmse_Lt = np.sqrt(mean_squared_error(y_Lt[split_Lt:], y_pred_Lt))
    forecast_input_Lt = df['Lt'].values[-lag_Lt:]
    forecast_Lt = []
    for _ in range(12):
        pred = model_Lt.predict(forecast_input_Lt.reshape(1, -1))[0]
        forecast_Lt.append(pred)
        forecast_input_Lt = np.append(forecast_input_Lt[1:], pred)

    # Ut modeling
    X_Ut, y_Ut = create_lagged_features(df['Ut'].values, lag_Ut)
    model_Ut = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    model_Ut.fit(X_Ut, y_Ut)
    split_Ut = -12
    y_pred_Ut = model_Ut.predict(X_Ut[split_Ut:])
    rmse_Ut = np.sqrt(mean_squared_error(y_Ut[split_Ut:], y_pred_Ut))
    forecast_input_Ut = df['Ut'].values[-lag_Ut:]
    forecast_Ut = []
    for _ in range(12):
        pred = model_Ut.predict(forecast_input_Ut.reshape(1, -1))[0]
        forecast_Ut.append(pred)
        forecast_input_Ut = np.append(forecast_input_Ut[1:], pred)

    # Print results
    print(f"\n{company_name} Forecasting Results:")
    print(f"RMSE (Lt - Last 12): {rmse_Lt:.5f}")
    print(f"RMSE (Ut - Last 12): {rmse_Ut:.5f}\n")
    print("Forecasted Lt (Next 12 Months):", np.round(forecast_Lt, 5))
    print("Forecasted Ut (Next 12 Months):", np.round(forecast_Ut, 5))

    # Plot merged graph
    plt.figure(figsize=(12, 5))
    plt.plot(df['Lt'].values, label="Actual Lt", color='blue')
    plt.plot(df['Ut'].values, label="Actual Ut", color='green')
    plt.plot(range(len(df), len(df) + 12), forecast_Lt, label="Forecasted Lt", linestyle='--', color='orange', marker='o')
    plt.plot(range(len(df), len(df) + 12), forecast_Ut, label="Forecasted Ut", linestyle='--', color='red', marker='x')
    plt.title(f"{company_name} - Lt and Ut Forecasts (Next 12 Months)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Backtesting Validation (Last 12 Known Points)
    print("\nBacktesting Validation (Last 12 Known Points):")
    # For Lt
    X_Lt_all, y_Lt_all = create_lagged_features(df['Lt'].values, lag_Lt)
    y_true_Lt = y_Lt_all[-12:]
    y_pred_Lt_known = model_Lt.predict(X_Lt_all[-12:])
    rmse_backtest_Lt = np.sqrt(mean_squared_error(y_true_Lt, y_pred_Lt_known))
    print(f"Backtest RMSE (Lt): {rmse_backtest_Lt:.5f}")
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 13), y_true_Lt, marker='o', label='Actual Lt')
    plt.plot(range(1, 13), y_pred_Lt_known, marker='x', label='Predicted Lt')
    plt.title(f"{company_name} Backtest - Lt (Last 12 Known Points)")
    plt.xlabel("Month")
    plt.ylabel("Lt")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # For Ut
    X_Ut_all, y_Ut_all = create_lagged_features(df['Ut'].values, lag_Ut)
    y_true_Ut = y_Ut_all[-12:]
    y_pred_Ut_known = model_Ut.predict(X_Ut_all[-12:])
    rmse_backtest_Ut = np.sqrt(mean_squared_error(y_true_Ut, y_pred_Ut_known))
    print(f"Backtest RMSE (Ut): {rmse_backtest_Ut:.5f}")
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 13), y_true_Ut, marker='o', label='Actual Ut')
    plt.plot(range(1, 13), y_pred_Ut_known, marker='x', label='Predicted Ut')
    plt.title(f"{company_name} Backtest - Ut (Last 12 Known Points)")
    plt.xlabel("Month")
    plt.ylabel("Ut")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sector_forecast(sector_name):
    print(f"\nRunning sector-wide forecast for {sector_name} (Merged Lt/Ut per company)...")
    companies = list(sector_map[sector_name].keys())
    for i, company in enumerate(companies):
        print(f"\n=== {company} ===")
        run_forecast(sector_map[sector_name][company], company)
        if i < len(companies)-1:
            user_input = input("\nType 'next' to continue to the next company, 'menu' to return to main menu, or 'exit' to quit: ").strip().lower()
            if user_input == "menu":
                return
            elif user_input == "exit":
                print("Exiting program.")
                sys.exit(0)
'''
def main_menu():
    while True:
        print("\nForecasting Application")
        print("1. Analyze a single company")
        print("2. Analyze entire sector")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            sector, company = select_sector_and_company()
            while True:
                run_forecast(sector_map[sector][company], company)
                user_input = input("\nType 'menu' to return to main menu, 'repeat' to see the graph again, or 'exit' to quit: ").strip().lower()
                if user_input == "menu":
                    break
                elif user_input == "exit":
                    print("Exiting program.")
                    sys.exit(0)
        elif choice == "2":
            print("Available Sectors:")
            for idx, sector in enumerate(sector_map.keys()):
                print(f"{idx+1}. {sector}")
            sector_choice = int(input("Select sector number: ")) - 1
            sector = list(sector_map.keys())[sector_choice]
            sector_forecast(sector)
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()
