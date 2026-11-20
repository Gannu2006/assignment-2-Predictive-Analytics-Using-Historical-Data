"""
Predictive Analytics Using Historical Data
Run: python main.py
The included CSV is synthetic sample data created for this assignment.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "historical_sales.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

def load_and_clean_data(path):
    data = pd.read_csv(path, parse_dates=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    print("\n--- DATA OVERVIEW ---")
    print(data.head())
    print("\nMissing values before cleaning:\n", data.isnull().sum())
    for column in data.select_dtypes(include=np.number).columns:
        data[column] = data[column].fillna(data[column].mean())
    print("\nMissing values after cleaning:\n", data.isnull().sum())
    return data

def create_features(data):
    frame = data.copy()
    frame["Month"] = frame["Date"].dt.month
    frame["Year"] = frame["Date"].dt.year
    frame["Lag_1"] = frame["Sales"].shift(1)
    frame["Lag_3"] = frame["Sales"].shift(3)
    frame["Lag_6"] = frame["Sales"].shift(6)
    frame["Rolling_Mean_3"] = frame["Sales"].shift(1).rolling(3).mean()
    frame = frame.dropna().reset_index(drop=True)
    features = [
        "Advertising_Spend", "Product_Price", "Promotion",
        "Month", "Year", "Lag_1", "Lag_3", "Lag_6", "Rolling_Mean_3"
    ]
    return frame, features

def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions)
    }

def main():
    data = load_and_clean_data(DATA_FILE)

    # Visualize historical sales
    plt.figure(figsize=(11, 5))
    plt.plot(data["Date"], data["Sales"], marker="o", linewidth=1.5)
    plt.title("Historical Monthly Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "historical_sales.png", dpi=150)
    plt.close()

    model_data, feature_cols = create_features(data)
    X = model_data[feature_cols]
    y = model_data["Sales"]

    # Chronological split: earliest 80% for training, latest 20% for testing
    split = int(len(model_data) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42
        )
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        scores = evaluate(y_test, pred)
        rows.append({"Model": name, **scores})
        print(f"\n{name} evaluation:")
        print(pd.Series(scores).round(3))

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model based on lowest RMSE: {best_name}")

    # Plot actual vs predicted values for the best model
    best_predictions = best_model.predict(X_test)
    plt.figure(figsize=(11, 5))
    plt.plot(model_data["Date"].iloc[split:], y_test.values, label="Actual", marker="o")
    plt.plot(model_data["Date"].iloc[split:], best_predictions, label="Predicted", marker="s")
    plt.title(f"Actual vs Predicted Sales — {best_name}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Iterative 12-month forecast. Future business inputs are held at latest values.
    history = data.copy()
    future_rows = []
    for step in range(1, 13):
        next_date = history["Date"].iloc[-1] + pd.offsets.MonthBegin(1)
        last_sales = history["Sales"].values
        row = {
            "Advertising_Spend": history["Advertising_Spend"].iloc[-1],
            "Product_Price": history["Product_Price"].iloc[-1],
            "Promotion": 0,
            "Month": next_date.month,
            "Year": next_date.year,
            "Lag_1": last_sales[-1],
            "Lag_3": last_sales[-3],
            "Lag_6": last_sales[-6],
            "Rolling_Mean_3": np.mean(last_sales[-3:])
        }
        predicted_sales = float(best_model.predict(pd.DataFrame([row])[feature_cols])[0])
        future_rows.append({"Date": next_date, "Predicted_Sales": round(predicted_sales, 2)})
        new_record = {
            "Date": next_date,
            "Advertising_Spend": row["Advertising_Spend"],
            "Product_Price": row["Product_Price"],
            "Promotion": 0,
            "Sales": predicted_sales
        }
        history = pd.concat([history, pd.DataFrame([new_record])], ignore_index=True)

    forecast = pd.DataFrame(future_rows)
    forecast.to_csv(OUTPUT / "next_12_month_forecast.csv", index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(data["Date"], data["Sales"], label="Historical Sales")
    plt.plot(forecast["Date"], forecast["Predicted_Sales"], label="12-Month Forecast", marker="o")
    plt.axvline(data["Date"].max(), linestyle="--", alpha=0.6, label="Forecast Start")
    plt.title("Historical Sales and Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "12_month_forecast.png", dpi=150)
    plt.close()

    print("\n--- MODEL COMPARISON ---")
    print(metrics.round(3).to_string(index=False))
    print("\n--- NEXT 12 MONTHS FORECAST ---")
    print(forecast.to_string(index=False))
    print(f"\nCompleted. Results and charts are saved in: {OUTPUT}")

if __name__ == "__main__":
    main()
