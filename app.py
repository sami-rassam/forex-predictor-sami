import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Forex Currency Predictor",
    layout="wide"
)

st.title("Forex Currency Predictor")
st.write(
    "This app uses saved machine learning models to forecast future foreign exchange rates."
)

MODEL_DIR = "models"


def forecast_future(model_package, horizon=30):
    model = model_package["model"]
    currency = model_package["currency"]
    history = model_package["last_data"].copy()

    forecasts = []

    for i in range(horizon):
        next_date = history["Date"].max() + pd.Timedelta(days=1)

        row = {
            "day": next_date.day,
            "month": next_date.month,
            "year": next_date.year,
            "dayofweek": next_date.dayofweek,
            "lag_1": history[currency].iloc[-1],
            "lag_7": history[currency].iloc[-7],
            "lag_30": history[currency].iloc[-30],
            "rolling_7": history[currency].tail(7).mean(),
            "rolling_30": history[currency].tail(30).mean()
        }

        X_future = pd.DataFrame([row])

        prediction = model.predict(X_future)[0]

        forecasts.append({
            "Date": next_date,
            "Forecast": prediction
        })

        new_row = {
            "Date": next_date,
            currency: prediction
        }

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)


if not os.path.exists(MODEL_DIR):
    st.error("The models folder does not exist. Please train and save models first.")
    st.stop()

model_files = [
    file for file in os.listdir(MODEL_DIR)
    if file.endswith("_model.pkl")
]

if len(model_files) == 0:
    st.error("No saved models found. Please run the notebook first.")
    st.stop()

currency_display_names = [
    file.replace("_model.pkl", "").replace("_", " ")
    for file in model_files
]

selected_currency = st.selectbox(
    "Select a currency",
    currency_display_names
)

forecast_horizon = st.slider(
    "Select forecast horizon in days",
    min_value=7,
    max_value=90,
    value=30
)

if st.button("Generate Forecast"):
    selected_index = currency_display_names.index(selected_currency)

    selected_model_file = model_files[selected_index]

    model_path = os.path.join(MODEL_DIR, selected_model_file)

    model_package = joblib.load(model_path)

    forecast_df = forecast_future(
        model_package,
        horizon=forecast_horizon
    )

    st.subheader("Model Used")
    st.write(model_package["model_name"])

    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

    st.subheader("Forecast Chart")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecast"],
        marker="o"
    )

    ax.set_title(f"{selected_currency} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Exchange Rate")
    plt.xticks(rotation=45)

    st.pyplot(fig)