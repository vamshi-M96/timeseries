import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Time Series Forecasting")

st.title("📊 Time Series Forecasting App")

# Upload data
uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    df.columns = [col.strip() for col in df.columns]

    st.write("### 📄 Raw Data")
    st.dataframe(df.head())

    # Select datetime column and target
    date_col = st.selectbox("🕒 Select Date Column", df.columns)
    target_col = st.selectbox("🎯 Select Target Column", [col for col in df.columns if col != date_col])

    # Convert and sort
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # Inference frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = 'D'  # default to daily
    st.write(f"⏱️ Detected Frequency: `{freq}`")

    # Show plot
    st.line_chart(df[target_col])

    # Stationarity Check
    st.subheader("🔍 Stationarity Test (ADF)")
    adf_result = adfuller(df[target_col])
    st.write({
        "ADF Statistic": round(adf_result[0], 4),
        "p-value": round(adf_result[1], 4),
        "Used lags": adf_result[2],
        "Number of observations": adf_result[3],
    })
    if adf_result[1] > 0.05:
        st.warning("⚠️ The data is NOT stationary (p-value ≥ 0.05)")
    else:
        st.success("✅ The data is stationary (p-value < 0.05)")

    # Seasonal Decomposition
    st.subheader("🧭 Seasonal Decomposition")
    seasonal_period = st.number_input("Set seasonal period (e.g., 12 for monthly)", min_value=2, value=12)
    try:
        result = seasonal_decompose(df[target_col], model='additive', period=seasonal_period)
        fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        result.observed.plot(ax=ax[0], title='Observed')
        result.trend.plot(ax=ax[1], title='Trend')
        result.seasonal.plot(ax=ax[2], title='Seasonal')
        result.resid.plot(ax=ax[3], title='Residual')
        st.pyplot(fig)
    except:
        st.error("Seasonal decomposition failed. Please adjust the seasonal period.")

    # Select forecasting model
    st.subheader("🧠 Forecasting Model")
    model_type = st.selectbox("Choose a model", ["Simple Exponential Smoothing", "Holt-Winters", "ARIMA"])

    # Model training
    with st.spinner("Training model..."):
        series = df[target_col]
        model_fit = None

        if model_type == "Simple Exponential Smoothing":
            model = SimpleExpSmoothing(series)
            model_fit = model.fit()
        elif model_type == "Holt-Winters":
            model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=seasonal_period)
            model_fit = model.fit()
        elif model_type == "ARIMA":
            p = st.number_input("ARIMA p", min_value=0, value=1)
            d = st.number_input("ARIMA d", min_value=0, value=1)
            q = st.number_input("ARIMA q", min_value=0, value=0)
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()

    st.success(f"✅ Model '{model_type}' trained successfully")

    # Forecasting section
    st.subheader("📅 Forecasting")

    last_date = df.index[-1]
    st.write(f"Last date in dataset: `{last_date.date()}`")

    # User-specified forecast end date (optional)
    default_forecast_days = 30
    default_end_date = (last_date + timedelta(days=default_forecast_days)).date()
    forecast_end_date = st.date_input("Select forecast end date", value=default_end_date)

    if forecast_end_date <= last_date.date():
        st.warning("⚠️ Selected end date is before last data date. Adjusted automatically.")
        forecast_end_date = default_end_date

    forecast_end_date = pd.to_datetime(forecast_end_date)

    # Create future index and compute periods
    future_index = pd.date_range(start=last_date, end=forecast_end_date, freq=freq)[1:]
    periods = len(future_index)

    if periods == 0:
        st.error("❌ No future periods to forecast.")
    else:
        forecast = model_fit.forecast(steps=periods)
        forecast.index = future_index
        forecast.name = "Forecast Value"

        # Show forecast
        st.subheader("🔮 Forecast Results")
        st.dataframe(forecast.reset_index().rename(columns={"index": "Forecast Date"}))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        df[target_col].plot(ax=ax, label='Historical', color='blue')
        forecast.plot(ax=ax, label='Forecast', linestyle='--', color='orange')
        ax.set_title(f"{model_type} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)


   

