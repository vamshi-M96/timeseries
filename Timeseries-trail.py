import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting App")

# Upload CSV file
file = st.file_uploader("Upload your time series CSV file", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    st.write("### Raw Data")
    st.dataframe(df.head())

    date_col = st.selectbox("Select the datetime column", df.columns)
    target_col = st.selectbox("Select the target value column", df.columns)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    st.line_chart(df[target_col])

    st.markdown("---")
    st.subheader("📊 Select Forecasting Model")

    model_type = st.selectbox("Choose a model", [
        "ARIMA",
        "Holt-Winters",
        "Simple Exponential Smoothing",
        "Moving Average",
        "Seasonal Naive"
    ])

    periods = st.number_input("Forecast periods (in time steps)", min_value=1, value=12)

    forecast = None

    if model_type == "ARIMA":
        st.info("Best for data with trend and seasonality. Works well when data is stationary after differencing.")
        p = st.slider("AR (p)", 0, 10, 1)
        d = st.slider("I (d)", 0, 2, 1)
        q = st.slider("MA (q)", 0, 10, 1)

        model = ARIMA(df[target_col], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)

    elif model_type == "Holt-Winters":
        st.info("Best for data with trend and seasonality components.")
        seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
        model = ExponentialSmoothing(df[target_col], trend="add", seasonal=seasonal, seasonal_periods=periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)

    elif model_type == "Simple Exponential Smoothing":
        st.info("Best for data with no clear trend or seasonality. Uses exponential weighting.")
        model = SimpleExpSmoothing(df[target_col])
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)

    elif model_type == "Moving Average":
        st.info("Best for smoothing out short-term fluctuations and highlighting long-term trends.")
        window = st.slider("Moving Average Window", 1, 30, 5)
        forecast = df[target_col].rolling(window=window).mean().iloc[-1]
        forecast = pd.Series([forecast]*periods, index=pd.date_range(df.index[-1], periods=periods+1, freq='D')[1:])

    elif model_type == "Seasonal Naive":
        st.info("Best for seasonal data. Forecast repeats the last observed season.")
        forecast = df[target_col].iloc[-periods:]
        forecast.index = pd.date_range(df.index[-1], periods=periods+1, freq='D')[1:]

    if forecast is not None:
        st.success(f"Forecast using {model_type} model")
        fig, ax = plt.subplots()
        df[target_col].plot(ax=ax, label='Original')
        forecast.plot(ax=ax, label='Forecast', linestyle='--')
        plt.legend()
        st.pyplot(fig)
        st.write(forecast)

    
        # Plot original + prediction
        fig, ax = plt.subplots()
        df[target_col].plot(ax=ax, label='Original', legend=True)
        forecast_df[target_col].plot(ax=ax, label='Forecast', legend=True, linestyle='--')
        plt.title(f"{model_type} Forecast")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📌 Model Use Cases Summary")
    st.markdown("""
    - **ARIMA**: Best for stationary data with autocorrelations. Suitable when trend and seasonality are removed or adjusted.
    - **Holt-Winters**: Best for data with seasonal and trend components.
    - **Simple Exponential Smoothing**: Best for short-term forecasting with no trend/seasonality.
    - **Moving Average**: Good for data smoothing and basic trend estimation.
    - **Seasonal Naive**: Best for strictly seasonal patterns (e.g., daily/weekly demand).
    """)

