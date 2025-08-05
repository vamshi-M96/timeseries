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

st.divider()
with st.expander("📘 Model Usage Guide (Click to Expand)"):
    st.markdown("""
| Model                     | Best Used For                                                                 |
|--------------------------|-------------------------------------------------------------------------------|
| **ARIMA**                | Stationary data with autocorrelations, after removing trend/seasonality.     |
| **Holt-Winters**         | Data with both trend and seasonal components.                                |
| **Simple Exp. Smoothing**| Short-term forecasting with no trend or seasonality.                         |
| **Moving Average**       | Smoothing noisy data and identifying basic trends.                           |
| **Seasonal Naive**       | Strictly seasonal patterns (e.g., same daily/weekly/monthly values).         |
""")


st.divider()
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

    
    # ----- Analysis: Stationarity & Seasonality -----
    st.markdown("---")
    st.subheader("🔍 Data Characteristics Analysis")
    
    # ADF Test
    st.markdown("**Augmented Dickey-Fuller (ADF) Test for Stationarity**")
    adf_result = adfuller(df[target_col])
    st.write({
        "ADF Statistic": round(adf_result[0], 4),
        "p-value": round(adf_result[1], 4),
        "Used lags": adf_result[2],
        "Number of observations": adf_result[3]
    })
    if adf_result[1] < 0.05:
        st.success("✅ The data is stationary (p-value < 0.05)")
    else:
        st.warning("⚠️ The data is NOT stationary (p-value ≥ 0.05)")
    
    # Seasonal Decomposition
    st.markdown("**Seasonal Decomposition (Trend & Seasonality Check)**")
    try:
        seasonal_period = st.number_input("Set seasonal period (e.g. 12 for monthly data)", value=12)
        decomposition = seasonal_decompose(df[target_col], model='additive', period=seasonal_period)
    
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        decomposition.observed.plot(ax=axs[0], title="Observed")
        decomposition.trend.plot(ax=axs[1], title="Trend")
        decomposition.seasonal.plot(ax=axs[2], title="Seasonal")
        decomposition.resid.plot(ax=axs[3], title="Residual")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Seasonal decomposition failed: {e}")


    
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
        st.success(f"✅ Forecast using {model_type} model")
    
        # Combine original and forecast for continuous plotting
        fig, ax = plt.subplots()
        
        # Plot original series
        df[target_col].plot(ax=ax, label='Original', color='blue')
    
        # Plot forecast (Series or DataFrame)
        if isinstance(forecast, pd.Series):
            forecast.plot(ax=ax, label='Forecast', linestyle='--', color='orange')
        elif isinstance(forecast, pd.DataFrame) and target_col in forecast.columns:
            forecast[target_col].plot(ax=ax, label='Forecast', linestyle='--', color='orange')
        else:
            forecast.plot(ax=ax, label='Forecast', linestyle='--', color='orange')


        # Show forecast values as table
        st.dataframe(forecast)
        
        ax.set_title(f"{model_type} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        plt.legend()
        st.pyplot(fig)
    
        


   

