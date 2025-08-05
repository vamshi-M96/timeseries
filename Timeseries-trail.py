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

# Upload CSV or Excel file
file = st.file_uploader("📂 Upload your time series CSV or Excel file", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    st.write("### 🧾 Raw Data Preview")
    st.dataframe(df.head())

    date_col = st.selectbox("📅 Select the **datetime column**", df.columns)
    target_col = st.selectbox("🎯 Select the **target value column**", df.columns)

    # Parse datetime and clean data
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # Data Preview
    st.line_chart(df[target_col])

    # ----- Analysis -----
    st.markdown("---")
    st.subheader("🔍 Data Characteristics Analysis")

    # ADF Test
    st.markdown("**📉 Augmented Dickey-Fuller (ADF) Test for Stationarity**")
    adf_result = adfuller(df[target_col])
    st.write({
        "ADF Statistic": round(adf_result[0], 4),
        "p-value": round(adf_result[1], 4),
        "Used lags": adf_result[2],
        "Number of observations": adf_result[3]
    })

    if adf_result[1] < 0.05:
        st.success("✅ The data is **stationary** (p-value < 0.05)")
    else:
        st.warning("⚠️ The data is **NOT stationary** (p-value ≥ 0.05)")

    # Seasonal Decomposition
    st.markdown("**📊 Seasonal Decomposition (Trend & Seasonality Check)**")
    try:
        seasonal_period = st.number_input("Set seasonal period (e.g. 12 for monthly, 7 for weekly)", value=12)
        decomposition = seasonal_decompose(df[target_col], model='additive', period=seasonal_period)

        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        decomposition.observed.plot(ax=axs[0], title="Observed")
        decomposition.trend.plot(ax=axs[1], title="Trend")
        decomposition.seasonal.plot(ax=axs[2], title="Seasonal")
        decomposition.resid.plot(ax=axs[3], title="Residual")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Seasonal decomposition failed: {e}")

    # Forecasting Section
    st.markdown("---")
    st.subheader("📈 Forecasting")

    model_type = st.selectbox("🔧 Choose a Forecasting Model", [
        "ARIMA", "Holt-Winters", "Simple Exponential Smoothing", "Moving Average", "Seasonal Naive"
    ])
    periods = st.number_input("🔮 Forecast periods (future steps)", min_value=1, value=12)

    forecast = None

    if model_type == "ARIMA":
        st.info("ARIMA: Good for trend + autocorrelation. Use on stationary series.")
        p = st.slider("AR (p)", 0, 10, 1)
        d = st.slider("I (d)", 0, 2, 1)
        q = st.slider("MA (q)", 0, 10, 1)

        model = ARIMA(df[target_col], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)

    elif model_type == "Holt-Winters":
        st.info("Holt-Winters: Best when data has trend & seasonality.")
        seasonal = st.selectbox("Seasonality Type", ["add", "mul", None], index=0)
        model = ExponentialSmoothing(
            df[target_col], trend="add", seasonal=seasonal if seasonal != "None" else None,
            seasonal_periods=seasonal_period
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)

    elif model_type == "Simple Exponential Smoothing":
        st.info("Simple Exp Smoothing: Use when there's no trend/seasonality.")
        model = SimpleExpSmoothing(df[target_col])
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)

    elif model_type == "Moving Average":
        st.info("Moving Average: Good for smoothing. Not predictive but shows trend.")
        window = st.slider("Moving Average Window", 1, 30, 5)
        last_avg = df[target_col].rolling(window=window).mean().iloc[-1]
        future_index = pd.date_range(df.index[-1], periods=periods + 1, freq='D')[1:]
        forecast = pd.Series([last_avg] * periods, index=future_index)

    elif model_type == "Seasonal Naive":
        st.info("Seasonal Naive: Repeats last season's values.")
        last_values = df[target_col].iloc[-periods:]
        future_index = pd.date_range(df.index[-1], periods=periods + 1, freq='D')[1:]
        forecast = pd.Series(np.tile(last_values.values, int(np.ceil(periods / len(last_values))))[:periods], index=future_index)

    # Plot forecast if available
    if forecast is not None:
        st.success(f"✅ Forecast using {model_type} model")

        st.markdown("#### 📋 Forecasted Values")
        st.dataframe(forecast.reset_index().rename(columns={"index": "Forecast Date", 0: "Forecast Value"}))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        df[target_col].plot(ax=ax, label='Historical', color='blue')
        forecast.plot(ax=ax, label='Forecast', linestyle='--', color='orange')
        ax.set_title(f"{model_type} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)

        


   

