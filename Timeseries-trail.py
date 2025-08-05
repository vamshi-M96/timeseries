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

file = st.file_uploader("📂 Upload your time series CSV or Excel file", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    st.write("### 🧾 Raw Data Preview")
    st.dataframe(df.head())

    date_col = st.selectbox("📅 Select the **datetime column**", df.columns)
    target_col = st.selectbox("🎯 Select the **target value column**", df.columns)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    st.line_chart(df[target_col])

    # Detect frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = 'D'  # fallback

    # ----- ADF Test -----
    st.markdown("---")
    st.subheader("🔍 Data Characteristics Analysis")
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

    # ----- Seasonal Decomposition -----
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

    # --------- Forecasting Section ----------
    st.markdown("---")
    st.subheader("📈 Forecasting")

    model_type = st.selectbox("🔧 Choose a Forecasting Model", [
        "ARIMA", "Holt-Winters", "Simple Exponential Smoothing", "Moving Average", "Seasonal Naive"
    ])

    # Forecast date input
    last_date = df.index[-1]
    st.write(f"📌 Last date in dataset: `{last_date.date()}` (freq: {freq})")

    forecast_end_date = st.date_input("📅 Select forecast end date", value=last_date.date())
    forecast_end_date = pd.to_datetime(forecast_end_date)

    if forecast_end_date <= last_date:
        st.error("⚠️ Forecast end date must be after the last date in the dataset.")
        st.stop()

    future_dates = pd.date_range(start=last_date, end=forecast_end_date, freq=freq)[1:]
    periods = len(future_dates)
    st.success(f"📈 Forecasting for `{periods}` periods until {forecast_end_date.date()}")

    forecast = None

    # --------- Model Fitting & Forecasting ----------
    if model_type == "ARIMA":
        st.info("ARIMA: Best for stationary data with autocorrelation.")
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
        st.info("Simple Exp Smoothing: No trend or seasonality.")
        model = SimpleExpSmoothing(df[target_col])
        model_fit = model.fit()
        forecast = model_fit.forecast(periods)

    elif model_type == "Moving Average":
        st.info("Moving Average: Smoothing only, no prediction capability.")
        window = st.slider("Moving Average Window", 1, 30, 5)
        last_avg = df[target_col].rolling(window=window).mean().iloc[-1]
        forecast = pd.Series([last_avg] * periods)

    elif model_type == "Seasonal Naive":
        st.info("Seasonal Naive: Repeats last season's pattern.")
        last_values = df[target_col].iloc[-periods:]
        repeated_values = np.tile(last_values.values, int(np.ceil(periods / len(last_values))))[:periods]
        forecast = pd.Series(repeated_values)

    # ---------- Display & Filter ----------
    if forecast is not None:
        # Apply datetime index
        forecast.index = future_dates
        forecast.name = "Forecast Value"

        st.success(f"✅ Forecast using {model_type} model")

        # 🎯 User selects range within forecast
        st.markdown("### 📆 Select Forecast Display Range")

        min_date = forecast.index.min().date()
        max_date = forecast.index.max().date()

        date_range = st.date_input(
            "Select forecast range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

            if start_date > end_date:
                st.error("⚠️ Start date must be before end date.")
                st.stop()

            filtered_forecast = forecast.loc[start_date:end_date]
        else:
            st.error("⚠️ Please select a valid date range.")
            st.stop()

        # 📋 Table
        st.markdown("#### 📋 Filtered Forecasted Values")
        st.dataframe(filtered_forecast.reset_index().rename(columns={"index": "Forecast Date"}))

        # 📈 Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        df[target_col].plot(ax=ax, label='Historical', color='blue')
        filtered_forecast.plot(ax=ax, label='Filtered Forecast', linestyle='--', color='orange')
        ax.set_title(f"{model_type} Forecast (Filtered View)")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)

   

