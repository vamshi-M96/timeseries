import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="📈 Time Series Forecasting",
    layout="wide"  # 👈 enables wide layout
)

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
            st.warning("🔍 SES is best for level-only data. If data has trend or seasonality, SES may underperform.")
            model = SimpleExpSmoothing(series)
            model_fit = model.fit()
        elif model_type == "Holt-Winters":
            st.success("✅ Holt-Winters works well when data has trend and/or seasonality.")
            model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=seasonal_period)
            model_fit = model.fit()
        elif model_type == "ARIMA":
            st.info("📈 ARIMA is powerful for stationary, trend-based time series.")
        
            use_auto = st.checkbox("🔄 Use Auto ARIMA (recommended for best p,d,q)", value=True)
        
            if use_auto:
                from pmdarima import auto_arima
        
                with st.spinner("Auto ARIMA running..."):
                    auto_model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
                    model_fit = auto_model
                    st.success(f"✅ Auto ARIMA selected order: {auto_model.order}")
                    st.code(str(auto_model.summary()))
            else:
                p = st.number_input("ARIMA p (AR term)", min_value=0, value=1)
                d = st.number_input("ARIMA d (Differencing)", min_value=0, value=1)
                q = st.number_input("ARIMA q (MA term)", min_value=0, value=0)
        
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                st.success(f"✅ ARIMA model trained with order ({p}, {d}, {q})")
                st.code(str(model_fit.summary()))


    st.success(f"✅ Model '{model_type}' trained successfully")

    # --- ACF and PACF Analysis for ARIMA ---

    st.subheader("📊 ACF & PACF for ARIMA Order Selection")
    
    # Stationarity suggestion
    result = adfuller(df[target_col].dropna())
    p_val = result[1]
    if p_val < 0.05:
        st.success("✅ Series is stationary. Suggested differencing (d) = 0")
        suggested_d = 0
    else:
        st.warning("⚠️ Series is not stationary. Suggested differencing (d) = 1")
        suggested_d = 1
    
    # Let user confirm differencing
    d_input = st.selectbox("Select differencing (d)", [0, 1, 2], index=suggested_d)
    
    # Difference the series
    series_diff = df[target_col].diff(d_input).dropna() if d_input > 0 else df[target_col]
    
    # ACF plot
    fig_acf, ax_acf = plt.subplots()
    plot_acf(series_diff, ax=ax_acf, lags=40)
    st.pyplot(fig_acf)
    
    # PACF plot
    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(series_diff, ax=ax_pacf, lags=40, method='ywm')
    st.pyplot(fig_pacf)
    
    st.info("""
    Use these plots to help choose ARIMA(p, d, q):
    - **PACF spikes** → p (AR term)
    - **ACF spikes** → q (MA term)
    """)


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


         # Combine historical and forecast into one DataFrame
        combined = pd.concat([df[[target_col]], forecast], axis=1)
    
        st.subheader("📈 Forecast Line Chart (Full View)")
        st.line_chart(combined)
        # Define default zoom range
        max_zoom_days = len(df)
        default_zoom_days = 60
        
        # Zoom slider
        zoom_days = st.slider("🔍 Zoom into last N days (historical + forecast)", min_value=10, max_value=max_zoom_days, value=default_zoom_days)
        
        # Set zoom range
        min_date = df.index[-zoom_days] if len(df) > zoom_days else df.index[0]
        max_date = forecast.index[-1]
        
        # Plot as line chart
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot historical as line
        ax.plot(df.index, df[target_col], label='Historical', color='blue', linewidth=2)
        
        # Plot forecast as line
        ax.plot(forecast.index, forecast, label='Forecast', linestyle='--', color='orange', linewidth=2)
        
        # Set x-axis limits for zoom
        ax.set_xlim([min_date, max_date])
        
        # Labels and legend
        ax.set_title(f"{model_type} Forecast (Line Chart View)")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        
        # Show plot in Streamlit
        st.pyplot(fig)


