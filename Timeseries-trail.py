import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from io import StringIO
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 General Time Series Analysis & Forecasting")

# Upload
file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if file:
    # Read file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select datetime and target
    with st.sidebar:
        st.subheader("📌 Column Selection")
        datetime_col = st.selectbox("Select Date/Time Column", df.columns)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        target_col = st.selectbox("Select Target Variable", numeric_cols)

    # Preprocessing
    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')

    df.set_index(datetime_col, inplace=True)
    df.sort_index(inplace=True)

    st.write(f"🕒 Date Range: {df.index.min()} → {df.index.max()}")

    # Resample and interpolate
    freq = st.sidebar.selectbox("Resample Frequency", ['D', 'W', 'M'], index=0)
    resampled = df[[target_col]].resample(freq).mean().interpolate()

    st.subheader("📉 Resampled and Interpolated Data")
    st.line_chart(resampled)

    # Decomposition
    st.subheader("🔍 Seasonal Decomposition")
    try:
        result = seasonal_decompose(resampled[target_col], model="additive", period=12)
        fig = result.plot()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Decomposition failed: {e}")

    # ACF/PACF
    st.subheader("📈 ACF & PACF")
    lags = st.slider("Select number of lags", 10, 60, 20)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(resampled[target_col], lags=lags, ax=ax[0])
    plot_pacf(resampled[target_col], lags=lags, ax=ax[1])
    st.pyplot(fig)

    # Lag plot
    st.subheader("🌀 Lag Plot")
    from pandas.plotting import lag_plot
    fig2, ax2 = plt.subplots()
    lag_plot(resampled[target_col], ax=ax2)
    st.pyplot(fig2)

    # Split
    st.subheader("✂️ Train-Test Split")
    split_ratio = st.slider("Training Data Ratio", 0.5, 0.95, 0.9)
    split_point = int(len(resampled) * split_ratio)
    train, test = resampled.iloc[:split_point], resampled.iloc[split_point:]
    st.write(f"Train size: {len(train)}, Test size: {len(test)}")

    # Holt-Winters Forecast
    st.subheader("🌡️ Holt-Winters Forecast")
    try:
        hw_model = ExponentialSmoothing(train[target_col], seasonal="add", trend="add", seasonal_periods=12).fit()
        hw_forecast = hw_model.forecast(len(test))
        fig3, ax3 = plt.subplots()
        train[target_col].plot(ax=ax3, label="Train")
        test[target_col].plot(ax=ax3, label="Test")
        hw_forecast.plot(ax=ax3, label="HW Forecast")
        ax3.legend()
        st.pyplot(fig3)
        st.write("MAPE:", np.mean(np.abs((test[target_col] - hw_forecast) / test[target_col])) * 100)
    except Exception as e:
        st.warning(f"Holt-Winters failed: {e}")

    # ARIMA Forecast
    st.subheader("📉 ARIMA Forecast")
    p = st.number_input("p", 0, 10, 2)
    d = st.number_input("d", 0, 2, 1)
    q = st.number_input("q", 0, 10, 2)

    try:
        arima_model = ARIMA(train[target_col], order=(p, d, q)).fit()
        arima_forecast = arima_model.forecast(len(test))
        fig4, ax4 = plt.subplots()
        train[target_col].plot(ax=ax4, label="Train")
        test[target_col].plot(ax=ax4, label="Test")
        arima_forecast.plot(ax=ax4, label="ARIMA Forecast")
        ax4.legend()
        st.pyplot(fig4)
        st.write("MAPE:", np.mean(np.abs((test[target_col] - arima_forecast) / test[target_col])) * 100)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

    # Download Forecasts
    st.subheader("📥 Download Forecasts")
    forecast_df = pd.DataFrame({
        "Actual": test[target_col],
        "HW_Forecast": hw_forecast if 'hw_forecast' in locals() else None,
        "ARIMA_Forecast": arima_forecast if 'arima_forecast' in locals() else None
    })
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("Download Forecasts CSV", csv, "forecast_results.csv", "text/csv")
