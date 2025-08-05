import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.plotting import lag_plot
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 General Time Series Analysis & Forecasting")

# Upload file
file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    with st.sidebar:
        st.subheader("📌 Column Selection")
        datetime_col = st.selectbox("Select Date/Time Column", df.columns)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        target_col = st.selectbox("Select Target Variable", numeric_cols)

    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
    df.set_index(datetime_col, inplace=True)
    df.sort_index(inplace=True)

    st.write(f"🕒 Date Range: {df.index.min()} → {df.index.max()}")

    freq = st.sidebar.selectbox("Resample Frequency", ['D', 'W', 'M'], index=0)
    resampled = df[[target_col]].resample(freq).mean().interpolate()

    st.subheader("📉 Resampled and Interpolated Data")
    st.line_chart(resampled)

    st.subheader("🔍 Seasonal Decomposition")
    try:
        result = seasonal_decompose(resampled[target_col], model="additive", period=12)
        fig = result.plot()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Decomposition failed: {e}")

    st.subheader("📈 ACF & PACF")
    lags = st.slider("Select number of lags", 10, 60, 20)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(resampled[target_col], lags=lags, ax=ax[0])
    plot_pacf(resampled[target_col], lags=lags, ax=ax[1])
    st.pyplot(fig)

    st.subheader("🌀 Lag Plot")
    fig2, ax2 = plt.subplots()
    lag_plot(resampled[target_col], ax=ax2)
    st.pyplot(fig2)

    st.subheader("✂️ Train-Test Split")
    split_ratio = st.slider("Training Data Ratio", 0.5, 0.95, 0.9)
    split_point = int(len(resampled) * split_ratio)
    train, test = resampled.iloc[:split_point], resampled.iloc[split_point:]
    st.write(f"Train size: {len(train)}, Test size: {len(test)}")

    st.subheader("📦 Choose Forecasting Models")
    use_hw = st.checkbox("Use Holt-Winters", True)
    use_arima = st.checkbox("Use ARIMA", True)
    use_moving_avg = st.checkbox("Use Moving Average", True)

    forecast_data = {"Actual": test[target_col]}

    if use_hw:
        st.subheader("🌡️ Holt-Winters Forecast")
        try:
            hw_model = ExponentialSmoothing(train[target_col], seasonal="add", trend="add", seasonal_periods=12).fit()
            hw_forecast = hw_model.forecast(len(test))
            forecast_data["HW_Forecast"] = hw_forecast
            fig3, ax3 = plt.subplots()
            train[target_col].plot(ax=ax3, label="Train")
            test[target_col].plot(ax=ax3, label="Test")
            hw_forecast.plot(ax=ax3, label="HW Forecast")
            ax3.legend()
            st.pyplot(fig3)
            st.write("MAPE (HW):", np.mean(np.abs((test[target_col] - hw_forecast) / test[target_col])) * 100)
        except Exception as e:
            st.warning(f"Holt-Winters failed: {e}")

    if use_arima:
        st.subheader("📉 ARIMA Forecast")
        p = st.number_input("p", 0, 10, 2)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 10, 2)
        try:
            arima_model = ARIMA(train[target_col], order=(p, d, q)).fit()
            arima_forecast = arima_model.forecast(len(test))
            forecast_data["ARIMA_Forecast"] = arima_forecast
            fig4, ax4 = plt.subplots()
            train[target_col].plot(ax=ax4, label="Train")
            test[target_col].plot(ax=ax4, label="Test")
            arima_forecast.plot(ax=ax4, label="ARIMA Forecast")
            ax4.legend()
            st.pyplot(fig4)
            st.write("MAPE (ARIMA):", np.mean(np.abs((test[target_col] - arima_forecast) / test[target_col])) * 100)
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")

    if use_moving_avg:
        st.subheader("🪄 Moving Average Forecast")
        try:
            window_size = st.slider("Moving Average Window", 2, 30, 12)
            moving_avg_forecast = train[target_col].rolling(window=window_size).mean().iloc[-1]
            ma_forecast = pd.Series([moving_avg_forecast] * len(test), index=test.index)
            forecast_data["Moving_Average"] = ma_forecast
            fig5, ax5 = plt.subplots()
            train[target_col].plot(ax=ax5, label="Train")
            test[target_col].plot(ax=ax5, label="Test")
            ma_forecast.plot(ax=ax5, label="MA Forecast")
            ax5.legend()
            st.pyplot(fig5)
            st.write("MAPE (MA):", np.mean(np.abs((test[target_col] - ma_forecast) / test[target_col])) * 100)
        except Exception as e:
            st.warning(f"Moving Average failed: {e}")

    st.subheader("📥 Download Forecasts")
    forecast_df = pd.DataFrame(forecast_data)
    st.dataframe(forecast_df)
    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("Download Forecasts CSV", csv, "forecast_results.csv", "text/csv")

    with st.expander("📘 Forecasting Model Guide"):
        st.markdown("""
        | Model            | Use When                                 | Best For                           |
        |------------------|------------------------------------------|------------------------------------|
        | Holt-Winters     | Data has seasonality & trend             | Sales, temperature, demand cycles  |
        | ARIMA            | Data has trend but not strong seasonality| Stock prices, economic indicators  |
        | Moving Average   | Simple smoothing without modeling         | Quick, rough short-term forecasts  |
        """)

