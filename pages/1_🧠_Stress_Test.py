import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(page_title='Homepage')
st.title('Prophet Stress Test🧠')

selected_stock = st.text_input("👇Input ticker dataset for prediction:", "MSFT")
st.info("Please, refer to Yahoo Finance for a ticker list of **S&P 500🎫** applicable ticker symbols. Type the symbol **EXACTLY** as provided by Yahoo Finance.")

# Date selection
START = st.date_input("📆Start date:", date(2000, 1, 1))
TODAY = date.today().strftime("%Y-%m-%d")

# Year range:
n_years = st.slider('⏳Day range of prediction:', 1, 365, 30)
period = n_years

if n_years >= 91:
    st.warning("The more days you select, the less accuracy will be.🤕")

st.subheader('Model stress settings🔧')

# Step 1: Noise robustness analysis
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

noise_level = st.slider('🔊Noise level:', 0.0, 10.0, 7.5)

# Step 2: Anomalous values analysis
min_anomaly_level, max_anomaly_level = st.slider('🔮Anomaly level:', 0, 100, (0, 50))

# Step 3: Impact of Data Size Analysis
data_size = st.slider('🔪Croped data level:', 90, 100, 95)

if st.button("Start Stress Test"):
    data = yf.download(selected_stock, START, TODAY)
    data.reset_index(inplace=True)

    st.subheader('Raw ['+ selected_stock +'] Dataset🥩')
    data_load_state = st.success('Dataset Uploaded successfully!✅')
    with st.expander("👀Check it Out"):
        st.write(data.tail())
    with st.expander("🤔Any missing Values?"):
        st.write(data.isnull().sum())
        st.success('We don`t have any missing values!✅')

    # Predict forecast with Prophet
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    with st.spinner('Loading data Forecast data 🎱...'):
        st.subheader('Forecast data 🎱')
        with st.expander("👀Check it Out"):
            st.write(forecast.tail())

    # Show last forecasted value
    last_forecasted_date = forecast.iloc[-1]['ds'].date()
    last_forecasted_value = forecast.iloc[-1]['yhat']

    # FBProphet Plot
    st.subheader('FBProphet Plot 🎯')
    with st.expander("💰What price will be?"):
        st.success(f'Stock price will be {last_forecasted_value:.2f}💲 after {period} day(s) on {last_forecasted_date}📅')
    fig1 = plot_plotly(m, forecast)

    # Update layout for the fig1
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price"
    )

    # Обмежуємо на останні 10 років
    last_date_forecast = forecast['ds'].iloc[-1]
    five_years_ago = last_date_forecast - pd.DateOffset(years=10)
    
    fig1.update_layout(xaxis_range=[five_years_ago, last_date_forecast])  
    st.plotly_chart(fig1)

    # Split data into train and test sets
    train_size = int(len(df_train) * 0.55)
    train_data = df_train[:train_size]
    test_data = df_train[train_size:]

    # Make predictions on test data
    results = []

    if not test_data.empty:
        forecast = m.predict(test_data)

        # Calculate Prophet's Test MAE, RMSE, and R-squared
        mae = mean_absolute_error(test_data['y'], forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
        mse = mean_squared_error(test_data['y'], forecast['yhat'])
        r2 = r2_score(test_data['y'], forecast['yhat'])

        # Show Prophet's Test Metrics
        st.subheader("Prophet's Test Metrics🧪")

        st.write("👁‍🗨Prediction trust factor:")
        if r2 >= 0.85:
            st.success('📗 - High level of trust factor.')
        elif r2 >= 0.75:
            st.info('📘 - Good level of trust factor.')
        elif r2 >= 0.7:
            st.warning('📒 - Satisfactory level of trust factor.')
        elif r2 >= 0.5:
            st.error('📙 - Low level of trust factor.')
        elif r2 >= -1:
            st.error('📕 - Very low level of trust factor.')

        st.write("R-squared:", r2)
        st.write("MAE (Mean Absolute Error):", mae)
        st.write("MSE (Mean Squared Error):", mse)
        st.write("RMSE (Root Mean Squared Error):", rmse)
        st.write(f"Stock Price: {last_forecasted_value:.2f}💲")
        
        results.append({
            'Test': 'Original',
            'R-squared': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Stock Price': f"{last_forecasted_value:.2f}💲"
        })

    else:
        st.warning("No test data available for the specified date range.")

    st.header('Stress Test Result🔬')

    # Noise robustness analysis
    noisy_data = add_noise(df_train['y'], noise_level)
    df_train_noisy = df_train.copy()
    df_train_noisy['y'] = noisy_data

    m_noisy = Prophet()
    m_noisy.fit(df_train_noisy)
    forecast_noisy = m_noisy.predict(future)

    fig_noisy = plot_plotly(m_noisy, forecast_noisy)
    fig_noisy.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis_range=[five_years_ago, last_date_forecast]
    )
    st.plotly_chart(fig_noisy)

    # Evaluate noise robustness
    if not test_data.empty:
        forecast_noisy_test = m_noisy.predict(test_data)
        mae_noisy = mean_absolute_error(test_data['y'], forecast_noisy_test['yhat'])
        mse_noisy = mean_squared_error(test_data['y'], forecast_noisy_test['yhat'])
        rmse_noisy = np.sqrt(mean_squared_error(test_data['y'], forecast_noisy_test['yhat']))
        r2_noisy = r2_score(test_data['y'], forecast_noisy_test['yhat'])

        st.subheader('Noise Robustness Analysis🔊')
        st.write("Noise Robustness Metrics:")
        st.write("R-squared:", r2_noisy)
        st.write("MAE (Mean Absolute Error):", mae_noisy)
        st.write("MSE (Mean Squared Error):", mse_noisy)
        st.write("RMSE (Root Mean Squared Error):", rmse_noisy)

        # Show last forecasted value after noise robustness analysis
        last_forecasted_date_noisy = forecast_noisy.iloc[-1]['ds'].date()
        last_forecasted_value_noisy = forecast_noisy.iloc[-1]['yhat']
        st.write(f"Stock Price (after noise): {last_forecasted_value_noisy:.2f}💲")
        
        results.append({
            'Test': 'Noise Robustness',
            'R-squared': r2_noisy,
            'MAE': mae_noisy,
            'MSE': mse_noisy,
            'RMSE': rmse_noisy,
            'Stock Price': f"{last_forecasted_value_noisy:.2f}💲"
        })

    # Anomalous values analysis
    num_anomalies = int(len(df_train) * max_anomaly_level / 100)
    min_num_anomalies = int(len(df_train) * min_anomaly_level / 100)
    anomalies = np.random.choice(df_train.index, num_anomalies, replace=False)

    anomaly_factor = np.random.uniform(0.1, 2.5, num_anomalies)
    df_train_anomalous = df_train.copy()
    df_train_anomalous.loc[anomalies, 'y'] *= anomaly_factor

    m_anomalous = Prophet()
    m_anomalous.fit(df_train_anomalous)
    forecast_anomalous = m_anomalous.predict(future)

    fig_anomalous = plot_plotly(m_anomalous, forecast_anomalous)
    fig_anomalous.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis_range=[five_years_ago, last_date_forecast]
    )
    st.plotly_chart(fig_anomalous)

    # Evaluate anomalous values robustness
    if not test_data.empty:
        forecast_anomalous_test = m_anomalous.predict(test_data)
        mae_anomalous = mean_absolute_error(test_data['y'], forecast_anomalous_test['yhat'])
        mse_anomalous = mean_squared_error(test_data['y'], forecast_anomalous_test['yhat'])
        rmse_anomalous = np.sqrt(mean_squared_error(test_data['y'], forecast_anomalous_test['yhat']))
        r2_anomalous = r2_score(test_data['y'], forecast_anomalous_test['yhat'])

        st.subheader('Anomalous Values Analysis🔮')
        st.write("R-squared:", r2_anomalous)
        st.write("MAE (Mean Absolute Error):", mae_anomalous)
        st.write("MSE (Mean Squared Error):", mse_anomalous)
        st.write("RMSE (Root Mean Squared Error):", rmse_anomalous)

        # Show last forecasted value after anomalous values analysis
        last_forecasted_date_anomalous = forecast_anomalous.iloc[-1]['ds'].date()
        last_forecasted_value_anomalous = forecast_anomalous.iloc[-1]['yhat']
        st.write(f"Stock Price (after anomaly: {last_forecasted_value_anomalous:.2f}💲")
        
        results.append({
            'Test': 'Anomalous Values',
            'R-squared': r2_anomalous,
            'MAE': mae_anomalous,
            'MSE': mse_anomalous,
            'RMSE': rmse_anomalous,
            'Stock Price': f"{last_forecasted_value_anomalous:.2f}💲"
        })

    # Impact of Data Size Analysis
    df_train_small = df_train.iloc[:int(len(df_train) * data_size / 100)]

    m_small = Prophet()
    m_small.fit(df_train_small)
    forecast_small = m_small.predict(future)

    fig_small = plot_plotly(m_small, forecast_small)
    fig_small.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis_range=[five_years_ago, last_date_forecast]
    )
    st.plotly_chart(fig_small)

    # Evaluate impact of data size
    if not test_data.empty:
        forecast_sampled_test = m_small.predict(test_data)
        mae_sampled = mean_absolute_error(test_data['y'], forecast_sampled_test['yhat'])
        mse_sampled = mean_squared_error(test_data['y'], forecast_sampled_test['yhat'])
        rmse_sampled = np.sqrt(mean_squared_error(test_data['y'], forecast_sampled_test['yhat']))
        r2_sampled = r2_score(test_data['y'], forecast_sampled_test['yhat'])

        st.subheader('Impact of Data Size Analysis🔪')
        st.write("R-squared:", r2_sampled)
        st.write("MAE (Mean Absolute Error):", mae_sampled)
        st.write("MSE (Mean Squared Error):", mse_sampled)
        st.write("RMSE (Root Mean Squared Error):", rmse_sampled)

        # Show last forecasted value after impact of data size analysis
        last_forecasted_date_sampled = forecast_small.iloc[-1]['ds'].date()
        last_forecasted_value_sampled = forecast_small.iloc[-1]['yhat']
        st.write(f"Stock Price (after cropping): {last_forecasted_value_sampled:.2f}💲")
        
        results.append({
            'Test': 'Impact of Data Size',
            'R-squared': r2_sampled,
            'MAE': mae_sampled,
            'MSE': mse_sampled,
            'RMSE': rmse_sampled,
            'Stock Price': f"{last_forecasted_value_sampled:.2f}💲"
        })

    # Display summary table
    st.header('Summary of Stress Test📊')
    results_df = pd.DataFrame(results)
    st.write(results_df)

    
