import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
from datetime import datetime, timedelta
from ta import add_all_ta_features
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

# Parameters
predict_days = 10

def load_stock_data(ticker, end_date, years=1):
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=years)
    adjusted_end_date = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=adjusted_end_date)
    return data

def preprocess_data(data):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    data[required_columns] = data[required_columns].fillna(0)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    features = ['Close', 'volume_adi', 'trend_macd', 'momentum_rsi', 'volatility_bbm']
    data = data[features].fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step, :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(60, 5)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def calculate_future_trading_date(start_date, trading_days):
    current_date = pd.to_datetime(start_date)
    added_days = 0
    while added_days < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            added_days += 1
    return current_date

def main():
    st.title("Stock Price Prediction using LSTM")
    st.write("This app predicts the closing price of a stock for the next 10 trading days.")

    ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):", value="AAPL")
    end_date = st.date_input("Enter the end date:", datetime.now()).strftime('%Y-%m-%d')

    if st.button("Predict"):
        try:
            stock_data = load_stock_data(ticker, end_date)
            scaled_data, scaler = preprocess_data(stock_data)

            X, y = create_dataset(scaled_data)
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            model = build_lstm_model()
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50, callbacks=[early_stopping], verbose=1)

            last_60_days = scaled_data[-60:]
            last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))
            predicted_prices = []

            for _ in range(predict_days):
                next_prediction = model.predict(last_60_days)
                predicted_prices.append(next_prediction[0, 0])
                new_row = np.zeros((1, 1, last_60_days.shape[2]))
                new_row[0, 0, 0] = next_prediction[0, 0]
                last_60_days = np.append(last_60_days[:, 1:, :], new_row, axis=1)

            dummy_array = np.zeros((len(predicted_prices), 5))
            dummy_array[:, 0] = np.array(predicted_prices).flatten()
            predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]

            target_date = calculate_future_trading_date(end_date, predict_days)
            close_price_end_date = stock_data.loc[end_date, 'Close']

            st.write(f"Close Price on {end_date}: {close_price_end_date:.2f}")
            st.write(f"Predicted Price on {target_date.strftime('%Y-%m-%d')}: {predicted_prices[-1]:.2f}")
            if predicted_prices[-1] > close_price_end_date:
                st.success("Prediction: UP 📈")
            else:
                st.error("Prediction: DOWN 📉")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
