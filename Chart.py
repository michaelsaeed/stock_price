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

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

# Parameters
end_date = '2024-09-30'
predict_days = 10

def load_stock_data(ticker, end_date, years=1):
    """Load the last 'years' years of stock data from Yahoo Finance."""
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=years)
    adjusted_end_date = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=adjusted_end_date)
    return data

def preprocess_data(data):
    """Preprocess the stock data with additional features."""
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    features = ['Close', 'volume_adi', 'trend_macd', 'momentum_rsi', 'volatility_bbm']
    data = data[features]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    """Create the dataset for LSTM."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step, :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model():
    """Build and compile the LSTM model."""
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
    """Calculate the future trading date by skipping weekends."""
    current_date = pd.to_datetime(start_date)
    added_days = 0
    while added_days < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Skip Saturday (5) and Sunday (6)
            added_days += 1
    return current_date

def main():
    ticker = 'ZETA'

    # Load stock data
    stock_data = load_stock_data(ticker, end_date)

    # Preprocess data
    scaled_data, scaler = preprocess_data(stock_data)

    # Create training dataset
    X, y = create_dataset(scaled_data)

    # Split data into training and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Build and train the LSTM model
    model = build_lstm_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50, callbacks=[early_stopping], verbose=1)

    # Predict the close price for the next 10 trading days
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))
    predicted_prices = []

    for _ in range(predict_days):
        next_prediction = model.predict(last_60_days)
        predicted_prices.append(next_prediction[0, 0])

        # Create a new row with the predicted value and zeros for other features
        new_row = np.zeros((1, 1, last_60_days.shape[2]))
        new_row[0, 0, 0] = next_prediction[0, 0]  # Set the predicted value for the 'Close' feature
        last_60_days = np.append(last_60_days[:, 1:, :], new_row, axis=1)

    # Create a dummy array with 5 features for inverse transformation
    dummy_array = np.zeros((len(predicted_prices), 5))
    dummy_array[:, 0] = np.array(predicted_prices).reshape(-1)  # Set the 'Close' feature
    predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]  # Inverse transform and extract 'Close' prices

    # Calculate the target date (10 trading days after the end date)
    target_date = calculate_future_trading_date(end_date, predict_days)

    # Get the close price on the end date
    close_price_end_date = stock_data.loc[end_date, 'Close']

    # Print the result
    print(f"Close Price on {end_date}: {close_price_end_date:.2f}")
    print(f"Predicted Price on {target_date.strftime('%Y-%m-%d')}: {predicted_prices[-1]:.2f}")
    if predicted_prices[-1] > close_price_end_date:
        print("UP")
    else:
        print("DOWN")

if __name__ == "__main__":
    main()