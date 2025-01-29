import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras_tuner import RandomSearch
from datetime import datetime, timedelta
import streamlit as st

def get_second_friday(start_date):
    """Find the second Friday after a given date."""
    days_until_friday = (4 - start_date.weekday()) % 7
    first_friday = start_date + timedelta(days=days_until_friday)
    second_friday = first_friday + timedelta(days=7)
    return second_friday

# Streamlit App
st.title("Stock Price Prediction using LSTM")

# Inputs for ticker and end_date
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL):", value="ASTS")
end_date = st.date_input("Select the end date:", value=datetime(2024, 10, 16))

# Convert end_date to datetime
end_date = datetime.combine(end_date, datetime.min.time())

# Calculate start_date (12 months prior)
start_date = end_date - timedelta(days=365)

# Download stock data using yfinance
st.write(f"Downloading data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))

# Handle cases where data is not available for the exact end_date
if end_date not in data.index:
    closest_date = data.index[data.index <= end_date][-1]
else:
    closest_date = end_date

# Extract the actual close price on or before the end_date
close_price_end_date = data.loc[closest_date, 'Close']

# Calculate technical indicators using pandas_ta
data['SMA_50'] = ta.sma(data['Close'], length=50)
data['SMA_200'] = ta.sma(data['Close'], length=200)
data['EMA_17'] = ta.ema(data['Close'], length=17)
data['RSI_14'] = ta.rsi(data['Close'], length=14)
data['Momentum'] = ta.mom(data['Close'], length=1)
data['Volume'] = data['Volume']

# Drop rows with NaN values
data = data.dropna()

# Ensure data is a copy of the DataFrame
data = data.copy()

# Prepare data for prediction
data['Days'] = (data.index - data.index[0]).days
X = data[['Days', 'SMA_50', 'SMA_200', 'EMA_17', 'RSI_14', 'Momentum', 'Volume']]
y = data['Close']

# Normalize the features (X) and target (y) separately
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape data for LSTM [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter Tuning with Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50),
                   return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', min_value=25, max_value=100, step=25), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

st.write("Performing hyperparameter tuning...")
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=2,
    directory='tuner_results',
    project_name='stock_price_prediction'
)

# Perform hyperparameter tuning
tuner.search(X_scaled, y_scaled, epochs=20, validation_split=0.2, verbose=1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Cross-Validation
st.write("Performing cross-validation...")
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    best_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Get the second Friday after end_date
second_friday = get_second_friday(end_date)

# Calculate the number of days from the start_date to the second Friday
target_day = (second_friday - data.index[0]).days

# Prepare the feature vector for prediction
target_features = pd.DataFrame([[target_day, data['SMA_50'].iloc[-1], data['SMA_200'].iloc[-1], data['EMA_17'].iloc[-1], data['RSI_14'].iloc[-1], data['Momentum'].iloc[-1], data['Volume'].iloc[-1]]],
                                columns=['Days', 'SMA_50', 'SMA_200', 'EMA_17', 'RSI_14', 'Momentum', 'Volume'])

# Normalize the target features using scaler_X
target_features_scaled = scaler_X.transform(target_features)
target_features_scaled = target_features_scaled.reshape((target_features_scaled.shape[0], 1, target_features_scaled.shape[1]))

# Predict the close price for the second Friday
predicted_close_price_scaled = best_model.predict(target_features_scaled)
predicted_close_price_target_date = scaler_y.inverse_transform(predicted_close_price_scaled)[0][0]

# Display results in Streamlit
st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")
st.write(f"Actual Close Price on End Date ({closest_date.strftime('%Y-%m-%d')}): {close_price_end_date:.2f}")
st.write(f"Second Friday: {second_friday.strftime('%Y-%m-%d')}")
st.write(f"Predicted Close Price on Second Friday ({second_friday.strftime('%Y-%m-%d')}): {predicted_close_price_target_date:.2f}")