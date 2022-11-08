import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import warnings
import os
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tqdm.keras import TqdmCallback



def load_data(ticker, period):
	data = yf.Ticker(ticker)
	data = data.history(period=period)
	return data

def get_train_n_test_data(data, percentage):
	train_data = data[:int(len(data) * percentage)]
	test_data = data[int(len(data) * percentage): len(data)]
	return train_data, test_data

def get_test_data(data):
	test_data = data[int(len(data) * 0.8): len(data)]
	return test_data


def prepare_data(prediction_days, value, train_data):
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(train_data[value].values.reshape(-1, 1))
	x_train = []
	y_train = []
	for x in range(prediction_days, len(scaled_data)):
		x_train.append(scaled_data[x-prediction_days:x, 0])
		y_train.append(scaled_data[x, 0])
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	return scaler, x_train, y_train


def build_model(x_train, y_train):
	model = Sequential()
	model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
	model.add(Dropout(0.2))
	model.add(LSTM(units=50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(units=25))
	model.add(Dropout(0.2))
	model.add(Dense(units=25))
	model.add(Dense(units=1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	print("\nTraining the LSTM Model...\n")
	model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0, callbacks=[TqdmCallback(verbose=0)])
	print("\n")
	return model


def test_the_data(train_data, test_data, value, prediction_days, scaler):
	actual_prices = test_data[value].values
	total_dataset = pd.concat((train_data[value], test_data[value]), axis=0)
	model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
	model_inputs = model_inputs.reshape(-1, 1)
	model_inputs = scaler.transform(model_inputs)
	return actual_prices, model_inputs


def make_predictions(prediction_days, model_inputs, model, scaler):
	x_test = []
	for x in range(prediction_days, len(model_inputs)):
		x_test.append(model_inputs[x-prediction_days:x, 0])
	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
	predicted_prices = model.predict(x_test)
	predicted_prices = scaler.inverse_transform(predicted_prices)
	return predicted_prices


def plot_predictions(actual_prices, predicted_prices, ticker):
	plt.plot(actual_prices, color="black", label=f"Actual {ticker} Prices")
	plt.plot(predicted_prices, color="green", label=f"Predicted {ticker} Prices")
	plt.title(f"{ticker} Share Price")
	plt.ylabel(f"{ticker} Share Price")
	plt.xlabel('Time')
	plt.legend()
	plt.show()


def predict_next_day_price(model_inputs, prediction_days, value, model, scaler):
	real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
	real_data = np.array(real_data)
	real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
	prediction = model.predict(real_data)
	prediction = scaler.inverse_transform(prediction)
	prediction = f"Prediction for the {value} value in the next day is {prediction[0][0]}"
	return prediction


def forecast_next_day(train_days, percentage, data, value):
	train_data, test_data = get_train_n_test_data(data, percentage)
	scaler, x_train, y_train = prepare_data(train_days, value, train_data)
	model = build_model(x_train, y_train)
	actual_prices, model_inputs = test_the_data(train_data, test_data, value, train_days, scaler)
	predicted_prices = make_predictions(train_days, model_inputs, model, scaler)
	# plot_predictions(actual_prices, predicted_prices, '^GSPC')
	print(predict_next_day_price(model_inputs, train_days, value, model, scaler))

data = load_data('^GSPC', '600d')
forecast_next_day(60, 0.8, data, 'Close')
