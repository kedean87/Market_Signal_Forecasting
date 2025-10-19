from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prophet import Prophet

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import matplotlib.pyplot as plt
import sys

class SentimentModel:
	def __init__(self, model_name):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

class ProphetRegressionModel:
	def __init__(self, data, stock_symbol):
		self.model = Prophet(daily_seasonality=True)
		self.model.add_regressor('sentiment')
		self.df = data
		
		self.stock_symbol = stock_symbol
		
		self.future = None
		self.forecast = None
		
		self.figure = None
	
	def _fit(self):
		self.model.fit(self.df)
	
	def update_future(self):
		self.future = self.model.make_future_dataframe(periods=30)
		self.future = self.future.merge(self.df[['ds', 'sentiment']], on='ds', how='left')
		self.future['sentiment'] = self.future['sentiment'].fillna(self.df['sentiment'].mean())
	
	def predict(self):
		self.forecast = self.model.predict(self.future)
	
	def _plot(self):
		self.figure = self.model.plot(self.forecast)
		plt.title(f"{self.stock_symbol} Forecast with News Sentiment Influence")
		plt.show()
		
	def _plot_components(self):
		self.figure = self.model.plot_components(self.forecast)
		plt.show()
	
	def evaluate_rmse(self):
		actual = self.df.set_index('ds')['y']
		pred = self.forecast.set_index('ds').loc[actual.index]['yhat']
		rmse = np.sqrt(np.mean((actual - pred) ** 2))
		print(f"RMSE on training period: {rmse:.2f}")

class NeuralNetworkRegression:
	def __init__(self, scaler, data=[], optimizer='adam', loss='mse'):
		self.model = None
		self.scaler = scaler
		
		if len(data) != 0:
			self.X = data['X']
			self.X_train = data['X_train']
			self.X_test = data['X_test']
			self.y_train = data['y_train']
			self.y_test = data['y_test']
			self.scaled = data['scaled']
		else:
			print("There is no data to create / train / predict with the model")
			sys.exit()
		
		self.optimizer = optimizer
		self.loss = loss
		
		self.y_pred = None
		self.y_pred_actual = None
		self.y_true_actual = None
	
	def define_model(self):
		self.model = tf.keras.Sequential([
			tf.keras.layers.LSTM(64, input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.LSTM(32),
			tf.keras.layers.Dense(1)
			])
	
	def train(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss)
		self.model.fit(
			self.X_train, self.y_train, 
			epochs=300, 
			batch_size=16, 
			verbose=1, 
			validation_split=0.1
			)
	
	def predict(self):
		self.y_pred = self.model.predict(self.X_test)

	def evaluate(self):
		# Reconstruct scaled arrays to original values
		inv_pred = np.zeros((len(self.y_pred), self.scaled.shape[1]))
		inv_pred[:, 0] = self.y_pred[:, 0]
		self.y_pred_actual = self.scaler.inverse_transform(inv_pred)[:, 0]

		inv_true = np.zeros((len(self.y_test), self.scaled.shape[1]))
		inv_true[:, 0] = self.y_test
		self.y_true_actual = self.scaler.inverse_transform(inv_true)[:, 0]

		# Evaluation
		mae = mean_absolute_error(self.y_true_actual, self.y_pred_actual)
		rmse = np.sqrt(mean_squared_error(self.y_true_actual, self.y_pred_actual))
		r2 = r2_score(self.y_true_actual, self.y_pred_actual)
		print(f"MAE   : {mae:.4f}")
		print(f"RMSE  : {rmse:.4f}")
		print(f"R²    : {r2:.4f}")

	def plot(self):
		plt.figure(figsize=(12,6))
		plt.plot(self.y_true_actual, label='True Price', alpha=0.7)
		plt.plot(self.y_pred_actual, label='Predicted Price', alpha=0.7)
		plt.legend()
		plt.title("Stock Price Prediction vs Actual")
		plt.xlabel("Time Steps")
		plt.ylabel("Price")
		plt.show()

		# Directional Accuracy
		true_signal = np.sign(np.diff(self.y_true_actual))
		pred_signal = np.sign(np.diff(self.y_pred_actual))
		accuracy = np.mean(true_signal == pred_signal)
		print(f"Directional Accuracy: {accuracy*100:.2f}%")
