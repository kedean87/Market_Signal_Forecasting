from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd

import pandas as pd
import numpy as np

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

from dataset import *
from sentiment_analysis import *
from merge_data import *
from model import *

print("Entering the serve script")

STOCK_SYMBOL = "AAPL"
START_DATE = "2025-09-20"

# get from https://newsapi.org/
NEWS_API_KEY = "fde2f948fd9d43a396cba6d354e3e204"  
COMPANY_NAME = "Apple"

def use_models(company, stock_symbol, prophet, lstm):
	ld = Dataset(
		company=COMPANY_NAME,
		stock_symbol=STOCK_SYMBOL,
		start_date=START_DATE,
		)
	
	ld.load_dataset()
	
	news_df = ld.df
	if news_df.empty:
		print("Error: No news found for the given period. Try a broader range or another ticker.")
		sys.exit()
	
	sa = SentimentAnalysis(
		model_name="yiyanghkust/finbert-tone"
		)
	news_df['sentiment'] = news_df['text'].apply(sa.get_sentiment)
	
	# Aggregate sentiment per day
	daily_sentiment = news_df.groupby('date', as_index=False)['sentiment'].mean()
	daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
	
	md = MergeData(daily_sentiment, ld.stock_data)
	md.merge_data()
	df = md.df
	
	rmp = prophet
	future = prophet.make_future_dataframe(periods=30)
	future = future.merge(df[['ds', 'sentiment']], on='ds', how='left')
	future['sentiment'] = future['sentiment'].fillna(df['sentiment'].mean())

	forecast = prophet.predict(future)
	actual = df.set_index('ds')['y']
	pred = forecast.set_index('ds').loc[actual.index]['yhat']
	rmse_prophet = np.sqrt(np.mean((actual - pred) ** 2))
	print("RMSE: ", rmse_prophet)
	
	# Scale data
	scaler = MinMaxScaler()
	scaled = scaler.fit_transform(df[['y', 'sentiment']])

	# Prepare sequences
	window = 10
	X, y = [], []
	for i in range(window, len(scaled)):
		X.append(scaled[i-window:i])
		y.append(scaled[i, 0])  # next-day close
	X, y = np.array(X), np.array(y)

	# --- 1. Split data into train/test (chronologically) ---
	train_size = int(len(X) * 0.8)
	X_train, X_test = X[:train_size], X[train_size:]
	y_train, y_test = y[:train_size], y[train_size:]
	
	y_pred = lstm.predict(X_test)
	
	# --- 4. Reconstruct scaled arrays to original values ---
	inv_pred = np.zeros((len(y_pred), scaled.shape[1]))
	inv_pred[:, 0] = y_pred[:, 0]
	y_pred_actual = scaler.inverse_transform(inv_pred)[:, 0]

	inv_true = np.zeros((len(y_test), scaled.shape[1]))
	inv_true[:, 0] = y_test
	y_true_actual = scaler.inverse_transform(inv_true)[:, 0]

	# --- 5. Evaluation ---
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
	mae = mean_absolute_error(y_true_actual, y_pred_actual)
	rmse_lstm = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
	r2 = r2_score(y_true_actual, y_pred_actual)

	print(f"📉 MAE  : {mae:.4f}")
	print(f"📉 RMSE : {rmse_lstm:.4f}")
	print(f"📈 R²    : {r2:.4f}")
	
	return {'prophet': rmse_prophet, 'lstm': rmse_lstm}

app = Flask(__name__)

print("Loading Models")
# Load models on startup
prophet = joblib.load("prophet_model.pkl")
lstm = tf.keras.models.load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(status="ok")

@app.route("/invocations", methods=["POST"])
def predict():
    data = request.get_json()
    company = data.get("company")
    symbol = data.get("symbol")
    
    rmse_results = use_models(company, symbol, prophet, lstm)
    
    return jsonify({"company": company, "forecast": rmse_results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
