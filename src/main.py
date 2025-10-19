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

STOCK_SYMBOL = "AAPL"
START_DATE = "2025-09-18"

# get from https://newsapi.org/
NEWS_API_KEY = "fde2f948fd9d43a396cba6d354e3e204"  
COMPANY_NAME = "Apple"

def main():
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
	
	rmp = RegressionModelProphet(
		data=df, 
		stock_symbol=STOCK_SYMBOL
		)
	rmp._fit()
	rmp.update_future()
	rmp.predict()
	rmp._plot()
	rmp._plot_components()
	rmp.evaluate_rmse()
	
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
	
	data = {}
	data['X'] = X
	data['X_train'] = X_train
	data['X_test'] = X_test
	data['y_train'] = y_train
	data['y_test'] = y_test
	data['scaled'] = scaled
	
	nnr = NeuralNetworkRegression(
		scaler=scaler,
		data=data
		)
	nnr.define_model()
	nnr.train()
	nnr.predict()
	nnr.evaluate()
	nnr.plot()
	

if __name__ == "__main__":
	main()
	
