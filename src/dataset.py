import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import requests

class Dataset:
	def __init__(self, company, stock_symbol, start_date, news_api_key, end_date=datetime.today().strftime("%Y-%m-%d")):
		self.url = f"https://newsapi.org/v2/everything?q={company}&from={start_date}&to={end_date}&language=en&sortBy=publishedAt&apiKey={news_api_key}"
		self.company = company
		self.stock_symbol = stock_symbol
		self.start_date = start_date
		self.end_date = end_date
		
		print(self.url, start_date, end_date)
		self.stock_data = yf.download(self.stock_symbol, start=start_date, end=end_date, interval="1d").reset_index()
		self.stock_data = self.stock_data[['Date', 'Close']]
		self.stock_data.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
		
		self.df = None
	
	def load_dataset(self):
		response = requests.get(self.url)
		data = response.json()

		# Check if API returned an error
		if data.get("status") != "ok":
			print(f"⚠️ Warning: NewsAPI request failed: {data.get('message')}")
			self.df = pd.DataFrame(columns=['date', 'text'])
			return self.df

		articles = data.get("articles", [])
		if len(articles) == 0:
			print("⚠️ No news articles found for this query.")
			self.df = pd.DataFrame(columns=['date', 'text'])
			return self.df
		
		# Ensure all keys exist
		df = pd.DataFrame(articles)
		keys = ['publishedAt', 'title', 'description']
		for col in keys:
			if col not in df.columns:
				df[col] = None
		
		df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce').dt.date
		df['text'] = df['title'].fillna('') + ". " + df['description'].fillna('')
		df = df[['date', 'text']]
		
		self.df = df
