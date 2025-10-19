import pandas as pd

class MergeData:
	def __init__(self, daily_sentiment, stock_data):
		self.daily_sentiment = daily_sentiment
		self.stock_data = stock_data
		
		self.df = None
	
	def merge_data(self):
		print("🔗 Merging data...")

		# 1. Reset index to make Date a column
		self.stock_data = self.stock_data.reset_index()

		# 2. Flatten columns if MultiIndex
		if isinstance(self.stock_data.columns, pd.MultiIndex):
			self.stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in self.stock_data.columns]

		# 3. Inspect column names
		print("Columns in stock_data after reset_index:", self.stock_data.columns)

		# 4. Try to find date and close columns intelligently
		date_col = next((c for c in self.stock_data.columns if 'date' in c.lower()), None)
		close_col = next((c for c in self.stock_data.columns if 'close' in c.lower()), None)

		if date_col and close_col:
			self.stock_data = self.stock_data[[date_col, close_col]]
		else:
			raise ValueError(f"❌ Could not find date/close columns in: {list(self.stock_data.columns)}")

		# 5. Rename columns to lowercase for merging
		self.stock_data.rename(columns={date_col: 'date', close_col: 'close'}, inplace=True)

		# 6. Ensure sentiment index is reset
		self.daily_sentiment = self.daily_sentiment.reset_index(drop=True)

		# 7. Merge
		merged = pd.merge(self.stock_data, self.daily_sentiment, on='date', how='left').fillna(0)

		# 8. Prepare final dataframe for modeling
		df = merged.rename(columns={'date': 'ds', 'close': 'y', 'sentiment': 'sentiment'})
		df['ds'] = pd.to_datetime(df['ds'])
		
		self.df = df.sort_values('ds')
