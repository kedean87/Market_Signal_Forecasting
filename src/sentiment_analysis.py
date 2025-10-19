from model import *
import pandas as pd
from scipy.special import softmax

class SentimentAnalysis:
	def __init__(self, model_name):
		self.sm = SentimentModel(model_name)
		self.daily_sentiment = None
	
	def get_sentiment(self, text):
		try:
			inputs = self.sm.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
			outputs = self.sm.model(**inputs)
			
			scores = softmax(outputs.logits.detach().numpy()[0])
			
			# weighted sentiment
			sentiment = np.dot(scores, [-1, 0, 1])
			
			self.daily_sentiment = sentiment
		
		except Exception:
			self.daily_sentiment = 0

		return self.daily_sentiment
