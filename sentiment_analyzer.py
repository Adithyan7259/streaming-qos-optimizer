from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List, Dict
import pickle
import os

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the sentiment analysis model with some basic training data
        """
        # Basic training data
        texts = [
            "Great stream quality!",
            "The game looks smooth",
            "No lag at all",
            "This stream is terrible",
            "Too much buffering",
            "The quality is poor",
            "Decent stream",
            "Okay quality",
            "Not bad",
            "Could be better"
        ]
        # Using integer labels: 2 for positive, 1 for neutral, 0 for negative
        labels = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train model
        self.model.fit(X, labels)
        
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of chat messages
        
        Args:
            texts: List of chat messages
            
        Returns:
            List of sentiment analysis results
        """
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Get predictions
        predictions = self.model.predict_proba(X)
        
        # Convert to sentiment labels
        results = []
        for pred in predictions:
            # Map probabilities to sentiment scores
            positive_prob = pred[2] if len(pred) > 2 else 0.0  # Class 2 (positive)
            neutral_prob = pred[1] if len(pred) > 1 else 0.0   # Class 1 (neutral)
            negative_prob = pred[0]                             # Class 0 (negative)
            
            sentiment = {
                'positive': float(positive_prob),
                'neutral': float(neutral_prob),
                'negative': float(negative_prob)
            }
            results.append(sentiment)
                
        return results
    
    def get_aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """
        Calculate aggregate sentiment metrics
        
        Args:
            sentiments: List of sentiment analysis results
            
        Returns:
            Dictionary of aggregate sentiment metrics
        """
        if not sentiments:
            return {
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'sentiment_score': 0.0
            }
            
        # Calculate ratios
        total = len(sentiments)
        positive_ratio = sum(1 for s in sentiments if s['positive'] > max(s['neutral'], s['negative'])) / total
        negative_ratio = sum(1 for s in sentiments if s['negative'] > max(s['neutral'], s['positive'])) / total
        neutral_ratio = sum(1 for s in sentiments if s['neutral'] > max(s['positive'], s['negative'])) / total
        
        # Calculate overall sentiment score (-1 to 1)
        sentiment_score = np.mean([
            s['positive'] - s['negative'] for s in sentiments
        ])
        
        return {
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'sentiment_score': sentiment_score
        } 