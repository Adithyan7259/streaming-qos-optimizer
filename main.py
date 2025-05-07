import pandas as pd
import numpy as np
from data_processor import StreamingDataProcessor
from satisfaction_classifier import SatisfactionClassifier
from sentiment_analyzer import SentimentAnalyzer
from qos_optimizer import QoSOptimizer, QoSParameters
from benchmark_data import BenchmarkDataset
import time
from typing import Dict, List
import json
from sklearn.metrics import mean_squared_error, r2_score

class StreamingQoSManager:
    def __init__(self):
        self.data_processor = StreamingDataProcessor()
        self.satisfaction_classifier = SatisfactionClassifier(input_dim=8)  # 6 streaming + 2 chat features
        self.sentiment_analyzer = SentimentAnalyzer()
        self.qos_optimizer = QoSOptimizer()
        
    def process_streaming_data(self, streaming_metrics: pd.DataFrame) -> np.ndarray:
        """
        Process streaming metrics data
        """
        return self.data_processor.process_streaming_metrics(streaming_metrics)[0]
    
    def process_chat_data(self, chat_data: List[Dict]) -> pd.DataFrame:
        """
        Process chat data
        """
        return self.data_processor.process_chat_data(chat_data)
    
    def analyze_sentiment(self, chat_messages: List[str]) -> Dict:
        """
        Analyze sentiment of chat messages
        """
        sentiments = self.sentiment_analyzer.analyze_sentiment(chat_messages)
        return self.sentiment_analyzer.get_aggregate_sentiment(sentiments)
    
    def optimize_qos(self,
                    current_params: QoSParameters,
                    streaming_metrics: pd.DataFrame,
                    chat_data: List[Dict]) -> QoSParameters:
        """
        Optimize QoS parameters based on current metrics and chat data
        """
        # Process streaming metrics
        processed_metrics = self.process_streaming_data(streaming_metrics)
        
        # Process chat data
        processed_chat = self.process_chat_data(chat_data)
        
        # Get satisfaction prediction
        combined_features = self.data_processor.create_training_data(
            processed_metrics,
            processed_chat,
            np.zeros(len(processed_metrics))  # Dummy labels for prediction
        )[0]
        satisfaction_score = self.satisfaction_classifier.predict(combined_features)[0][0]
        
        # Get sentiment analysis
        chat_messages = [msg['message'] for msg in chat_data]
        sentiment_metrics = self.analyze_sentiment(chat_messages)
        
        # Optimize QoS parameters
        return self.qos_optimizer.optimize_parameters(
            current_params,
            satisfaction_score,
            sentiment_metrics['sentiment_score'],
            current_params.network_quality
        )
    
    def evaluate_performance(self, 
                           streaming_metrics: pd.DataFrame,
                           chat_data: List[Dict],
                           ground_truth: np.ndarray) -> Dict:
        """
        Evaluate the performance of the QoS optimization system
        
        Args:
            streaming_metrics: DataFrame containing streaming metrics
            chat_data: List of chat messages
            ground_truth: Ground truth satisfaction scores
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Process data
        processed_metrics = self.process_streaming_data(streaming_metrics)
        processed_chat = self.process_chat_data(chat_data)
        
        # Get predictions
        combined_features = self.data_processor.create_training_data(
            processed_metrics,
            processed_chat,
            np.zeros(len(processed_metrics))
        )[0]
        predictions = self.satisfaction_classifier.predict(combined_features)
        
        # Calculate metrics
        mse = mean_squared_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mean_prediction': np.mean(predictions),
            'mean_ground_truth': np.mean(ground_truth)
        }

def main():
    # Initialize QoS manager and benchmark dataset
    qos_manager = StreamingQoSManager()
    benchmark = BenchmarkDataset()
    
    # Load benchmark data
    streaming_metrics, chat_data, ground_truth = benchmark.load_benchmark_data()
    
    # Evaluate system performance
    evaluation_results = qos_manager.evaluate_performance(
        streaming_metrics,
        chat_data,
        ground_truth
    )
    
    print("\nSystem Performance Evaluation:")
    print(f"Mean Squared Error: {evaluation_results['mse']:.4f}")
    print(f"R² Score: {evaluation_results['r2_score']:.4f}")
    print(f"Mean Predicted Satisfaction: {evaluation_results['mean_prediction']:.4f}")
    print(f"Mean Ground Truth Satisfaction: {evaluation_results['mean_ground_truth']:.4f}")
    
    # Example optimization with a single sample
    current_params = QoSParameters(
        bitrate=4000,
        resolution=(1280, 720),
        fps=30,
        buffer_size=1000000,
        network_quality=0.8
    )
    
    # Use first sample from benchmark data
    sample_metrics = streaming_metrics.iloc[0:1]
    sample_chat = chat_data[:10]  # Use first 10 chat messages
    
    # Optimize QoS parameters
    optimized_params = qos_manager.optimize_qos(
        current_params,
        sample_metrics,
        sample_chat
    )
    
    print("\nOptimized QoS Parameters:")
    print(f"Bitrate: {optimized_params.bitrate} kbps")
    print(f"Resolution: {optimized_params.resolution}")
    print(f"FPS: {optimized_params.fps}")
    print(f"Buffer Size: {optimized_params.buffer_size} bytes")
    print(f"Network Quality: {optimized_params.network_quality}")

if __name__ == "__main__":
    main() 