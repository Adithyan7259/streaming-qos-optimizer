import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List

class StreamingDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_streaming_metrics(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Process streaming metrics including bitrate, latency, frame drops, etc.
        
        Args:
            data: DataFrame containing streaming metrics
            
        Returns:
            Tuple of processed features and metadata
        """
        # Extract relevant features
        features = [
            'bitrate', 'latency', 'frame_drops', 'fps',
            'buffer_size', 'network_quality'
        ]
        
        # Handle missing values
        data = data.fillna(method='ffill')
        
        # Scale numerical features
        scaled_features = self.scaler.fit_transform(data[features])
        
        # Create metadata dictionary
        metadata = {
            'feature_names': features,
            'scaler_params': {
                'mean_': self.scaler.mean_,
                'scale_': self.scaler.scale_
            }
        }
        
        return scaled_features, metadata
    
    def process_chat_data(self, chat_data: List[Dict]) -> pd.DataFrame:
        """
        Process chat messages and engagement metrics
        
        Args:
            chat_data: List of chat messages with metadata
            
        Returns:
            DataFrame with processed chat metrics
        """
        # Convert chat data to DataFrame
        df = pd.DataFrame(chat_data)
        
        # Calculate engagement metrics
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Calculate message frequency per hour
        message_freq = df.groupby('hour').size().reset_index(name='message_count')
        
        # Calculate emote usage
        df['emote_count'] = df['message'].str.count(r':\w+:')
        
        # Create a summary DataFrame with the metrics we need
        summary_df = pd.DataFrame({
            'message_count': [len(df)],  # Total message count
            'emote_count': [df['emote_count'].sum()]  # Total emote count
        })
        
        return summary_df
    
    def create_training_data(self, 
                           streaming_metrics: np.ndarray,
                           chat_metrics: pd.DataFrame,
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine streaming metrics and chat data for model training
        
        Args:
            streaming_metrics: Processed streaming metrics
            chat_metrics: Processed chat data
            labels: Ground truth labels
            
        Returns:
            Tuple of features and labels for training
        """
        # Get chat features
        chat_features = chat_metrics[['message_count', 'emote_count']].values
        
        # Repeat chat features to match the number of streaming metric samples
        num_samples = streaming_metrics.shape[0]
        repeated_chat_features = np.repeat(chat_features, num_samples, axis=0)
        
        # Combine features
        combined_features = np.hstack([streaming_metrics, repeated_chat_features])
        
        return combined_features, labels 