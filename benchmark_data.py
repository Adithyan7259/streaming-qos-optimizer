import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import random

class BenchmarkDataset:
    def __init__(self):
        self.resolutions = [(1280, 720), (1920, 1080), (854, 480)]
        self.bitrates = [2000, 4000, 6000, 8000]
        self.fps_options = [30, 60]
        self.buffer_sizes = [1000000, 2000000, 4000000]
        
    def generate_streaming_metrics(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic streaming metrics data
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame containing streaming metrics
        """
        data = {
            'bitrate': np.random.choice(self.bitrates, num_samples),
            'latency': np.random.normal(100, 20, num_samples),  # Mean 100ms, std 20ms
            'frame_drops': np.random.poisson(2, num_samples),  # Poisson distribution for frame drops
            'fps': np.random.choice(self.fps_options, num_samples),
            'buffer_size': np.random.choice(self.buffer_sizes, num_samples),
            'network_quality': np.random.uniform(0.5, 1.0, num_samples)
        }
        return pd.DataFrame(data)
    
    def generate_chat_data(self, num_messages: int = 100) -> List[Dict]:
        """
        Generate synthetic chat data with timestamps and messages
        
        Args:
            num_messages: Number of chat messages to generate
            
        Returns:
            List of chat message dictionaries
        """
        base_time = datetime.now()
        messages = [
            "Great stream quality! :D",
            "The game looks smooth",
            "No lag at all",
            "Stream is a bit choppy",
            "Perfect quality!",
            "Having some buffering issues",
            "The stream is crystal clear",
            "Getting some frame drops",
            "Amazing stream quality",
            "Stream is running perfectly"
        ]
        
        chat_data = []
        for i in range(num_messages):
            timestamp = base_time + timedelta(seconds=i*5)  # 5 seconds between messages
            message = random.choice(messages)
            chat_data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'message': message
            })
        
        return chat_data
    
    def generate_ground_truth(self, streaming_metrics: pd.DataFrame) -> np.ndarray:
        """
        Generate ground truth satisfaction scores based on streaming metrics
        
        Args:
            streaming_metrics: DataFrame containing streaming metrics
            
        Returns:
            Array of satisfaction scores (0-1)
        """
        # Calculate satisfaction based on various factors
        latency_score = 1 - (streaming_metrics['latency'] / 200)  # Normalize latency
        frame_drop_score = 1 - (streaming_metrics['frame_drops'] / 10)  # Normalize frame drops
        fps_score = streaming_metrics['fps'] / 60  # Normalize FPS
        network_score = streaming_metrics['network_quality']
        
        # Combine scores with weights
        satisfaction = (
            0.3 * latency_score +
            0.3 * frame_drop_score +
            0.2 * fps_score +
            0.2 * network_score
        )
        
        # Clip to [0, 1] range
        return np.clip(satisfaction, 0, 1)
    
    def load_benchmark_data(self) -> Tuple[pd.DataFrame, List[Dict], np.ndarray]:
        """
        Load or generate benchmark dataset
        
        Returns:
            Tuple of (streaming_metrics, chat_data, ground_truth)
        """
        streaming_metrics = self.generate_streaming_metrics()
        chat_data = self.generate_chat_data()
        ground_truth = self.generate_ground_truth(streaming_metrics)
        
        return streaming_metrics, chat_data, ground_truth 