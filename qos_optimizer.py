import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class QoSParameters:
    bitrate: float
    resolution: Tuple[int, int]
    fps: int
    buffer_size: int
    network_quality: float

class QoSOptimizer:
    def __init__(self):
        self.min_bitrate = 1000  # kbps
        self.max_bitrate = 8000  # kbps
        self.resolution_presets = {
            'low': (640, 360),
            'medium': (1280, 720),
            'high': (1920, 1080)
        }
        self.fps_presets = [30, 60]
        
    def optimize_parameters(self,
                          current_params: QoSParameters,
                          satisfaction_score: float,
                          sentiment_score: float,
                          network_quality: float) -> QoSParameters:
        """
        Optimize streaming parameters based on satisfaction and sentiment scores
        
        Args:
            current_params: Current QoS parameters
            satisfaction_score: Predicted satisfaction score (0-1)
            sentiment_score: Aggregate sentiment score (-1 to 1)
            network_quality: Current network quality score (0-1)
            
        Returns:
            Optimized QoS parameters
        """
        # Calculate combined score (weighted average)
        combined_score = 0.7 * satisfaction_score + 0.3 * (sentiment_score + 1) / 2
        
        # Adjust bitrate based on scores and network quality
        target_bitrate = self._optimize_bitrate(
            current_params.bitrate,
            combined_score,
            network_quality
        )
        
        # Select appropriate resolution and fps
        resolution, fps = self._select_quality_preset(
            target_bitrate,
            network_quality
        )
        
        # Calculate optimal buffer size
        buffer_size = self._calculate_buffer_size(
            target_bitrate,
            network_quality
        )
        
        return QoSParameters(
            bitrate=target_bitrate,
            resolution=resolution,
            fps=fps,
            buffer_size=buffer_size,
            network_quality=network_quality
        )
    
    def _optimize_bitrate(self,
                         current_bitrate: float,
                         combined_score: float,
                         network_quality: float) -> float:
        """
        Optimize bitrate based on scores and network quality
        """
        # Calculate target bitrate
        target_bitrate = current_bitrate
        
        if combined_score < 0.5:
            # Reduce bitrate if satisfaction is low
            target_bitrate *= 0.8
        elif combined_score > 0.8 and network_quality > 0.7:
            # Increase bitrate if satisfaction is high and network is good
            target_bitrate *= 1.2
            
        # Ensure bitrate stays within bounds
        target_bitrate = np.clip(
            target_bitrate,
            self.min_bitrate,
            self.max_bitrate
        )
        
        return target_bitrate
    
    def _select_quality_preset(self,
                             target_bitrate: float,
                             network_quality: float) -> Tuple[Tuple[int, int], int]:
        """
        Select appropriate resolution and fps based on bitrate and network quality
        """
        if target_bitrate < 2500 or network_quality < 0.5:
            resolution = self.resolution_presets['low']
            fps = self.fps_presets[0]
        elif target_bitrate < 5000 or network_quality < 0.7:
            resolution = self.resolution_presets['medium']
            fps = self.fps_presets[0]
        else:
            resolution = self.resolution_presets['high']
            fps = self.fps_presets[1]
            
        return resolution, fps
    
    def _calculate_buffer_size(self,
                             target_bitrate: float,
                             network_quality: float) -> int:
        """
        Calculate optimal buffer size based on bitrate and network quality
        """
        # Base buffer size in seconds
        base_buffer = 5
        
        # Adjust buffer size based on network quality
        if network_quality < 0.5:
            buffer_multiplier = 2.0
        elif network_quality < 0.7:
            buffer_multiplier = 1.5
        else:
            buffer_multiplier = 1.0
            
        # Calculate buffer size in bytes
        buffer_size = int(
            base_buffer * buffer_multiplier * target_bitrate * 1000 / 8
        )
        
        return buffer_size 