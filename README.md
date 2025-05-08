Adithyan R
RA2211027010040







# Streaming QoS Optimizer

An intelligent Quality of Service (QoS) optimization system for live video game streaming platforms. This system uses machine learning and sentiment analysis to optimize streaming parameters in real-time, balancing technical performance with viewer satisfaction.

## Features

- Real-time QoS parameter optimization
- Machine learning-based satisfaction prediction
- Sentiment analysis of viewer chat
- Benchmark dataset generation
- Performance evaluation metrics
- Support for multiple streaming parameters:
  - Bitrate (2000-8000 kbps)
  - Resolution (480p, 720p, 1080p)
  - FPS (30/60)
  - Buffer size
  - Network quality settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streaming-qos-optimizer.git
cd streaming-qos-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

This will:
1. Load/generate benchmark data
2. Evaluate system performance
3. Show optimization results

## Project Structure

```
streaming-qos-optimizer/
├── main.py                 # Main application entry point
├── data_processor.py       # Data processing and feature engineering
├── satisfaction_classifier.py  # ML model for satisfaction prediction
├── sentiment_analyzer.py   # Chat sentiment analysis
├── qos_optimizer.py        # QoS parameter optimization
├── benchmark_data.py       # Benchmark dataset generation
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Components

### Data Processor
- Processes streaming metrics (bitrate, latency, frame drops, etc.)
- Analyzes chat messages for viewer feedback
- Combines technical and social metrics

### Satisfaction Classifier
- Predicts viewer satisfaction based on streaming metrics
- Uses machine learning to understand quality-satisfaction relationships
- Helps make informed decisions about quality adjustments

### Sentiment Analyzer
- Analyzes chat messages for viewer sentiment
- Provides additional feedback beyond technical metrics
- Helps understand viewer experience from comments

### QoS Optimizer
- Adjusts streaming parameters based on predictions and feedback
- Balances quality with technical constraints
- Optimizes for viewer satisfaction

## Benchmark System

The benchmark system generates synthetic data for testing and evaluation:
- 1000 samples of technical data
- 100 chat messages
- Ground truth satisfaction scores

## Evaluation Metrics

- Mean Squared Error (MSE) for prediction accuracy
- R² score for model fit
- Comparison of predicted vs. actual satisfaction

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python 3.9+
- Uses scikit-learn for machine learning
- Implements sentiment analysis for chat processing 
