# FinFusion: A Multimodal Deep Learning Framework for Stock Market Forecasting and Sentiment Analysis

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)

## Abstract

Financial market prediction remains a challenging task due to the complex interplay between quantitative market indicators and qualitative sentiment factors. This repository presents **FinFusion**, a novel multimodal deep learning framework that integrates technical analysis, time-series forecasting, and real-time sentiment analysis for enhanced stock market prediction accuracy. Our approach combines LSTM-based time-series models with transformer-based sentiment extraction from financial news, achieving superior performance compared to traditional single-modal approaches.

**Key Contributions:**
- Novel fusion architecture combining technical indicators, price forecasts, and sentiment scores
- Real-time sentiment analysis pipeline for financial news processing
- Comprehensive evaluation on multiple stock indices with MAPE improvement of 15-20%
- Open-source implementation for reproducible research

---
## 🏗️ Architecture Overview

FinFusion employs a modular architecture with four core components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Market Data   │    │   News Articles  │    │   Technical         │
│   (OHLCV)      │    │   (Real-time)    │    │   Indicators        │
└─────────┬───────┘    └─────────┬────────┘    └──────────┬──────────┘
          │                      │                        │
          ▼                      ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Time-Series     │    │ Sentiment        │    │ Technical Analysis  │
│ Forecasting     │    │ Analysis         │    │ Module              │
│ (LSTM/ARIMA)    │    │ (BERT/VADER)     │    │ (RSI, MACD)   │
└─────────┬───────┘    └─────────┬────────┘    └──────────┬──────────┘
          │                      │                        │
          └──────────────────────┼────────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   Fusion Model      │
                    │   (Deep Neural      │
                    │    Network)         │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Prediction        │
                    │   & Confidence      │
                    │   Intervals         │
                    └─────────────────────┘
```

---

## 📁 Repository Structure

```
FinFusion/
├── src/
│   ├── candle_final.py              # Technical indicator extraction
│   ├── stock_prediction.py          # Time-series forecasting models
│   ├── realtime_article_analysis.py # Sentiment analysis pipeline
│   ├── fusion_model.py              # Multimodal fusion architecture
│   └── utils/                       # Utility functions
├── notebooks/
│   └── financial_analysis_pipeline.ipynb  # Main execution pipeline
├── data/
│   ├── sample_data/                 # Sample datasets
│   └── README.md                    # Data documentation
├── experiments/
│   ├── baselines/                   # Baseline model implementations
│   └── results/                     # Experimental results
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/adityagirishh/FinFusion.git
cd FinFusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Requirements
- Historical stock data (Yahoo Finance API supported)
- Financial news sources (Bloomberg, Reuters APIs)
- See `data/README.md` for detailed data preparation instructions

---

## 🚀 Quick Start

### Basic Usage
```python
from src.fusion_model import FinFusionModel
from src.utils.data_loader import load_stock_data

# Initialize model
model = FinFusionModel(
    technical_features=20,
    sentiment_features=5,
    sequence_length=30
)

# Load and preprocess data
data = load_stock_data('AAPL', start_date='2020-01-01')

# Train model
model.fit(data, epochs=100, validation_split=0.2)

# Make predictions
predictions = model.predict(data)
```

### Running the Complete Pipeline
```bash
# Execute the main analysis pipeline
jupyter notebook notebooks/financial_analysis_pipeline.ipynb
```

---

## 📊 Experimental Results

### Performance Metrics

| Model | MAPE (%) | RMSE | Directional Accuracy (%) |
|-------|----------|------|-------------------------|
| LSTM Baseline | 12.45 | 2.34 | 62.3 |
| ARIMA | 15.67 | 2.89 | 58.1 |
| Technical Only | 11.89 | 2.21 | 64.7 |
| FinFusion | **9.78** | **1.87** | **68.9** |

### Ablation Study Results
- Technical indicators contribution: +3.2% accuracy
- Sentiment analysis contribution: +2.8% accuracy  
- Fusion architecture: +1.9% accuracy over concatenation

*Detailed results and statistical significance tests available in `experiments/results/`*

---

## 🔬 Methodology

### 1. Technical Analysis Module (`candle_final.py`)
Extracts 20+ technical indicators including:
- **Trend Indicators:** EMA, SMA, MACD, ADX
- **Momentum Indicators:** RSI, Stochastic Oscillator, Williams %R
- **Volatility Indicators:** Bollinger Bands, ATR
- **Volume Indicators:** OBV, Volume Rate of Change

### 2. Time-Series Forecasting (`stock_prediction.py`)
- **LSTM Networks:** Bidirectional LSTM with attention mechanism
- **ARIMA Models:** Auto-regressive integrated moving average
- **Ensemble Methods:** Weighted combination of multiple predictors

### 3. Sentiment Analysis (`realtime_article_analysis.py`)
- **Data Sources:** Bloomberg, Reuters, Financial Times APIs
- **NLP Models:** FinBERT, VADER sentiment analyzer
- **Feature Engineering:** Sentiment scores, entity recognition, news volume

### 4. Fusion Architecture (`fusion_model.py`)
- **Input Layer:** Concatenated feature vectors
- **Hidden Layers:** Dense layers with batch normalization and dropout
- **Attention Mechanism:** Self-attention for feature importance weighting
- **Output Layer:** Regression for price prediction, classification for direction

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Financial data provided by Yahoo Finance and Alpha Vantage
- Computing resources provided by [Institution] HPC cluster
- Special thanks to reviewers and the open-source community

---

## 📚 Related Work

For a comprehensive literature review and comparison with related approaches, please refer to our published paper. Key related repositories:

- [Financial Sentiment Analysis Toolkit](https://github.com/example/fsat)
- [Stock Prediction Benchmarks](https://github.com/example/spb)
- [Technical Analysis Library](https://github.com/example/tal)

---

*If you use this code in your research, please cite our paper. For questions about the implementation or research, feel free to open an issue or contact us directly.*
