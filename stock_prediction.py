import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import seaborn as sns
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define hyperparameters - focusing on TSLA initially
TICKERS = ["TSLA"]  # Focus on TSLA as our main test case
LOOK_BACK = 60  # Number of days to look back for prediction
FORECAST_HORIZON = 5  # Days to forecast in the future
BATCH_SIZE = 32
EPOCHS = 40  # Increased epochs for better model convergence
LEARNING_RATE = 0.0001
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8
DROPOUT = 0.2  # Slightly reduced dropout for better performance
D_FF = 1024
MAX_LEN = 1000
N_SPLITS = 5
FEATURE_DIM = 64  # Output dimension for fusion model compatibility

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers, d_ff, dropout, feature_dim=FEATURE_DIM):
        super(StockTransformer, self).__init__()
        
        # Input embedding with layer normalization and residual connection
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding with learned parameters
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder with layer normalization and residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Stock embedding with learned parameters
        self.stock_embedding = nn.Sequential(
            nn.Embedding(len(TICKERS), d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Feature extraction for fusion model (64-dimensional output)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x, stock_idx):
        # Add stock embedding
        stock_emb = self.stock_embedding(stock_idx).unsqueeze(1)
        
        # Input embedding with residual connection
        x_emb = self.embedding(x)
        x = x_emb + self.pos_encoding(x_emb)
        
        # Add stock embedding to each time step
        x = x + stock_emb
        
        # Transformer with residual connection
        x_trans = self.transformer(x)
        x = x + x_trans
        
        # Take the last time step's output
        x = x[:, -1, :]
        
        # Extract features for fusion model
        features = self.feature_extractor(x)
        
        # Price prediction
        price = self.price_head(features)
        
        # Confidence estimation
        confidence = self.confidence_head(features)
        
        return price, features, confidence

class StockDataset(Dataset):
    def __init__(self, x, y, stock_idx):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.stock_idx = torch.tensor(stock_idx, dtype=torch.long)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.stock_idx[idx]

def fetch_financial_data(tickers, period="5y"):
    """Fetch stock data for multiple tickers with enhanced error handling and validation"""
    all_data = {}
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}...")
            df = yf.download(ticker, period=period)
            
            if df.empty:
                logger.error(f"No data available for {ticker}")
                continue
                
            logger.info(f"Processing {ticker} data...")
            logger.info(f"Initial shape: {df.shape}")
            
            # Create a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Calculate technical indicators
            # Moving averages
            close_series = df['Close'].squeeze()
            df.loc[:, 'MA_5'] = close_series.rolling(window=5, min_periods=1).mean()
            df.loc[:, 'MA_20'] = close_series.rolling(window=20, min_periods=1).mean()
            df.loc[:, 'MA_50'] = close_series.rolling(window=50, min_periods=1).mean()
            
            # MACD
            df.loc[:, 'EMA_12'] = close_series.ewm(span=12, adjust=False, min_periods=1).mean()
            df.loc[:, 'EMA_26'] = close_series.ewm(span=26, adjust=False, min_periods=1).mean()
            df.loc[:, 'MACD'] = df['EMA_12'] - df['EMA_26']
            df.loc[:, 'MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            
            # RSI
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df.loc[:, 'BB_Middle'] = df['MA_20']
            df.loc[:, 'BB_Std'] = close_series.rolling(window=20, min_periods=1).std()
            df.loc[:, 'BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df.loc[:, 'BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            
            # Volume indicators
            volume_series = df['Volume'].squeeze()
            df.loc[:, 'Volume_MA_5'] = volume_series.rolling(window=5, min_periods=1).mean()
            df.loc[:, 'Volume_Ratio'] = (volume_series / df['Volume_MA_5'].replace(0, 1e-10)).fillna(1.0)
            df.loc[:, 'Volume_1d_Change'] = volume_series.pct_change().fillna(0)
            
            # Volatility
            df.loc[:, 'Daily_Return'] = close_series.pct_change().fillna(0)
            df.loc[:, 'Volatility_20d'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
            
            # Additional features
            df.loc[:, 'Price_MA_Ratio'] = (close_series / df['MA_20'].replace(0, 1e-10)).fillna(1.0)
            df.loc[:, 'BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'].replace(0, 1e-10)).fillna(0)
            df.loc[:, 'MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Fill NaN values
            price_cols = ['Close', 'MA_5', 'MA_20', 'MA_50', 'EMA_12', 'EMA_26', 
                         'MACD', 'MACD_Signal', 'BB_Middle', 'BB_Upper', 'BB_Lower']
            df[price_cols] = df[price_cols].fillna(method='ffill')
            
            other_cols = ['RSI', 'BB_Std', 'Volume_MA_5', 'Volume_Ratio', 'Volatility_20d',
                         'Price_MA_Ratio', 'BB_Width', 'MACD_Hist']
            df[other_cols] = df[other_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Replace any remaining NaN and infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)  # Final fallback to zero
            
            # Verify data quality
            if df.isnull().sum().sum() > 0:
                logger.warning(f"Data for {ticker} still contains NaN values after processing. Filling with zeros.")
                df = df.fillna(0)
                
            if (df == np.inf).any().any() or (df == -np.inf).any().any():
                logger.warning(f"Data for {ticker} still contains infinite values after processing. Replacing with large values.")
                df = df.replace([np.inf], 1e9).replace([-np.inf], -1e9)
            
            logger.info(f"After processing shape: {df.shape}")
            
            if len(df) > LOOK_BACK + FORECAST_HORIZON:
                all_data[ticker] = df
                logger.info(f"Successfully processed {ticker} data: {len(df)} samples")
            else:
                logger.error(f"Insufficient data for {ticker}: only {len(df)} samples")
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No valid data available for any of the tickers")
    
    return all_data

def prepare_multi_stock_data(data_dict, look_back=LOOK_BACK, forecast_horizon=FORECAST_HORIZON):
    """Prepare data for multiple stocks with enhanced validation"""
    X_sequences = []
    y_sequences = []
    stock_indices = []
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 
                'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 
                'Volatility_20d', 'Price_MA_Ratio', 'BB_Width', 'MACD_Hist']
    
    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))  # Better for transformer models
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # First pass: fit scalers
    all_features = []
    all_targets = []
    
    for ticker, data in data_dict.items():
        try:
            # Verify all features exist
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                logger.error(f"Missing features for {ticker}: {missing_features}")
                continue
                
            # Select features
            X = data[features].astype(float).values
            y = data[['Close']].shift(-forecast_horizon).values[:-forecast_horizon]
            X = X[:-forecast_horizon]
            
            # Check for invalid values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning(f"Invalid values found in {ticker} features. Attempting to clean...")
                X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
                
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                logger.warning(f"Invalid values found in {ticker} targets. Attempting to clean...")
                y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9)
                
            all_features.append(X)
            all_targets.append(y)
            
        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {str(e)}")
            continue
    
    if not all_features:
        raise ValueError("No valid features could be extracted from the data")
    
    # Fit scalers on all data
    all_features = np.vstack(all_features)
    all_targets = np.vstack(all_targets)
    feature_scaler.fit(all_features)
    target_scaler.fit(all_targets)
    
    # Second pass: transform data and create sequences
    for stock_idx, (ticker, data) in enumerate(data_dict.items()):
        try:
            # Select features
            X = data[features].astype(float).values
            y = data[['Close']].shift(-forecast_horizon).values[:-forecast_horizon]
            X = X[:-forecast_horizon]
            
            # Clean any invalid values
            X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
            y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Scale the data
            X_scaled = feature_scaler.transform(X)
            y_scaled = target_scaler.transform(y)
            
            # Create sequences
            for i in range(len(X_scaled) - look_back):
                X_sequences.append(X_scaled[i:i+look_back])
                y_sequences.append(y_scaled[i+look_back-1])
                stock_indices.append(stock_idx)
                
            logger.info(f"Added {len(X_scaled) - look_back} sequences for {ticker}")
            
        except Exception as e:
            logger.error(f"Error creating sequences for {ticker}: {str(e)}")
            continue
    
    if not X_sequences:
        raise ValueError("No valid sequences could be created from the data")
    
    return np.array(X_sequences), np.array(y_sequences), np.array(stock_indices), feature_scaler, target_scaler

def train_model(model, train_loader, val_loader, device, epochs):
    """Train the model with enhanced optimization and monitoring"""
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # Define loss functions
    price_criterion = nn.MSELoss()
    confidence_criterion = nn.BCELoss()
    
    # Define optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_price_loss = 0
        train_conf_loss = 0
        
        for X_batch, y_batch, stock_idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            X_batch, y_batch, stock_idx = X_batch.to(device), y_batch.to(device), stock_idx.to(device)
            
            optimizer.zero_grad()
            price_pred, features, confidence = model(X_batch, stock_idx)
            
            # Ensure shapes match
            if price_pred.shape != y_batch.shape:
                price_pred = price_pred.squeeze()
            
            # Confidence target is based on prediction error 
            # (1 for perfect prediction, 0 for bad prediction)
            with torch.no_grad():
                error = torch.abs(price_pred - y_batch)
                max_error = torch.max(error)
                if max_error > 0:
                    confidence_target = 1 - (error / max_error)
                else:
                    confidence_target = torch.ones_like(error)
            
            # Calculate losses
            p_loss = price_criterion(price_pred, y_batch)
            c_loss = confidence_criterion(confidence, confidence_target)
            
            # Combined loss with dynamic weighting
            loss = p_loss + 0.2 * c_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_price_loss += p_loss.item()
            train_conf_loss += c_loss.item()
        
        train_loss /= len(train_loader)
        train_price_loss /= len(train_loader)
        train_conf_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_price_loss = 0
        val_conf_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch, stock_idx in val_loader:
                X_batch, y_batch, stock_idx = X_batch.to(device), y_batch.to(device), stock_idx.to(device)
                price_pred, features, confidence = model(X_batch, stock_idx)
                
                # Ensure shapes match
                if price_pred.shape != y_batch.shape:
                    price_pred = price_pred.squeeze()
                
                # Confidence target
                error = torch.abs(price_pred - y_batch)
                max_error = torch.max(error)
                if max_error > 0:
                    confidence_target = 1 - (error / max_error)
                else:
                    confidence_target = torch.ones_like(error)
                
                # Calculate losses
                p_loss = price_criterion(price_pred, y_batch)
                c_loss = confidence_criterion(confidence, confidence_target)
                
                loss = p_loss + 0.2 * c_loss
                
                val_loss += loss.item()
                val_price_loss += p_loss.item()
                val_conf_loss += c_loss.item()
        
        val_loss /= len(val_loader)
        val_price_loss /= len(val_loader)
        val_conf_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_stock_transformer.pth')
            logger.info(f"Epoch {epoch+1}: Saved new best model with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.6f} (Price: {train_price_loss:.6f}, Conf: {train_conf_loss:.6f}), "
                   f"Val Loss: {val_loss:.6f} (Price: {val_price_loss:.6f}, Conf: {val_conf_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()
    
    # Load best model
    model.load_state_dict(torch.load('best_stock_transformer.pth'))
    return model

def evaluate_model(model, test_loader, target_scaler, device):
    """Evaluate model with detailed metrics and uncertainty estimation"""
    model.eval()
    all_preds = []
    all_targets = []
    all_features = []
    all_confidence = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_indices in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_indices = batch_indices.to(device)
            
            price_pred, features, confidence = model(batch_x, batch_indices)
            
            # Store predictions, targets, features and confidence for metrics
            all_preds.extend(price_pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features = np.array(all_features)
    all_confidence = np.array(all_confidence)
    
    # Inverse transform predictions and targets
    if len(all_preds.shape) == 1:
        all_preds = all_preds.reshape(-1, 1)
    if len(all_targets.shape) == 1:
        all_targets = all_targets.reshape(-1, 1)
    
    all_preds_original = target_scaler.inverse_transform(all_preds)
    all_targets_original = target_scaler.inverse_transform(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets_original, all_preds_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_original, all_preds_original)
    r2 = r2_score(all_targets_original, all_preds_original)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((all_targets_original - all_preds_original) / all_targets_original)) * 100
    
    # Accuracy based on direction prediction
    direction_correct = np.sum((np.diff(all_targets_original, axis=0) > 0) == 
                              (np.diff(all_preds_original, axis=0) > 0))
    direction_accuracy = direction_correct / (len(all_targets_original) - 1)
    
    # Confidence analysis
    mean_confidence = np.mean(all_confidence)
    confidence_correlation = np.corrcoef(all_confidence.flatten(), 
                                       1 - np.abs(all_targets - all_preds).flatten())[0, 1]
    
    # Print metrics
    logger.info(f"Test Metrics:")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R2 Score: {r2:.6f}")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"Direction Accuracy: {direction_accuracy:.2f}")
    logger.info(f"Mean Confidence: {mean_confidence:.4f}")
    logger.info(f"Confidence-Error Correlation: {confidence_correlation:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets_original[:100], label='Actual')
    plt.plot(all_preds_original[:100], label='Predicted')
    plt.title('Stock Price Prediction (Test Set Sample)')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.close()
    
    # Plot confidence vs error
    plt.figure(figsize=(10, 6))
    abs_errors = np.abs(all_targets - all_preds).flatten()
    plt.scatter(all_confidence.flatten(), 1 - abs_errors, alpha=0.5)
    plt.title('Model Confidence vs Prediction Accuracy')
    plt.xlabel('Model Confidence')
    plt.ylabel('1 - Absolute Error')
    plt.grid(True)
    plt.savefig('confidence_vs_error.png')
    plt.close()
    
    return mse, mae, r2, all_features, all_confidence

def extract_timeseries_features(model, data_dict, feature_scaler, target_scaler, device, lookback=LOOK_BACK):
    """Extract time series features for the fusion model"""
    model.eval()
    stock_features = {}
    
    for ticker, df in data_dict.items():
        try:
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 
                        'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 
                        'Volatility_20d', 'Price_MA_Ratio', 'BB_Width', 'MACD_Hist']
            
            # Verify features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing features for {ticker}: {missing_features}")
                continue
                
            # Get most recent data
            recent_data = df[features].iloc[-lookback:].values
            
            # Handle potential issues with the data
            recent_data = np.nan_to_num(recent_data, nan=0.0, posinf=1e9, neginf=-1e9)
            
            # Scale the data
            X_scaled = feature_scaler.transform(recent_data)
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)  # Add batch dimension
            
            # Get stock index
            stock_idx = torch.tensor([TICKERS.index(ticker)], dtype=torch.long).to(device)
            
            # Extract features
            with torch.no_grad():
                _, feature_vector, confidence = model(X_tensor, stock_idx)
                
                # Create output dictionary with all necessary information for fusion model
                stock_features[ticker] = {
                    'timeseries_feat': feature_vector.cpu().numpy()[0],  # Extract the 64-dim feature vector
                    'confidence': confidence.item(),
                    'last_price': df['Close'].iloc[-1],
                    'volatility': df['Volatility_20d'].iloc[-1],
                    'rsi': df['RSI'].iloc[-1],
                    'timestamp': df.index[-1].strftime('%Y-%m-%d')
                }
                
            logger.info(f"Successfully extracted features for {ticker}")
            
        except Exception as e:
            logger.error(f"Error extracting features for {ticker}: {str(e)}")
            continue
    
    return stock_features

def predict_future(model, data_dict, feature_scaler, target_scaler, days_to_predict, device):
    """Make future predictions for multiple stocks"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50', 
                'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_Ratio', 
                'Volatility_20d', 'Price_MA_Ratio', 'BB_Width', 'MACD_Hist']
    
    predictions = {}
    for ticker, data in data_dict.items():
        try:
            # Get the last look_back days of data
            X = data[features].iloc[-LOOK_BACK:].values
            
            # Scale the data
            X = feature_scaler.transform(X)
            
            # Convert to tensor and add batch dimension
            X = torch.FloatTensor(X).unsqueeze(0).to(device)
            stock_idx = torch.LongTensor([list(data_dict.keys()).index(ticker)]).to(device)
            
            # Make prediction
            with torch.no_grad():
                price_pred, _, confidence = model(X, stock_idx)
                price_pred = target_scaler.inverse_transform(price_pred.cpu().numpy())
                confidence = confidence.cpu().numpy()
            
            predictions[ticker] = {
                'price': price_pred[0][0],
                'confidence': confidence[0][0]
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {str(e)}")
            continue
    
    return predictions

if __name__ == "__main__":
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Fetch data
        logger.info("Fetching stock data...")
        data_dict = fetch_financial_data(TICKERS)
        
        if not data_dict:
            raise ValueError("No data could be fetched for any ticker")

        # Prepare data
        logger.info("Preparing data...")
        X, y, stock_indices, feature_scaler, target_scaler = prepare_multi_stock_data(data_dict)
        
        # Create datasets
        dataset = StockDataset(X, y, stock_indices)
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        input_size = X.shape[2]  # Number of features
        model = StockTransformer(
            input_size=input_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT
        ).to(device)
        
        # Train model
        logger.info("Starting model training...")
        train_model(model, train_loader, val_loader, device, EPOCHS)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluate_model(model, test_loader, target_scaler, device)
        
        # Make future predictions
        logger.info("Making future predictions...")
        predictions = predict_future(model, data_dict, feature_scaler, target_scaler, FORECAST_HORIZON, device)
        
        # Print predictions
        for ticker, pred in predictions.items():
            logger.info(f"{ticker} - Predicted Price: ${pred['price']:.2f} (Confidence: {pred['confidence']:.2%})")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

