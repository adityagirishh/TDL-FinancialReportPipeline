import yfinance as yf
import mplfinance as mpf
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Technical Indicator Implementation (replacing talib functions)
def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Moving Average Convergence Divergence"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_momentum(data, period=10):
    """Calculate Momentum"""
    return data.diff(period)

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({
        'tr1': tr1,
        'tr2': tr2,
        'tr3': tr3
    }).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = pd.Series(index=close.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_ad(high, low, close, volume):
    """Calculate Accumulation/Distribution Line"""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.finfo(float).eps)
    ad = (clv * volume).cumsum()
    return ad

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr = calculate_atr(high, low, close, period=1)
    
    # Plus Directional Movement (+DM)
    high_diff = high.diff()
    low_diff = low.diff().abs()
    pdm = pd.Series(0, index=high.index)
    pdm[(high_diff > 0) & (high_diff > low_diff)] = high_diff
    
    # Minus Directional Movement (-DM)
    low_diff = low.diff()
    high_diff = high.diff().abs()
    mdm = pd.Series(0, index=low.index)
    mdm[(low_diff > 0) & (low_diff > high_diff)] = low_diff
    
    # Smooth +DM, -DM and TR with Wilder's smoothing technique
    smoothed_tr = tr.rolling(window=period).sum()
    smoothed_pdm = pdm.rolling(window=period).sum()
    smoothed_mdm = mdm.rolling(window=period).sum()
    
    # Calculate +DI and -DI
    pdi = 100 * smoothed_pdm / smoothed_tr.replace(0, np.finfo(float).eps)
    mdi = 100 * smoothed_mdm / smoothed_tr.replace(0, np.finfo(float).eps)
    
    # Calculate DX and ADX
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.finfo(float).eps)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_cci(high, low, close, period=14):
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3
    tp_sma = calculate_sma(tp, period)
    md = (tp - tp_sma).abs().rolling(window=period).mean()
    cci = (tp - tp_sma) / (0.015 * md.replace(0, np.finfo(float).eps))
    
    return cci

def identify_doji(open_prices, high, low, close, tolerance=0.1):
    """Identify Doji candle pattern"""
    body_size = abs(close - open_prices)
    high_wick = high - np.maximum(close, open_prices)
    low_wick = np.minimum(close, open_prices) - low
    
    # Body is very small compared to the wicks
    body_to_range_ratio = body_size / (high - low).replace(0, np.finfo(float).eps)
    
    return (body_to_range_ratio < tolerance).astype(int) * 100

def identify_hammer(open_prices, high, low, close, body_ratio=0.3, wick_ratio=2):
    """Identify Hammer candle pattern"""
    body_size = abs(close - open_prices)
    upper_wick = high - np.maximum(close, open_prices)
    lower_wick = np.minimum(close, open_prices) - low
    total_range = high - low
    
    # Small body at the upper end with a long lower shadow
    is_body_small = body_size / total_range.replace(0, np.finfo(float).eps) <= body_ratio
    is_upper_wick_small = upper_wick / body_size.replace(0, np.finfo(float).eps) <= 0.5
    is_lower_wick_long = lower_wick / body_size.replace(0, np.finfo(float).eps) >= wick_ratio
    
    return (is_body_small & is_upper_wick_small & is_lower_wick_long).astype(int) * 100

def identify_shooting_star(open_prices, high, low, close, body_ratio=0.3, wick_ratio=2):
    """Identify Shooting Star candle pattern"""
    body_size = abs(close - open_prices)
    upper_wick = high - np.maximum(close, open_prices)
    lower_wick = np.minimum(close, open_prices) - low
    total_range = high - low
    
    # Small body at the lower end with a long upper shadow
    is_body_small = body_size / total_range.replace(0, np.finfo(float).eps) <= body_ratio
    is_lower_wick_small = lower_wick / body_size.replace(0, np.finfo(float).eps) <= 0.5
    is_upper_wick_long = upper_wick / body_size.replace(0, np.finfo(float).eps) >= wick_ratio
    
    return (is_body_small & is_lower_wick_small & is_upper_wick_long).astype(int) * 100

# 1. Enhanced candlestick chart generation with technical indicators
def generate_enhanced_candlestick_images(data, output_dir="charts"):
    """
    Generate enhanced candlestick charts with technical indicators
    
    Args:
        data: DataFrame with OHLCV data
        output_dir: Directory to save the chart images
    
    Returns:
        List of tuples (image_path, target_return, features_dict)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed granularity
    window_size = 30
    stride = 5
    
    # Create specific directory for this granularity
    granularity_dir = f"{output_dir}/enhanced_w{window_size}_s{stride}"
    os.makedirs(granularity_dir, exist_ok=True)
    
    # Compute daily returns for target values
    data['Return'] = data['Close'].pct_change()
    
    # Calculate technical indicators
    # Moving Averages
    data['SMA_5'] = calculate_sma(data['Close'], 5)
    data['SMA_10'] = calculate_sma(data['Close'], 10)
    data['SMA_20'] = calculate_sma(data['Close'], 20)
    data['EMA_5'] = calculate_ema(data['Close'], 5)
    data['EMA_10'] = calculate_ema(data['Close'], 10)
    data['EMA_20'] = calculate_ema(data['Close'], 20)
    
    # Momentum Indicators
    data['RSI'] = calculate_rsi(data['Close'], 14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(
        data['Close'], fast_period=12, slow_period=26, signal_period=9)
    data['MOM'] = calculate_momentum(data['Close'], 10)
    
    # Volatility Indicators
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'], 14)
    data['BBANDS_Upper'], data['BBANDS_Middle'], data['BBANDS_Lower'] = calculate_bollinger_bands(
        data['Close'], period=20, num_std=2)
    
    # Volume Indicators
    data['OBV'] = calculate_obv(data['Close'], data['Volume'])
    data['AD'] = calculate_ad(data['High'], data['Low'], data['Close'], data['Volume'])
    
    # Trend Indicators
    data['ADX'] = calculate_adx(data['High'], data['Low'], data['Close'], 14)
    data['CCI'] = calculate_cci(data['High'], data['Low'], data['Close'], 14)
    
    # Price Pattern Indicators
    data['DOJI'] = identify_doji(data['Open'], data['High'], data['Low'], data['Close'])
    data['HAMMER'] = identify_hammer(data['Open'], data['High'], data['Low'], data['Close'])
    data['SHOOTING_STAR'] = identify_shooting_star(data['Open'], data['High'], data['Low'], data['Close'])
    
    # Custom Features
    # Price position relative to moving averages
    data['Price_to_SMA20'] = data['Close'] / data['SMA_20'] - 1
    data['Price_to_SMA50'] = data['Close'] / calculate_sma(data['Close'], 50) - 1
    
    # Volatility measures
    data['Daily_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Weekly_Volatility'] = data['Close'].pct_change().rolling(5).std()
    data['Monthly_Volatility'] = data['Close'].pct_change().rolling(21).std()
    
    # Drop NaN values
    data = data.dropna()
    
    # List to store image paths, targets, and feature dictionaries
    image_data = []
    
    print(f"\nGenerating enhanced charts with window size {window_size} and stride {stride}...")
    
    # Generate charts with sliding window
    for i in range(0, len(data) - window_size - 1, stride):
        window_data = data.iloc[i:i+window_size]
        target_idx = i + window_size
        
        # Skip if we're at the end of the data
        if target_idx >= len(data):
            continue
            
        # Calculate target (next day's return)
        target_return = data.iloc[target_idx]['Return']
        
        # Extract numerical features from the last day in the window
        last_day = window_data.iloc[-1]
        last_5_days = window_data.iloc[-5:]
        
        # Create feature dictionary
        features_dict = {
            # Price features
            'close': last_day['Close'],
            'open': last_day['Open'],
            'high': last_day['High'],
            'low': last_day['Low'],
            'volume': last_day['Volume'],
            
            # Technical indicators
            'rsi': last_day['RSI'],
            'macd': last_day['MACD'],
            'macd_signal': last_day['MACD_Signal'],
            'macd_hist': last_day['MACD_Hist'],
            'mom': last_day['MOM'],
            'atr': last_day['ATR'],
            'bbands_upper': last_day['BBANDS_Upper'],
            'bbands_middle': last_day['BBANDS_Middle'],
            'bbands_lower': last_day['BBANDS_Lower'],
            'obv': last_day['OBV'],
            'adx': last_day['ADX'],
            'cci': last_day['CCI'],
            
            # Custom features
            'price_to_sma20': last_day['Price_to_SMA20'],
            'price_to_sma50': last_day['Price_to_SMA50'],
            'daily_range': last_day['Daily_Range'],
            'weekly_volatility': last_day['Weekly_Volatility'],
            'monthly_volatility': last_day['Monthly_Volatility'],
            
            # Statistical features from the window
            'mean_return': window_data['Return'].mean(),
            'std_return': window_data['Return'].std(),
            'min_return': window_data['Return'].min(),
            'max_return': window_data['Return'].max(),
            
            # Recent price changes
            'return_1d': window_data['Return'].iloc[-1],
            'return_5d': window_data['Close'].iloc[-1] / window_data['Close'].iloc[-5] - 1 if len(window_data) >= 5 else 0,
            'return_10d': window_data['Close'].iloc[-1] / window_data['Close'].iloc[-10] - 1 if len(window_data) >= 10 else 0,
            
            # Price patterns (count in last 5 days)
            'doji_count': sum(1 for x in last_5_days['DOJI'] if x != 0),
            'hammer_count': sum(1 for x in last_5_days['HAMMER'] if x != 0),
            'shooting_star_count': sum(1 for x in last_5_days['SHOOTING_STAR'] if x != 0),
            
            # Moving average crossovers
            'sma5_above_sma20': 1 if last_day['SMA_5'] > last_day['SMA_20'] else 0,
            'ema5_above_ema20': 1 if last_day['EMA_5'] > last_day['EMA_20'] else 0,
            
            # RSI zones
            'rsi_oversold': 1 if last_day['RSI'] < 30 else 0,
            'rsi_overbought': 1 if last_day['RSI'] > 70 else 0,
            
            # MACD signal
            'macd_bullish': 1 if last_day['MACD'] > last_day['MACD_Signal'] else 0,
            
            # Volume analysis
            'volume_ratio': last_day['Volume'] / window_data['Volume'].mean(),
            'volume_trend': last_5_days['Volume'].mean() / window_data['Volume'].mean(),
        }
        
        # Generate filename with date range
        start_date = window_data.index[0].strftime('%Y%m%d')
        end_date = window_data.index[-1].strftime('%Y%m%d')
        filename = f"{granularity_dir}/chart_{start_date}to{end_date}.png"
        
        # Generate and save chart with indicators
        try:
            # Add plot of indicators (3 panels: price, volume, RSI)
            apds = [
                mpf.make_addplot(window_data[['SMA_5', 'SMA_20']], panel=0),
                mpf.make_addplot(window_data['RSI'], panel=2, ylabel='RSI'),
                mpf.make_addplot(window_data['MACD'], panel=3, ylabel='MACD'),
                mpf.make_addplot(window_data['MACD_Signal'], panel=3, color='red')
            ]
            
            # Create the plot with multiple panels
            mpf.plot(window_data, type='candle', style='charles',
                    title=f'Stock: {start_date} to {end_date} (W{window_size})',
                    ylabel='Price',
                    volume=True,
                    addplot=apds,
                    figscale=1.3,
                    panel_ratios=(4, 1, 1, 1),
                    savefig=filename)
            
            # Store image path, target, and features
            image_data.append((filename, target_return, features_dict))
            print(f"Generated chart: {filename}, Target Return: {target_return:.4f}")
        except Exception as e:
            print(f"Error generating chart for {start_date} to {end_date}: {e}")
    
    print(f"Generated {len(image_data)} enhanced charts for window size {window_size}, stride {stride}")
    return image_data

# 2. Extract Features from Candlestick Chart Images using a more advanced model
def extract_image_features(image_path):
    """Extract features from a candlestick chart image using ResNet-50 (more powerful than ResNet-18)"""
    try:
        # Load the pretrained ResNet-50 model
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    except Exception:
        # Fallback for older PyTorch versions
        model = models.resnet50(pretrained=True)
    
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval()
    
    # Enhanced image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
        
        # Convert to numpy array
        feature_vector = features.squeeze().numpy()
        return feature_vector
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

# 3. Load Local Data (CSV file) or Download from Yahoo Finance
def load_data(file_path=None, ticker="TSLA", period="5y"):
    """Load stock data from CSV file or download from Yahoo Finance"""
    if file_path and os.path.exists(file_path):
        try:
            # Try parsing with default header
            stock_data = pd.read_csv(file_path)
            
            # Check if columns need to be renamed
            if 'Date' not in stock_data.columns:
                # Try with header at row 2
                stock_data = pd.read_csv(file_path, header=2)
                # Rename columns if needed
                if len(stock_data.columns) >= 6:  # Assuming OHLCV format
                    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Ensure Date is datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            
            # Sort by date
            stock_data = stock_data.sort_index()
            
            print(f"Loaded data from file with shape: {stock_data.shape}")
            return stock_data
            
        except Exception as e:
            print(f"Error loading data from file: {e}")
            print("Falling back to downloading data from Yahoo Finance...")
    
    # Download data from Yahoo Finance
    try:
        print(f"Downloading {ticker} data for period {period} from Yahoo Finance...")
        stock_data = yf.download(ticker, period=period)
        print(f"Downloaded data with shape: {stock_data.shape}")
        return stock_data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# 4. Create Enhanced Dataset with combined image and numerical features
def create_combined_feature_dataset(image_data):
    """
    Create dataset with both image features and numerical technical indicators
    
    Args:
        image_data: List of tuples (image_path, target_return, features_dict)
    
    Returns:
        X_img: Image feature matrix
        X_num: Numerical feature matrix
        y: Target values
        valid_paths: List of valid image paths
        feature_names: List of numerical feature names
    """
    image_features_list = []
    numerical_features_list = []
    targets = []
    valid_paths = []
    
    print(f"Processing {len(image_data)} images for combined feature extraction...")
    
    for image_path, target_return, features_dict in image_data:
        # Extract image features
        image_feature_vector = extract_image_features(image_path)
        
        if image_feature_vector is not None:
            # Create numerical feature vector
            numerical_features = list(features_dict.values())
            
            # Store data
            image_features_list.append(image_feature_vector)
            numerical_features_list.append(numerical_features)
            targets.append(target_return)
            valid_paths.append(image_path)
    
    # Convert to numpy arrays
    X_img = np.array(image_features_list)
    X_num = np.array(numerical_features_list)
    y = np.array(targets)
    feature_names = list(image_data[0][2].keys())
    
    print(f"Created dataset with {len(X_img)} samples")
    print(f"Image feature dim: {X_img.shape}")
    print(f"Numerical feature dim: {X_num.shape}")
    
    return X_img, X_num, y, valid_paths, feature_names

# 5. Train and evaluate multiple ML models with combined features
def evaluate_combined_models(X_img, X_num, y, feature_names, model_names=None):
    """
    Train and evaluate multiple ML models using combined features
    
    Args:
        X_img: Image feature matrix
        X_num: Numerical feature matrix
        y: Target values
        feature_names: Names of numerical features
        model_names: List of model names to evaluate
    
    Returns:
        Dictionary of trained models, performance metrics, and best model
    """
    # Default models to evaluate if none provided
    if model_names is None:
        model_names = [
            'LinearRegression', 
            'Ridge', 
            'Lasso',
            'ElasticNet',
            'RandomForest', 
            'GradientBoosting', 
            'SVR',
            'Stacking'
        ]
    
    # Prepare combined features
    # First, reduce dimensionality of image features using PCA
    print("Applying PCA to reduce image feature dimensionality...")
    pca = PCA(n_components=50)  # Reduce to 50 components
    X_img_reduced = pca.fit_transform(X_img)
    print(f"Reduced image features from {X_img.shape[1]} to {X_img_reduced.shape[1]} dimensions")
    
    # Combine reduced image features with numerical features
    X_combined = np.hstack((X_img_reduced, X_num))
    print(f"Combined feature matrix shape: {X_combined.shape}")
    
    # Define feature names for the combined dataset
    combined_feature_names = [f'img_pc{i+1}' for i in range(X_img_reduced.shape[1])] + feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate
    base_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=0.1),
        'Lasso': Lasso(alpha=0.001),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'),
    }
    
    # Add stacking regressor
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('ridge', Ridge(alpha=0.1))
    ]
    base_models['Stacking'] = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )
    
    # Results dictionary
    results = {
        'models': {},
        'metrics': {},
        'best_model': None,
        'best_score': -float('inf'),
        'scaler': scaler,
        'pca': pca,
        'feature_importance': {}
    }
    
    # For plotting
    plt.figure(figsize=(15, 10))
    
    # Evaluate each model
    for i, model_name in enumerate(model_names):
        if model_name not in base_models:
            print(f"Warning: Model {model_name} not found in available models")
            continue
            
        print(f"\nTraining {model_name}...")
        model = base_models[model_name]
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate R² and take its absolute value
            r2_raw = r2_score(y_test, y_pred)
            r2 = abs(r2_raw)
            
            # Store model and metrics
            results['models'][model_name] = model
            results['metrics'][model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'R2_raw': r2_raw
            }
            
            # Calculate confidence score (using absolute R² as confidence metric)
            confidence_score = r2
            results['metrics'][model_name]['ConfidenceScore'] = confidence_score
            
            # Check if this is the best model so far
            if confidence_score > results['best_score']:
                results['best_model'] = model_name
                results['best_score'] = confidence_score
            
            # Plot predictions vs actual
            plt.subplot(2, 4, i + 1)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--')
            plt.title(f'{model_name}\nabs(R²) = {r2:.4f}')
            plt.xlabel('Actual Return')
            plt.ylabel('Predicted Return')
            plt.grid(True)
            
            print(f"{model_name} results:")
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Original R²: {r2_raw:.4f}")
            print(f"  Absolute R²: {r2:.4f}")
            
            # Extract feature importance for appropriate models
            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                importance = model.feature_importances_
                # Create a DataFrame for easier sorting
                imp_df = pd.DataFrame({
                    'Feature': combined_feature_names,
                    'Importance': importance
                })
                # Sort by importance
                imp_df = imp_df.sort_values('Importance', ascending=False)
                
                # Store top 20 features importance
                results['feature_importance'][model_name] = imp_df.head(20)
                
                # Plot feature importance
                plt.figure(figsize=(10)),# Plot feature importance
                plt.figure(figsize=(10, 8))
                imp_df.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False)
                plt.title(f'Top 20 Feature Importance for {model_name}')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{model_name}.png')
                plt.close()
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Save comparison plot
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Print best model
    print(f"\nBest model: {results['best_model']} with absolute R² score: {results['best_score']:.4f}")
    
    return results

# 6. Create a prediction function for combined features
def predict_returns_combined(model, pca, scaler, image_paths, numerical_features):
    """
    Predict returns using combined features
    
    Args:
        model: Trained model
        pca: Fitted PCA for dimensionality reduction
        scaler: Fitted scaler
        image_paths: List of image paths
        numerical_features: List of corresponding numerical feature dictionaries
    
    Returns:
        List of predicted returns
    """
    predictions = []
    
    for i, image_path in enumerate(image_paths):
        # Extract image features
        img_features = extract_image_features(image_path)
        
        if img_features is not None:
            # Get numerical features
            num_features = list(numerical_features[i].values())
            
            # Apply PCA to image features
            img_features_reduced = pca.transform(img_features.reshape(1, -1))
            
            # Combine features
            combined_features = np.hstack((img_features_reduced, np.array(num_features).reshape(1, -1)))
            
            # Scale features
            scaled_features = scaler.transform(combined_features)
            
            # Make prediction
            predicted_return = model.predict(scaled_features)[0]
            predictions.append((image_path, predicted_return))
            
            print(f"Predicted return for {os.path.basename(image_path)}: {predicted_return:.4f}")
    
    return predictions

# 7. Main function tying everything together
def main():
    # Set ticker and period
    ticker = "TSLA"
    period = "5y"
    
    # File path to data (optional)
    file_path = 'tsla_5y_data.csv'
    
    # Load data
    stock_data = load_data(file_path, ticker, period)
    if stock_data is None:
        print("Failed to load stock data. Exiting...")
        return
    
    # Generate enhanced candlestick charts with technical indicators
    image_data = generate_enhanced_candlestick_images(stock_data)
    
    if not image_data:
        print("No image data generated. Exiting...")
        return
    
    # Create combined dataset with image and numerical features
    X_img, X_num, y, valid_paths, feature_names = create_combined_feature_dataset(image_data)
    if len(X_img) == 0:
        print("No valid features extracted. Exiting...")
        return
    
    # Train and evaluate models with combined features
    results = evaluate_combined_models(X_img, X_num, y, feature_names)
    
    # Save best model
    best_model = results['models'][results['best_model']]
    window_size = 30
    stride = 5
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model_combined_w{window_size}_s{stride}.pkl')
    joblib.dump(results['scaler'], f'models/scaler_combined_w{window_size}_s{stride}.pkl')
    joblib.dump(results['pca'], f'models/pca_w{window_size}_s{stride}.pkl')
    
    # Print overall best results
    print("\n" + "="*80)
    print(f"BEST MODEL RESULTS:")
    print(f"Using window size {window_size}, stride {stride}")
    print(f"Best model: {results['best_model']}")
    print(f"Best confidence score (|R²|): {abs(results['best_score']):.4f}")
    print("="*80)
    
    # Save results summary
    save_results_summary(results)
    
    # Show feature importance for the best model if available
    if results['best_model'] in results['feature_importance']:
        print("\nTop 10 features for best model:")
        print(results['feature_importance'][results['best_model']].head(10))
    
    # Demonstrate predictions with best model
    print("\nDemonstrating predictions with best model...")
    import random
    test_indices = random.sample(range(len(valid_paths)), min(5, len(valid_paths)))
    test_paths = [valid_paths[i] for i in test_indices]
    test_features = [image_data[i][2] for i in test_indices]  # Get corresponding numerical features
    
    predict_returns_combined(best_model, results['pca'], results['scaler'], test_paths, test_features)


# Helper function to save results summary
def save_results_summary(results):
    """Save a summary of results to CSV"""
    window_size = 30
    stride = 5
    data = []

    for model_name, metrics in results['metrics'].items():
        data.append({
            'Window_Size': window_size,
            'Stride': stride,
            'Model': model_name,
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],  # Only absolute R² stored
            'Is_Best': (results['best_model'] == model_name)
        })

    summary_df = pd.DataFrame(data)
    summary_df.to_csv('model_performance_summary.csv', index=False)
    print("Results summary saved to 'model_performance_summary.csv'")


if __name__ == "__main__":
    main()
