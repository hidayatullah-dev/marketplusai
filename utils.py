import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'current_data' not in st.session_state:
        st.session_state.current_data = pd.DataFrame()
        
    if 'data_timestamp' not in st.session_state:
        st.session_state.data_timestamp = time.time()
        
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
        
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = pd.DataFrame()

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if pd.isna(num):
        return "N/A"
    
    abs_num = abs(num)
    sign = -1 if num < 0 else 1
    
    if abs_num < 1000:
        return f"{num:.2f}"
    elif abs_num < 1000000:
        return f"{sign * abs_num / 1000:.2f}K"
    elif abs_num < 1000000000:
        return f"{sign * abs_num / 1000000:.2f}M"
    else:
        return f"{sign * abs_num / 1000000000:.2f}B"

def calculate_returns(price_data, period='daily'):
    """
    Calculate returns for the given price data
    
    Args:
        price_data (pd.DataFrame): DataFrame with 'Close' price data
        period (str): Period for returns calculation ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        pd.Series: Series with returns
    """
    if period == 'daily':
        returns = price_data['Close'].pct_change()
    elif period == 'weekly':
        returns = price_data['Close'].resample('W').last().pct_change()
    elif period == 'monthly':
        returns = price_data['Close'].resample('M').last().pct_change()
    elif period == 'yearly':
        returns = price_data['Close'].resample('Y').last().pct_change()
    else:
        raise ValueError(f"Invalid period: {period}")
    
    return returns.dropna()

def calculate_volatility(returns, window=21):
    """
    Calculate rolling volatility from returns
    
    Args:
        returns (pd.Series): Series with returns
        window (int): Window size for rolling volatility
        
    Returns:
        pd.Series: Series with volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    """
    Calculate the Sharpe ratio
    
    Args:
        returns (pd.Series): Series with returns
        risk_free_rate (float): Annualized risk-free rate
        periods (int): Number of periods in a year
        
    Returns:
        float: Sharpe ratio
    """
    # Daily risk-free rate
    rf_daily = (1 + risk_free_rate) ** (1 / periods) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_daily
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(periods) * excess_returns.mean() / returns.std()
    
    return sharpe_ratio

def calculate_drawdowns(price_data):
    """
    Calculate drawdowns from price data
    
    Args:
        price_data (pd.DataFrame): DataFrame with 'Close' price data
        
    Returns:
        pd.Series: Series with drawdowns
    """
    # Calculate the running maximum
    running_max = price_data['Close'].cummax()
    
    # Calculate drawdowns
    drawdowns = (price_data['Close'] / running_max) - 1
    
    return drawdowns

def calculate_technical_indicators(price_data):
    """
    Calculate common technical indicators
    
    Args:
        price_data (pd.DataFrame): DataFrame with OHLC price data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    df = price_data.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df
