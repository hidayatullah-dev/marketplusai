import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import time
import random

# Dictionary of popular tickers
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
    "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "DIS", "NFLX", "INTC"
]

POPULAR_FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", 
    "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "USDINR=X", "USDBRL=X"
]

POPULAR_COMMODITIES = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "CL=F",  # Crude Oil
    "NG=F",  # Natural Gas
    "HG=F",  # Copper
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F",  # Soybeans
    "KC=F",  # Coffee
    "CT=F"   # Cotton
]

# Cache to store fetched data and reduce API calls
data_cache = {}
cache_expiry = 3600  # Cache expiry in seconds (1 hour)

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first
    current_time = time.time()
    if cache_key in data_cache and current_time - data_cache[cache_key]['timestamp'] < cache_expiry:
        print(f"Using cached data for {ticker}")
        return data_cache[cache_key]['data']
    
    # Add retry logic for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {ticker} from {start_date} to {end_date}")
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Check if data is empty
            if ticker_data.empty:
                print(f"Warning: No data returned for {ticker}")
                # For demo purposes, generate sample data if API fails
                return generate_sample_data(ticker, start_date, end_date)
            
            # Cache the result
            data_cache[cache_key] = {
                'data': ticker_data,
                'timestamp': current_time
            }
            
            return ticker_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                # Add a small delay before retrying
                time.sleep(2 * (attempt + 1))
            else:
                print(f"All attempts failed for {ticker}, generating sample data")
                # Return sample data on failure
                return generate_sample_data(ticker, start_date, end_date)

def fetch_forex_data(ticker, start_date, end_date):
    """Fetch historical forex data from Yahoo Finance"""
    return fetch_stock_data(ticker, start_date, end_date)

def fetch_commodity_data(ticker, start_date, end_date):
    """Fetch historical commodity data from Yahoo Finance"""
    return fetch_stock_data(ticker, start_date, end_date)

def get_available_tickers():
    """Return a list of popular stock tickers"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 
        'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 
        'PG', 'UNH', 'HD', 'BAC', 'MA',
        'DIS', 'NVDA', 'PYPL', 'ADBE', 'CRM'
    ]

def get_available_forex():
    """Return a list of popular forex pairs"""
    return [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 
        'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'EURJPY=X', 
        'EURGBP=X', 'EURCHF=X'
    ]

def get_available_commodities():
    """Return a list of popular commodities"""
    return [
        'GC=F',    # Gold
        'SI=F',    # Silver
        'CL=F',    # Crude Oil
        'HG=F',    # Copper
        'NG=F',    # Natural Gas
        'ZC=F',    # Corn
        'ZW=F',    # Wheat
        'ZS=F',    # Soybean
        'KC=F',    # Coffee
        'CT=F'     # Cotton
    ]

def generate_sample_data(ticker, start_date, end_date):
    """Generate sample data for demonstration when API fails"""
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Set base price based on ticker
    if ticker == 'AAPL':
        base_price = 150.0
    elif ticker == 'MSFT':
        base_price = 280.0
    elif ticker == 'GOOGL':
        base_price = 2800.0
    elif ticker.startswith('GC'):  # Gold
        base_price = 1800.0
    elif ticker.startswith('SI'):  # Silver
        base_price = 25.0
    elif ticker.startswith('CL'):  # Oil
        base_price = 75.0
    elif ticker.startswith('EUR'):  # Euro forex
        base_price = 1.1
    else:
        base_price = 100.0
    
    # Set random seed based on ticker for consistent results
    random.seed(hash(ticker) % 10000)
    
    # Generate prices with trend and volatility
    n_days = len(date_range)
    trend = random.uniform(-0.0001, 0.0002)  # Slight random trend
    
    prices = []
    current_price = base_price
    
    for i in range(n_days):
        # Add trend and random movement
        daily_return = trend + random.normalvariate(0, 0.015)  # 1.5% daily volatility
        current_price *= (1 + daily_return)
        prices.append(current_price)
    
    # Create DataFrame
    df = pd.DataFrame(index=date_range)
    df['Open'] = [p * (1 - random.uniform(0, 0.01)) for p in prices]
    df['High'] = [p * (1 + random.uniform(0, 0.015)) for p in prices]
    df['Low'] = [p * (1 - random.uniform(0, 0.015)) for p in prices]
    df['Close'] = prices
    df['Volume'] = [int(random.uniform(1000000, 5000000)) for _ in range(n_days)]
    
    print(f"Generated sample data for {ticker} with {len(df)} rows")
    return df

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_ticker_info(ticker):
    """
    Get information about a ticker from Yahoo Finance
    
    Args:
        ticker (str): Ticker symbol
        
    Returns:
        dict: Dictionary with ticker information
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Extract relevant information
        relevant_info = {}
        
        if 'shortName' in info:
            relevant_info['name'] = info['shortName']
        elif 'longName' in info:
            relevant_info['name'] = info['longName']
        else:
            relevant_info['name'] = ticker
            
        if 'sector' in info:
            relevant_info['sector'] = info['sector']
        
        if 'industry' in info:
            relevant_info['industry'] = info['industry']
            
        if 'marketCap' in info:
            relevant_info['market_cap'] = info['marketCap']
            
        if 'forwardPE' in info:
            relevant_info['forward_pe'] = info['forwardPE']
            
        if 'dividendYield' in info and info['dividendYield'] is not None:
            relevant_info['dividend_yield'] = info['dividendYield'] * 100  # Convert to percentage
            
        return relevant_info
    except Exception as e:
        # Return a basic info dictionary if there's an error
        return {'name': ticker, 'error': str(e)}
