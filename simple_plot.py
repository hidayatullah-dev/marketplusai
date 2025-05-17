import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import sys

def fetch_data(ticker="AAPL", days=365):
    """Fetch stock data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if df.empty:
            print(f"No data found for ticker {ticker}")
            return None
            
        print(f"Fetched {len(df)} data points")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def simple_predict(data, days=7):
    """Simple linear prediction function"""
    # Check that we have sufficient data
    if len(data) < 5:
        print("Insufficient data for prediction")
        return None
    
    # Get the last 30 days (or less if not available)
    window = min(30, len(data))
    close_values = data['Close'].dropna().values[-window:]
    
    # Ensure we have data to work with
    if len(close_values) < 5:
        print("Insufficient non-NaN data for prediction")
        return None
    
    x = np.arange(len(close_values))
    
    # Fit a simple linear regression
    z = np.polyfit(x, close_values, 1)
    slope = z[0]
    intercept = z[1]
    
    # Predict future values
    future_x = np.arange(len(close_values), len(close_values) + days)
    future_y = slope * future_x + intercept
    
    # Create prediction dataframe
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    pred_df = pd.DataFrame(index=future_dates)
    pred_df['Close'] = future_y
    
    # Add confidence intervals (simple approach)
    std_dev = data['Close'].dropna().pct_change().dropna().std()
    # Handle the case where std_dev might be a Series
    if isinstance(std_dev, pd.Series):
        std_dev = std_dev.iloc[0] if not std_dev.empty else 0.01
    if pd.isna(std_dev) or std_dev == 0:
        # Use a default small volatility if we can't calculate it
        std_dev = 0.01
    
    volatility = std_dev * np.sqrt(np.arange(1, days + 1))
    pred_df['Lower'] = pred_df['Close'] * (1 - 1.96 * volatility)
    pred_df['Upper'] = pred_df['Close'] * (1 + 1.96 * volatility)
    
    return pred_df

def plot_data(data, prediction_data=None, ticker="AAPL", show_sma=True, show_ema=False, show_bollinger=False):
    """Plot the stock data and predictions"""
    if data is None:
        return
    
    # Create figure with 2 subplots if prediction data is available, otherwise 1
    if prediction_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    # Add technical indicators if requested
    if show_sma:
        sma20 = data['Close'].rolling(window=20).mean()
        sma50 = data['Close'].rolling(window=50).mean()
        ax1.plot(data.index, sma20, label='SMA 20', linestyle='--', color='orange')
        ax1.plot(data.index, sma50, label='SMA 50', linestyle='-.', color='green')
    
    if show_ema:
        ema20 = data['Close'].ewm(span=20, adjust=False).mean()
        ema50 = data['Close'].ewm(span=50, adjust=False).mean()
        ax1.plot(data.index, ema20, label='EMA 20', linestyle='--', color='purple')
        ax1.plot(data.index, ema50, label='EMA 50', linestyle='-.', color='brown')
    
    if show_bollinger:
        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        ax1.plot(data.index, upper_band, label='Upper BB', linestyle=':', color='gray')
        ax1.plot(data.index, lower_band, label='Lower BB', linestyle=':', color='gray')
        ax1.fill_between(data.index, lower_band, upper_band, alpha=0.1, color='gray')
    
    # Add labels and title
    ax1.set_title(f"{ticker} - Historical Data")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    
    # Plot prediction data if available
    if prediction_data is not None:
        # Last 30 days of historical + prediction
        last_30_days = data.tail(30)
        
        ax2.plot(last_30_days.index, last_30_days['Close'], label='Historical', color='blue')
        ax2.plot(prediction_data.index, prediction_data['Close'], label='Prediction', color='red')
        
        # Confidence interval
        ax2.fill_between(
            prediction_data.index,
            prediction_data['Lower'],
            prediction_data['Upper'],
            alpha=0.2,
            color='red',
            label='95% Confidence'
        )
        
        # Add labels and title
        ax2.set_title(f"{ticker} - Price Prediction ({len(prediction_data)} days)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Default values
    ticker = "AAPL"
    days = 365
    prediction_days = 7
    show_sma = True
    show_ema = False
    show_bollinger = False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            days = int(sys.argv[2])
        except:
            print(f"Invalid number of days: {sys.argv[2]}. Using default: 365")
    if len(sys.argv) > 3:
        try:
            prediction_days = int(sys.argv[3])
        except:
            print(f"Invalid prediction days: {sys.argv[3]}. Using default: 7")
    
    # Fetch data
    data = fetch_data(ticker, days)
    
    if data is None:
        print("No data to plot. Exiting.")
        return
    
    # Generate predictions
    prediction_data = simple_predict(data, prediction_days)
    
    # Plot data
    plot_data(data, prediction_data, ticker, show_sma, show_ema, show_bollinger)

if __name__ == "__main__":
    print("Simple Stock Plotter - Command Line Version")
    print("Usage: python simple_plot.py [ticker] [days] [prediction_days]")
    print("Example: python simple_plot.py AAPL 365 7")
    print("-" * 50)
    main() 