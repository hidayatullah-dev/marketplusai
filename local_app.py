import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Simple prediction function (same as in app.py)
def simple_predict(data, days=7):
    """Simple linear prediction function"""
    # Check that we have sufficient data
    if len(data) < 5:
        raise ValueError("Insufficient data for prediction")
    
    # Get the last 30 days (or less if not available)
    window = min(30, len(data))
    close_values = data['Close'].dropna().values[-window:]
    
    # Ensure we have data to work with
    if len(close_values) < 5:
        raise ValueError("Insufficient non-NaN data for prediction")
    
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
    if pd.isna(std_dev) or std_dev == 0:
        # Use a default small volatility if we can't calculate it
        std_dev = 0.01
    
    volatility = std_dev * np.sqrt(np.arange(1, days + 1))
    pred_df['Lower'] = pred_df['Close'] * (1 - 1.96 * volatility)
    pred_df['Upper'] = pred_df['Close'] * (1 + 1.96 * volatility)
    
    return pred_df

# Data fetching functions
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        return df.dropna()
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

# List of popular tickers
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"
]

# Main application class
class FinancialForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Market Predictor - Local Version")
        self.root.geometry("1200x800")
        
        # Create frame for controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Ticker selection
        ttk.Label(control_frame, text="Select Stock:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_var = tk.StringVar(value="AAPL")
        self.ticker_combo = ttk.Combobox(control_frame, textvariable=self.ticker_var, values=POPULAR_STOCKS)
        self.ticker_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Time period selection
        ttk.Label(control_frame, text="Time Period:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.period_var = tk.StringVar(value="1 Year")
        period_options = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"]
        period_combo = ttk.Combobox(control_frame, textvariable=self.period_var, values=period_options)
        period_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Prediction days
        ttk.Label(control_frame, text="Prediction Days:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.days_var = tk.StringVar(value="7")
        days_entry = ttk.Entry(control_frame, textvariable=self.days_var, width=5)
        days_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        # Fetch button
        self.fetch_button = ttk.Button(control_frame, text="Fetch Data", command=self.fetch_and_plot)
        self.fetch_button.grid(row=0, column=6, padx=20, pady=5)
        
        # Technical indicators
        ttk.Label(control_frame, text="Technical Indicators:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.show_sma = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="SMA", variable=self.show_sma).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.show_ema = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="EMA", variable=self.show_ema).grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        self.show_bollinger = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Bollinger Bands", variable=self.show_bollinger).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Create tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Historical tab
        self.historical_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.historical_frame, text="Historical Data")
        
        # Prediction tab
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Predictions")
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_var.set(f"Ready. Last updated: {current_time}")
        
        # Data holders
        self.df = None
        self.prediction_data = None
        
        # Create initial plots
        self.create_initial_plots()
    
    def create_initial_plots(self):
        # Historical plot
        self.historical_fig, self.historical_ax = plt.subplots(figsize=(12, 6))
        self.historical_canvas = FigureCanvasTkAgg(self.historical_fig, master=self.historical_frame)
        self.historical_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Prediction plot
        self.prediction_fig, self.prediction_ax = plt.subplots(figsize=(12, 6))
        self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, master=self.prediction_frame)
        self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def fetch_and_plot(self):
        ticker = self.ticker_var.get()
        period = self.period_var.get()
        prediction_days = int(self.days_var.get())
        
        # Convert period to days
        period_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730
        }
        days = period_map.get(period, 365)
        
        # Set status
        self.status_var.set(f"Fetching data for {ticker}...")
        self.root.update_idletasks()
        
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        self.df = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if self.df.empty:
            self.status_var.set(f"No data available for {ticker}")
            return
        
        # Generate predictions
        try:
            self.prediction_data = simple_predict(self.df, days=prediction_days)
            
            # Plot historical data
            self.plot_historical_data()
            
            # Plot predictions
            self.plot_predictions()
            
            # Update status
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.status_var.set(f"Data updated. Last fetch: {current_time}")
            
        except Exception as e:
            self.status_var.set(f"Error generating predictions: {str(e)}")
    
    def plot_historical_data(self):
        if self.df is None or self.df.empty:
            return
        
        # Clear previous plot
        self.historical_ax.clear()
        
        # Plot price data
        self.historical_ax.plot(self.df.index, self.df['Close'], label='Close Price')
        
        # Add SMA if selected
        if self.show_sma.get():
            sma20 = self.df['Close'].rolling(window=20).mean()
            sma50 = self.df['Close'].rolling(window=50).mean()
            self.historical_ax.plot(self.df.index, sma20, label='SMA 20', linestyle='--')
            self.historical_ax.plot(self.df.index, sma50, label='SMA 50', linestyle='-.')
        
        # Add EMA if selected
        if self.show_ema.get():
            ema20 = self.df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = self.df['Close'].ewm(span=50, adjust=False).mean()
            self.historical_ax.plot(self.df.index, ema20, label='EMA 20', linestyle='--')
            self.historical_ax.plot(self.df.index, ema50, label='EMA 50', linestyle='-.')
        
        # Add Bollinger Bands if selected
        if self.show_bollinger.get():
            sma20 = self.df['Close'].rolling(window=20).mean()
            std20 = self.df['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            self.historical_ax.plot(self.df.index, upper_band, label='Upper BB', linestyle=':')
            self.historical_ax.plot(self.df.index, lower_band, label='Lower BB', linestyle=':')
            self.historical_ax.fill_between(self.df.index, lower_band, upper_band, alpha=0.1)
        
        # Set labels and title
        ticker = self.ticker_var.get()
        period = self.period_var.get()
        self.historical_ax.set_title(f"{ticker} - {period} Historical Data")
        self.historical_ax.set_xlabel("Date")
        self.historical_ax.set_ylabel("Price")
        self.historical_ax.legend()
        self.historical_ax.grid(True)
        
        # Format dates on x-axis
        self.historical_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.historical_fig.autofmt_xdate()
        
        # Update canvas
        self.historical_canvas.draw()
    
    def plot_predictions(self):
        if self.df is None or self.df.empty or self.prediction_data is None or self.prediction_data.empty:
            return
        
        # Clear previous plot
        self.prediction_ax.clear()
        
        # Plot historical data
        self.prediction_ax.plot(self.df.index, self.df['Close'], label='Historical', color='blue')
        
        # Plot prediction
        self.prediction_ax.plot(self.prediction_data.index, self.prediction_data['Close'], label='Prediction', color='red')
        
        # Plot confidence interval
        self.prediction_ax.fill_between(
            self.prediction_data.index,
            self.prediction_data['Lower'],
            self.prediction_data['Upper'],
            alpha=0.2,
            color='red',
            label='95% Confidence'
        )
        
        # Set labels and title
        ticker = self.ticker_var.get()
        days = self.days_var.get()
        self.prediction_ax.set_title(f"{ticker} - {days} Day Prediction")
        self.prediction_ax.set_xlabel("Date")
        self.prediction_ax.set_ylabel("Price")
        self.prediction_ax.legend()
        self.prediction_ax.grid(True)
        
        # Format dates on x-axis
        self.prediction_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.prediction_fig.autofmt_xdate()
        
        # Update canvas
        self.prediction_canvas.draw()

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialForecastApp(root)
    root.mainloop() 