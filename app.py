import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sqlite3
import json
import io
from email_service import email_service
from prediction_models import get_prediction_model

# Import custom modules
from data_fetcher import (
    fetch_stock_data, 
    fetch_forex_data, 
    fetch_commodity_data, 
    get_available_tickers,
    get_available_forex,
    get_available_commodities
)
from utils import initialize_session_state, format_large_number

# Database initialization
def init_db():
    """Initialize SQLite database for contacts and subscriptions"""
    conn = sqlite3.connect('finance_app.db')
    c = conn.cursor()
    
    # Create contacts table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create subscriptions table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        plan_type TEXT NOT NULL,
        plan_price REAL NOT NULL,
        payment_method TEXT NOT NULL,
        active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# Email functionality
def send_email(recipient, subject, body):
    """Send email using SMTP"""
    try:
        # Email configuration
        sender_email = "your-email@gmail.com"  # Replace with your email
        sender_password = "your-app-password"  # Replace with your app password
        
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient
        message["Subject"] = subject
        
        # Add body to email
        message.attach(MIMEText(body, "html"))
        
        # Connect to SMTP server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# Contact form handler
def handle_contact_form(name, email, message):
    """Store contact form submissions and send confirmation emails"""
    try:
        # Store in database
        conn = sqlite3.connect('finance_app.db')
        c = conn.cursor()
        c.execute("INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)",
                 (name, email, message))
        conn.commit()
        conn.close()
        
        # Send confirmation email to user
        user_subject = "We received your message - MarketPulse AI"
        user_body = f"""
        <html>
        <body>
            <h2>Thank you for contacting us!</h2>
            <p>Hello {name},</p>
            <p>We have received your message and will get back to you as soon as possible.</p>
            <p>Your message:</p>
            <blockquote>{message}</blockquote>
            <p>Best regards,<br>MarketPulse AI Team</p>
        </body>
        </html>
        """
        send_email(email, user_subject, user_body)
        
        # Forward message to admin
        admin_email = "hidayatullah2269@gmail.com"  # Admin email
        admin_subject = f"New Contact Form Submission from {name}"
        admin_body = f"""
        <html>
        <body>
            <h2>New Contact Form Submission</h2>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Email:</strong> {email}</p>
            <p><strong>Message:</strong></p>
            <blockquote>{message}</blockquote>
        </body>
        </html>
        """
        send_email(admin_email, admin_subject, admin_body)
        
        return True
    except Exception as e:
        st.error(f"Error processing contact form: {str(e)}")
        return False

# Simple prediction function 
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

# Advanced prediction with ARIMA
def arima_predict(data, days=7):
    """ARIMA model for time series prediction"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Check stationarity
        close_values = data['Close'].dropna()
        
        # Use appropriate differencing if needed
        differenced = False
        if adfuller(close_values)[1] > 0.05:
            # Non-stationary, use differencing
            close_values = close_values.diff().dropna()
            differenced = True
            
        # Fit ARIMA model
        model = ARIMA(close_values, order=(5,1,0))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=days)
        
        # Create prediction dataframe
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        pred_df = pd.DataFrame(index=future_dates)
        
        # Undo differencing if applied
        if differenced:
            last_value = data['Close'].iloc[-1]
            cumulative_sum = np.cumsum(forecast)
            pred_values = last_value + cumulative_sum
            pred_df['Close'] = pred_values
        else:
            pred_df['Close'] = forecast
        
        # Add confidence intervals
        pred_df['Lower'] = pred_df['Close'] * 0.95
        pred_df['Upper'] = pred_df['Close'] * 1.05
        
        return pred_df
    except ImportError:
        st.warning("statsmodels package not available. Using simple prediction instead.")
        return simple_predict(data, days)

# Machine learning prediction
def ml_predict(data, days=7):
    """Machine learning based prediction using XGBoost"""
    try:
        import xgboost as xgb
        from sklearn.preprocessing import MinMaxScaler
        
        # Feature engineering
        df = data.copy()
        
        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Daily_Return'] = df['Close'].pct_change()
        df['Price_Volatility'] = df['Close'].rolling(window=10).std()
        
        # Create lagged features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        features = ['SMA_5', 'SMA_20', 'RSI', 'Daily_Return', 'Price_Volatility'] + [f'Close_Lag_{i}' for i in range(1, 6)]
        X = df[features].values
        y = df['Close'].values
        
        # Scale features
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_scaled, y_scaled)
        
        # Prepare prediction features
        last_row = df.iloc[-1:].copy()
        prediction_rows = []
        
        for i in range(days):
            # Create new row with shifted features
            new_row = last_row.copy()
            
            # Make prediction
            pred_features = new_row[features].values
            pred_features_scaled = scaler_X.transform(pred_features)
            pred_scaled = model.predict(pred_features_scaled)[0]
            prediction = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            # Update last row for next prediction
            new_row['Close'] = prediction
            new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
            
            # Update technical indicators
            if i > 0:
                historical_close = list(df['Close'].values) + [r['Close'] for r in prediction_rows]
                new_row['SMA_5'] = np.mean(historical_close[-5:])
                new_row['SMA_20'] = np.mean(historical_close[-20:])
                
                # Update lagged values
                for j in range(1, 6):
                    if j == 1:
                        new_row[f'Close_Lag_{j}'] = last_row['Close'].values[0]
                    elif j <= i+1:
                        new_row[f'Close_Lag_{j}'] = prediction_rows[i-j+1]['Close']
            
            prediction_rows.append(new_row)
            last_row = new_row
        
        # Create prediction dataframe
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        pred_df = pd.DataFrame(index=future_dates)
        pred_df['Close'] = [row['Close'] for row in prediction_rows]
        
        # Add confidence intervals
        std_dev = data['Close'].pct_change().std()
        confidence_interval = 1.96 * std_dev * np.sqrt(np.arange(1, days + 1))
        
        pred_df['Lower'] = pred_df['Close'] * (1 - confidence_interval)
        pred_df['Upper'] = pred_df['Close'] * (1 + confidence_interval)
        
        return pred_df
    except ImportError:
        st.warning("XGBoost or scikit-learn not available. Using simple prediction instead.")
        return simple_predict(data, days)

# Calculate RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    """Calculate RSI technical indicator"""
    deltas = prices.diff()
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

# API endpoint for contact form
def api_contact_form():
    """API endpoint for handling contact form submissions"""
    st.markdown("""
    <script>
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(contactForm);
            
            // Show loading state
            const submitButton = contactForm.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            submitButton.innerHTML = '<span class="loading-spinner"></span> Sending...';
            submitButton.disabled = true;
            
            // Make AJAX request to submit form
            fetch('/api/contact', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Hide form and show success message
                    contactForm.style.display = 'none';
                    const successMessage = document.querySelector('.success-message');
                    if (successMessage) {
                        successMessage.style.display = 'block';
                        
                        // Add animation classes to success message elements
                        const checkmark = successMessage.querySelector('.success-checkmark');
                        if (checkmark) {
                            checkmark.classList.add('animate-fade-in');
                        }
                    }
                } else {
                    // Show error
                    alert('Error: ' + (data.error || 'Unknown error occurred'));
                    submitButton.innerHTML = originalText;
                    submitButton.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error submitting form:', error);
                alert('An error occurred. Please try again later.');
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
            });
        });
    }
    </script>
    """, unsafe_allow_html=True)

# Process subscription
def process_subscription(user_email, plan_type, plan_price, payment_method):
    """Process user subscription and store in database"""
    try:
        conn = sqlite3.connect('finance_app.db')
        c = conn.cursor()
        
        # Check if user already has an active subscription
        c.execute("SELECT * FROM subscriptions WHERE user_email = ? AND active = 1", (user_email,))
        existing = c.fetchone()
        
        # Calculate expiration date based on plan
        now = datetime.now()
        if plan_type == "monthly":
            expires_at = now + timedelta(days=30)
        elif plan_type == "yearly":
            expires_at = now + timedelta(days=365)
        else:  # lifetime
            expires_at = now + timedelta(days=36500)  # ~100 years
        
        if existing:
            # Update existing subscription
            c.execute("""
            UPDATE subscriptions 
            SET plan_type = ?, plan_price = ?, payment_method = ?, expires_at = ?
            WHERE user_email = ? AND active = 1
            """, (plan_type, plan_price, payment_method, expires_at, user_email))
        else:
            # Create new subscription
            c.execute("""
            INSERT INTO subscriptions 
            (user_email, plan_type, plan_price, payment_method, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """, (user_email, plan_type, plan_price, payment_method, expires_at))
        
        conn.commit()
        conn.close()
        
        # Send confirmation email
        subject = "Your MarketPulse AI PRO Subscription"
        body = f"""
        <html>
        <body>
            <h2>Thank you for subscribing to MarketPulse AI PRO!</h2>
            <p>Hello,</p>
            <p>Your subscription has been activated successfully.</p>
            <p><strong>Plan:</strong> {plan_type.capitalize()}</p>
            <p><strong>Price:</strong> ${plan_price}</p>
            <p><strong>Valid until:</strong> {expires_at.strftime('%Y-%m-%d')}</p>
            <p>Enjoy your premium features!</p>
            <p>Best regards,<br>MarketPulse AI Team</p>
        </body>
        </html>
        """
        send_email(user_email, subject, body)
        
        return True
    except Exception as e:
        st.error(f"Error processing subscription: {str(e)}")
        return False

# Server configuration for local development
# When deploying to Streamlit Cloud, these will be ignored
import os
if os.environ.get('LOCAL_DEV') == 'true':
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'

# Set page configuration
st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - add a custom banner with new logo
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .logo-text {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        background: linear-gradient(90deg, #9333EA 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .logo-emoji {
        font-size: 2.2rem;
        margin-right: 10px;
    }
    .logo-tagline {
        font-size: 0.9rem;
        color: #CCCCCC;
        margin-left: 15px;
        font-style: italic;
    }
</style>

<div class="main-header">
    <span class="logo-emoji">🔮</span>
    <span class="logo-text">MarketPulse AI</span>
    <span class="logo-tagline">Advanced Financial Intelligence</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state and database
initialize_session_state()
init_db()

# Add API routes for contact form
api_contact_form()

# Sidebar for controls with enhanced styling
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <h1 style="color:#9333EA; font-weight:bold;">🔮 Market Navigator</h1>
        <p style="font-style:italic; font-size:0.9em;">Real-time financial analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a divider
    st.markdown("<hr style='margin:0; padding:0; height:1px; border:none; background-color:#555555;'>", unsafe_allow_html=True)
    
    # Asset type selection with better styling
    st.subheader("🔍 Select Market")
    asset_type = st.radio(
        "",  # Remove label as we use subheader
        ["Stocks", "Forex", "Commodities"],
        index=0,
        help="Choose the type of financial asset to analyze"
    )
    
    # Asset selector based on type
    if asset_type == "Stocks":
        available_assets = get_available_tickers()
        default_asset = "AAPL"
    elif asset_type == "Forex":
        available_assets = get_available_forex()
        default_asset = "EURUSD=X"
    else:  # Commodities
        available_assets = get_available_commodities()
        default_asset = "GC=F"  # Gold
    
    try:
        default_index = available_assets.index(default_asset)
    except:
        default_index = 0
        
    selected_asset = st.selectbox(
        f"Select {asset_type[:-1]}",
        available_assets,
        index=default_index
    )
    
    # Time period selection
    time_period = st.select_slider(
        "Historical Data Period",
        options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
        value="1 Year"
    )
    
    # Convert period to days
    period_map = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }
    days = period_map[time_period]
    
    # Prediction horizon
    prediction_days = st.slider(
        "Prediction Horizon (Days)",
        min_value=1,
        max_value=30,
        value=7
    )
    
    # Prediction model selector
    st.subheader("Prediction Model")
    prediction_model = st.radio(
        "",
        ["Simple", "ARIMA", "Machine Learning"],
        index=0,
        help="Choose the prediction model to use"
    )
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    show_ema = st.checkbox("Exponential Moving Average (EMA)", value=False)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)
    show_rsi = st.checkbox("Relative Strength Index (RSI)", value=False)
    show_macd = st.checkbox("MACD", value=False)
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.data_timestamp = time.time()
        st.rerun()

# Main content - we're not using the default title anymore since we have a custom header
# st.title("MarketPulse AI")

# Current time display with better styling
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div style="background-color:#262730; padding:10px; border-radius:5px; margin-bottom:15px;">
    <h4 style="color:#9333EA; margin:0;">
        <span style="color:#FAFAFA;">Live Data</span> • Last updated: {current_time}
        <span style="float:right; font-size:0.8em; background-color:#9333EA; color:white; padding:3px 8px; border-radius:10px;">REAL-TIME</span>
    </h4>
</div>
""", unsafe_allow_html=True)

# Fetch data based on selection
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

with st.spinner(f"Fetching data for {selected_asset}..."):
    try:
        if asset_type == "Stocks":
            df = fetch_stock_data(selected_asset, start_date, end_date)
            asset_full_name = f"{selected_asset} Stock"
        elif asset_type == "Forex":
            df = fetch_forex_data(selected_asset, start_date, end_date)
            asset_full_name = f"{selected_asset.replace('=X', '')} Exchange Rate"
        else:  # Commodities
            df = fetch_commodity_data(selected_asset, start_date, end_date)
            asset_full_name = f"{selected_asset.replace('=F', '')} Commodity"
        
        # Store data in session state
        st.session_state.current_data = df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Check if data is empty
if df.empty:
    st.warning("No data available for the selected time period.")
    st.stop()

# Add fallback data if API request failed
if 'Close' not in df.columns:
    st.warning("Data API connection issue. Using sample data for demonstration.")
    # Generate sample data
    date_range = pd.date_range(end=datetime.now(), periods=365)
    base_price = 150.0  # Sample starting price
    
    # Generate somewhat realistic price movements
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0005, 0.02, len(date_range))
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    prices = base_price * cumulative_returns
    
    # Create a sample dataframe
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(100000, 10000000, len(date_range))
    }, index=date_range)
    
    # Store data in session state
    st.session_state.current_data = df

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

# Convert pandas Series to Python float to avoid formatting issues
latest_price = df['Close'].iloc[-1]
if isinstance(latest_price, pd.Series):
    latest_price = latest_price.iloc[0]
latest_price = float(latest_price)

prev_price = df['Close'].iloc[-2]
if isinstance(prev_price, pd.Series):
    prev_price = prev_price.iloc[0]
prev_price = float(prev_price)

daily_change = latest_price - prev_price
daily_change_pct = (daily_change / prev_price) * 100

period_start_price = df['Close'].iloc[0]
if isinstance(period_start_price, pd.Series):
    period_start_price = period_start_price.iloc[0]
period_start_price = float(period_start_price)

period_change = latest_price - period_start_price
period_change_pct = (period_change / period_start_price) * 100

# Display metrics
col1.metric(
    "Current Price",
    f"${latest_price:.2f}" if asset_type != "Forex" else f"{latest_price:.4f}",
    f"{daily_change_pct:.2f}% Today"
)

col2.metric(
    f"{time_period} Change",
    f"${period_change:.2f}" if asset_type != "Forex" else f"{period_change:.4f}",
    f"{period_change_pct:.2f}%"
)

# Handle volume safely
volume_display = "N/A"
if 'Volume' in df.columns:
    # Check if volume column has values
    all_null = True
    try:
        all_null_check = df['Volume'].isnull().all()
        if isinstance(all_null_check, bool):
            all_null = all_null_check
        elif isinstance(all_null_check, pd.Series) and len(all_null_check) > 0:
            all_null = all_null_check.iloc[0]
    except:
        all_null = True
    
    if not all_null:
        avg_volume = df['Volume'].mean()
        if isinstance(avg_volume, pd.Series):
            avg_volume = avg_volume.iloc[0]
        avg_volume = float(avg_volume)
        volume_display = format_large_number(avg_volume)

col3.metric("Volume (Avg)", volume_display)

# Calculate volatility safely
try:
    volatility = df['Close'].pct_change().dropna().std() * 100
    if isinstance(volatility, pd.Series):
        volatility = volatility.iloc[0]
    volatility = float(volatility)
except:
    volatility = 0.0
    
col4.metric("Volatility", f"{volatility:.2f}%")

# Create tabs for visualizations
tab1, tab2 = st.tabs(["📈 Historical Data", "🔮 Predictions"])

with tab1:
    st.subheader(f"Historical Price Data for {asset_full_name}")
    
    # Create enhanced chart with more professional colors
    fig = go.Figure()
    
    # Create range slider for zooming
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Add candlestick chart if OHLC data is available
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',  # Green for increasing
            decreasing_line_color='#ef5350',  # Red for decreasing
            increasing_fillcolor='rgba(38, 166, 154, 0.3)',  # Semi-transparent green
            decreasing_fillcolor='rgba(239, 83, 80, 0.3)'    # Semi-transparent red
        ))
    else:
        # Enhanced line chart for close price only
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2196f3', width=2),  # Professional blue
            fill='tozeroy',  # Fill area below the line
            fillcolor='rgba(33, 150, 243, 0.1)'  # Semi-transparent blue
        ))
    
    # Add technical indicators
    if show_sma:
        sma_20 = df['Close'].rolling(window=20).mean()
        sma_50 = df['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma_20,
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma_50,
            mode='lines',
            name='SMA 50',
            line=dict(color='magenta', width=1)
        ))
    
    if show_ema:
        ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_20,
            mode='lines',
            name='EMA 20',
            line=dict(color='yellow', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_50,
            mode='lines',
            name='EMA 50',
            line=dict(color='purple', width=1)
        ))
    
    if show_bollinger:
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=upper_band,
            mode='lines',
            name='Upper Bollinger Band',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=lower_band,
            mode='lines',
            name='Lower Bollinger Band',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.05)'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_full_name} - {time_period} Historical Data",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart if available
    has_volume = False
    if 'Volume' in df.columns:
        # Check if volume column has values
        all_null = True
        try:
            all_null_check = df['Volume'].isnull().all()
            if isinstance(all_null_check, bool):
                all_null = all_null_check
            elif isinstance(all_null_check, pd.Series) and len(all_null_check) > 0:
                all_null = all_null_check.iloc[0]
            has_volume = not all_null
        except:
            has_volume = False
            
    if has_volume:
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(46, 184, 184, 0.5)'
        ))
        fig_volume.update_layout(
            title='Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_dark',
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_volume, use_container_width=True)

with tab2:
    st.subheader(f"Price Predictions for {asset_full_name}")
    
    # Generate predictions
    with st.spinner("Generating predictions..."):
        try:
            # Filter out any NaN values from the data
            clean_df = df.copy()
            clean_df = clean_df.dropna(subset=['Close'])
            
            if len(clean_df) > 5:  # Need at least 5 data points for prediction
                # Choose prediction model based on selection
                if prediction_model == "Simple":
                    prediction_data = simple_predict(clean_df, days=prediction_days)
                elif prediction_model == "ARIMA":
                    prediction_data = arima_predict(clean_df, days=prediction_days)
                else:  # Machine Learning
                    prediction_data = ml_predict(clean_df, days=prediction_days)
                
                # Create enhanced prediction chart
                fig_pred = go.Figure()
                
                # Add a marker at the current price point for better transition
                last_point_x = clean_df.index[-1]
                last_point_y = clean_df['Close'].iloc[-1]
                
                # Historical data with improved styling
                fig_pred.add_trace(go.Scatter(
                    x=clean_df.index,
                    y=clean_df['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#2196f3', width=2.5),  # Professional blue
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Add marker at transition point
                fig_pred.add_trace(go.Scatter(
                    x=[last_point_x],
                    y=[last_point_y],
                    mode='markers',
                    marker=dict(color='white', size=8, line=dict(color='#2196f3', width=2)),
                    name='Current Price',
                    hovertemplate='Current Price<br>%{x}<br>$%{y:.2f}<extra></extra>'
                ))
                
                # Prediction line with improved styling
                fig_pred.add_trace(go.Scatter(
                    x=prediction_data.index,
                    y=prediction_data['Close'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#9333EA', width=2.5, dash='dot'),  # Theme color with dotted line
                    hovertemplate='%{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ))
                
                # Add upper confidence bound
                fig_pred.add_trace(go.Scatter(
                    x=prediction_data.index,
                    y=prediction_data['Upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add lower confidence bound with fill
                fig_pred.add_trace(go.Scatter(
                    x=prediction_data.index,
                    y=prediction_data['Lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(147, 51, 234, 0.3)',  # Matching theme color
                    name='95% Confidence',
                    hovertemplate='%{x}<br>Range: $%{y:.2f} to $' + 
                                  prediction_data['Upper'].apply(lambda x: f'{x:.2f}').to_list() + 
                                  '<extra></extra>'
                ))
                
                # Update layout
                fig_pred.update_layout(
                    title=f"{asset_full_name} - {prediction_days} Day Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction metrics
                st.markdown("### Prediction Details")
                metrics_df = pd.DataFrame({
                    'Metric': ['Prediction Method', 'Confidence Level', 'Prediction Horizon'],
                    'Value': [f'{prediction_model} Model', '95%', f'{prediction_days} days']
                })
                st.dataframe(metrics_df)
                
                # Market insights section
                st.markdown("### 📈 Market Insights & Technical Analysis")
                
                # Calculate additional technical indicators
                if len(clean_df) >= 14:  # Need at least 14 data points for RSI
                    # Calculate RSI
                    delta = clean_df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    latest_rsi = rsi.iloc[-1]
                    
                    # Calculate MACD
                    ema12 = clean_df['Close'].ewm(span=12).mean()
                    ema26 = clean_df['Close'].ewm(span=26).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9).mean()
                    latest_macd = macd.iloc[-1]
                    latest_signal = signal.iloc[-1]
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # RSI indicator with color coding
                    rsi_color = "#26a69a" if latest_rsi < 70 else ("#ef5350" if latest_rsi > 70 else "#26a69a")
                    rsi_text = "Oversold" if latest_rsi < 30 else ("Overbought" if latest_rsi > 70 else "Neutral")
                    col1.metric(
                        "RSI (14)", 
                        f"{latest_rsi:.2f}", 
                        rsi_text,
                        delta_color=("normal" if latest_rsi > 30 and latest_rsi < 70 else "inverse")
                    )
                    
                    # MACD indicator
                    macd_status = "Bullish" if latest_macd > latest_signal else "Bearish"
                    col2.metric(
                        "MACD", 
                        f"{latest_macd:.2f}", 
                        macd_status,
                        delta_color=("normal" if latest_macd > latest_signal else "inverse")
                    )
                    
                    # Trend strength
                    sma20 = clean_df['Close'].rolling(window=20).mean().iloc[-1]
                    sma50 = clean_df['Close'].rolling(window=50).mean().iloc[-1]
                    trend = "Uptrend" if sma20 > sma50 else "Downtrend"
                    col3.metric(
                        "Trend (SMA20/50)", 
                        trend, 
                        f"{((clean_df['Close'].iloc[-1] / clean_df['Close'].iloc[-20]) - 1) * 100:.2f}%",
                        delta_color=("normal" if trend == "Uptrend" else "inverse")
                    )
                    
                    # Add a summary of the prediction
                    st.markdown("#### AI Forecast Summary")
                    
                    # Calculate percentage change from current to predicted
                    current_price = clean_df['Close'].iloc[-1]
                    predicted_end_price = prediction_data['Close'].iloc[-1]
                    percent_change = ((predicted_end_price / current_price) - 1) * 100
                    direction = "increase" if percent_change > 0 else "decrease"
                    
                    summary_text = f"""
                    <div style="background-color:#262730; padding:15px; border-radius:5px; margin:15px 0;">
                        <p>Based on the analysis of historical data and current market conditions, the model predicts 
                        that <strong>{selected_asset}</strong> will likely <span style="color:{'#26a69a' if percent_change > 0 else '#ef5350'}; font-weight:bold;">
                        {direction} by {abs(percent_change):.2f}%</span> over the next {prediction_days} days.</p>
                        
                        <p>The forecast considers recent price movements, volatility patterns, and technical indicators. 
                        The confidence interval represents the range within which the price is likely to move with 95% probability.</p>
                    </div>
                    """
                    st.markdown(summary_text, unsafe_allow_html=True)
                    
                # Add email forecast functionality
                email_forecast()
                
                # Export predictions
                if st.button("Export Predictions to CSV", key="export_button"):
                    csv = prediction_data.to_csv(index=True)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"{selected_asset}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Insufficient data points to generate predictions. Try another asset or time period.")
                
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")

def email_forecast():
    """Add email forecast functionality to predictions tab"""
    st.markdown("### 📧 Receive Forecast Report by Email")
    
    with st.form("email_forecast_form"):
        st.write("Get this forecast delivered to your inbox:")
        
        email = st.text_input("Email Address")
        include_chart = st.checkbox("Include chart image", value=True)
        
        submit_button = st.form_submit_button("Send Forecast Report")
        
        if submit_button:
            if not email or '@' not in email:
                st.error("Please enter a valid email address")
            else:
                # Get current prediction data
                if 'current_data' not in st.session_state:
                    st.error("No forecast data available. Please generate a forecast first.")
                    return
                
                try:
                    # Get current data and prediction model
                    data = st.session_state.current_data
                    
                    # Get prediction days from UI
                    days = prediction_days if 'prediction_days' in locals() else 7
                    
                    # Get prediction model type from sidebar
                    model_type = prediction_model if 'prediction_model' in locals() else "Simple"
                    
                    # Get selected asset name
                    asset_name = selected_asset if 'selected_asset' in locals() else "Asset"
                    
                    # Generate prediction using the selected model
                    model = get_prediction_model(model_type, data)
                    prediction_data = model.predict(days=days)
                    
                    # Generate chart image for email
                    if include_chart:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(prediction_data.index, prediction_data['Close'], 
                                label='Prediction', color='#9333EA')
                        plt.fill_between(
                            prediction_data.index,
                            prediction_data['Lower'],
                            prediction_data['Upper'],
                            alpha=0.2,
                            color='#9333EA',
                            label='95% Confidence'
                        )
                        
                        # Add recent historical data
                        historical_range = 30
                        if len(data) > historical_range:
                            recent_data = data.iloc[-historical_range:]
                        else:
                            recent_data = data
                            
                        plt.plot(recent_data.index, recent_data['Close'], 
                                label='Historical', color='#4F46E5')
                        
                        plt.title(f"{asset_name} Price Forecast ({model_type} Model)")
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Save chart to temporary file
                        temp_file = f"temp_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                        plt.savefig(temp_file)
                        plt.close()
                        
                        # Send email with chart
                        result = email_service.send_forecast_report(
                            email, asset_name, prediction_data, temp_file)
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    else:
                        # Send email without chart
                        result = email_service.send_forecast_report(
                            email, asset_name, prediction_data)
                    
                    if result:
                        st.success(f"Forecast report sent to {email}!")
                    else:
                        st.error("Failed to send email. Please try again later.")
                        
                except Exception as e:
                    st.error(f"Error generating forecast report: {str(e)}")

# Add About section at the bottom with a button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

# Initialize session state for showing about section if it doesn't exist
if 'show_about' not in st.session_state:
    st.session_state.show_about = False

# Create a centered button that toggles the about section visibility
with col2:
    if st.button("ℹ️ About MarketPulse AI", use_container_width=True):
        st.session_state.show_about = not st.session_state.show_about

# Display the about section when show_about is True
if st.session_state.show_about:
    st.markdown("""
    <div style="background-color:#262730; padding:20px; border-radius:10px; margin-top:20px;">
        <h2 style="color:#9333EA; text-align:center;">About MarketPulse AI</h2>
        
        <h3>What is MarketPulse AI?</h3>
        
        <p><strong>MarketPulse AI</strong> is an advanced financial market analysis and prediction platform that combines real-time data with machine learning to forecast price movements across stocks, forex, and commodities.</p>
        
        <h3>Key Features</h3>
        
        <ul>
            <li><strong>Multi-Market Analysis</strong>: Track and predict prices for stocks, forex pairs, and commodities</li>
            <li><strong>Advanced Prediction Models</strong>: Choose from simple regression, ARIMA statistical models, or machine learning algorithms</li>
            <li><strong>Technical Indicators</strong>: Visualize key indicators like SMA, EMA, Bollinger Bands, RSI, MACD, and more</li>
            <li><strong>Interactive Visualization</strong>: Explore historical data with interactive charts and custom timeframes</li>
            <li><strong>Email Reports</strong>: Receive detailed forecast reports directly to your inbox</li>
            <li><strong>Premium Features</strong>: Unlock advanced indicators and prediction models with PRO subscription</li>
        </ul>
        
        <h3>How It Works</h3>
        
        <ol>
            <li><strong>Select Your Market</strong>: Choose between stocks, forex, or commodities from the sidebar</li>
            <li><strong>Pick an Asset</strong>: Select the specific asset you want to analyze</li>
            <li><strong>Configure Analysis</strong>: Set the historical data period and prediction horizon</li>
            <li><strong>Choose Your Model</strong>: Select from three prediction models (Simple, ARIMA, or Machine Learning)</li>
            <li><strong>Analyze Results</strong>: View the generated predictions, confidence intervals, and technical analysis</li>
        </ol>
        
        <h3>Data Sources</h3>
        
        <p>MarketPulse AI uses financial data from reputable providers including Yahoo Finance for stocks and commodities, and forex data from leading market exchanges. All predictions are generated using statistical analysis of historical patterns and are for informational purposes only.</p>
        
        <h3>Prediction Models</h3>
        
        <ul>
            <li><strong>Simple</strong>: Linear regression model that identifies trends in historical data</li>
            <li><strong>ARIMA</strong>: Time series forecasting that accounts for seasonality and cyclical patterns</li>
            <li><strong>Machine Learning</strong>: XGBoost algorithm that incorporates multiple technical indicators and patterns</li>
        </ul>
        
        <div style="background-color:#333740; padding:15px; border-radius:5px; margin-top:15px;">
            <p><strong>Disclaimer</strong>: Financial market predictions involve risk. MarketPulse AI forecasts should be used as one of many tools in your investment decision process, not as the sole basis for financial decisions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add a close button
    if st.button("Close", key="close_about"):
        st.session_state.show_about = False
        st.rerun()