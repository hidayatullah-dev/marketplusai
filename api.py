from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from email_service import email_service
import json
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Import the prediction function from app.py
from app import simple_predict

# Import data fetchers
from data_fetcher import (
    fetch_stock_data, 
    fetch_forex_data, 
    fetch_commodity_data, 
    get_available_tickers,
    get_available_forex,
    get_available_commodities
)

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Database setup
def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('market_predictor.db')
    cursor = conn.cursor()
    
    # Create contacts table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        subject TEXT,
        message TEXT NOT NULL,
        subscribe BOOLEAN DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create subscribers table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subscribers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT,
        subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        active BOOLEAN DEFAULT 1
    )
    ''')
    
    # Create payments table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        plan TEXT NOT NULL,
        amount REAL NOT NULL,
        payment_method TEXT NOT NULL,
        transaction_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create users table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT,
        is_pro BOOLEAN DEFAULT 0,
        pro_expires_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

# Email sending function
def send_email(to_email, subject, body):
    """Send an email (configured for development/demo)"""
    # For demonstration, we'll log the email instead of sending it
    # In production, you would use SMTP to send real emails
    app.logger.info(f"Email would be sent to: {to_email}")
    app.logger.info(f"Subject: {subject}")
    app.logger.info(f"Body: {body}")
    
    # Uncomment this to actually send emails in production
    """
    sender_email = "your-email@gmail.com"  # Replace with your actual email
    password = "your-password"  # Replace with your actual password
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, to_email, message.as_string())
    server.quit()
    """
    
    return True

# Helper function to save contact form data
def save_contact_form(name, email, subject, message, subscribe):
    """Save contact form data to the database"""
    conn = sqlite3.connect('market_predictor.db')
    cursor = conn.cursor()
    
    # Insert into contacts table
    cursor.execute(
        "INSERT INTO contacts (name, email, subject, message, subscribe) VALUES (?, ?, ?, ?, ?)",
        (name, email, subject, message, 1 if subscribe else 0)
    )
    
    # If subscribe is checked, add to subscribers table
    if subscribe:
        cursor.execute(
            "INSERT OR IGNORE INTO subscribers (email, name) VALUES (?, ?)",
            (email, name)
        )
    
    conn.commit()
    conn.close()
    
    return True

# Helper function to calculate volatility
def calculate_volatility(price_data):
    """Calculate the volatility from price data"""
    returns = price_data.pct_change().dropna()
    return float(returns.std() * 100)

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# API endpoints
@app.route('/api/assets')
def get_assets():
    """Get available assets for each type"""
    return jsonify({
        'stocks': get_available_tickers(),
        'forex': get_available_forex(),
        'commodities': get_available_commodities()
    })

@app.route('/api/data')
def get_data():
    """Get market data for a specific asset"""
    asset_type = request.args.get('type', 'stock')
    symbol = request.args.get('symbol', 'AAPL')
    days = int(request.args.get('days', 365))
    
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    # Fetch data based on asset type
    if asset_type == 'stock':
        df = fetch_stock_data(symbol, start_date, end_date)
        asset_name = f"{symbol} Stock"
    elif asset_type == 'forex':
        df = fetch_forex_data(symbol, start_date, end_date)
        asset_name = f"{symbol.replace('=X', '')} Exchange Rate"
    else:  # commodities
        df = fetch_commodity_data(symbol, start_date, end_date)
        asset_name = f"{symbol.replace('=F', '')} Commodity"
    
    if df.empty or 'Close' not in df.columns:
        # Generate sample data for demo
        app.logger.warning(f"No data found for {symbol} or API limited. Using sample data.")
        
        # Generate sample data
        date_range = pd.date_range(start=start_date, end=end_date)
        base_price = 100.0 if symbol != 'AAPL' else 150.0
        if symbol.startswith('GC'):
            base_price = 1800.0  # Gold
        elif symbol.startswith('SI'):
            base_price = 25.0  # Silver
        elif symbol.startswith('CL'):
            base_price = 70.0  # Oil
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.015, len(date_range))
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

    # Calculate metrics
    latest_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    daily_change = ((latest_price / prev_price) - 1) * 100
    
    start_price = float(df['Close'].iloc[0])
    period_change = latest_price - start_price
    period_change_pct = ((latest_price / start_price) - 1) * 100
    
    # Calculate volatility
    volatility = calculate_volatility(df['Close'])
    
    # Calculate average volume
    volume_avg = "N/A"
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        volume_avg = float(df['Volume'].mean())
    
    # Format the dataframe for JSON
    df_json = df.reset_index()
    
    # Convert dates to string format
    df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d')
    
    # Handle NaN values
    df_json = df_json.fillna("null").replace([np.inf, -np.inf], "null")
    
    # Prepare response
    response = {
        'symbol': symbol,
        'name': asset_name,
        'data': df_json.to_dict(orient='records'),
        'metrics': {
            'currentPrice': latest_price,
            'dailyChange': daily_change,
            'periodChange': period_change,
            'periodChangePercent': period_change_pct,
            'volatility': volatility,
            'volumeAvg': volume_avg
        }
    }
    
    return jsonify(response)

@app.route('/api/predict')
def get_prediction():
    """Get market predictions for a specific asset"""
    asset_type = request.args.get('type', 'stock')
    symbol = request.args.get('symbol', 'AAPL')
    days = int(request.args.get('days', 365))
    prediction_days = int(request.args.get('prediction_days', 7))
    
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    # Fetch data based on asset type
    if asset_type == 'stock':
        df = fetch_stock_data(symbol, start_date, end_date)
    elif asset_type == 'forex':
        df = fetch_forex_data(symbol, start_date, end_date)
    else:  # commodities
        df = fetch_commodity_data(symbol, start_date, end_date)
    
    if df.empty or 'Close' not in df.columns:
        # Generate sample data for demo
        app.logger.warning(f"No data found for {symbol} or API limited. Using sample data for prediction.")
        
        # Generate sample data
        date_range = pd.date_range(start=start_date, end=end_date)
        base_price = 100.0 if symbol != 'AAPL' else 150.0
        if symbol.startswith('GC'):
            base_price = 1800.0  # Gold
        elif symbol.startswith('SI'):
            base_price = 25.0  # Silver
        elif symbol.startswith('CL'):
            base_price = 70.0  # Oil
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.015, len(date_range))
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
    
    # Generate predictions
    try:
        clean_df = df.copy().dropna(subset=['Close'])
        if len(clean_df) > 5:
            prediction_data = simple_predict(clean_df, days=prediction_days)
            
            # Calculate technical indicators
            # RSI
            delta = clean_df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = float(rsi.iloc[-1])
            
            if latest_rsi < 30:
                rsi_status = "Oversold"
            elif latest_rsi > 70:
                rsi_status = "Overbought"
            else:
                rsi_status = "Neutral"
            
            # MACD
            ema12 = clean_df['Close'].ewm(span=12).mean()
            ema26 = clean_df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            latest_macd = float(macd.iloc[-1])
            latest_signal = float(signal.iloc[-1])
            
            macd_status = "Bullish" if latest_macd > latest_signal else "Bearish"
            
            # Trend
            sma20 = clean_df['Close'].rolling(window=20).mean().iloc[-1]
            sma50 = clean_df['Close'].rolling(window=50).mean().iloc[-1]
            trend = "Uptrend" if sma20 > sma50 else "Downtrend"
            trend_change = ((clean_df['Close'].iloc[-1] / clean_df['Close'].iloc[-20]) - 1) * 100
            
            # Format prediction dataframe
            pred_json = prediction_data.reset_index()
            pred_json['Date'] = pred_json['Date'].dt.strftime('%Y-%m-%d')
            pred_json = pred_json.fillna("null").replace([np.inf, -np.inf], "null")
            
            # Prepare response
            response = {
                'symbol': symbol,
                'predictions': pred_json.to_dict(orient='records'),
                'insights': {
                    'rsi': latest_rsi,
                    'rsiStatus': rsi_status,
                    'macd': latest_macd,
                    'macdStatus': macd_status,
                    'trend': trend,
                    'trendChange': trend_change
                }
            }
            
            return jsonify(response)
        else:
            return jsonify({
                'error': "Insufficient data points for prediction"
            }), 400
    except Exception as e:
        return jsonify({
            'error': f"Error generating predictions: {str(e)}"
        }), 500

@app.route('/api/contact', methods=['POST'])
def contact_form():
    """Handle contact form submission"""
    try:
        data = request.form or request.json
        
        name = data.get('name', '')
        email = data.get('email', '')
        subject = data.get('subject', 'Contact Form Submission')
        message = data.get('message', '')
        subscribe = data.get('subscribe', False)
        recipient = data.get('recipient', 'hidayatullah2269@gmail.com')
        
        # Validate required fields
        if not name or not email or not message:
            return jsonify({
                'error': 'Missing required fields'
            }), 400
        
        # Save to database
        save_contact_form(name, email, subject, message, subscribe)
        
        # Forward to specified email
        email_body = f"""
        New contact form submission:
        
        Name: {name}
        Email: {email}
        Subject: {subject}
        Message: {message}
        Subscribe to updates: {'Yes' if subscribe else 'No'}
        
        This message was sent from MarketPulse AI.
        """
        
        send_email(recipient, f"MarketPulse AI: {subject}", email_body)
        
        return jsonify({
            'success': True,
            'message': 'Your message has been sent successfully!'
        })
    
    except Exception as e:
        app.logger.error(f"Error in contact form: {str(e)}")
        return jsonify({
            'error': 'Failed to process your request. Please try again later.'
        }), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    """Handle newsletter subscription"""
    try:
        data = request.form or request.json
        
        email = data.get('email', '')
        name = data.get('name', '')
        
        if not email:
            return jsonify({
                'error': 'Email is required'
            }), 400
        
        # Add to subscribers
        conn = sqlite3.connect('market_predictor.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR IGNORE INTO subscribers (email, name) VALUES (?, ?)",
            (email, name)
        )
        
        conn.commit()
        conn.close()
        
        # Send confirmation email
        email_body = f"""
        Thank you for subscribing to MarketPulse AI updates!
        
        You'll now receive the latest market insights, tips, and feature announcements.
        
        If you didn't request this subscription, please ignore this email.
        """
        
        send_email(email, "Welcome to MarketPulse AI!", email_body)
        
        return jsonify({
            'success': True,
            'message': 'Successfully subscribed!'
        })
    
    except Exception as e:
        app.logger.error(f"Error in subscription: {str(e)}")
        return jsonify({
            'error': 'Failed to process your subscription. Please try again later.'
        }), 500

@app.route('/api/payment', methods=['POST'])
def process_payment():
    """Handle payment processing (demo only)"""
    try:
        data = request.form or request.json
        
        name = data.get('name', '')
        email = data.get('email', '')
        plan = data.get('plan', '')
        payment_method = data.get('payment_method', '')
        amount = 0
        
        # Set amount based on plan
        if plan == 'monthly':
            amount = 29.00
        elif plan == 'yearly':
            amount = 199.00
        elif plan == 'lifetime':
            amount = 499.00
        
        # In a real app, you would process the payment with a payment provider here
        
        # Demo: Save payment info to database
        conn = sqlite3.connect('market_predictor.db')
        cursor = conn.cursor()
        
        # Check if user exists, create if not
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute(
                "INSERT INTO users (name, email, is_pro) VALUES (?, ?, 1)",
                (name, email)
            )
            user_id = cursor.lastrowid
        else:
            user_id = user[0]
            cursor.execute(
                "UPDATE users SET is_pro = 1 WHERE id = ?",
                (user_id,)
            )
        
        # Set expiration date based on plan
        if plan == 'monthly':
            expires_at = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "UPDATE users SET pro_expires_at = ? WHERE id = ?",
                (expires_at, user_id)
            )
        elif plan == 'yearly':
            expires_at = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "UPDATE users SET pro_expires_at = ? WHERE id = ?",
                (expires_at, user_id)
            )
        elif plan == 'lifetime':
            cursor.execute(
                "UPDATE users SET pro_expires_at = NULL WHERE id = ?",
                (user_id,)
            )
        
        # Record the payment
        transaction_id = f"DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cursor.execute(
            "INSERT INTO payments (user_id, plan, amount, payment_method, transaction_id) VALUES (?, ?, ?, ?, ?)",
            (user_id, plan, amount, payment_method, transaction_id)
        )
        
        conn.commit()
        conn.close()
        
        # Send confirmation email
        email_body = f"""
        Thank you for upgrading to MarketPulse AI PRO!
        
        Plan: {plan.title()}
        Amount: ${amount:.2f}
        Transaction ID: {transaction_id}
        
        You now have access to all premium features. If you have any questions, please contact support.
        """
        
        send_email(email, "Welcome to MarketPulse AI PRO!", email_body)
        
        # Also notify the admin
        admin_email = "hidayatullah2269@gmail.com"
        admin_body = f"""
        New PRO upgrade:
        
        Name: {name}
        Email: {email}
        Plan: {plan.title()}
        Amount: ${amount:.2f}
        Payment Method: {payment_method}
        Transaction ID: {transaction_id}
        """
        
        send_email(admin_email, "New PRO Subscription!", admin_body)
        
        return jsonify({
            'success': True,
            'message': 'Payment processed successfully!',
            'transaction_id': transaction_id
        })
    
    except Exception as e:
        app.logger.error(f"Error processing payment: {str(e)}")
        return jsonify({
            'error': 'Failed to process payment. Please try again later.'
        }), 500

# Helper functions for API endpoints
def init_api():
    """Initialize API routes"""
    api_routes = {
        '/api/contact': handle_contact_form,
        '/api/subscribe': handle_subscription,
        '/api/export': handle_export_data,
        '/api/email-forecast': handle_email_forecast
    }
    return api_routes

def handle_contact_form(request_data):
    """API endpoint to handle contact form submissions"""
    try:
        # Extract form data
        name = request_data.get('name', '')
        email = request_data.get('email', '')
        message = request_data.get('message', '')
        
        # Validate input
        if not name or not email or not message:
            return {'success': False, 'error': 'All fields are required'}
        
        # Store in database
        conn = sqlite3.connect('finance_app.db')
        c = conn.cursor()
        c.execute("INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)",
                 (name, email, message))
        conn.commit()
        conn.close()
        
        # Send confirmation email to user
        email_service.send_contact_confirmation(name, email, message)
        
        # Notify admin
        email_service.notify_admin_contact(name, email, message)
        
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_subscription(request_data):
    """API endpoint to handle subscription purchases"""
    try:
        # Extract data
        email = request_data.get('email', '')
        plan_type = request_data.get('plan_type', '')
        plan_price = request_data.get('plan_price', 0)
        payment_method = request_data.get('payment_method', '')
        card_info = request_data.get('card_info', {})
        
        # Validate input
        if not email or not plan_type or not plan_price:
            return {'success': False, 'error': 'Required fields missing'}
        
        # Calculate expiration date
        now = datetime.now()
        if plan_type == "monthly":
            expires_at = now + timedelta(days=30)
        elif plan_type == "yearly":
            expires_at = now + timedelta(days=365)
        else:  # lifetime
            expires_at = now + timedelta(days=36500)  # ~100 years
        
        # Process payment (in a real app, connect to payment gateway)
        # For demo, we'll assume payment was successful
        
        # Store subscription in database
        conn = sqlite3.connect('finance_app.db')
        c = conn.cursor()
        
        # Check if user already has an active subscription
        c.execute("SELECT * FROM subscriptions WHERE user_email = ? AND active = 1", (email,))
        existing = c.fetchone()
        
        if existing:
            # Update existing subscription
            c.execute("""
            UPDATE subscriptions 
            SET plan_type = ?, plan_price = ?, payment_method = ?, expires_at = ?
            WHERE user_email = ? AND active = 1
            """, (plan_type, float(plan_price), payment_method, expires_at, email))
        else:
            # Create new subscription
            c.execute("""
            INSERT INTO subscriptions 
            (user_email, plan_type, plan_price, payment_method, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """, (email, plan_type, float(plan_price), payment_method, expires_at))
        
        conn.commit()
        conn.close()
        
        # Send confirmation email
        email_service.send_subscription_confirmation(email, plan_type, plan_price, expires_at)
        
        return {
            'success': True, 
            'subscription': {
                'email': email,
                'plan': plan_type,
                'expires_at': expires_at.strftime('%Y-%m-%d'),
                'is_active': True
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_export_data(request_data):
    """API endpoint to export data to CSV or JSON"""
    try:
        # Extract data
        data_type = request_data.get('data_type', 'historical')  # historical or prediction
        format_type = request_data.get('format', 'csv')  # csv or json
        data = request_data.get('data', [])
        
        if not data:
            return {'success': False, 'error': 'No data provided'}
        
        # Convert data to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            return {'success': False, 'error': 'Invalid data format'}
        
        # Generate export content
        if format_type.lower() == 'csv':
            content = df.to_csv(index=True)
            mime_type = 'text/csv'
            file_ext = 'csv'
        else:  # json
            content = df.to_json(orient='records', date_format='iso')
            mime_type = 'application/json'
            file_ext = 'json'
        
        filename = f"{data_type}_data_{datetime.now().strftime('%Y%m%d')}.{file_ext}"
        
        return {
            'success': True,
            'content': content,
            'filename': filename,
            'mime_type': mime_type
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_email_forecast(request_data):
    """API endpoint to email forecast reports"""
    try:
        # Extract data
        email = request_data.get('email', '')
        asset_name = request_data.get('asset_name', '')
        forecast_data = request_data.get('forecast_data', [])
        chart_data = request_data.get('chart_image', None)
        
        # Validate input
        if not email or not asset_name or not forecast_data:
            return {'success': False, 'error': 'Required fields missing'}
        
        # Convert forecast data to DataFrame if it's not already
        if not isinstance(forecast_data, pd.DataFrame):
            forecast_data = pd.DataFrame(forecast_data)
        
        # Generate chart image if not provided
        chart_path = None
        if chart_data:
            # If chart data is provided as base64
            if isinstance(chart_data, str) and chart_data.startswith('data:image'):
                # Extract base64 data
                base64_data = chart_data.split(',')[1]
                image_data = base64.b64decode(base64_data)
                
                # Save to temp file
                temp_file = f"temp_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                with open(temp_file, 'wb') as f:
                    f.write(image_data)
                chart_path = temp_file
            # If chart data is not provided, generate a simple chart
            else:
                # Create simple chart
                plt.figure(figsize=(10, 6))
                plt.plot(forecast_data.index, forecast_data['Close'], label='Prediction')
                plt.fill_between(
                    forecast_data.index,
                    forecast_data['Lower'],
                    forecast_data['Upper'],
                    alpha=0.2,
                    label='95% Confidence'
                )
                plt.title(f"{asset_name} Price Forecast")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save to temp file
                temp_file = f"temp_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                plt.savefig(temp_file)
                plt.close()
                chart_path = temp_file
        
        # Send email with forecast
        result = email_service.send_forecast_report(email, asset_name, forecast_data, chart_path)
        
        # Clean up temp file if it exists
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
        
        if result:
            return {'success': True}
        else:
            return {'success': False, 'error': 'Failed to send email'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Initialize the API
api_routes = init_api()

# Initialize the database when the app starts
with app.app_context():
    init_db()

if __name__ == '__main__':
    # Determine port - use PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 