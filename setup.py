import sqlite3
import os
import sys

def init_db():
    """Initialize SQLite database for contacts and subscriptions"""
    print("Initializing database...")
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
    print("Database initialization complete!")

def setup():
    """Complete setup for MarketPulse AI app"""
    print("Setting up MarketPulse AI...")
    
    # Initialize database
    init_db()
    
    print("\nSetup complete! You can now run the app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    setup() 