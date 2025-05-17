#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python -c "
import sqlite3

def init_db():
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

init_db()
"

echo "Setup complete!" 