import os
import webbrowser
import threading
import time
from flask import Flask

# Import the Flask app from api.py
from api import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("Starting Financial Market Predictor...")
    print("Opening browser window...")
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port, debug=False) 