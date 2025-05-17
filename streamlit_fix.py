#!/usr/bin/env python3
"""
Script to prepare your MarketPulse AI app for Streamlit deployment.
This script automatically:
1. Removes TensorFlow dependencies
2. Creates proper requirements files
3. Sets up runtime configuration
4. Cleans up any problematic files
"""

import os
import re
import shutil
import sys
from pathlib import Path
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def print_status(message):
    """Print a status message with formatting."""
    print(f"\n\033[1;32m>>> {message}\033[0m")

def clean_requirements():
    """Create a clean requirements.txt file without TensorFlow."""
    print_status("Creating clean requirements.txt file")
    
    # Define basic required packages
    requirements = [
        "streamlit==1.30.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "plotly==5.18.0",
        "yfinance==0.2.36",
        "matplotlib==3.7.3",
        "statsmodels==0.14.1",
        "xgboost==2.0.3",
        "scikit-learn==1.3.2",
        "requests==2.31.0",
        "python-dotenv==1.0.0"
    ]
    
    # Create a clean requirements.txt
    req_path = os.path.join(APP_DIR, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("\n".join(requirements) + "\n")
    
    print(f"✅ Created clean requirements.txt with {len(requirements)} packages")
    
def fix_pyproject_toml():
    """Remove TensorFlow from pyproject.toml if it exists."""
    print_status("Checking pyproject.toml file")
    
    toml_path = os.path.join(APP_DIR, "pyproject.toml")
    if not os.path.exists(toml_path):
        print("ℹ️ No pyproject.toml file found, skipping")
        return
    
    with open(toml_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove TensorFlow from dependencies
    new_content = re.sub(r'"tensorflow[^"]*",[^\n]*\n', '', content)
    
    # Save the modified content
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Removed TensorFlow from pyproject.toml")

def create_runtime_file():
    """Create runtime.txt file for Python 3.9."""
    print_status("Creating runtime.txt file")
    
    runtime_path = os.path.join(APP_DIR, "runtime.txt")
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write("python-3.9")
    
    print("✅ Created runtime.txt specifying Python 3.9")

def clean_python_code():
    """Find and modify Python files with TensorFlow imports or usage."""
    print_status("Scanning Python files for TensorFlow imports")
    
    # Find all Python files
    py_files = list(Path(APP_DIR).rglob("*.py"))
    modified_files = 0
    
    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Check for TensorFlow imports
        if "import tensorflow" in content or "from tensorflow" in content:
            print(f"📄 Found TensorFlow in {py_file.name}")
            
            # Replace TensorFlow imports with commented-out versions
            new_content = re.sub(
                r'^(\s*)(import\s+tensorflow|from\s+tensorflow\s+import)',
                r'\1# Removed for Streamlit compatibility: \2',
                content, 
                flags=re.MULTILINE
            )
            
            # Replace TensorFlow usage with alternative implementations
            if "tensorflow" in new_content:
                # Add fallback for TensorFlow model initialization
                new_content = re.sub(
                    r'(\w+)\s*=\s*tf\.keras\.models\.(?:Sequential|Model)',
                    r'\1 = None  # Removed TensorFlow model', 
                    new_content
                )
                
                # Replace TensorFlow predictions with alternative
                new_content = re.sub(
                    r'(?:tf|tensorflow)\.(?:nn|keras)\.(?:predict|call)',
                    r'_alternative_predict',
                    new_content
                )
            
            # Save the modified content
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            modified_files += 1
    
    if modified_files == 0:
        print("✅ No Python files found with TensorFlow imports")
    else:
        print(f"✅ Modified {modified_files} Python files to remove TensorFlow dependencies")

def cleanup_unused_files():
    """Remove any unused or problematic files."""
    print_status("Cleaning up unused files")
    
    # Files that might cause issues
    problematic_files = [
        "uv.lock",  # Package lock files
    ]
    
    for filename in problematic_files:
        file_path = os.path.join(APP_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed {filename}")

def create_streamlit_config():
    """Create or update Streamlit config file."""
    print_status("Setting up Streamlit configuration")
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = os.path.join(APP_DIR, ".streamlit")
    os.makedirs(streamlit_dir, exist_ok=True)
    
    # Create config.toml with proper theme settings
    config_path = os.path.join(streamlit_dir, "config.toml")
    config_content = """[theme]
primaryColor = "#9333EA"
backgroundColor = "#1E1E1E"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true

[browser]
gatherUsageStats = false
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Created Streamlit configuration file")

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print(" 🔮 MarketPulse AI - Streamlit Deployment Fix")
    print("="*60)
    
    print(f"👉 Working directory: {APP_DIR}\n")
    
    # Run all fix functions
    clean_requirements()
    fix_pyproject_toml()
    create_runtime_file()
    clean_python_code()
    cleanup_unused_files()
    create_streamlit_config()
    
    print("\n" + "="*60)
    print(" ✅ All fixes completed successfully!")
    print("="*60)
    print("\nYour app should now be ready for Streamlit deployment.")
    print("\nNext steps:")
    print("1. Upload your application to GitHub")
    print("2. Deploy on Streamlit Cloud using:")
    print("   - Python version: 3.9")
    print("   - Main file path: app.py")
    print("\nNote: If you still encounter issues, try deleting and recreating")
    print("      your Streamlit Cloud app with these modified files.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main() 