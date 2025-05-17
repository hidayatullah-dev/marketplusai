import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def find_similar_patterns(df, pattern_length=30, top_n=3, normalize=True):
    """
    Find historical patterns similar to the most recent pattern
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        pattern_length (int): Length of the pattern to compare
        top_n (int): Number of similar patterns to return
        normalize (bool): Whether to normalize patterns for comparison
        
    Returns:
        list: List of tuples (similarity score, start index) of similar patterns
    """
    if len(df) < pattern_length * 2:
        st.warning(f"Not enough historical data to find patterns of length {pattern_length}")
        return []
    
    # Get the closing prices
    prices = df['Close'].values
    
    # Current pattern (most recent n days)
    current_pattern = prices[-pattern_length:]
    
    # Normalize the current pattern if requested
    if normalize:
        current_pattern = (current_pattern - np.mean(current_pattern)) / np.std(current_pattern)
    
    # Initialize list to store similarity scores
    similarities = []
    
    # Iterate through historical data to find similar patterns
    for i in range(len(prices) - 2 * pattern_length):
        # Extract historical pattern
        historical_pattern = prices[i:i+pattern_length]
        
        # Normalize if requested
        if normalize:
            historical_pattern = (historical_pattern - np.mean(historical_pattern)) / np.std(historical_pattern)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([current_pattern], [historical_pattern])[0][0]
        
        # Add to list
        similarities.append((similarity, i))
    
    # Sort by similarity (descending) and get top_n
    similarities.sort(reverse=True)
    
    # Return top_n patterns, excluding the most recent one (which would be identical)
    filtered_similarities = []
    for similarity, idx in similarities:
        # Check if this pattern doesn't overlap too much with the current pattern
        if idx < len(prices) - (pattern_length * 1.5):
            filtered_similarities.append((similarity, idx))
            if len(filtered_similarities) >= top_n:
                break
    
    return filtered_similarities

def create_pattern_embedding(data, window_size=10):
    """
    Create embeddings for vector-based similarity search (for Pinecone integration)
    
    Args:
        data (pd.DataFrame): DataFrame with time series data
        window_size (int): Size of the sliding window
        
    Returns:
        np.ndarray: Array of embeddings
    """
    # Get normalized price changes
    price_changes = data['Close'].pct_change().dropna().values
    
    # Create sliding windows
    embeddings = []
    timestamps = []
    
    for i in range(len(price_changes) - window_size + 1):
        window = price_changes[i:i+window_size]
        # Normalize window
        window = (window - np.mean(window)) / (np.std(window) + 1e-8)
        embeddings.append(window)
        timestamps.append(data.index[i+window_size-1])
    
    return np.array(embeddings), timestamps

def simulate_pinecone_search(embeddings, timestamps, query_embedding, top_k=5):
    """
    Simulate Pinecone vector search (without using actual Pinecone)
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        timestamps (list): List of timestamps corresponding to embeddings
        query_embedding (np.ndarray): Query embedding
        top_k (int): Number of results to return
        
    Returns:
        list: List of tuples (similarity score, timestamp) of similar patterns
    """
    # Calculate cosine similarity with each embedding
    similarities = []
    
    for i, embedding in enumerate(embeddings):
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append((similarity, timestamps[i]))
    
    # Sort by similarity (descending) and get top_k
    similarities.sort(reverse=True)
    
    return similarities[:top_k]
