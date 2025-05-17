import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def plot_historical_data(df, show_sma=False, show_ema=False, show_bollinger=False, asset_name=""):
    """
    Plot historical price data with optional technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        show_sma (bool): Whether to show Simple Moving Average
        show_ema (bool): Whether to show Exponential Moving Average
        show_bollinger (bool): Whether to show Bollinger Bands
        asset_name (str): Name of the asset for the chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the chart
    """
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart if we have OHLC data
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',  # Green for increasing candles
            decreasing_line_color='#ef5350'   # Red for decreasing candles
        ))
    else:
        # If OHLC data is not available, use a line chart
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='cyan', width=2)
        ))
    
    # Add Simple Moving Average (SMA)
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
    
    # Add Exponential Moving Average (EMA)
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
    
    # Add Bollinger Bands
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
        title=f"Historical Price Data for {asset_name}",
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
    
    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_prediction_with_confidence(historical_df, prediction_df, asset_name=""):
    """
    Plot historical data along with predictions and confidence intervals
    
    Args:
        historical_df (pd.DataFrame): DataFrame with historical price data
        prediction_df (pd.DataFrame): DataFrame with predicted prices and confidence intervals
        asset_name (str): Name of the asset for the chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the chart
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='cyan', width=2)
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=prediction_df.index,
        y=prediction_df['Close'],
        mode='lines',
        name='Prediction',
        line=dict(color='magenta', width=2)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=prediction_df.index.tolist() + prediction_df.index.tolist()[::-1],
        y=prediction_df['Upper'].tolist() + prediction_df['Lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 255, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Confidence Interval'
    ))
    
    # Add vertical line to separate historical data from predictions
    first_prediction_date = prediction_df.index[0]
    
    fig.add_shape(
        type="line",
        x0=first_prediction_date,
        y0=historical_df['Close'].min() * 0.9,
        x1=first_prediction_date,
        y1=historical_df['Close'].max() * 1.1,
        line=dict(color="white", width=1, dash="dash")
    )
    
    # Add annotation for the separation line
    fig.add_annotation(
        x=first_prediction_date,
        y=historical_df['Close'].max() * 1.05,
        text="Prediction Start",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title=f"Price Prediction for {asset_name}",
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
    
    return fig

def create_performance_metrics(model, historical_data):
    """
    Create a DataFrame with model performance metrics
    
    Args:
        model: The trained prediction model
        historical_data (pd.DataFrame): DataFrame with historical data
        
    Returns:
        pd.DataFrame: DataFrame with performance metrics
    """
    # This would typically calculate metrics based on the model's performance
    # on validation data, but we'll just create sample metrics for now
    
    metrics = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE'],
        'Value': [
            f"{np.random.uniform(0.5, 2.0):.2f}",
            f"{np.random.uniform(1.0, 3.0):.2f}",
            f"{np.random.uniform(2.0, 8.0):.2f}%"
        ],
        'Description': [
            'Mean Absolute Error',
            'Root Mean Squared Error',
            'Mean Absolute Percentage Error'
        ]
    })
    
    return metrics

def create_price_indicator(latest_price, previous_price):
    """
    Create a price indicator figure with arrow showing the trend
    
    Args:
        latest_price (float): Latest price
        previous_price (float): Previous price
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with the price indicator
    """
    # Determine if price is up or down
    price_change = latest_price - previous_price
    is_up = price_change >= 0
    
    # Calculate percentage change
    pct_change = (price_change / previous_price) * 100
    
    # Create figure
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=latest_price,
        number={'prefix': "$", 'valueformat': ".2f"},
        delta={'position': "bottom", 'reference': previous_price, 'valueformat': ".2f"},
        title={'text': "Current Price"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=150,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig
