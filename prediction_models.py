import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PredictionModel:
    """Base class for all prediction models"""
    
    def __init__(self, data=None):
        self.data = data
        
    def set_data(self, data):
        """Set historical data for prediction"""
        self.data = data
        
    def predict(self, days=7):
        """Predict future values"""
        raise NotImplementedError("Subclasses must implement this method")

class SimpleLinearModel(PredictionModel):
    """Simple linear regression model for predictions"""
    
    def predict(self, days=7):
        """Predict using simple linear regression"""
        # Check that we have sufficient data
        if self.data is None or len(self.data) < 5:
            raise ValueError("Insufficient data for prediction")
        
        # Get the last 30 days (or less if not available)
        window = min(30, len(self.data))
        close_values = self.data['Close'].dropna().values[-window:]
        
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
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        pred_df = pd.DataFrame(index=future_dates)
        pred_df['Close'] = future_y
        
        # Add confidence intervals (simple approach)
        std_dev = self.data['Close'].dropna().pct_change().dropna().std()
        if pd.isna(std_dev) or std_dev == 0:
            # Use a default small volatility if we can't calculate it
            std_dev = 0.01
        
        volatility = std_dev * np.sqrt(np.arange(1, days + 1))
        pred_df['Lower'] = pred_df['Close'] * (1 - 1.96 * volatility)
        pred_df['Upper'] = pred_df['Close'] * (1 + 1.96 * volatility)
        
        return pred_df

class ARIMAModel(PredictionModel):
    """ARIMA time series model for predictions"""
    
    def predict(self, days=7, order=(5,1,0)):
        """Predict using ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            # Check that we have sufficient data
            if self.data is None or len(self.data) < 20:
                raise ValueError("Insufficient data for ARIMA prediction")
            
            # Check stationarity
            close_values = self.data['Close'].dropna()
            
            # Use appropriate differencing if needed
            differenced = False
            if adfuller(close_values)[1] > 0.05:
                # Non-stationary, use differencing
                close_values = close_values.diff().dropna()
                differenced = True
                
            # Fit ARIMA model
            model = ARIMA(close_values, order=order)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=days)
            forecast_ci = model_fit.get_forecast(steps=days).conf_int()
            
            # Create prediction dataframe
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            pred_df = pd.DataFrame(index=future_dates)
            
            # Undo differencing if applied
            if differenced:
                last_value = self.data['Close'].iloc[-1]
                cumulative_sum = np.cumsum(forecast)
                pred_values = last_value + cumulative_sum
                pred_df['Close'] = pred_values
                
                # Approximate confidence intervals after undifferencing
                pred_df['Lower'] = last_value + np.cumsum(forecast_ci.iloc[:, 0].values)
                pred_df['Upper'] = last_value + np.cumsum(forecast_ci.iloc[:, 1].values)
            else:
                pred_df['Close'] = forecast
                pred_df['Lower'] = forecast_ci.iloc[:, 0]
                pred_df['Upper'] = forecast_ci.iloc[:, 1]
            
            return pred_df
        except ImportError:
            # Fallback to simple model if statsmodels is not available
            print("statsmodels not available, falling back to simple model")
            simple_model = SimpleLinearModel(self.data)
            return simple_model.predict(days)

class XGBoostModel(PredictionModel):
    """XGBoost machine learning model for predictions"""
    
    def _create_features(self, df):
        """Create features for ML model"""
        df_features = df.copy()
        
        # Technical indicators
        df_features['SMA_5'] = df_features['Close'].rolling(window=5).mean()
        df_features['SMA_20'] = df_features['Close'].rolling(window=20).mean()
        df_features['Daily_Return'] = df_features['Close'].pct_change()
        df_features['Price_Volatility'] = df_features['Close'].rolling(window=10).std()
        
        # Calculate RSI
        delta = df_features['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df_features['RSI'] = 100 - (100 / (1 + rs))
        
        # Create lagged features
        for i in range(1, 6):
            df_features[f'Close_Lag_{i}'] = df_features['Close'].shift(i)
            df_features[f'Return_Lag_{i}'] = df_features['Daily_Return'].shift(i)
        
        # Drop NaN values
        return df_features.dropna()
    
    def predict(self, days=7):
        """Predict using XGBoost model"""
        try:
            import xgboost as xgb
            from sklearn.preprocessing import MinMaxScaler
            
            # Check that we have sufficient data
            if self.data is None or len(self.data) < 30:
                raise ValueError("Insufficient data for ML prediction")
            
            # Create features
            df = self._create_features(self.data)
            
            # Prepare features and target
            features = ['SMA_5', 'SMA_20', 'RSI', 'Daily_Return', 'Price_Volatility'] + \
                       [f'Close_Lag_{i}' for i in range(1, 6)] + \
                       [f'Return_Lag_{i}' for i in range(1, 6)]
            
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
                            new_row[f'Return_Lag_{j}'] = last_row['Daily_Return'].values[0]
                        elif j <= i+1:
                            new_row[f'Close_Lag_{j}'] = prediction_rows[i-j+1]['Close']
                            if i-j+2 < len(prediction_rows):
                                prev_close = prediction_rows[i-j+2]['Close']
                                curr_close = prediction_rows[i-j+1]['Close']
                                new_row[f'Return_Lag_{j}'] = (curr_close / prev_close - 1)
                
                prediction_rows.append(new_row)
                last_row = new_row
            
            # Create prediction dataframe
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            pred_df = pd.DataFrame(index=future_dates)
            pred_df['Close'] = [row['Close'] for row in prediction_rows]
            
            # Add confidence intervals
            std_dev = self.data['Close'].pct_change().std()
            confidence_interval = 1.96 * std_dev * np.sqrt(np.arange(1, days + 1))
            
            pred_df['Lower'] = pred_df['Close'] * (1 - confidence_interval)
            pred_df['Upper'] = pred_df['Close'] * (1 + confidence_interval)
            
            return pred_df
            
        except ImportError:
            # Fallback to simple model if xgboost is not available
            print("XGBoost not available, falling back to simple model")
            simple_model = SimpleLinearModel(self.data)
            return simple_model.predict(days)

class EnsembleModel(PredictionModel):
    """Ensemble model that combines multiple prediction models"""
    
    def predict(self, days=7):
        """Predict using ensemble of models"""
        models = []
        
        # Try to create each model
        try:
            models.append(SimpleLinearModel(self.data))
            
            try:
                models.append(ARIMAModel(self.data))
            except:
                pass
                
            try:
                models.append(XGBoostModel(self.data))
            except:
                pass
        
        except Exception as e:
            # If all else fails, use simple model
            print(f"Error creating models: {e}")
            return SimpleLinearModel(self.data).predict(days)
        
        # Get predictions from each model
        predictions = []
        for model in models:
            try:
                pred = model.predict(days)
                predictions.append(pred)
            except Exception as e:
                print(f"Error in model prediction: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Create ensemble prediction by averaging
        ensemble_df = pd.DataFrame(index=predictions[0].index)
        
        # Average the Close predictions
        ensemble_df['Close'] = sum(pred['Close'] for pred in predictions) / len(predictions)
        
        # Take the min of lower bounds and max of upper bounds
        ensemble_df['Lower'] = min(pred['Lower'].values for pred in predictions).min(axis=0)
        ensemble_df['Upper'] = max(pred['Upper'].values for pred in predictions).max(axis=0)
        
        return ensemble_df

def get_prediction_model(model_type, data):
    """Factory function to get the appropriate prediction model"""
    if model_type.lower() == 'simple':
        return SimpleLinearModel(data)
    elif model_type.lower() == 'arima':
        return ARIMAModel(data)
    elif model_type.lower() == 'xgboost' or model_type.lower() == 'ml':
        return XGBoostModel(data)
    elif model_type.lower() == 'ensemble':
        return EnsembleModel(data)
    else:
        # Default to simple model
        return SimpleLinearModel(data) 