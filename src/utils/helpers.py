"""
Helper Utilities for Natural Gas Forecasting

This module contains various helper functions used across the project.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3

# Configure logging
logging.basicConfig(
    filename='helpers.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_csv(file_path):
    """Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
        raise

def load_data_from_db(db_path, table_name='natural_gas_data'):
    """Load data from a SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database.
        table_name (str, optional): Name of the table to query.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        logger.info(f"Loading data from database {db_path}, table {table_name}")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
            
        # Create a connection to the database
        conn = sqlite3.connect(db_path)
        
        # Load data from the table
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        
        # Close the connection
        conn.close()
        
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            
        logger.info(f"Loaded {len(df)} records from database")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise

def create_date_features(df, date_column='date'):
    """Create date-based features from a date column.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        date_column (str, optional): Name of the date column.
        
    Returns:
        pd.DataFrame: Dataframe with additional date features.
    """
    try:
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in dataframe")
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure date column is datetime
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Create features
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['day_of_week'] = result_df[date_column].dt.dayofweek
        result_df['day_of_year'] = result_df[date_column].dt.dayofyear
        result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
        
        # Add season
        result_df['season'] = result_df['month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else 'Fall'
        )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error creating date features: {e}")
        raise

def calculate_metrics(true_values, predictions):
    """Calculate common evaluation metrics.
    
    Args:
        true_values (array-like): Actual values.
        predictions (array-like): Predicted values.
        
    Returns:
        dict: Dictionary with evaluation metrics.
    """
    try:
        # Convert to numpy arrays
        y_true = np.array(true_values)
        y_pred = np.array(predictions)
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
        
        # Return metrics dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def export_results_to_json(data, filepath):
    """Export results to a JSON file.
    
    Args:
        data (dict): Data to export.
        filepath (str): Path to save the JSON file.
    """
    try:
        logger.info(f"Exporting results to {filepath}")
        
        # Handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=DateTimeEncoder)
            
        logger.info(f"Results exported successfully to {filepath}")
        
    except Exception as e:
        logger.error(f"Error exporting results to JSON: {e}")
        raise

def generate_date_range(start_date, periods=12, freq='MS'):
    """Generate a sequence of dates.
    
    Args:
        start_date (str or datetime): Start date.
        periods (int, optional): Number of periods to generate.
        freq (str, optional): Frequency string (MS=month start).
        
    Returns:
        pd.DatetimeIndex: Generated date range.
    """
    try:
        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        # Generate date range
        date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        return date_range
        
    except Exception as e:
        logger.error(f"Error generating date range: {e}")
        raise 