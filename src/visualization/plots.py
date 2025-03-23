"""
Plotting Utilities for Natural Gas Data

This module contains functions for creating visualizations of natural gas data and forecasts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import logging

# Configure logging
logging.basicConfig(
    filename='visualization.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

def plot_time_series(data, date_col='date', value_col='value', title='Natural Gas Time Series', 
                    xlabel='Date', ylabel='Value', color='#1f77b4', figsize=(12, 6)):
    """Plot a time series.
    
    Args:
        data (pd.DataFrame): DataFrame containing time series data.
        date_col (str, optional): Name of date column.
        value_col (str, optional): Name of value column.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        color (str, optional): Line color.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    try:
        logger.info(f"Creating time series plot with {len(data)} data points")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the time series
        ax.plot(data[date_col], data[value_col], color=color, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Format date axis
        date_format = DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        raise

def plot_forecast(historical_data, forecast_data, date_col='date', hist_col='value', 
                 forecast_col='forecast', title='Natural Gas Forecast', xlabel='Date', 
                 ylabel='Value', figsize=(12, 6)):
    """Plot historical data with forecast.
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical data.
        forecast_data (pd.DataFrame): DataFrame with forecast data.
        date_col (str, optional): Name of date column.
        hist_col (str, optional): Name of historical value column.
        forecast_col (str, optional): Name of forecast value column.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    try:
        logger.info(f"Creating forecast plot with {len(historical_data)} historical points "
                   f"and {len(forecast_data)} forecast points")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(historical_data[date_col], historical_data[hist_col], 
                color='#1f77b4', linewidth=2, label='Historical')
        
        # Plot forecast data
        ax.plot(forecast_data[date_col], forecast_data[forecast_col], 
                color='#ff7f0e', linewidth=2, label='Forecast')
        
        # Add vertical line to separate historical and forecast data
        if not forecast_data.empty and not historical_data.empty:
            separation_date = historical_data[date_col].iloc[-1]
            ax.axvline(x=separation_date, color='black', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Format date axis
        date_format = DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating forecast plot: {e}")
        raise

def plot_seasonal_decomposition(data, results, title='Seasonal Decomposition', figsize=(12, 10)):
    """Plot seasonal decomposition components.
    
    Args:
        data (pd.DataFrame): Original time series data.
        results (statsmodels.tsa.seasonal.DecomposeResult): Decomposition results.
        title (str, optional): Plot title.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    try:
        logger.info("Creating seasonal decomposition plot")
        
        # Create figure and axes
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot each component
        axes[0].plot(results.observed, linewidth=2)
        axes[0].set_title('Original Time Series')
        
        axes[1].plot(results.trend, linewidth=2)
        axes[1].set_title('Trend')
        
        axes[2].plot(results.seasonal, linewidth=2)
        axes[2].set_title('Seasonality')
        
        axes[3].plot(results.resid, linewidth=2)
        axes[3].set_title('Residuals')
        
        # Add grid to all subplots
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating seasonal decomposition plot: {e}")
        raise

def plot_model_evaluation(historical, predictions, title='Model Evaluation', figsize=(12, 10)):
    """Plot model evaluation charts.
    
    Args:
        historical (array-like): Historical values.
        predictions (array-like): Predicted values.
        title (str, optional): Plot title.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    try:
        logger.info("Creating model evaluation plots")
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Calculate residuals
        residuals = np.array(historical) - np.array(predictions)
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(historical, predictions, alpha=0.5)
        max_val = max(max(historical), max(predictions))
        min_val = min(min(historical), min(predictions))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        
        # Plot 2: Residuals
        axes[0, 1].plot(residuals)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals')
        
        # Plot 3: Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, alpha=0.5, density=True)
        sns.kdeplot(residuals, ax=axes[1, 0])
        axes[1, 0].set_title('Residuals Distribution')
        
        # Plot 4: QQ plot
        from scipy import stats
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        # Add grid to all subplots
        for row in axes:
            for ax in row:
                ax.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model evaluation plots: {e}")
        raise

def save_plot(fig, filename, dpi=300):
    """Save a matplotlib figure to a file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filename (str): Output filename.
        dpi (int, optional): DPI for the output image.
    """
    try:
        logger.info(f"Saving plot to {filename}")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Error saving plot: {e}")
        raise 