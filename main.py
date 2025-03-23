"""
Natural Gas Forecasting - Main Entry Point

This script serves as the main entry point for the natural gas forecasting project.
It connects the ETL, modeling, and visualization components.
"""

import os
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from src.data.etl import run_etl_pipeline
from src.models.forecasting import ARIMAModel, SARIMAXModel
from src.visualization.plots import plot_time_series, plot_forecast, save_plot
from src.utils.helpers import load_data_from_csv, load_data_from_db, calculate_metrics, export_results_to_json

# Configure logging
logging.basicConfig(
    filename='natural_gas_forecasting.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Natural Gas Forecasting')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['etl', 'model', 'visualize', 'full'],
                        help='Mode to run the pipeline (default: full)')
                        
    parser.add_argument('--start-year', type=int,
                        default=datetime.now().year - 5,
                        help='Start year for data extraction (default: 5 years ago)')
                        
    parser.add_argument('--end-year', type=int,
                        default=datetime.now().year,
                        help='End year for data extraction (default: current year)')
                        
    parser.add_argument('--forecast-periods', type=int, default=12,
                        help='Number of periods to forecast (default: 12)')
                        
    parser.add_argument('--model-type', type=str, default='arima',
                        choices=['arima', 'sarimax'],
                        help='Type of forecasting model to use (default: arima)')
                        
    parser.add_argument('--data-source', type=str, default='db',
                        choices=['csv', 'db', 'api'],
                        help='Source of data for modeling (default: db)')
                        
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files (default: output)')
    
    return parser.parse_args()

def run_etl(start_year, end_year):
    """Run the ETL process.
    
    Args:
        start_year (int): Start year for data extraction.
        end_year (int): End year for data extraction.
        
    Returns:
        pd.DataFrame: Processed data.
    """
    logger.info(f"Running ETL from {start_year} to {end_year}")
    
    try:
        # Convert years to date strings for the ETL pipeline
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        # Run the ETL pipeline
        processed_data = run_etl_pipeline(start_date=start_date, end_date=end_date)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error running ETL: {e}")
        raise

def load_modeling_data(data_source, start_year=None, end_year=None):
    """Load data for modeling.
    
    Args:
        data_source (str): Source of data ('csv', 'db', or 'api').
        start_year (int, optional): Start year for API data.
        end_year (int, optional): End year for API data.
        
    Returns:
        pd.DataFrame: Data for modeling.
    """
    logger.info(f"Loading data from {data_source}")
    
    try:
        if data_source == 'csv':
            # Load from CSV
            data = load_data_from_csv('cleaned_energy_data.csv')
            
            # Ensure value column exists
            if 'value' not in data.columns and 'total-consumption' in data.columns:
                data['value'] = pd.to_numeric(data['total-consumption'], errors='coerce')
                
        elif data_source == 'db':
            # Load from database
            data = load_data_from_db('energy_data.db')
            
            # Ensure value column exists
            if 'value' not in data.columns and 'total-consumption' in data.columns:
                data['value'] = pd.to_numeric(data['total-consumption'], errors='coerce')
                
        elif data_source == 'api':
            # Run ETL to get fresh data
            if start_year is None:
                start_year = datetime.now().year - 5
            if end_year is None:
                end_year = datetime.now().year
                
            data = run_etl(start_year, end_year)
        else:
            raise ValueError(f"Invalid data source: {data_source}")
            
        # Verify data has required columns
        if 'date' not in data.columns:
            raise ValueError("Data must contain 'date' column")
            
        if 'value' not in data.columns:
            if 'total-consumption' in data.columns:
                data['value'] = pd.to_numeric(data['total-consumption'], errors='coerce')
            else:
                raise ValueError("Data must contain 'value' or 'total-consumption' column")
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading modeling data: {e}")
        raise

def run_forecasting(data, model_type, forecast_periods):
    """Run forecasting models.
    
    Args:
        data (pd.DataFrame): Input data.
        model_type (str): Type of model ('arima' or 'sarimax').
        forecast_periods (int): Number of periods to forecast.
        
    Returns:
        tuple: (forecast_df, model, train_data, test_data, metrics)
    """
    logger.info(f"Running {model_type.upper()} forecasting for {forecast_periods} periods")
    
    try:
        # Prepare data for modeling
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
            
        # Sort by date
        data = data.sort_values('date')
        
        # Split data into training and testing
        train_size = 0.8
        split_idx = int(len(data) * train_size)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")
        
        # Initialize and fit model
        if model_type == 'arima':
            model = ARIMAModel(p=1, d=1, q=1)
            model.fit(train_data, value_column='value')
        elif model_type == 'sarimax':
            model = SARIMAXModel(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
            model.fit(train_data, value_column='value')
        else:
            raise ValueError(f"Invalid model type: {model_type}")
            
        # Generate in-sample predictions for test data
        test_predictions = model.predict(periods=len(test_data))
        
        # Calculate metrics
        metrics = calculate_metrics(test_data['value'].values, test_predictions)
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Generate future forecast
        last_date = data['date'].iloc[-1]
        forecast_df = model.get_forecast_df(
            start_date=last_date + pd.Timedelta(days=1),
            periods=forecast_periods
        )
        
        return forecast_df, model, train_data, test_data, metrics
        
    except Exception as e:
        logger.error(f"Error running forecasting: {e}")
        raise

def create_visualizations(data, forecast_df, train_data, test_data, test_predictions, output_dir):
    """Create visualizations.
    
    Args:
        data (pd.DataFrame): Full historical data.
        forecast_df (pd.DataFrame): Forecast data.
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        test_predictions (array-like): Predictions for test data.
        output_dir (str): Directory to save output files.
    """
    logger.info("Creating visualizations")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot historical data
        historical_fig = plot_time_series(
            data, 
            date_col='date', 
            value_col='value',
            title='Natural Gas Historical Data'
        )
        save_plot(historical_fig, os.path.join(output_dir, 'historical_data.png'))
        
        # 2. Plot forecasts
        forecast_fig = plot_forecast(
            data,
            forecast_df,
            date_col='date',
            hist_col='value',
            forecast_col='forecast',
            title='Natural Gas Forecast'
        )
        save_plot(forecast_fig, os.path.join(output_dir, 'forecast.png'))
        
        # 3. Create test data prediction plot
        # Create a DataFrame with test predictions
        test_pred_df = test_data.copy()
        test_pred_df['prediction'] = test_predictions
        
        test_fig = plot_forecast(
            train_data,
            test_pred_df,
            date_col='date',
            hist_col='value',
            forecast_col='prediction',
            title='Model Validation - Test Set Predictions'
        )
        save_plot(test_fig, os.path.join(output_dir, 'model_validation.png'))
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise

def export_results(forecast_df, metrics, output_dir):
    """Export results to files.
    
    Args:
        forecast_df (pd.DataFrame): Forecast data.
        metrics (dict): Model evaluation metrics.
        output_dir (str): Directory to save output files.
    """
    logger.info("Exporting results")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export forecast to CSV
        forecast_csv_path = os.path.join(output_dir, 'forecast.csv')
        forecast_df.to_csv(forecast_csv_path, index=False)
        logger.info(f"Forecast exported to {forecast_csv_path}")
        
        # Export metrics to JSON
        metrics_json_path = os.path.join(output_dir, 'metrics.json')
        export_results_to_json(metrics, metrics_json_path)
        logger.info(f"Metrics exported to {metrics_json_path}")
        
        # Create a summary report
        report_data = {
            'forecast_summary': {
                'periods': len(forecast_df),
                'start_date': forecast_df['date'].min().isoformat(),
                'end_date': forecast_df['date'].max().isoformat(),
                'min_value': float(forecast_df['forecast'].min()),
                'max_value': float(forecast_df['forecast'].max()),
                'mean_value': float(forecast_df['forecast'].mean())
            },
            'model_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        report_json_path = os.path.join(output_dir, 'report.json')
        export_results_to_json(report_data, report_json_path)
        logger.info(f"Summary report exported to {report_json_path}")
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise

def main():
    """Main function."""
    args = parse_args()
    
    try:
        logger.info("Starting Natural Gas Forecasting pipeline")
        logger.info(f"Mode: {args.mode}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run in specified mode
        if args.mode in ['etl', 'full']:
            data = run_etl(args.start_year, args.end_year)
        
        if args.mode in ['model', 'visualize', 'full']:
            if args.mode != 'etl':
                # Load data if not already loaded from ETL
                data = load_modeling_data(args.data_source, args.start_year, args.end_year)
                
            # Run forecasting
            forecast_df, model, train_data, test_data, metrics = run_forecasting(
                data, 
                args.model_type, 
                args.forecast_periods
            )
        
        if args.mode in ['visualize', 'full']:
            # Generate predictions for test data
            test_predictions = model.predict(periods=len(test_data))
            
            # Create visualizations
            create_visualizations(
                data, 
                forecast_df, 
                train_data, 
                test_data, 
                test_predictions,
                args.output_dir
            )
            
            # Export results
            export_results(forecast_df, metrics, args.output_dir)
        
        logger.info("Natural Gas Forecasting pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 