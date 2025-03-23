"""
Script to create a new cleaned CSV file and print information about it.
"""

import os
import pandas as pd
from datetime import datetime
from src.data.etl import run_etl_pipeline, DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_new_cleaned_csv():
    """Create a new cleaned CSV file with a different name."""
    try:
        # Set date range
        start_year = 2015
        end_year = 2023
        
        # Convert to date strings for the ETL pipeline
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        # Run ETL pipeline to get the processed data
        processed_data = run_etl_pipeline(start_date=start_date, end_date=end_date)
        
        # Create a custom data loader with the new filename
        new_csv_path = "new_cleaned_energy_data.csv"
        loader = DataLoader()
        loader.save_to_csv(processed_data, output_path=new_csv_path)
        
        logger.info(f"Created new CSV file: {new_csv_path}")
        
        return new_csv_path
    
    except Exception as e:
        logger.error(f"Error creating new CSV file: {e}")
        raise

def print_csv_info(csv_path):
    """Print detailed information about the CSV file."""
    try:
        logger.info(f"Loading and analyzing CSV file: {csv_path}")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        # Get file stats
        file_size_bytes = os.path.getsize(csv_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'period' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        # Print file information
        print("\n" + "="*80)
        print(f"CSV FILE INFORMATION: {csv_path}")
        print("="*80)
        print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
        print(f"Number of Rows: {len(df):,}")
        print(f"Number of Columns: {len(df.columns):,}")
        
        # Print column information
        print("\nCOLUMN INFORMATION:")
        print("-"*80)
        for col in df.columns:
            dtype = df[col].dtype
            num_nulls = df[col].isna().sum()
            percent_nulls = (num_nulls / len(df)) * 100
            
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                print(f"{col}: {dtype} | Nulls: {num_nulls:,} ({percent_nulls:.2f}%) | Min: {min_val} | Max: {max_val} | Mean: {mean_val:.2f}")
            else:
                unique_vals = df[col].nunique()
                print(f"{col}: {dtype} | Nulls: {num_nulls:,} ({percent_nulls:.2f}%) | Unique Values: {unique_vals:,}")
        
        # Print date range if date column exists
        if date_columns:
            print("\nDATE RANGE:")
            print("-"*80)
            for date_col in date_columns:
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                print(f"{date_col}: {min_date} to {max_date}")
        
        # Print sample data
        print("\nSAMPLE DATA (First 5 rows):")
        print("-"*80)
        print(df.head().to_string())
        
        # Print summary statistics for numerical columns
        print("\nSUMMARY STATISTICS:")
        print("-"*80)
        print(df.describe().to_string())
        
        # Print information about natural gas data
        if 'fuel' in df.columns and 'natural gas' in df['fuel'].values:
            ng_data = df[df['fuel'] == 'natural gas']
            print("\nNATURAL GAS DATA SUMMARY:")
            print("-"*80)
            print(f"Number of Natural Gas Records: {len(ng_data):,}")
            
            # If consumption data exists
            if 'total-consumption' in df.columns:
                print(f"Total Consumption (Sum): {ng_data['total-consumption'].sum():,.2f}")
                
                # Group by year or location if available
                if 'year' in df.columns:
                    yearly_data = ng_data.groupby('year')['total-consumption'].sum()
                    print("\nYearly Natural Gas Consumption:")
                    print(yearly_data.to_string())
                
                if 'location' in df.columns:
                    location_data = ng_data.groupby('location')['total-consumption'].sum().sort_values(ascending=False).head(10)
                    print("\nTop 10 Locations by Natural Gas Consumption:")
                    print(location_data.to_string())
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error analyzing CSV file: {e}")
        raise

if __name__ == "__main__":
    try:
        # Create new CSV file
        new_csv_path = create_new_cleaned_csv()
        
        # Print information about the new CSV file
        print_csv_info(new_csv_path)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"Error: {e}") 