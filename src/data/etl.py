"""
ETL Pipeline for Natural Gas Data

This module handles the extraction, transformation, and loading of natural gas data
from various sources, particularly the EIA API.
"""

import os
import json
import logging
import sqlite3
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("EIA_API_KEY")
if not API_KEY:
    raise ValueError("EIA_API_KEY not found in environment variables")

class EIADataExtractor:
    """Class to extract data from the EIA API."""
    
    BASE_URL = "https://api.eia.gov/v2/"
    
    def __init__(self, api_key=None):
        """Initialize the EIA data extractor.
        
        Args:
            api_key (str, optional): API key for EIA. Defaults to environment variable.
        """
        self.api_key = api_key or API_KEY
        
    def get_electricity_data(self, start_year=2010, end_year=None, limit=5000):
        """Get electricity data using pagination.
        
        Args:
            start_year (int, optional): Start year for data.
            end_year (int, optional): End year for data.
            limit (int, optional): Number of rows per request.
            
        Returns:
            pd.DataFrame: DataFrame containing the electricity data.
        """
        if not end_year:
            end_year = datetime.now().year
        
        endpoint = ('electricity/electric-power-operational-data/data/?frequency=annual'
                   '&data[0]=consumption-for-eg&data[1]=consumption-uto&data[2]=cost'
                   '&data[3]=generation&data[4]=heat-content&data[5]=receipts&data[6]=stocks'
                   '&data[7]=total-consumption&data[8]=total-consumption-btu'
                   '&facets[location][]=AK&facets[location][]=AL&facets[location][]=AR'
                   '&facets[location][]=AZ&facets[location][]=CA&facets[location][]=CO'
                   '&facets[location][]=CT&facets[location][]=DC&facets[location][]=DE'
                   '&facets[location][]=FL&facets[location][]=GA&facets[location][]=HI'
                   '&facets[location][]=IA&facets[location][]=ID&facets[location][]=IL'
                   '&facets[location][]=IN&facets[location][]=KS&facets[location][]=KY'
                   '&facets[location][]=LA&facets[location][]=MA&facets[location][]=MD'
                   '&facets[location][]=ME&facets[location][]=MI&facets[location][]=MN'
                   '&facets[location][]=MO&facets[location][]=MS&facets[location][]=MT'
                   '&facets[location][]=NC&facets[location][]=ND&facets[location][]=NE'
                   '&facets[location][]=NH&facets[location][]=NJ&facets[location][]=NM'
                   '&facets[location][]=NV&facets[location][]=NY&facets[location][]=OH'
                   '&facets[location][]=OK&facets[location][]=OR&facets[location][]=PA'
                   '&facets[location][]=PR&facets[location][]=RI&facets[location][]=SC'
                   '&facets[location][]=SD&facets[location][]=TN&facets[location][]=TX'
                   '&facets[location][]=US&facets[location][]=UT&facets[location][]=VA'
                   '&facets[location][]=VT&facets[location][]=WA&facets[location][]=WI'
                   '&facets[location][]=WV&facets[location][]=WY'
                   f'&start={start_year}&end={end_year}'
                   '&sort[0][column]=period&sort[0][direction]=asc'
                   '&sort[1][column]=consumption-for-eg&sort[1][direction]=asc')
        
        offset = 0
        all_data = []
        
        try:
            logger.info(f"Requesting electricity data from {start_year} to {end_year}")
            
            while True:
                # Construct the URL with updated offset and length parameters
                url = f"{self.BASE_URL}{endpoint}&offset={offset}&length={limit}"
                logger.info(f"Requesting data with offset: {offset}")
                
                response = requests.get(url, params={'api_key': self.api_key})
                response.raise_for_status()
                
                data = response.json()
                batch = data.get("response", {}).get("data", [])
                
                # Break the loop if no more data is returned
                if not batch:
                    break
                
                all_data.extend(batch)
                logger.info(f"Retrieved {len(batch)} records")
                
                # If the batch size is less than the limit, we've reached the last page
                if len(batch) < limit:
                    break
                
                # Increase the offset for the next iteration
                offset += limit
            
            logger.info(f"Total records retrieved: {len(all_data)}")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Convert period to date
            if 'period' in df.columns:
                df['date'] = pd.to_datetime(df['period'])
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from EIA API: {e}")
            raise
    
    def get_natural_gas_data(self, start_year=2010, end_year=None):
        """Get natural gas data.
        
        This is a wrapper around get_electricity_data that filters for natural gas.
        
        Args:
            start_year (int, optional): Start year for data.
            end_year (int, optional): End year for data.
            
        Returns:
            pd.DataFrame: DataFrame containing natural gas data.
        """
        # Get the complete dataset
        data = self.get_electricity_data(start_year, end_year)
        
        # Filter for natural gas data
        if 'fuel' in data.columns:
            natural_gas_data = data[data['fuel'] == 'natural gas'].copy()
            logger.info(f"Filtered {len(natural_gas_data)} natural gas records")
            return natural_gas_data
        else:
            logger.warning("No 'fuel' column found in data, returning all data")
            return data

class DataTransformer:
    """Class to transform the extracted data."""
    
    def __init__(self):
        """Initialize the data transformer."""
        pass
        
    def clean_data(self, df):
        """Clean the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        cleaned_df = cleaned_df.fillna(method='ffill')
        
        # Sort by date
        if 'date' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('date')
            
        return cleaned_df
        
    def add_features(self, df):
        """Add features to the dataframe for analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with additional features.
        """
        # Make a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Add year and month columns
        if 'date' in enhanced_df.columns:
            enhanced_df['year'] = enhanced_df['date'].dt.year
            enhanced_df['month'] = enhanced_df['date'].dt.month
            
            # Add season
            enhanced_df['season'] = enhanced_df['month'].apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                'Spring' if x in [3, 4, 5] else
                'Summer' if x in [6, 7, 8] else 'Fall'
            )
            
            # Convert value column if needed
            value_col = 'total-consumption'
            if value_col in enhanced_df.columns:
                # Ensure numeric data type
                enhanced_df['value'] = pd.to_numeric(enhanced_df[value_col], errors='coerce')
                
                # Add rolling averages
                enhanced_df['rolling_avg_3m'] = enhanced_df['value'].rolling(window=3).mean()
                enhanced_df['rolling_avg_12m'] = enhanced_df['value'].rolling(window=12).mean()
            
        return enhanced_df

class DataLoader:
    """Class to load the transformed data."""
    
    def __init__(self, db_path="energy_data.db"):
        """Initialize the data loader.
        
        Args:
            db_path (str, optional): Path to the SQLite database.
        """
        self.db_path = db_path
        
    def save_to_csv(self, df, output_path="cleaned_energy_data.csv"):
        """Save the dataframe to a CSV file.
        
        Args:
            df (pd.DataFrame): Dataframe to save.
            output_path (str, optional): Path to save the CSV file.
        """
        try:
            logger.info(f"Saving data to {output_path}")
            df.to_csv(output_path, index=False)
            logger.info(f"Data successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            raise
            
    def save_to_database(self, df, table_name="natural_gas_data"):
        """Save the dataframe to a SQLite database.
        
        Args:
            df (pd.DataFrame): Dataframe to save.
            table_name (str, optional): Name of the table.
        """
        try:
            logger.info(f"Saving data to database {self.db_path}, table {table_name}")
            
            # Create a connection to the database
            conn = sqlite3.connect(self.db_path)
            
            # Save the dataframe to the database
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
            # Close the connection
            conn.close()
            
            logger.info(f"Data successfully saved to database")
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            raise

def run_etl_pipeline(start_date=None, end_date=None):
    """Run the complete ETL pipeline.
    
    Args:
        start_date (str, optional): Start date in YYYY-MM-DD format.
        end_date (str, optional): End date in YYYY-MM-DD format.
        
    Returns:
        pd.DataFrame: The final processed dataframe.
    """
    try:
        logger.info("Starting ETL pipeline")
        
        # Convert dates to years for the API call
        start_year = 2010
        end_year = datetime.now().year
        
        if start_date:
            start_year = datetime.strptime(start_date, "%Y-%m-%d").year
        if end_date:
            end_year = datetime.strptime(end_date, "%Y-%m-%d").year
        
        # Extract
        extractor = EIADataExtractor()
        raw_data = extractor.get_natural_gas_data(start_year, end_year)
        logger.info(f"Extracted {len(raw_data)} records")
        
        # Transform
        transformer = DataTransformer()
        cleaned_data = transformer.clean_data(raw_data)
        processed_data = transformer.add_features(cleaned_data)
        logger.info("Data transformation complete")
        
        # Load
        loader = DataLoader()
        loader.save_to_csv(processed_data)
        loader.save_to_database(processed_data)
        logger.info("Data loading complete")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in ETL pipeline: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    today = datetime.now().strftime("%Y-%m-%d")
    one_year_ago = f"{datetime.now().year - 1}-{datetime.now().month:02d}-{datetime.now().day:02d}"
    
    processed_data = run_etl_pipeline(start_date=one_year_ago, end_date=today)
    print(f"Processed {len(processed_data)} records") 