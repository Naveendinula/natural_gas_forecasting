"""
Script to replace cost values with zero for specified renewable fuel types in the energy dataset.
"""

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of renewable fuel types that should have cost set to zero
RENEWABLE_FUEL_TYPES = [
    'renewable',
    'wind',
    'conventional hydroelectric',
    'solar photovoltaic',
    'other renewables',
    'hydro-electric pumped storage',
    'on shore wind turbine',
    'solar',
    'solar thermal',
    'estimated small scale solar photovoltaic',
    'estimated total solar',
    'estimated total solar photovoltaic',
    'offshore wind turbine'
]

def update_renewable_costs(input_csv, output_csv):
    """
    Update the energy dataset to set cost=0 for specified renewable fuel types.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_csv (str): Path to the output CSV file
    """
    try:
        print(f"Reading data from {input_csv}")
        logger.info(f"Reading data from {input_csv}")
        
        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"ERROR: Input file not found: {input_csv}")
            raise FileNotFoundError(f"Input file not found: {input_csv}")
        
        # Read the CSV file
        print("Loading CSV file...")
        df = pd.read_csv(input_csv)
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns in the dataset: {df.columns.tolist()}")
        logger.info(f"Original data shape: {df.shape}")
        
        # Using fuelTypeDescription instead of fuel
        fuel_column = 'fuelTypeDescription'
        
        # Check if 'fuelTypeDescription' column exists
        if fuel_column not in df.columns:
            print(f"WARNING: '{fuel_column}' column not found in the dataset")
            print(f"Available columns: {df.columns.tolist()}")
            logger.warning(f"'{fuel_column}' column not found in the dataset")
            return
        
        # Check if 'cost' column exists
        if 'cost' not in df.columns:
            print("WARNING: 'cost' column not found in the dataset")
            print(f"Available columns: {df.columns.tolist()}")
            logger.warning("'cost' column not found in the dataset")
            return
        
        # Count records before update
        print("Identifying renewable fuel types...")
        renewable_mask = df[fuel_column].str.lower().isin([fuel.lower() for fuel in RENEWABLE_FUEL_TYPES])
        records_to_update = renewable_mask.sum()
        
        print(f"Found {records_to_update} records to update")
        logger.info(f"Found {records_to_update} records to update")
        
        # Update cost to zero for renewable fuel types (case insensitive)
        if records_to_update > 0:
            # Create a backup of the original values
            df['original_cost'] = df['cost']
            
            # Set cost to zero for renewable fuel types
            df.loc[renewable_mask, 'cost'] = 0
            
            print(f"Updated {records_to_update} records with cost set to zero")
            logger.info(f"Updated {records_to_update} records with cost set to zero")
        else:
            print("No records found for the specified renewable fuel types")
            print("Sample fuel types in the dataset:")
            print(df[fuel_column].value_counts().head(10))
            logger.info("No records found for the specified renewable fuel types")
        
        # Save the updated data to the output CSV file
        print(f"Saving updated data to {output_csv}")
        logger.info(f"Saving updated data to {output_csv}")
        df.to_csv(output_csv, index=False)
        
        # Print summary of changes
        print("Summary of changes:")
        logger.info("Summary of changes:")
        for fuel_type in RENEWABLE_FUEL_TYPES:
            count = df[df[fuel_column].str.lower() == fuel_type.lower()].shape[0]
            if count > 0:
                print(f"  - {fuel_type}: {count} records updated")
                logger.info(f"  - {fuel_type}: {count} records updated")
        
        print(f"Data successfully saved to {output_csv}")
        logger.info(f"Data successfully saved to {output_csv}")
        
        return df
        
    except Exception as e:
        print(f"ERROR updating renewable costs: {e}")
        import traceback
        print(traceback.format_exc())
        logger.error(f"Error updating renewable costs: {e}")
        raise

if __name__ == "__main__":
    print("Starting renewable cost update process...")
    
    # Define input and output file paths
    input_csv = "new_cleaned_energy_data.csv"
    output_csv = "renewable_cost_updated.csv"
    
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    
    # Update renewable costs
    try:
        updated_df = update_renewable_costs(input_csv, output_csv)
        
        # Print sample of updated data
        if updated_df is not None:
            fuel_column = 'fuelTypeDescription'
            renewable_mask = updated_df[fuel_column].str.lower().isin([fuel.lower() for fuel in RENEWABLE_FUEL_TYPES])
            updated_sample = updated_df[renewable_mask].head(10)
            
            if not updated_sample.empty:
                print("\nSample of updated renewable records:")
                print(updated_sample[[fuel_column, 'cost', 'original_cost']].to_string())
        
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"ERROR: Process failed: {e}")
        import traceback
        print(traceback.format_exc()) 