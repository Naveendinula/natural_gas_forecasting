# Natural Gas Forecasting Project

A comprehensive system for forecasting natural gas consumption using time series analysis and forecasting techniques.

## Project Structure

```
natural_gas_forecasting/
├── src/                      # Source code
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py
│   │   └── etl.py            # ETL pipeline
│   ├── models/               # Forecasting models
│   │   ├── __init__.py
│   │   └── forecasting.py    # Time series models
│   ├── visualization/        # Visualization modules
│   │   ├── __init__.py
│   │   └── plots.py          # Plotting utilities
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py        # Helper functions
│   └── __init__.py
├── main.py                   # Main entry point
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables (API keys)
├── README.md                 # Project documentation
└── output/                   # Output directory for results (created at runtime)
```

## Features

- **ETL Pipeline**: Extract natural gas data from the EIA API, transform it, and load it into CSV and SQLite database
- **Forecasting Models**: ARIMA and SARIMAX time series forecasting models
- **Visualization**: Create plots of historical data, forecasts, and model evaluations
- **Helper Utilities**: Functions for data loading, metrics calculation, and result exporting

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/natural_gas_forecasting.git
   cd natural_gas_forecasting
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your EIA API key:
   ```
   # .env
   EIA_API_KEY=your_api_key_here
   ```

## Usage

The project can be run in different modes using the `main.py` script:

1. Run the complete pipeline (ETL + modeling + visualization):
   ```
   python main.py --mode full
   ```

2. Run only the ETL process:
   ```
   python main.py --mode etl
   ```

3. Run only the modeling process:
   ```
   python main.py --mode model
   ```

4. Run only the visualization process:
   ```
   python main.py --mode visualize
   ```

### Command Line Options

- `--mode`: Pipeline mode (`etl`, `model`, `visualize`, or `full`)
- `--start-year`: Start year for data extraction (default: 5 years ago)
- `--end-year`: End year for data extraction (default: current year)
- `--forecast-periods`: Number of periods to forecast (default: 12)
- `--model-type`: Type of forecasting model (`arima` or `sarimax`)
- `--data-source`: Source of data for modeling (`csv`, `db`, or `api`)
- `--output-dir`: Directory to save output files (default: `output`)

Example:
```
python main.py --mode full --model-type sarimax --forecast-periods 24 --start-year 2015 --end-year 2023 --output-dir results
```

## Output

The pipeline generates the following outputs in the specified output directory:

- **Historical data plot**: `historical_data.png`
- **Forecast plot**: `forecast.png`
- **Model validation plot**: `model_validation.png`
- **Forecast CSV**: `forecast.csv`
- **Metrics JSON**: `metrics.json`
- **Summary report**: `report.json`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [U.S. Energy Information Administration (EIA)](https://www.eia.gov/) for providing the API and data
- [statsmodels](https://www.statsmodels.org/) for time series modeling
- [pandas](https://pandas.pydata.org/) for data manipulation
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualization
