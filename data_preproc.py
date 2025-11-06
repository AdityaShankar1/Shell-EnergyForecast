import pandas as pd

FILE_PATH = "World Energy Consumption.csv"

def load_and_inspect(file_path):
    """Loads the CSV and performs an initial inspection."""
    print(f"--- 1. Loading and Inspecting {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    print(f"Shape: {df.shape} (Rows, Columns)")
    
    # Show all columns, their types, and non-null counts
    print("\nDataFrame Info (Top 10 columns):")
    print(df.info(max_cols=10, verbose=True))
    
    # Get a report of missing data (this dataset has a lot)
    print("\nMissing Value Report (Top 10 columns with most missing data):")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing_pct.head(10))
    
    return df

def preprocess_for_analysis(df):
    """Filters, selects, and cleans the data for a specific analysis."""
    print("\n--- 2. Preprocessing for Analysis ---")
    
    # This dataset has over 100 columns. Let's select a few key ones.
    columns_of_interest = [
        'country', 'year', 'population', 'gdp',
        'electricity_generation', 'fossil_fuel_consumption',
        'solar_consumption', 'wind_consumption', 'hydro_consumption',
        'energy_per_capita', 'energy_per_gdp'
    ]
    
    # Filter out non-existent columns (in case the CSV is different)
    columns_to_keep = [col for col in columns_of_interest if col in df.columns]
    
    if len(columns_to_keep) < 3: # Need at least country and year
        print("Error: Key columns 'country' or 'year' not found.")
        return None
        
    df_clean = df[columns_to_keep].copy()
    
    # This dataset includes continents and "World". Let's filter for a single country.
    # Since you are in India, let's analyze 'India'
    df_india = df_clean[df_clean['country'] == 'India'].copy()
    
    if df_india.empty:
        print("Error: Could not find data for 'India'.")
        return None
        
    # For time-series, we should sort by year and set it as the index
    df_india = df_india.sort_values(by='year')
    
    # Convert 'year' to a proper datetime object (YYYY-01-01)
    df_india['year'] = pd.to_datetime(df_india['year'], format='%Y')
    df_india.set_index('year', inplace=True)
    
    # Fill missing consumption data with 0 (a reasonable assumption for this dataset)
    df_india.fillna(0, inplace=True)
    
    print("Successfully preprocessed and filtered data for 'India'.")
    return df_india

if __name__ == "__main__":
    raw_df = load_and_inspect(FILE_PATH)
    
    if raw_df is not None:
        processed_df = preprocess_for_analysis(raw_df)
        
        if processed_df is not None:
            print("\n--- 3. Preprocessed Data for 'India' (Last 5 Years) ---")
            print(processed_df.tail().to_markdown())
