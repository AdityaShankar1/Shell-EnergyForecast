import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv('romania_energy.csv')

# Display initial info
print(f"\nOriginal data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"\nCleaned columns: {list(df.columns)}")

# Convert datetime - IT'S ALREADY IN ISO FORMAT!
print("\nConverting datetime...")
df['datetime'] = pd.to_datetime(df['datetime'])  # No format needed!

# Check for any parsing errors
if df['datetime'].isnull().any():
    print(f"WARNING: {df['datetime'].isnull().sum()} datetime values failed to parse")
    print(df[df['datetime'].isnull()])
else:
    print("✓ All datetime values parsed successfully")

# Sort by datetime
df = df.sort_values('datetime')

# Check for duplicates
duplicates = df.duplicated(subset=['datetime']).sum()
if duplicates > 0:
    print(f"\nWARNING: Found {duplicates} duplicate timestamps")
    print("Removing duplicates, keeping first occurrence...")
    df = df.drop_duplicates(subset=['datetime'], keep='first')
else:
    print("✓ No duplicate timestamps")

print(f"\nFinal data shape: {df.shape}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Total hours: {len(df)}")
print(f"Total days: {len(df) / 24:.1f}")

# Basic data quality checks
print("\n" + "="*50)
print("DATA QUALITY CHECKS")
print("="*50)

print("\nValue ranges:")
for col in ['consumption', 'production', 'nuclear', 'wind', 'hydroelectric', 'coal', 'solar']:
    print(f"{col:15s}: {df[col].min():6.0f} to {df[col].max():6.0f} MW")

print("\nNegative values check:")
for col in df.columns:
    if col != 'datetime':
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  {col}: {neg_count} negative values")

# Connect to PostgreSQL
print("\n" + "="*50)
print("CONNECTING TO POSTGRESQL")
print("="*50)

# UPDATE THIS WITH YOUR PASSWORD!
engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')

# Test connection
try:
    with engine.connect() as conn:
        print("✓ Connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nPlease update the password in the script!")
    exit()

# Load data into PostgreSQL
print("\nLoading data to PostgreSQL...")
try:
    df.to_sql('romania_energy', 
              engine, 
              if_exists='replace',  # This will drop and recreate the table
              index=False, 
              method='multi', 
              chunksize=1000)
    print(f"✓ Successfully loaded {len(df)} rows into 'romania_energy' table")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

# Verify the load
print("\n" + "="*50)
print("VERIFYING DATA IN POSTGRESQL")
print("="*50)

verify_query = """
SELECT 
    COUNT(*) as total_rows,
    MIN(datetime) as start_date,
    MAX(datetime) as end_date,
    ROUND(AVG(consumption)::numeric, 2) as avg_consumption,
    ROUND(AVG(production)::numeric, 2) as avg_production,
    MIN(wind) as min_wind,
    MAX(wind) as max_wind
FROM romania_energy
"""

verification = pd.read_sql(verify_query, engine)
print("\nDatabase verification:")
print(verification.to_string(index=False))

# Sample data check
sample_query = "SELECT * FROM romania_energy ORDER BY datetime LIMIT 5"
sample = pd.read_sql(sample_query, engine)
print("\nFirst 5 rows in database:")
print(sample)

print("\n" + "="*50)
print("DATA LOAD COMPLETE!")
print("="*50)
print(f"\n✓ {len(df)} rows loaded")
print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"✓ {len(df) / 24:.1f} days of hourly data")