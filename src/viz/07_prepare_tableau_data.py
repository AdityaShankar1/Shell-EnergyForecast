import pandas as pd
from sqlalchemy import create_engine
import os

os.makedirs('data/tableau', exist_ok=True)

print("="*70)
print("PREPARING DATA FOR TABLEAU DASHBOARD")
print("="*70)

engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')

# ============================================
# 1. MAIN CONSUMPTION DATA (sampled for performance)
# ============================================
print("\n[1/5] Preparing main consumption dataset...")

# Load full processed data, sample every 6 hours for Tableau performance
query = """
SELECT 
    datetime,
    consumption,
    production,
    nuclear,
    wind,
    hydroelectric,
    oil_and_gas,
    coal,
    solar,
    biomass,
    renewable,
    fossil,
    renewable_pct,
    fossil_pct,
    nuclear_pct,
    balance,
    hour,
    day_of_week,
    month,
    season,
    is_weekend
FROM romania_energy_processed
ORDER BY datetime
"""

df_full = pd.read_sql(query, engine)
df_full['datetime'] = pd.to_datetime(df_full['datetime'])

# For Tableau: Use every 4th hour (6 hours of data per day for faster loading)
df_tableau = df_full.iloc[::4].copy()

print(f"✓ Original: {len(df_full):,} rows")
print(f"✓ Sampled:  {len(df_tableau):,} rows (every 4 hours)")

df_tableau.to_csv('data/tableau/01_consumption_data.csv', index=False)
print("✓ Saved: data/tableau/01_consumption_data.csv")

# ============================================
# 2. FORECAST RESULTS (All models)
# ============================================
print("\n[2/5] Preparing forecast comparison data...")

# Load ARIMA/SARIMA results
arima_results = pd.read_csv('data/processed/arima_results.csv')
prophet_results = pd.read_csv('data/processed/prophet_results.csv')

# Combine forecasts
forecast_combined = pd.DataFrame({
    'datetime': pd.to_datetime(arima_results['datetime']),
    'actual': arima_results['actual'],
    'arima_forecast': arima_results['arima_forecast'],
    'sarima_forecast': arima_results['sarima_forecast'],
    'prophet_forecast': prophet_results['prophet_forecast'],
    'arima_error': arima_results['arima_error'],
    'sarima_error': arima_results['sarima_error'],
    'prophet_error': prophet_results['prophet_error']
})

forecast_combined.to_csv('data/tableau/02_forecast_comparison.csv', index=False)
print("✓ Saved: data/tableau/02_forecast_comparison.csv")

# ============================================
# 3. MODEL PERFORMANCE METRICS
# ============================================
print("\n[3/5] Preparing model performance data...")

model_comparison = pd.read_csv('data/processed/model_comparison_final.csv')
model_comparison.to_csv('data/tableau/03_model_performance.csv', index=False)
print("✓ Saved: data/tableau/03_model_performance.csv")

# ============================================
# 4. AGGREGATED STATISTICS (for faster dashboard)
# ============================================
print("\n[4/5] Creating aggregated statistics...")

# Hourly aggregates
hourly_stats = df_full.groupby('hour').agg({
    'consumption': ['mean', 'min', 'max', 'std'],
    'renewable_pct': 'mean',
    'fossil_pct': 'mean'
}).round(2)
hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
hourly_stats = hourly_stats.reset_index()
hourly_stats.to_csv('data/tableau/04_hourly_stats.csv', index=False)
print("✓ Saved: data/tableau/04_hourly_stats.csv")

# Daily aggregates
df_full['date'] = df_full['datetime'].dt.date
daily_stats = df_full.groupby('date').agg({
    'consumption': 'mean',
    'production': 'mean',
    'renewable': 'mean',
    'fossil': 'mean',
    'nuclear': 'mean',
    'renewable_pct': 'mean'
}).round(2).reset_index()
daily_stats.to_csv('data/tableau/05_daily_stats.csv', index=False)
print("✓ Saved: data/tableau/05_daily_stats.csv")

# Monthly aggregates
df_full['year'] = df_full['datetime'].dt.year
df_full['month'] = df_full['datetime'].dt.month

monthly_stats = df_full.groupby(['year', 'month']).agg({
    'consumption': 'mean',
    'renewable_pct': 'mean',
    'fossil_pct': 'mean',
    'nuclear_pct': 'mean'
}).round(2).reset_index()
monthly_stats.to_csv('data/tableau/06_monthly_stats.csv', index=False)
print("✓ Saved: data/tableau/06_monthly_stats.csv")

# ============================================
# 5. ENERGY MIX BREAKDOWN (for stacked charts)
# ============================================
print("\n[5/5] Creating energy mix breakdown...")

# Reshape data for Tableau stacked area charts
df_energy_mix = df_tableau[['datetime', 'nuclear', 'coal', 'hydroelectric', 
                             'oil_and_gas', 'wind', 'solar', 'biomass']].copy()

# Melt for Tableau
df_energy_mix_long = df_energy_mix.melt(
    id_vars=['datetime'],
    var_name='energy_source',
    value_name='production_mw'
)

df_energy_mix_long['energy_type'] = df_energy_mix_long['energy_source'].map({
    'nuclear': 'Nuclear',
    'coal': 'Fossil',
    'oil_and_gas': 'Fossil',
    'wind': 'Renewable',
    'solar': 'Renewable',
    'hydroelectric': 'Renewable',
    'biomass': 'Renewable'
})

df_energy_mix_long.to_csv('data/tableau/07_energy_mix_breakdown.csv', index=False)
print("✓ Saved: data/tableau/07_energy_mix_breakdown.csv")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)
print("\n✅ 7 CSV files created in 'data/tableau/'")
print("✅ Data optimized for Tableau performance")
print("✅ Ready to import into Tableau Public")

print("\n" + "-"*70)
print("FILES CREATED:")
print("-"*70)
print("01_consumption_data.csv      - Main time series (sampled)")
print("02_forecast_comparison.csv   - All model forecasts")
print("03_model_performance.csv     - Model metrics comparison")
print("04_hourly_stats.csv          - Hourly aggregates")
print("05_daily_stats.csv           - Daily aggregates")
print("06_monthly_stats.csv         - Monthly aggregates")
print("07_energy_mix_breakdown.csv  - Energy source breakdown")

print("\n" + "="*70)
print("NEXT STEP: Open Tableau Public and import these files!")
print("="*70)