import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*60)
print("PART 6: FEATURE ENGINEERING")
print("="*60)

# Load data
engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')
df = pd.read_sql("SELECT * FROM romania_energy ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"\n✓ Loaded {len(df):,} rows")
print(f"✓ Original columns: {len(df.columns)}")

# ============================================
# TIME-BASED FEATURES
# ============================================
print("\n[8/8] Creating features...")

# Basic time features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['day_of_year'] = df.index.dayofyear
df['week_of_year'] = df.index.isocalendar().week
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year

# Binary indicators
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
df['is_peak_morning'] = df['hour'].isin([7, 8, 9, 10]).astype(int)
df['is_peak_evening'] = df['hour'].isin([17, 18, 19, 20, 21]).astype(int)

# Season mapping
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
df['season'] = df['month'].map(season_map)

# Cyclical encoding (for neural nets, but good to have)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("✓ Time features created")

# ============================================
# ENERGY MIX FEATURES
# ============================================

# Handle negative wind (take absolute for production calculation)
df['wind_abs'] = df['wind'].abs()

# Renewable vs Fossil breakdown
df['renewable'] = df['wind_abs'] + df['hydroelectric'] + df['solar'] + df['biomass']
df['fossil'] = df['coal'] + df['oil_and_gas']

# Percentages (avoid division by zero)
df['renewable_pct'] = np.where(df['production'] > 0, 
                                (df['renewable'] / df['production']) * 100, 
                                0)
df['fossil_pct'] = np.where(df['production'] > 0,
                             (df['fossil'] / df['production']) * 100,
                             0)
df['nuclear_pct'] = np.where(df['production'] > 0,
                              (df['nuclear'] / df['production']) * 100,
                              0)

# Individual source percentages
for source in ['nuclear', 'coal', 'hydroelectric', 'oil_and_gas', 'solar', 'biomass']:
    df[f'{source}_pct'] = np.where(df['production'] > 0,
                                    (df[source] / df['production']) * 100,
                                    0)
df['wind_pct'] = np.where(df['production'] > 0,
                          (df['wind_abs'] / df['production']) * 100,
                          0)

# Energy balance
df['balance'] = df['production'] - df['consumption']
df['is_importing'] = (df['balance'] < 0).astype(int)
df['is_exporting'] = (df['balance'] > 0).astype(int)

print("✓ Energy mix features created")

# ============================================
# LAGGED FEATURES (For ML models, not ARIMA)
# ============================================

# Consumption lags
df['consumption_lag_1h'] = df['consumption'].shift(1)
df['consumption_lag_24h'] = df['consumption'].shift(24)
df['consumption_lag_48h'] = df['consumption'].shift(48)
df['consumption_lag_168h'] = df['consumption'].shift(168)

# Production lags
df['production_lag_1h'] = df['production'].shift(1)
df['production_lag_24h'] = df['production'].shift(24)

# Renewable lags (weather-dependent)
df['renewable_lag_1h'] = df['renewable'].shift(1)
df['renewable_lag_24h'] = df['renewable'].shift(24)

print("✓ Lagged features created")

# ============================================
# ROLLING STATISTICS
# ============================================

# Rolling means (moving averages)
df['consumption_rolling_3h'] = df['consumption'].rolling(window=3, min_periods=1).mean()
df['consumption_rolling_24h'] = df['consumption'].rolling(window=24, min_periods=1).mean()
df['consumption_rolling_168h'] = df['consumption'].rolling(window=168, min_periods=1).mean()

# Rolling standard deviations (volatility)
df['consumption_rolling_std_24h'] = df['consumption'].rolling(window=24, min_periods=1).std()

# Rate of change
df['consumption_change_1h'] = df['consumption'].diff(1)
df['consumption_change_24h'] = df['consumption'].diff(24)
df['consumption_pct_change_1h'] = df['consumption'].pct_change(1) * 100
df['consumption_pct_change_24h'] = df['consumption'].pct_change(24) * 100

print("✓ Rolling statistics created")

# ============================================
# SUMMARY STATISTICS
# ============================================

print(f"\n✓ Total features created: {len(df.columns)}")
print(f"✓ Original: 10, Added: {len(df.columns) - 10}")

print("\n" + "-"*60)
print("FEATURE CATEGORIES:")
print("-"*60)
print(f"Time features:         {sum('hour' in col or 'day' in col or 'week' in col or 'month' in col or 'year' in col or 'season' in col or 'quarter' in col or 'is_' in col or 'sin' in col or 'cos' in col for col in df.columns)}")
print(f"Energy mix features:   {sum('renewable' in col or 'fossil' in col or 'pct' in col or 'balance' in col for col in df.columns)}")
print(f"Lagged features:       {sum('lag' in col for col in df.columns)}")
print(f"Rolling features:      {sum('rolling' in col or 'change' in col for col in df.columns)}")

# ============================================
# DATA QUALITY CHECK
# ============================================

print("\n" + "-"*60)
print("DATA QUALITY CHECK:")
print("-"*60)

# Missing values (only lagged features should have NaN)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print(f"\nColumns with missing values:")
    print(missing.head(10))
    print(f"\n(Lagged and rolling features will have NaN at the start - this is expected)")
else:
    print("\n✓ No missing values!")

# ============================================
# VISUALIZE KEY FEATURES
# ============================================

print("\n" + "-"*60)
print("CREATING FEATURE VISUALIZATIONS:")
print("-"*60)

# 1. Energy Mix Over Time
fig, ax = plt.subplots(figsize=(14, 6))

# Sample data (plot every 24th hour for readability)
df_sample = df.iloc[::24].copy()

ax.stackplot(df_sample.index,
            df_sample['nuclear'],
            df_sample['coal'],
            df_sample['hydroelectric'],
            df_sample['oil_and_gas'],
            df_sample['renewable'] - df_sample['hydroelectric'],  # Wind+Solar+Biomass
            labels=['Nuclear', 'Coal', 'Hydroelectric', 'Oil & Gas', 'Wind/Solar/Biomass'],
            colors=['#FF6B35', '#8B4513', '#4ECDC4', '#95E1D3', '#F38181'],
            alpha=0.8)

ax.set_title('Energy Production Mix Evolution (Daily Averages)', fontsize=14, fontweight='bold')
ax.set_ylabel('Production (MW)', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/07_energy_mix_evolution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/07_energy_mix_evolution.png")
plt.show()

# 2. Renewable Percentage Over Time
fig, ax = plt.subplots(figsize=(14, 6))

df_sample['renewable_pct'].plot(ax=ax, color='#06A77D', linewidth=0.5, alpha=0.7)
df_sample['renewable_pct'].rolling(168).mean().plot(ax=ax, color='#E63946', 
                                                     linewidth=2, label='7-day average')

ax.set_title('Renewable Energy Percentage Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Renewable %', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/08_renewable_percentage.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/08_renewable_percentage.png")
plt.show()

# 3. Consumption patterns by time features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By hour
df.groupby('hour')['consumption'].mean().plot(kind='bar', ax=axes[0,0], color='steelblue', alpha=0.8)
axes[0,0].set_title('Average Consumption by Hour', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Hour')
axes[0,0].set_ylabel('MW')
axes[0,0].grid(axis='y', alpha=0.3)

# By day of week
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df.groupby('day_of_week')['consumption'].mean().plot(kind='bar', ax=axes[0,1], 
                                                       color=['steelblue']*5 + ['coral']*2,
                                                       alpha=0.8)
axes[0,1].set_title('Average Consumption by Day of Week', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('Day')
axes[0,1].set_ylabel('MW')
axes[0,1].set_xticklabels(day_names, rotation=45)
axes[0,1].grid(axis='y', alpha=0.3)

# By month
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df.groupby('month')['consumption'].mean().plot(kind='bar', ax=axes[1,0], color='forestgreen', alpha=0.8)
axes[1,0].set_title('Average Consumption by Month', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('MW')
axes[1,0].set_xticklabels(month_names, rotation=45)
axes[1,0].grid(axis='y', alpha=0.3)

# By season
df.groupby('season')['consumption'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall']).plot(
    kind='bar', ax=axes[1,1], 
    color=['#A8DADC', '#457B9D', '#E63946', '#F4A261'],
    alpha=0.8)
axes[1,1].set_title('Average Consumption by Season', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('Season')
axes[1,1].set_ylabel('MW')
axes[1,1].set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'], rotation=45)
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/09_consumption_by_time_features.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/09_consumption_by_time_features.png")
plt.show()

# ============================================
# SAVE PROCESSED DATA
# ============================================

print("\n" + "-"*60)
print("SAVING PROCESSED DATA:")
print("-"*60)

# Save to PostgreSQL
df.to_sql('romania_energy_processed', engine, if_exists='replace', index=True)
print("✓ Saved to PostgreSQL: 'romania_energy_processed'")

# Save to CSV
df.to_csv('data/processed/romania_energy_processed.csv')
print("✓ Saved to CSV: 'data/processed/romania_energy_processed.csv'")

# Save feature list for documentation
feature_list = pd.DataFrame({
    'Feature': df.columns,
    'Data Type': df.dtypes.values,
    'Missing Values': df.isnull().sum().values,
    'Missing %': (df.isnull().sum() / len(df) * 100).values
})
feature_list.to_csv('data/processed/feature_list.csv', index=False)
print("✓ Saved feature list: 'data/processed/feature_list.csv'")

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE!")
print("="*60)
print(f"\n✅ Total features: {len(df.columns)}")
print(f"✅ Data ready for modeling")
print(f"✅ Files saved in 'data/processed/'")

print("\n" + "-"*60)
print("NEXT STEPS:")
print("-"*60)
print("📌 DAY 3: Build ARIMA and Prophet models")
print("📌 DAY 4: Compare models and tune parameters")
print("📌 DAY 5: Create Dagster pipeline")
print("📌 DAY 6: Build Tableau dashboard")
print("📌 DAY 7: Documentation and presentation")