import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*60)
print("DAY 2: EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING")
print("="*60)

# ============================================
# LOAD DATA FROM POSTGRESQL
# ============================================
print("\n[1/8] Loading data from PostgreSQL...")
engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')

df = pd.read_sql("SELECT * FROM romania_energy ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"✓ Loaded {len(df):,} rows")
print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
print(f"✓ Duration: {(df.index.max() - df.index.min()).days} days")

# ============================================
# SEASONAL DECOMPOSITION
# ============================================
print("\n[2/8] Performing seasonal decomposition...")
print("   (This may take 1-2 minutes for 54k rows...)")

# Use weekly seasonality (24 hours * 7 days = 168)
decomposition = seasonal_decompose(df['consumption'], 
                                   model='additive',  # Additive: Trend + Seasonal + Residual
                                   period=24*7,       # Weekly pattern
                                   extrapolate_trend='freq')

# Create the decomposition plot
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Observed data
decomposition.observed.plot(ax=axes[0], color='#2E86AB', linewidth=0.5)
axes[0].set_ylabel('Observed', fontsize=11, fontweight='bold')
axes[0].set_title('Seasonal Decomposition of Electricity Consumption', 
                  fontsize=14, fontweight='bold', pad=20)
axes[0].grid(alpha=0.3)

# Trend component
decomposition.trend.plot(ax=axes[1], color='#E63946', linewidth=1)
axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.3)

# Seasonal component
decomposition.seasonal.plot(ax=axes[2], color='#06A77D', linewidth=0.5)
axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
axes[2].grid(alpha=0.3)

# Residual component
decomposition.resid.plot(ax=axes[3], color='#8B4513', linewidth=0.3, alpha=0.7)
axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_seasonality_decomposition.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/01_seasonality_decomposition.png")
plt.show()

# Explain what we see
print("\n" + "-"*60)
print("INTERPRETATION:")
print("-"*60)
print(f"• Trend: Shows long-term changes (increasing/decreasing demand)")
print(f"• Seasonal: Weekly repeating pattern (visible waves)")
print(f"• Residual: Random noise after removing trend + seasonal")
print(f"\nResidual std dev: {decomposition.resid.std():.2f} MW")
print(f"As % of mean consumption: {(decomposition.resid.std() / df['consumption'].mean() * 100):.2f}%")

# ============================================
# HOURLY CONSUMPTION PATTERNS
# ============================================
print("\n[3/8] Analyzing hourly patterns...")

df['hour'] = df.index.hour
hourly_stats = df.groupby('hour')['consumption'].agg(['mean', 'std', 'min', 'max'])

# Create hourly pattern plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Average consumption by hour
axes[0].bar(hourly_stats.index, hourly_stats['mean'], 
           color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('Average Electricity Consumption by Hour of Day', 
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hour of Day', fontsize=11)
axes[0].set_ylabel('Average Consumption (MW)', fontsize=11)
axes[0].set_xticks(range(24))
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(hourly_stats['mean']):
    axes[0].text(i, v + 50, f'{v:.0f}', ha='center', fontsize=8)

# Boxplot showing distribution by hour
df.boxplot(column='consumption', by='hour', ax=axes[1], 
          patch_artist=True, grid=False)
axes[1].set_title('Consumption Distribution by Hour (Boxplot)', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hour of Day', fontsize=11)
axes[1].set_ylabel('Consumption (MW)', fontsize=11)
axes[1].get_figure().suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('outputs/02_hourly_patterns.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/02_hourly_patterns.png")
plt.show()

# Print key insights
print("\n" + "-"*60)
print("HOURLY INSIGHTS:")
print("-"*60)
peak_hours = hourly_stats['mean'].nlargest(3)
low_hours = hourly_stats['mean'].nsmallest(3)

print("\n🔺 Peak Consumption Hours:")
for hour, consumption in peak_hours.items():
    print(f"   {hour:02d}:00 - {consumption:.0f} MW")

print("\n🔻 Low Consumption Hours:")
for hour, consumption in low_hours.items():
    print(f"   {hour:02d}:00 - {consumption:.0f} MW")

peak_to_trough = hourly_stats['mean'].max() - hourly_stats['mean'].min()
print(f"\n📊 Peak-to-Trough Difference: {peak_to_trough:.0f} MW ({peak_to_trough/hourly_stats['mean'].mean()*100:.1f}%)")

# ============================================
# DAY OF WEEK PATTERNS
# ============================================
print("\n[4/8] Analyzing weekly patterns...")

df['day_of_week'] = df.index.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_stats = df.groupby('day_of_week')['consumption'].agg(['mean', 'std'])

# Create weekly pattern plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
axes[0].bar(range(7), weekly_stats['mean'], 
           color=['#1f77b4']*5 + ['#ff7f0e']*2,  # Blue for weekdays, orange for weekend
           alpha=0.7, edgecolor='black')
axes[0].set_title('Average Consumption by Day of Week', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Day of Week', fontsize=11)
axes[0].set_ylabel('Average Consumption (MW)', fontsize=11)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(day_names, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(weekly_stats['mean']):
    axes[0].text(i, v + 30, f'{v:.0f}', ha='center', fontsize=9)

# Hourly heatmap by day of week
hourly_by_day = df.pivot_table(values='consumption', 
                                index='hour', 
                                columns='day_of_week', 
                                aggfunc='mean')

sns.heatmap(hourly_by_day, cmap='YlOrRd', ax=axes[1], 
           cbar_kws={'label': 'Consumption (MW)'}, fmt='.0f')
axes[1].set_title('Consumption Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day of Week', fontsize=11)
axes[1].set_ylabel('Hour of Day', fontsize=11)
axes[1].set_xticklabels(day_names, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('outputs/03_weekly_patterns.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/03_weekly_patterns.png")
plt.show()

# Print insights
print("\n" + "-"*60)
print("WEEKLY INSIGHTS:")
print("-"*60)
weekday_avg = df[df['day_of_week'] < 5]['consumption'].mean()
weekend_avg = df[df['day_of_week'] >= 5]['consumption'].mean()
difference = weekday_avg - weekend_avg
pct_difference = (difference / weekend_avg) * 100

print(f"\n📅 Weekday Average:  {weekday_avg:.0f} MW")
print(f"📅 Weekend Average:  {weekend_avg:.0f} MW")
print(f"📊 Difference:       {difference:.0f} MW ({pct_difference:.1f}%)")
print(f"\n💡 Weekdays consume {pct_difference:.1f}% MORE than weekends")
print(f"   (Due to industrial/commercial activity)")

# ============================================
# MONTHLY & SEASONAL PATTERNS
# ============================================
print("\n[5/8] Analyzing seasonal patterns...")

df['month'] = df.index.month
df['year'] = df.index.year

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Monthly statistics
monthly_stats = df.groupby('month')['consumption'].agg(['mean', 'std'])

# Create seasonal plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Average consumption by month
axes[0, 0].bar(range(1, 13), monthly_stats['mean'], 
              color='forestgreen', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Average Consumption by Month', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Month', fontsize=11)
axes[0, 0].set_ylabel('Average Consumption (MW)', fontsize=11)
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(month_names, rotation=45, ha='right')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Consumption by month over years (to see trends)
for year in df['year'].unique():
    year_data = df[df['year'] == year].groupby('month')['consumption'].mean()
    axes[0, 1].plot(year_data.index, year_data.values, 
                   marker='o', label=str(year), alpha=0.7)

axes[0, 1].set_title('Monthly Consumption Trends by Year', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Month', fontsize=11)
axes[0, 1].set_ylabel('Average Consumption (MW)', fontsize=11)
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, rotation=45, ha='right')
axes[0, 1].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].grid(alpha=0.3)

# 3. Seasonal comparison
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

seasonal_avg = []
seasonal_names = []
for season, months in seasons.items():
    avg = df[df['month'].isin(months)]['consumption'].mean()
    seasonal_avg.append(avg)
    seasonal_names.append(season)

axes[1, 0].bar(seasonal_names, seasonal_avg, 
              color=['#A8DADC', '#457B9D', '#E63946', '#F4A261'], 
              edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Average Consumption by Season', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Average Consumption (MW)', fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(seasonal_avg):
    axes[1, 0].text(i, v + 30, f'{v:.0f} MW', ha='center', fontsize=10, fontweight='bold')

# 4. Monthly boxplot to show variability
df.boxplot(column='consumption', by='month', ax=axes[1, 1], patch_artist=True, grid=False)
axes[1, 1].set_title('Consumption Distribution by Month', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Month', fontsize=11)
axes[1, 1].set_ylabel('Consumption (MW)', fontsize=11)
axes[1, 1].set_xticklabels(month_names, rotation=45, ha='right')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('outputs/04_seasonal_patterns.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/04_seasonal_patterns.png")
plt.show()

# Print seasonal insights
print("\n" + "-"*60)
print("SEASONAL INSIGHTS:")
print("-"*60)
for season, months in seasons.items():
    avg = df[df['month'].isin(months)]['consumption'].mean()
    print(f"{season:10s}: {avg:,.0f} MW")

highest_season = max(seasons.keys(), key=lambda s: df[df['month'].isin(seasons[s])]['consumption'].mean())
lowest_season = min(seasons.keys(), key=lambda s: df[df['month'].isin(seasons[s])]['consumption'].mean())
print(f"\n🔥 Highest: {highest_season}")
print(f"❄️  Lowest:  {lowest_season}")

