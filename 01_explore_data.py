import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to database
engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')

# Load data
print("Loading data from PostgreSQL...")
df = pd.read_sql("SELECT * FROM romania_energy ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"Data loaded: {len(df)} rows")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Quick visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Consumption over time
axes[0].plot(df.index, df['consumption'], linewidth=0.5)
axes[0].set_title('Electricity Consumption Over Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MW')
axes[0].grid(alpha=0.3)

# Plot 2: Energy mix stacked area
energy_cols = ['nuclear', 'coal', 'hydroelectric', 'wind', 'solar', 'oil_and_gas', 'biomass']
axes[1].stackplot(df.index, 
                  df[energy_cols].T, 
                  labels=energy_cols,
                  alpha=0.8)
axes[1].set_title('Energy Production Mix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MW')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].grid(alpha=0.3)

# Plot 3: Production vs Consumption
axes[2].plot(df.index, df['consumption'], label='Consumption', linewidth=0.5)
axes[2].plot(df.index, df['production'], label='Production', linewidth=0.5)
axes[2].set_title('Production vs Consumption', fontsize=12, fontweight='bold')
axes[2].set_ylabel('MW')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('initial_exploration.png', dpi=150)
print("\nPlot saved as 'initial_exploration.png'")
plt.show()

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())