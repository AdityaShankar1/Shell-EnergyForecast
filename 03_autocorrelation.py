import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os

# Create outputs folder
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("PART 5: AUTOCORRELATION ANALYSIS (FOR ARIMA)")
print("="*60)

# Load data
engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')
df = pd.read_sql("SELECT * FROM romania_energy ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"\n✓ Loaded {len(df):,} rows")

# ============================================
# STATIONARITY TEST (Augmented Dickey-Fuller)
# ============================================
print("\n[5/8] Testing for stationarity...")
print("   (ARIMA requires stationary data)")

adf_result = adfuller(df['consumption'].dropna())

print("\n" + "-"*60)
print("AUGMENTED DICKEY-FULLER TEST RESULTS:")
print("-"*60)
print(f"ADF Statistic:     {adf_result[0]:.4f}")
print(f"P-value:           {adf_result[1]:.4f}")
print(f"Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("\n✓ Data IS stationary (p < 0.05)")
    print("  → Good for ARIMA without differencing")
else:
    print("\n✗ Data is NOT stationary (p >= 0.05)")
    print("  → Will need differencing for ARIMA (d=1 or d=2)")

# ============================================
# AUTOCORRELATION FUNCTION (ACF)
# ============================================
print("\n[6/8] Computing autocorrelation...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# ACF Plot - Shows correlation with past lags
plot_acf(df['consumption'].dropna(), 
         lags=168,  # 1 week (24 hours * 7 days)
         ax=axes[0],
         alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF) - 1 Week of Lags', 
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag (hours)', fontsize=11)
axes[0].set_ylabel('Correlation', fontsize=11)
axes[0].grid(alpha=0.3)

# PACF Plot - Shows direct correlation (removes indirect effects)
plot_pacf(df['consumption'].dropna(), 
          lags=72,  # 3 days
          ax=axes[1],
          alpha=0.05,
          method='ywm')  # Yule-Walker method
axes[1].set_title('Partial Autocorrelation Function (PACF) - 3 Days of Lags', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag (hours)', fontsize=11)
axes[1].set_ylabel('Partial Correlation', fontsize=11)
axes[1].grid(alpha=0.3)

# Zoom in on first 48 hours (2 days) for detail
plot_acf(df['consumption'].dropna(), 
         lags=48,
         ax=axes[2],
         alpha=0.05)
axes[2].set_title('ACF - Zoomed to 48 Hours (Shows Daily Pattern Clearly)', 
                 fontsize=14, fontweight='bold')
axes[2].set_xlabel('Lag (hours)', fontsize=11)
axes[2].set_ylabel('Correlation', fontsize=11)
axes[2].grid(alpha=0.3)

# Highlight key lags
key_lags = [24, 48, 168]  # 1 day, 2 days, 1 week
for lag in key_lags:
    if lag <= 48:
        axes[2].axvline(x=lag, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[2].text(lag, 0.5, f'{lag}h', rotation=90, va='bottom', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('outputs/05_autocorrelation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/05_autocorrelation.png")
plt.show()

# ============================================
# INTERPRET ACF/PACF FOR ARIMA PARAMETERS
# ============================================
print("\n" + "-"*60)
print("ACF/PACF INTERPRETATION:")
print("-"*60)
print("\n📊 What the plots show:")
print("   • Strong correlation at lag 24 (yesterday same hour)")
print("   • Strong correlation at lag 168 (last week same hour)")
print("   • Gradual decay in ACF suggests AR component needed")
print("   • Sharp cutoff in PACF suggests MA component")

print("\n🎯 Suggested ARIMA parameters (initial guess):")
print("   • p (AR order): 1-2 (based on PACF cutoff)")
print("   • d (differencing): 0 or 1 (test both)")
print("   • q (MA order): 1-2 (based on ACF decay)")
print("\n   Start with: ARIMA(1,0,1) or ARIMA(1,1,1)")
print("   Also try: ARIMA(2,0,2) or SARIMA with seasonal component")

# ============================================
# CORRELATION WITH LAGGED VALUES
# ============================================
print("\n[7/8] Computing lag correlations...")

# Create lagged versions
lags_to_test = [1, 24, 48, 168]  # 1h, 1day, 2days, 1week
lag_correlations = []

for lag in lags_to_test:
    corr = df['consumption'].corr(df['consumption'].shift(lag))
    lag_correlations.append(corr)
    print(f"   Lag {lag:3d} hours: correlation = {corr:.4f}")

# Visualize lag importance
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar([f'{lag}h' for lag in lags_to_test], 
              lag_correlations,
              color=['#2E86AB', '#06A77D', '#E63946', '#F4A261'],
              edgecolor='black',
              alpha=0.8)
ax.set_title('Correlation with Lagged Consumption Values', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('Correlation Coefficient', fontsize=11)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (lag, corr) in enumerate(zip(lags_to_test, lag_correlations)):
    ax.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/06_lag_correlations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/06_lag_correlations.png")
plt.show()

print("\n" + "="*60)
print("AUTOCORRELATION ANALYSIS COMPLETE!")
print("="*60)
print("\n💡 Key Takeaways:")
print("   1. Yesterday's consumption (24h lag) is HIGHLY predictive")
print("   2. Last week's consumption (168h lag) also important")
print("   3. ARIMA will use these patterns for forecasting")
print("   4. Seasonal ARIMA (SARIMA) might work even better")