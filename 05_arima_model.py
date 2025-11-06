import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import os
import time

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*70)
print("DAY 3: ARIMA FORECASTING MODEL")
print("="*70)

# ============================================
# LOAD PROCESSED DATA
# ============================================
print("\n[1/6] Loading processed data...")

engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')
df = pd.read_sql("SELECT datetime, consumption FROM romania_energy_processed ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"✓ Loaded {len(df):,} rows")
print(f"✓ Date range: {df.index.min()} to {df.index.max()}")

# ============================================
# TRAIN-TEST SPLIT
# ============================================
print("\n[2/6] Splitting data into train and test sets...")

# Use last 7 days (168 hours) for testing
test_size = 168  # 1 week
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

print(f"✓ Training set: {len(train):,} rows ({train.index.min()} to {train.index.max()})")
print(f"✓ Test set:     {len(test):,} rows ({test.index.min()} to {test.index.max()})")
print(f"✓ Train/Test split: {len(train)/len(df)*100:.1f}% / {len(test)/len(df)*100:.1f}%")

# ============================================
# BASELINE MODEL (Naive Forecast)
# ============================================
print("\n[3/6] Creating baseline model (for comparison)...")

# Naive forecast: Use last week's same hour
baseline_predictions = []
for i in range(len(test)):
    # Get consumption from 168 hours ago (same day/hour last week)
    baseline_pred = train.iloc[-168 + i]['consumption']
    baseline_predictions.append(baseline_pred)

baseline_mae = mean_absolute_error(test['consumption'], baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(test['consumption'], baseline_predictions))
baseline_mape = mean_absolute_percentage_error(test['consumption'], baseline_predictions) * 100

print(f"\n📊 Baseline Model Performance (Last Week Same Hour):")
print(f"   MAE:  {baseline_mae:.2f} MW")
print(f"   RMSE: {baseline_rmse:.2f} MW")
print(f"   MAPE: {baseline_mape:.2f}%")
print(f"\n   → Our ARIMA model should beat this!")

# ============================================
# ARIMA MODEL 1: Simple ARIMA(1,0,1)
# ============================================
print("\n[4/6] Training ARIMA(1,0,1) model...")
print("   (This may take 5-10 minutes for 50k+ data points...)")

start_time = time.time()

# Fit ARIMA model
arima_model = ARIMA(train['consumption'], order=(1, 0, 1))
arima_fitted = arima_model.fit()

train_time = time.time() - start_time
print(f"✓ Model trained in {train_time:.1f} seconds")

# Model summary
print("\n" + "-"*70)
print("ARIMA MODEL SUMMARY:")
print("-"*70)
print(arima_fitted.summary())

# Make predictions
print("\n[5/6] Generating forecasts...")
arima_forecast = arima_fitted.forecast(steps=len(test))
arima_forecast_series = pd.Series(arima_forecast.values, index=test.index)

# Calculate metrics
arima_mae = mean_absolute_error(test['consumption'], arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test['consumption'], arima_forecast))
arima_mape = mean_absolute_percentage_error(test['consumption'], arima_forecast) * 100

print(f"\n📊 ARIMA(1,0,1) Performance:")
print(f"   MAE:  {arima_mae:.2f} MW")
print(f"   RMSE: {arima_rmse:.2f} MW")
print(f"   MAPE: {arima_mape:.2f}%")

# Compare with baseline
improvement = ((baseline_mape - arima_mape) / baseline_mape) * 100
print(f"\n   🎯 Improvement over baseline: {improvement:.1f}%")

# ============================================
# SARIMA MODEL: With Daily Seasonality
# ============================================
print("\n[4b/6] Training SARIMA(1,0,1)(1,0,1,24) model...")
print("   (This includes 24-hour seasonal component...)")

start_time = time.time()

# SARIMA with 24-hour seasonality
sarima_model = SARIMAX(train['consumption'], 
                       order=(1, 0, 1),
                       seasonal_order=(1, 0, 1, 24),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_fitted = sarima_model.fit(disp=False)

train_time = time.time() - start_time
print(f"✓ SARIMA model trained in {train_time:.1f} seconds")

# Make predictions
sarima_forecast = sarima_fitted.forecast(steps=len(test))
sarima_forecast_series = pd.Series(sarima_forecast.values, index=test.index)

# Calculate metrics
sarima_mae = mean_absolute_error(test['consumption'], sarima_forecast)
sarima_rmse = np.sqrt(mean_squared_error(test['consumption'], sarima_forecast))
sarima_mape = mean_absolute_percentage_error(test['consumption'], sarima_forecast) * 100

print(f"\n📊 SARIMA(1,0,1)(1,0,1,24) Performance:")
print(f"   MAE:  {sarima_mae:.2f} MW")
print(f"   RMSE: {sarima_rmse:.2f} MW")
print(f"   MAPE: {sarima_mape:.2f}%")

# Compare with baseline
sarima_improvement = ((baseline_mape - sarima_mape) / baseline_mape) * 100
print(f"\n   🎯 Improvement over baseline: {sarima_improvement:.1f}%")

# ============================================
# VISUALIZE FORECASTS
# ============================================
print("\n[6/6] Creating visualizations...")

# Plot 1: Full test period forecast
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ARIMA forecast
axes[0].plot(test.index, test['consumption'], 
            label='Actual', color='black', linewidth=2, alpha=0.7)
axes[0].plot(test.index, arima_forecast_series, 
            label='ARIMA(1,0,1)', color='#E63946', linewidth=1.5, linestyle='--')
axes[0].fill_between(test.index, 
                     test['consumption'], 
                     arima_forecast_series,
                     alpha=0.2, color='red')
axes[0].set_title('ARIMA(1,0,1) Forecast vs Actual (7-Day Test Period)', 
                 fontsize=14, fontweight='bold')
axes[0].set_ylabel('Consumption (MW)', fontsize=11)
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(alpha=0.3)

# Add metrics as text
axes[0].text(0.02, 0.95, 
            f'MAPE: {arima_mape:.2f}%\nMAE: {arima_mae:.0f} MW\nRMSE: {arima_rmse:.0f} MW',
            transform=axes[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# SARIMA forecast
axes[1].plot(test.index, test['consumption'], 
            label='Actual', color='black', linewidth=2, alpha=0.7)
axes[1].plot(test.index, sarima_forecast_series, 
            label='SARIMA(1,0,1)(1,0,1,24)', color='#06A77D', linewidth=1.5, linestyle='--')
axes[1].fill_between(test.index, 
                     test['consumption'], 
                     sarima_forecast_series,
                     alpha=0.2, color='green')
axes[1].set_title('SARIMA with 24-Hour Seasonality Forecast vs Actual', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Consumption (MW)', fontsize=11)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(alpha=0.3)

# Add metrics as text
axes[1].text(0.02, 0.95, 
            f'MAPE: {sarima_mape:.2f}%\nMAE: {sarima_mae:.0f} MW\nRMSE: {sarima_rmse:.0f} MW',
            transform=axes[1].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/10_arima_sarima_forecast.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/10_arima_sarima_forecast.png")
plt.show()

# Plot 2: Zoom into first 48 hours (2 days)
fig, ax = plt.subplots(figsize=(16, 6))

test_48h = test.iloc[:48]
arima_48h = arima_forecast_series.iloc[:48]
sarima_48h = sarima_forecast_series.iloc[:48]

ax.plot(test_48h.index, test_48h['consumption'], 
       label='Actual', color='black', linewidth=2.5, marker='o', markersize=3)
ax.plot(test_48h.index, arima_48h, 
       label='ARIMA(1,0,1)', color='#E63946', linewidth=2, 
       linestyle='--', marker='s', markersize=3)
ax.plot(test_48h.index, sarima_48h, 
       label='SARIMA(1,0,1)(1,0,1,24)', color='#06A77D', linewidth=2, 
       linestyle='--', marker='^', markersize=3)

ax.set_title('Detailed View: First 48 Hours of Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Date & Time', fontsize=11)
ax.set_ylabel('Consumption (MW)', fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.grid(alpha=0.3)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/11_forecast_48h_detail.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/11_forecast_48h_detail.png")
plt.show()

# Plot 3: Residual analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ARIMA residuals
arima_residuals = test['consumption'].values - arima_forecast
axes[0, 0].plot(test.index, arima_residuals, color='#E63946', linewidth=0.5, alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_title('ARIMA Residuals (Forecast Errors)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Error (MW)')
axes[0, 0].grid(alpha=0.3)

# ARIMA residual histogram
axes[0, 1].hist(arima_residuals, bins=30, color='#E63946', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('ARIMA Residual Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Error (MW)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(alpha=0.3)

# SARIMA residuals
sarima_residuals = test['consumption'].values - sarima_forecast
axes[1, 0].plot(test.index, sarima_residuals, color='#06A77D', linewidth=0.5, alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_title('SARIMA Residuals (Forecast Errors)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Error (MW)')
axes[1, 0].grid(alpha=0.3)

# SARIMA residual histogram
axes[1, 1].hist(sarima_residuals, bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('SARIMA Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Error (MW)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/12_residual_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/12_residual_analysis.png")
plt.show()

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "-"*70)
print("SAVING RESULTS:")
print("-"*70)

# Create results dataframe
results_df = pd.DataFrame({
    'datetime': test.index,
    'actual': test['consumption'].values,
    'arima_forecast': arima_forecast,
    'sarima_forecast': sarima_forecast,
    'arima_error': arima_residuals,
    'sarima_error': sarima_residuals,
    'arima_error_abs': np.abs(arima_residuals),
    'sarima_error_abs': np.abs(sarima_residuals)
})

results_df.to_csv('data/processed/arima_results.csv', index=False)
print("✓ Saved: data/processed/arima_results.csv")

# Save model comparison
model_comparison = pd.DataFrame({
    'Model': ['Baseline (Last Week)', 'ARIMA(1,0,1)', 'SARIMA(1,0,1)(1,0,1,24)'],
    'MAE': [baseline_mae, arima_mae, sarima_mae],
    'RMSE': [baseline_rmse, arima_rmse, sarima_rmse],
    'MAPE': [baseline_mape, arima_mape, sarima_mape]
})

model_comparison.to_csv('data/processed/model_comparison.csv', index=False)
print("✓ Saved: data/processed/model_comparison.csv")

# Print comparison table
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(model_comparison.to_string(index=False))

# Determine best model
best_model = model_comparison.loc[model_comparison['MAPE'].idxmin(), 'Model']
print(f"\n🏆 Best Model: {best_model}")

print("\n" + "="*70)
print("ARIMA MODELING COMPLETE!")
print("="*70)
print("\n✅ Models trained and evaluated")
print("✅ Forecasts generated for 7-day test period")
print("✅ Visualizations saved in 'outputs/'")
print("✅ Results saved in 'data/processed/'")

print("\n" + "-"*70)
print("NEXT STEPS:")
print("-"*70)
print("📌 Run Prophet model (06_prophet_model.py)")
print("📌 Compare all three approaches")
print("📌 Select best model for final dashboard")