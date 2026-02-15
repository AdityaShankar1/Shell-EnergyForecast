import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import os
import time

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

print("="*70)
print("DAY 3: PROPHET FORECASTING MODEL")
print("="*70)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/5] Loading data...")

engine = create_engine('postgresql://postgres:09012004@localhost:5432/Energy Forecasting')
df = pd.read_sql("SELECT datetime, consumption FROM romania_energy_processed ORDER BY datetime", engine)
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"✓ Loaded {len(df):,} rows")
print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ============================================
# PREPARE DATA FOR PROPHET
# ============================================
print("\n[2/5] Preparing data for Prophet...")

# Prophet requires columns named 'ds' (datestamp) and 'y' (value)
df_prophet = df.rename(columns={'datetime': 'ds', 'consumption': 'y'})

# Train-test split (same as ARIMA - last 168 hours)
test_size = 168
train_prophet = df_prophet.iloc[:-test_size]
test_prophet = df_prophet.iloc[-test_size:]

print(f"✓ Training set: {len(train_prophet):,} rows")
print(f"✓ Test set:     {len(test_prophet):,} rows")

# ============================================
# TRAIN PROPHET MODEL
# ============================================
print("\n[3/5] Training Prophet model...")
print("   (This may take 2-3 minutes...)")

start_time = time.time()

# Initialize Prophet with custom parameters
prophet_model = Prophet(
    changepoint_prior_scale=0.05,  # Flexibility in trend changes
    seasonality_prior_scale=10,     # Strength of seasonality
    seasonality_mode='additive',    # Additive seasonality (vs multiplicative)
    daily_seasonality=True,         # Enable daily patterns
    weekly_seasonality=True,        # Enable weekly patterns
    yearly_seasonality=True,        # Enable yearly patterns
    interval_width=0.95             # 95% confidence intervals
)

# Add custom seasonalities for better accuracy
prophet_model.add_seasonality(
    name='hourly',
    period=1,           # 1 day
    fourier_order=8     # Complexity of pattern
)

# Fit the model
prophet_model.fit(train_prophet)

train_time = time.time() - start_time
print(f"✓ Prophet model trained in {train_time:.1f} seconds")

# ============================================
# GENERATE FORECASTS
# ============================================
print("\n[4/5] Generating forecasts...")

# Create future dataframe for prediction
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='H')

# Make predictions
forecast = prophet_model.predict(future)

# Extract test period predictions
test_forecast = forecast.iloc[-len(test_prophet):]

# Calculate metrics
prophet_predictions = test_forecast['yhat'].values
actual_values = test_prophet['y'].values

prophet_mae = mean_absolute_error(actual_values, prophet_predictions)
prophet_rmse = np.sqrt(mean_squared_error(actual_values, prophet_predictions))
prophet_mape = mean_absolute_percentage_error(actual_values, prophet_predictions) * 100

print(f"\n📊 Prophet Model Performance:")
print(f"   MAE:  {prophet_mae:.2f} MW")
print(f"   RMSE: {prophet_rmse:.2f} MW")
print(f"   MAPE: {prophet_mape:.2f}%")

# Compare with previous models
print("\n" + "-"*70)
print("COMPARISON WITH PREVIOUS MODELS:")
print("-"*70)

# Load previous results
model_comparison = pd.read_csv('data/processed/model_comparison.csv')
print(model_comparison.to_string(index=False))

# Add Prophet to comparison
prophet_row = pd.DataFrame({
    'Model': ['Prophet'],
    'MAE': [prophet_mae],
    'RMSE': [prophet_rmse],
    'MAPE': [prophet_mape]
})

model_comparison_updated = pd.concat([model_comparison, prophet_row], ignore_index=True)
print(f"\n{'Prophet':<25s} {prophet_mae:>10.2f} {prophet_rmse:>10.2f} {prophet_mape:>10.2f}")

# Determine best model overall
best_model_idx = model_comparison_updated['MAPE'].idxmin()
best_model = model_comparison_updated.loc[best_model_idx, 'Model']
best_mape = model_comparison_updated.loc[best_model_idx, 'MAPE']

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model} (MAPE: {best_mape:.2f}%)")
print("="*70)

# ============================================
# VISUALIZATIONS
# ============================================
print("\n[5/5] Creating visualizations...")

# Plot 1: Prophet forecast with components
fig = prophet_model.plot(forecast)
fig.set_size_inches(16, 8)
axes = fig.get_axes()
axes[0].set_title('Prophet Forecast - Full Time Series', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Consumption (MW)', fontsize=11)

# Add test period highlight
test_start = test_prophet['ds'].iloc[0]
test_end = test_prophet['ds'].iloc[-1]
axes[0].axvline(x=test_start, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Test Period Start')
axes[0].legend()

plt.tight_layout()
plt.savefig('outputs/13_prophet_full_forecast.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/13_prophet_full_forecast.png")
plt.show()

# Plot 2: Prophet components (trend, seasonality)
fig = prophet_model.plot_components(forecast)
fig.set_size_inches(14, 10)
plt.tight_layout()
plt.savefig('outputs/14_prophet_components.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/14_prophet_components.png")
plt.show()

# Plot 3: Test period detailed comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Full test period
axes[0].plot(test_prophet['ds'], actual_values, 
            label='Actual', color='black', linewidth=2, alpha=0.8)
axes[0].plot(test_prophet['ds'], prophet_predictions, 
            label='Prophet Forecast', color='#2E86AB', linewidth=1.5, linestyle='--')
axes[0].fill_between(test_prophet['ds'], 
                     test_forecast['yhat_lower'], 
                     test_forecast['yhat_upper'],
                     alpha=0.2, color='blue', label='95% Confidence Interval')
axes[0].set_title('Prophet Forecast vs Actual (7-Day Test Period)', 
                 fontsize=14, fontweight='bold')
axes[0].set_ylabel('Consumption (MW)', fontsize=11)
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(alpha=0.3)

# Add metrics box
axes[0].text(0.02, 0.95, 
            f'MAPE: {prophet_mape:.2f}%\nMAE: {prophet_mae:.0f} MW\nRMSE: {prophet_rmse:.0f} MW',
            transform=axes[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# First 48 hours detail
test_48h = test_prophet.iloc[:48]
pred_48h = prophet_predictions[:48]
lower_48h = test_forecast['yhat_lower'].values[:48]
upper_48h = test_forecast['yhat_upper'].values[:48]

axes[1].plot(test_48h['ds'], test_48h['y'], 
            label='Actual', color='black', linewidth=2.5, marker='o', markersize=4)
axes[1].plot(test_48h['ds'], pred_48h, 
            label='Prophet Forecast', color='#2E86AB', linewidth=2, 
            linestyle='--', marker='s', markersize=3)
axes[1].fill_between(test_48h['ds'], lower_48h, upper_48h,
                     alpha=0.2, color='blue', label='95% CI')
axes[1].set_title('Detailed View: First 48 Hours', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date & Time', fontsize=11)
axes[1].set_ylabel('Consumption (MW)', fontsize=11)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(alpha=0.3)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/15_prophet_test_detail.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/15_prophet_test_detail.png")
plt.show()

# Plot 4: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

models = model_comparison_updated['Model'].values
mae_values = model_comparison_updated['MAE'].values
rmse_values = model_comparison_updated['RMSE'].values
mape_values = model_comparison_updated['MAPE'].values

# MAE comparison
axes[0].bar(range(len(models)), mae_values, 
           color=['gray', '#E63946', '#06A77D', '#2E86AB'],
           alpha=0.8, edgecolor='black')
axes[0].set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('MAE (MW)', fontsize=11)
axes[0].set_xticks(range(len(models)))
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(mae_values):
    axes[0].text(i, v + 20, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

# RMSE comparison
axes[1].bar(range(len(models)), rmse_values, 
           color=['gray', '#E63946', '#06A77D', '#2E86AB'],
           alpha=0.8, edgecolor='black')
axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('RMSE (MW)', fontsize=11)
axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(rmse_values):
    axes[1].text(i, v + 30, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

# MAPE comparison (most important!)
colors = ['gray', '#E63946', '#06A77D', '#2E86AB']
axes[2].bar(range(len(models)), mape_values, 
           color=colors,
           alpha=0.8, edgecolor='black')
axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('MAPE (%)', fontsize=11)
axes[2].set_xticks(range(len(models)))
axes[2].set_xticklabels(models, rotation=45, ha='right')
axes[2].grid(axis='y', alpha=0.3)

# Highlight best model
best_idx = np.argmin(mape_values)
axes[2].bar(best_idx, mape_values[best_idx], 
           color='gold', alpha=0.9, edgecolor='black', linewidth=2)

for i, v in enumerate(mape_values):
    axes[2].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/16_model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/16_model_comparison.png")
plt.show()

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "-"*70)
print("SAVING RESULTS:")
print("-"*70)

# Save Prophet predictions
prophet_results = pd.DataFrame({
    'datetime': test_prophet['ds'],
    'actual': actual_values,
    'prophet_forecast': prophet_predictions,
    'prophet_lower': test_forecast['yhat_lower'].values,
    'prophet_upper': test_forecast['yhat_upper'].values,
    'prophet_error': actual_values - prophet_predictions,
    'prophet_error_abs': np.abs(actual_values - prophet_predictions)
})

prophet_results.to_csv('data/processed/prophet_results.csv', index=False)
print("✓ Saved: data/processed/prophet_results.csv")

# Save updated model comparison
model_comparison_updated.to_csv('data/processed/model_comparison_final.csv', index=False)
print("✓ Saved: data/processed/model_comparison_final.csv")

# Save forecast components for analysis
forecast_components = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 
                                 'trend', 'weekly', 'yearly']].tail(168)
forecast_components.to_csv('data/processed/prophet_components.csv', index=False)
print("✓ Saved: data/processed/prophet_components.csv")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("FINAL MODEL COMPARISON")
print("="*70)
print(model_comparison_updated.to_string(index=False))

print("\n" + "="*70)
print(f"🏆 WINNER: {best_model}")
print("="*70)
print(f"\n✅ Best MAPE: {best_mape:.2f}%")

# Improvement calculations
baseline_mape = model_comparison_updated.loc[0, 'MAPE']
improvement = ((baseline_mape - best_mape) / baseline_mape) * 100
print(f"✅ Improvement over baseline: {improvement:.1f}%")

# Business impact
avg_consumption = df['consumption'].mean()
avg_error_mw = best_mape / 100 * avg_consumption
print(f"\n💡 Business Impact:")
print(f"   Average consumption: {avg_consumption:.0f} MW")
print(f"   Average forecast error: ±{avg_error_mw:.0f} MW")
print(f"   This enables better grid balancing and trading decisions")

print("\n" + "="*70)
print("PROPHET MODELING COMPLETE!")
print("="*70)
print("\n✅ All 4 models trained and compared")
print("✅ Best model identified")
print("✅ Visualizations saved")
print("✅ Results ready for dashboard")

print("\n" + "-"*70)
print("WHAT'S NEXT:")
print("-"*70)
print("📌 DAY 4: Model tuning (optional - if you want to improve further)")
print("📌 DAY 5: Dagster pipeline (data orchestration)")
print("📌 DAY 6: Tableau dashboard (visualization)")
print("📌 DAY 7: Documentation and presentation")
print("\n💡 For now, you have working models! Let's move to pipeline/dashboard.")