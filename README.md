⚡ Electricity Consumption Forecasting for Grid Optimization (Romania)
📘 Project Overview

This project addresses a key operational challenge in modern energy systems: accurately forecasting short-term electricity demand to improve grid balancing, energy market efficiency, and renewable energy integration.

Using 6 years of hourly consumption data (2019–2025) from the Romanian national grid, a statistical time-series forecasting model was developed to predict electricity demand 168 hours (1 week) ahead.

📊 Key Results & Business Impact
🔧 Primary Metric (MAPE)

11.64% Mean Absolute Percentage Error
→ Achieved industry-standard accuracy for week-ahead load forecasting without external features (e.g., weather).

📉 Performance Improvement

19.5% improvement over the Naive Baseline (Last Week Same Hour).
→ Reduces operational risk and reliance on expensive spinning reserves.

💶 Quantified Savings

€24–39 Million projected annual optimized value, based on:

Improved day-ahead market bidding

Reduced fuel consumption

Lower reserve capacity usage

⚙️ Operational Impact

Reduced average forecast error: ±749 MW
→ Enables more precise unit commitment, scheduling, and maintenance planning for power plants.

🛠️ Technical Stack

Language: Python 3.10

Core Libraries: Pandas, NumPy, Statsmodels (SARIMA), Prophet (for comparison)

Database: PostgreSQL (optimized for time-series storage/indexing)

Visualization: Matplotlib, Seaborn

Version Control: Git

📈 Methodology
1. Data Acquisition & Preparation

Collected 54,160 hourly readings spanning 6 years.

Performed rigorous data cleaning and anomaly removal.

2. Exploratory Data Analysis (EDA)

Identified strong multi-level seasonality:

24-Hour Cycle: 96% correlation — morning/evening peaks

168-Hour Weekly Cycle: 85% correlation — industrial & commercial patterns

Annual Cycle: W-shaped — driven by winter heating and summer cooling loads

3. Feature Engineering

Created 80+ engineered features, including:

Lag variables

Rolling window statistics

Day/week/month flags

Cyclical time encodings (sine/cosine)

4. Model Selection

Evaluated multiple forecasting models:

Model	Purpose
Naive Baseline	Benchmark
ARIMA	Non-seasonal modeling
Prophet	Trend/seasonality heuristic
SARIMA	Chosen due to best fit and lowest AIC/BIC

SARIMA effectively captured the strong 24-hour seasonality and achieved the highest accuracy.

5. Validation Strategy

Used walk-forward validation.

Final evaluation on a completely held-out 168-hour (1 week) period to mimic real deployment.

🚀 Future Enhancements

To further improve accuracy (target: <10% MAPE), upcoming improvements will include:

🌦️ 1. Weather Integration (SARIMAX)

Temperature

Humidity

Wind speed
→ Captures HVAC-driven load variations.

🎉 2. Holiday & Event Indicators

Flags for major Romanian holidays
→ Reduces systematic over-forecasting during low-activity periods.

☀️ 3. Renewable Production Data

Incorporate wind & solar generation forecasts
→ Enables net-load forecasting, more relevant for grid operators.
