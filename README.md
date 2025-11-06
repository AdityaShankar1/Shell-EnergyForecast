⚡ Electricity Consumption Forecasting for Grid Optimization (Romania)

Project Overview

This project focuses on solving a critical challenge in energy market operations: accurately forecasting short-term electricity demand to enable efficient grid balancing, optimized energy trading, and successful integration of variable renewable energy sources (RES).

Using a comprehensive dataset of 6 years of hourly consumption data from the Romanian national grid (2019-2025), a statistical time-series model was developed to predict electricity demand 168 hours (one week) into the future.

Key Results & Business Impact

Metric

Result

Value Creation

Primary Metric (MAPE)

11.64% (Mean Absolute Percentage Error)

Achieved industry-acceptable week-ahead forecasting accuracy, even without external factors like weather data.

Performance Improvement

19.5% improvement over the Naive Forecast (Last Week Same Hour).

Significant reduction in operational risk and minimized reliance on high-cost emergency reserves.

Quantified Savings

Projected €24–39 Million in annual optimized value.

Savings derived from optimized day-ahead market trading, reduced fuel consumption, and lower spinning reserve requirements.

Operational Impact

Reduced average forecast error to ±749 MW.

Enables precise unit commitment and maintenance scheduling for power plants.

🛠️ Technical Stack

Language: Python 3.10

Core Libraries: Pandas, NumPy, Statsmodels (SARIMA), Prophet (Comparison)

Database: PostgreSQL (structured time-series data storage and indexing)

Visualization: Matplotlib, Seaborn

Version Control: Git

📈 Methodology

Data Acquisition & Preparation: Utilized 6 years (54,160 hourly observations) of raw consumption data. Conducted rigorous quality assessment to remove erroneous readings.

Exploratory Data Analysis (EDA): Identified and statistically validated strong multi-level seasonality:

24-Hour Cycle: High correlation (96%) with peak demand during morning/evening.

168-Hour Cycle: Strong weekly pattern (85% correlation) reflecting industrial/commercial activity.

Annual Cycle: W-shaped seasonal pattern driven by winter heating and summer cooling demands.

Feature Engineering: Generated 80+ derived features including lagged consumption values, rolling statistics, day/week/month flags, and cyclical encoding (sine/cosine transforms) to improve model robustness.

Model Selection: Evaluated multiple time-series models (Naive Baseline, ARIMA, Prophet) and selected the Seasonal Auto-Regressive Integrated Moving Average (SARIMA) model due to its superior performance in capturing the 24-hour cycle and achieving the lowest statistical error (AIC/BIC).

Validation: Employed a strict walk-forward validation approach, training on all historical data and testing on a completely held-out final week (168 hours) to simulate real-world operational deployment.

🚀 Future Enhancements

The current model relies purely on historical consumption patterns. Future work will integrate external factors for a sub-10% MAPE performance:

Weather Integration: Incorporate temperature, humidity, and wind speed forecasts (exogenous variables - SARIMAX) to capture HVAC-driven demand spikes.

Holiday Indicators: Add binary flags for major Romanian holidays to mitigate systematic over-forecasting during public closures.

Production Data: Integrate actual renewable generation (Wind/Solar) forecasts to directly model the net load, improving stability for grid operators.
