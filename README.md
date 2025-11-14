# Electricity Consumption Forecasting for Grid Optimization (Romania)
## 1. Project Overview

This project focuses on short-term electricity demand forecasting to support grid balancing, market operations, and renewable energy integration.
A statistical time-series model was built using 6 years (2019–2025) of hourly national grid consumption data from Romania.
The model generates week-ahead (168-hour) forecasts for operational planning.

## 2. Key Results and Business Impact
### 2.1 Forecasting Accuracy

MAPE: 11.64%

Achieves industry-standard performance for week-ahead forecasting without external drivers such as weather.

### 2.2 Improvement Over Baseline

19.5% improvement compared to a Naive baseline (Last Week Same Hour).

Reduces operational uncertainty and lowers dependency on costly spinning reserves.

### 2.3 Financial Impact

Estimated annual savings: €24–39 million

Savings derive from:

Optimized day-ahead energy market decisions

Reduced fuel usage

Lower reserve activation requirements

### 2.4 Operational Reliability

Average absolute error: ±749 MW

Supports improved unit commitment, dispatch planning, and maintenance scheduling.

## 3. Technical Stack

Language: Python 3.10

Libraries: Pandas, NumPy, Statsmodels (SARIMA), Prophet (comparison only)

Database: PostgreSQL (time-series structured storage)

Visualization: Matplotlib, Seaborn

Version Control: Git

## 4. Methodology
### 4.1 Data Preparation

Used 54,160 hourly observations covering 6 years.

Performed quality checks to remove abnormal or corrupted readings.

### 4.2 Exploratory Data Analysis

Identified strong seasonal patterns:

Daily Seasonality (24 hours): High correlation (96%), reflecting standard load fluctuations.

Weekly Seasonality (168 hours): Strong pattern (85%), influenced by business and industrial cycles.

Annual Seasonality: W-shaped pattern due to heating and cooling demands.

### 4.3 Feature Engineering

Created more than 80 features, including:

Lagged consumption values

Rolling means and variances

Day-of-week and month indicators

Cyclical encodings (sine/cosine) for hours and seasons

These features improved model stability and seasonal capture.

### 4.4 Model Evaluation and Selection

Tested multiple approaches:

Model	Notes
Naive Baseline	Benchmark reference
ARIMA	Unable to capture multi-seasonality
Prophet	Good for trend, less effective here
SARIMA (Selected)	Best overall performance, lowest AIC/BIC

SARIMA was selected because it best captured daily periodicity and provided the lowest statistical error.

### 4.5 Validation Strategy

Applied walk-forward validation, training on historical data and evaluating on unseen periods.

Final performance measured on a held-out 168-hour test window, representing real operational usage.

## 5. Future Enhancements
### 5.1 Weather-Driven Forecasting

Integrate temperature, humidity, and wind speed using SARIMAX to capture HVAC-driven variability and improve accuracy.

### 5.2 Holiday and Event Indicators

Introduce binary markers for national holidays and major events to reduce systematic over-prediction during reduced commercial activity.

### 5.3 Renewable Generation Forecasts

Add solar and wind generation data to forecast net load, improving relevance for transmission operators and market planning.
