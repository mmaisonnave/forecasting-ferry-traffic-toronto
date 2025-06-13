# Technical Test for Applied Scientist (Data Scientist)

__Opportunity No. 44081__

## 1. Task

This project tackles a forecasting task using the Toronto Island Ferry Ticket dataset, with two objectives: (1) improve a base model for predicting daily ticket redemptions, and (2) build a new model to forecast daily ticket sales. A 365-day forecast horizon is used, as defined in the base model, limiting the use of future exogenous variables (such as weather), which are unreliable that far ahead. Models relied on historical patterns to capture strong seasonality. Calendar features (holidays, school breaks) were excluded but noted for future work.

## 2. Data
The original dataset records ferry ticket sales and redemptions at 15-minute intervals. As done with the base model, this data was aggregated to daily frequency, producing two daily time series: sales and redemption counts.

The series exhibit strong seasonal patterns, peaking during warmer months. Model performance was evaluated using an expanding window with four train/test splits, where each training set grew incrementally and the subsequent 365 days served as unseen test data. To prevent data leakage and ensure realistic forecasting, only features available at training time were used, with test data reserved strictly for evaluation, assuming no knowledge of future exogenous variables.

## 3. Models

Six forecasting models were implemented for both target variables:

- Historical Average by Day: A simple benchmark using daily averages from the training set as predictions.
- ARIMA-based Models (ARIMAX, ARIMAX on Residuals, SARIMAX): Autoregressive models with exogenous inputs. To forecast one target (sales or redemptions), the other was used as a regressor. Since future values were unavailable for the 365-day test period, test-time regressors were replaced with historical daily averages from the training set. ARIMAX on Residuals was applied after removing seasonality; the seasonal component for the test set was estimated using the Base Model and added back to create the final forecasts. SARIMAX extended ARIMAX by incorporating seasonal terms.
- Prophet: A forecasting tool developed by Meta that fits non-linear trends with additive yearly, weekly, and daily seasonality.
- Ensemble: An average of forecasts from ARIMAX, SARIMAX, Prophet, and the Historical Average. ARIMAX on Residuals was excluded due to reduced performance.

Hyperparameters were tuned using greedy search, guided by exploratory techniques such as stationarity checks (ADF), decomposition (STL), and autocorrelation analysis (ACF/PACF), see Exploratory Data Analysis notebook.

## 4. Results

The ensemble model delivered the best performance in both tasks. Performance was measured using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE) on the 365-day test horizon averaged over four splits.

**Redemption Forecasting Models:**

| Metric   | Model    | Mean           | Std Dev        | 95% Confidence Interval             |
|----------|----------|----------------|----------------|-------------------------------------|
| **MAPE** | Base     | 0.8652         | 0.0194         | [0.8344, 0.8961]                    |
|          | Ensemble | 0.5919         | 0.1834         | [0.3000, 0.8838]                    |
| **MAE**  | Base     | 2,468.64       | 515.08         | [1,649.03, 3,288.26]                |
|          | Ensemble | 1,496.95       | 104.98         | [1,329.91, 1,663.99]                |
| **MSE**  | Base     | 14,659,607.33  | 5,655,109.52   | [5,661,066.13, 23,658,148.54]       |
|          | Ensemble | 7,053,227.89   | 2,297,804.76   | [3,396,907.76, 10,709,548.03]       |


**Sales Forecasting Models:**

| Metric   | Model    | Mean          | Std Dev      | 95% Confidence Interval       |
| -------- | -------- | ------------- | ------------ | ----------------------------- |
| **MAPE** | Base     | 0.8411        | 0.0170       | [0.8139, 0.8682]              |
|          | Ensemble | 0.8841        | 0.1675       | [0.6175, 1.1507]              |
| **MAE**  | Base     | 2,326.43      | 591.40       | [1,385.39, 3,267.47]          |
|          | Ensemble | 1,403.18      | 200.44       | [1,084.24, 1,722.13]          |
| **MSE**  | Base     | 12,792,470.29 | 5,660,765.93 | [3,784,928.48, 21,800,012.10] |
|          | Ensemble | 5,982,077.00  | 2,466,846.79 | [2,056,773.27, 9,907,380.73]  |

All results here.

## 5. Conclusions and Future Work
The experiments produced effective models for both tasks. Simple intepretable models like SARIMAX outperformed advanced ones like Prophet, showing that in long forecasting windows with limited features, simpler methods suffice. A simple ensemble achieved the best results, cutting MAE by ~39% for sales and redemptions forecasting. Future work could include known events (e.g., holidays), explore neural models (LSTMs, Transformers), and adopt probabilistic forecasting for uncertainty quantification.