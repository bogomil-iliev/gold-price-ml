
# Data (not included)

**Dataset link:** https://www.kaggle.com/datasets/somyaagarwal69/gold-forecasting

Source used in this study:
- Kaggle: somyaagarwal69/gold-forecasting (INR), 2000-10-01 → 2020-08-01.  
  Columns include: Gold_Price (target), CPI, Sensex, USD_INR, Interest_Rate, USD_Index, Crude_Oil, etc.

This repo trains on **CPI, Sensex, USD_INR** and predicts **Gold_Price**.  
We standardise features for Linear Regression and use a **QuantileTransformer** on the target for the Decision Tree (via `TransformedTargetRegressor`).
