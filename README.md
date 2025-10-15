# Gold Price Forecasting (INR) — scikit-learn

![Python](https://img.shields.io/badge/Python-3.10+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end ML pipeline to forecast **gold prices in INR** using:
- **Multivariate Linear Regression** on scaled features
- **Decision Tree** with **QuantileTransformer** on the target for robust distribution handling

Includes EDA, feature selection (**CPI, Sensex, USD_INR**), reproducible scripts, and result figures.  
Report: `docs/report.pdf`. 

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Data
Put your CSV at data/GoldUP.csv (see data/README.md).

## Train both models
```python
python scripts/train_lr.py
python scripts/train_tree.py
```
or run **scripts/Full_Script.py**

## Results 
  - **Linear Regression:** ~0.96 R² on test.
  - **Decision Tree (+QuantileTransformer):** ~0.99 R² on test and better overlap of actual vs predicted.

See results/figures/*.png.

<img width="751" height="543" alt="Distribution_Real_vs_Predicted_MLR" src="https://github.com/user-attachments/assets/65b72ed8-b495-41df-8724-03fd600f44c0" />
<img width="744" height="556" alt="Distribution_Real_vs_Predicted_DT" src="https://github.com/user-attachments/assets/8d732d6c-128d-4986-9528-22c753c6b88c" />
<img width="412" height="86" alt="Report_of_Results_MLR" src="https://github.com/user-attachments/assets/98ac62eb-15e6-4091-b8b8-27fc0c3ddeca" />
<img width="407" height="68" alt="Report_of_Results_DT" src="https://github.com/user-attachments/assets/63ae1e80-6131-41f8-99be-6abfcdee80cc" />


## Repo map
```pgsql
scripts/ … train_lr.py, train_tree.py, Full_Script.py
data/    … README only (no CSV in git)
docs/    … project report + EDA figure(s)
results/ … results and generated plots
```

## Notes
  - Research/education only.
  - Do not commit proprietary datasets; keep data paths relative.
