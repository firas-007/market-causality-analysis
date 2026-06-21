# Stock Market Causality Analysis: Influence of CAC40 and S&P500 on the Moroccan MASI Index

## Project Overview

This project is an intermediate-to-advanced Python econometrics study of causal links between the Moroccan MASI stock index, the French CAC40 index, the S&P500 index, and two exchange-rate series: EUR/MAD and USD/MAD. The workflow is built for Jupyter Notebook and separates analysis notebooks from reusable Python modules.

The project does not use machine learning, web development, or dashboards. It focuses on financial time-series preprocessing, classical econometric testing, VAR modeling, Granger causality, and robustness discussion.

## Research Question

Do international equity markets and exchange-rate movements help explain short-run movements in the Moroccan MASI index, and is there evidence of feedback between MASI and Moroccan exchange-rate variables?

## Article Motivation

The project is inspired by a scientific study that analyzes causal relationships between MASI, CAC40, S&P500, USD/MAD, and EUR/MAD using VAR, Granger causality, and a robust VAR-Gini approach. The article motivation is especially relevant because financial series often contain extreme observations, non-normal returns, and volatility episodes that can affect classical OLS-based VAR estimates.

## Dataset Description

Place the following CSV files in `data/raw/`:

- `MASI.csv`
- `CAC40.csv`
- `SP500.csv`
- `EUR_MAD.csv`
- `USD_MAD.csv`

Each CSV must contain at least a date column and a closing-price column. The loader detects common names:

- Date columns: `Date`, `date`, `DATE`
- Price columns: `Close`, `close`, `Adj Close`, `Price`, `Dernier`

The merged dataset uses the intersection of dates across all series to keep VAR inputs aligned.

## Methods Used

- Financial time-series preprocessing
- First differences, simple returns, and log returns
- Descriptive statistics
- Jarque-Bera normality test
- IQR and z-score outlier detection
- ADF unit-root test
- KPSS stationarity test
- Johansen cointegration test
- Vector Autoregression (VAR)
- Granger causality testing
- Impulse response functions
- Forecast error variance decomposition
- VAR residual diagnostics and stability checks

## Project Structure

```text
market-causality-analysis/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_descriptive_statistics_outliers.ipynb
│   ├── 03_stationarity_cointegration.ipynb
│   └── 04_var_granger_causality.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── statistics.py
│   ├── outliers.py
│   ├── stationarity.py
│   ├── cointegration.py
│   ├── var_model.py
│   ├── diagnostics.py
│   ├── visualization.py
│   └── utils.py
├── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## How To Run The Notebooks

1. Add the five required CSV files to `data/raw/`.
2. Start Jupyter from the project root:

```bash
jupyter notebook
```

3. Run the notebooks in order:

- `01_data_preparation.ipynb`
- `02_descriptive_statistics_outliers.ipynb`
- `03_stationarity_cointegration.ipynb`
- `04_var_granger_causality.ipynb`

## Expected Outputs

Processed datasets are saved in `data/processed/`:

- `market_data_clean.csv`
- `market_data_first_difference.csv`
- `market_data_returns.csv`
- `market_data_log_returns.csv`

Model tables and test results are saved in `data/results/`. Figures are saved in `figures/`.

## Limitations

- Results depend on data frequency, sample period, data vendor conventions, and holiday alignment.
- Classical VAR is estimated using OLS and can be sensitive to outliers.
- Granger causality identifies predictive content, not structural economic causality.
- Johansen cointegration conclusions are sensitive to lag order and deterministic-term assumptions.
- This project discusses VAR-Gini but does not implement the full estimator.

## Future Improvements

- Implement a complete VAR-Gini estimator or another robust multivariate time-series estimator.
- Add rolling-window Granger causality tests to study time-varying relationships.
- Compare daily, weekly, and monthly sampling frequencies.
- Add structural break tests around major market events.
- Extend diagnostics with ARCH effects and robust covariance estimates.

## CV Description

Built a modular Python/Jupyter econometrics project studying causal relationships between MASI, CAC40, S&P500, EUR/MAD, and USD/MAD. Implemented robust data ingestion, financial preprocessing, normality and outlier analysis, ADF/KPSS stationarity testing, Johansen cointegration, VAR modeling, Granger causality, impulse response functions, variance decomposition, and model diagnostics using pandas, scipy, statsmodels, matplotlib, and seaborn.
