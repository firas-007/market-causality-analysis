# Market Causality Analysis

This project examines relationships between the Moroccan MASI index, CAC40, S&P 500, EUR/MAD, and USD/MAD using financial time-series methods.

The analysis covers:

- Data cleaning and return calculation
- Descriptive statistics and outlier detection
- Stationarity and cointegration tests
- Vector Autoregression (VAR)
- Granger causality tests
- Impulse response functions and forecast error variance decomposition

## Data

Place these files in `data/raw/`:

- `MASI.csv`
- `CAC40.csv`
- `SP500.csv`
- `EUR_MAD.csv`
- `USD_MAD.csv`

Each file should contain a date column and a closing-price column.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Usage

Start Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in order:

1. `01_data_preparation.ipynb`
2. `02_descriptive_statistics_outliers.ipynb`
3. `03_stationarity_cointegration.ipynb`
4. `04_var_granger_causality.ipynb`

Generated datasets and test results are saved under `data/`, while charts are saved in `figures/`.

## Note

Granger causality measures predictive relationships and does not prove structural economic causation.
