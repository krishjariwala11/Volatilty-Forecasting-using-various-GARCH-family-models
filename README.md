# Stock Volatility Forecasting Dashboard 📈

🔗 Live Demo: [GARCH Volatility Model App](https://garch-krish.streamlit.app/)

A premium Streamlit application to forecast stock market volatility using GARCH-family models (GARCH, EGARCH, GJR-GARCH).

## Features
- **Multi-Stock Support**: Forecast volatility for any ticker supported by Yahoo Finance (e.g., `^NSEI`, `RELIANCE.NS`, `AAPL`, `TSLA`).
- **Advanced Modeling**:
  - GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
  - EGARCH (Exponential GARCH)
  - GJR-GARCH (Glosten-Jagannathan-Runkle GARCH)
- **Interactive Visualizations**:
  - Realized vs. Predicted Volatility comparison.
  - Daily Returns vs. Conditional Volatility.
  - Multi-day future volatility forecasts.
- **Backtesting & Evaluation**: Automatic computation of RMSE and MAE against realized volatility.
- **Premium UI**: Sleek dark-mode dashboard with styled parameter tables and metrics.

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-volatility-forecast.git
   cd stock-volatility-forecast
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

## How it Works
1. **Data Ingestion**: Fetches historical close prices via `yfinance`.
2. **Returns Calculation**: Computes log returns of the asset.
3. **Volatility Proxy**: Calculates 21-day rolling standard deviation (annualized) as a proxy for "Realized Volatility".
4. **Model Fitting**: Uses the `arch` library to fit the selected GARCH model to the returns.
5. **Forecasting**: Generates multi-day variance forecasts and converts them to annualized volatility.

## Screenshots
*(Add screenshots here after running the app)*

## Disclaimer
This tool is for educational and research purposes only. Trading involves significant risk, and volatility forecasts should not be used as sole investment advice.
