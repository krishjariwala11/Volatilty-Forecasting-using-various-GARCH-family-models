import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_loader import fetch_data, preprocess_data
from model import fit_garch, get_forecasts, evaluate_model
import numpy as np

# Page Config
st.set_page_config(page_title="Stock Volatility Forecast", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
    }
    .stTable {
        background-color: #1e2130;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock Volatility Forecasting Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI", help="Enter yfinance ticker like RELIANCE.NS or AAPL")

end_date = datetime.today()
start_date = end_date - timedelta(days=365*5) # 5 years

selected_start = st.sidebar.date_input("Start Date", start_date)
selected_end = st.sidebar.date_input("End Date", end_date)

model_type = st.sidebar.selectbox("Model Type", ["GARCH", "EGARCH", "GJR-GARCH"])
dist_type = st.sidebar.selectbox("Distribution", ["Normal", "t", "skewt"])
p = st.sidebar.slider("p (Lagged Volatility)", 1, 5, 1)
q = st.sidebar.slider("q (Lagged Returns)", 1, 5, 1)

horizon = st.sidebar.number_input("Forecast Horizon (Days)", 1, 90, 30)

if st.button("Run Forecast"):
    with st.spinner(f"Fetching data for {ticker} and fitting model..."):
        try:
            # 1. Fetch Data
            df = fetch_data(ticker, selected_start, selected_end)
            df = preprocess_data(df)
            
            # 2. Fit Model
            res = fit_garch(df['Returns'], p=p, q=q, model_type=model_type, dist=dist_type)
            
            # 3. Forecast
            forecast_vol = get_forecasts(res, horizon=horizon)
            
            # 4. Evaluation
            df['Predicted_Vol'] = res.conditional_volatility * np.sqrt(252)
            metrics = evaluate_model(df['Returns'], df['Predicted_Vol'])
            
            # UI Layout
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Volatility (Realized)", f"{df['Realized_Vol'].iloc[-1]:.2f}%")
            col2.metric("Forecasted Volatility (Avg)", f"{np.mean(forecast_vol):.2f}%")
            col3.metric("Model RMSE", f"{metrics['RMSE']:.4f}")
            
            # Stats Summary in Neat Format
            st.subheader("Model Parameters")
            params = res.params
            pvalues = res.pvalues
            std_err = res.std_err
            tvalues = res.tvalues
            
            param_df = pd.DataFrame({
                "Coefficient": params.values,
                "Std. Error": std_err.values,
                "t-stat": tvalues.values,
                "p-value": pvalues.values
            }, index=params.index)
            
            # Style the dataframe for a neat look
            st.dataframe(param_df.style.format("{:.4f}").highlight_between(left=0, right=0.05, subset=['p-value'], color='#00d4ff22'), use_container_width=True)

            # Plots
            st.subheader(f"{ticker} Volatility: Realized vs. Predicted")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Realized_Vol'], name="Realized Volatility (21d)", line=dict(color='#00d4ff', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Predicted_Vol'], name=f"{model_type} Fitted Volatility", line=dict(color='#ff4b4b', width=1.5, dash='dot')))
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns Plot
            st.subheader(f"{ticker} Returns & Fitted Volatility")
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Bar(x=df.index, y=df['Returns'], name="Daily Returns (%)", marker_color='rgba(255, 255, 255, 0.3)'))
            fig_ret.add_trace(go.Scatter(x=df.index, y=df['Predicted_Vol']/10, name="Annualized Vol (scaled)", line=dict(color='#ff4b4b', width=2)))
            fig_ret.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_ret, use_container_width=True)
            
            # Forecast Plot
            st.subheader(f"Next {horizon} Days Volatility Forecast for {ticker}")
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=future_dates, y=forecast_vol, mode='lines+markers', name="Forecast", line=dict(color='#00ff88')))
            fig_f.update_layout(template="plotly_dark", xaxis_title="Future Dates", yaxis_title="Annualized Volatility (%)")
            st.plotly_chart(fig_f, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Ensure the ticker symbol is correct (e.g., RELIANCE.NS, TSLA).")
else:
    st.info("Enter a stock ticker and click 'Run Forecast' to start.")
