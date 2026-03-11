from arch import arch_model
import pandas as pd
import numpy as np

def fit_garch(returns, p=1, q=1, model_type='GARCH', dist='Normal'):
    """
    Fit a GARCH-family model to returns.
    model_type: 'GARCH', 'EGARCH', 'GJR-GARCH'
    dist: 'Normal', 't', 'skewt'
    """
    # Map model types to 'arch' library parameters
    vol = 'GARCH'
    if model_type == 'EGARCH':
        vol = 'EGARCH'
    elif model_type == 'GJR-GARCH':
        vol = 'GARCH' # GJR-GARCH is GARCH with o=1
        
    o = 1 if model_type == 'GJR-GARCH' else 0
    
    am = arch_model(returns, vol=vol, p=p, o=o, q=q, dist=dist)
    res = am.fit(disp='off')
    return res

def get_forecasts(res, horizon=30):
    """Generate volatility forecasts."""
    forecasts = res.forecast(horizon=horizon, reindex=False)
    # Volatility is square root of variance
    # Annualize it (sqrt(252))
    predicted_vol = np.sqrt(forecasts.variance.values[-1] * 252)
    return predicted_vol

def evaluate_model(actual_returns, predicted_vol_series):
    """
    Compute relevant parameters and compare vs actual value.
    Note: Realized vol is often computed as absolute returns or squared returns 
    when compared to conditional variance.
    """
    # Simple evaluation: compare predicted vol with realized vol proxy
    # Here we assume predicted_vol_series is aligned with actual_returns
    
    # Realized Vol (Rolling 21-day std annualized)
    realized_vol = actual_returns.rolling(window=21).std() * np.sqrt(252)
    
    # Metrics
    mask = ~np.isnan(realized_vol) & ~np.isnan(predicted_vol_series)
    y_true = realized_vol[mask]
    y_pred = predicted_vol_series[mask]
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Realized': y_true,
        'Predicted': y_pred
    }
