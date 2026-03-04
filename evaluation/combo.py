import numpy as np
import pandas as pd


def fit_linear_combo(features: pd.DataFrame, target: pd.Series):
    """Fit a linear model with intercept via least squares.

    Returns coefficients as numpy array [intercept, beta_1, ...].
    """
    aligned = pd.concat([features, target.rename("target")], axis=1).dropna()
    if aligned.empty:
        return None

    x = aligned[features.columns].to_numpy(dtype=float)
    y = aligned["target"].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return coef


def predict_linear_combo(features: pd.DataFrame, coef: np.ndarray):
    """Predict using coefficients from fit_linear_combo."""
    if coef is None:
        return pd.Series(index=features.index, dtype=float)

    x = features.to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    y_hat = x_design @ coef
    return pd.Series(y_hat, index=features.index)
