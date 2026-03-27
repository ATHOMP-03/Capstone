"""
ml_analysis.py
Python equivalent of src/r/ml_analysis.R

Doubly-robust ATE estimation via DoubleML Partially Linear Regression (PLR).
Uses XGBoost with CUDA GPU acceleration for the nuisance models in place of
ranger. Drop-in swap to CPU: change device="cuda" -> device="cpu" below.

Run 1: Average daily Twitter sentiment (twitter_sent, continuous [-1, 1])
Run 2: Negative tweet volume (twitter_count * I(twitter_sent < 0))
"""

import numpy as np
import pandas as pd
import doubleml as dml
from xgboost import XGBRegressor
from pathlib import Path

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
IN_FILE  = ROOT / "data" / "processed" / "panel_long.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
long = pd.read_csv(IN_FILE, parse_dates=["date"])

# ---------------------------------------------------------------------------
# XGBoost learner factory
# Change device="cpu" if running without a CUDA GPU
# ---------------------------------------------------------------------------
def make_xgb():
    return XGBRegressor(
        n_estimators   = 500,
        learning_rate  = 0.05,
        max_depth      = 6,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        device         = "cuda",
        tree_method    = "hist",
        random_state   = 42,
        n_jobs         = -1,
    )


# ===========================================================================
# RUN 1 — Average daily Twitter sentiment as continuous treatment
#
# Treatment: twitter_sent (continuous, uniform [-1, 1])
# Outcome:   return
# Confounders: all except px_open, px_close, identifiers, outcome, treatment
# ===========================================================================

confounders_1 = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_count", "news_sent", "rsi_30", "ma_50",
    "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7",
]

df1 = (
    long[["return", "twitter_sent"] + confounders_1]
    .dropna()
    .reset_index(drop=True)
)

data_1 = dml.DoubleMLData(
    df1,
    y_col  = "return",
    d_cols = "twitter_sent",
    x_cols = confounders_1,
)

dml_1 = dml.DoubleMLPLR(
    data_1,
    ml_l     = make_xgb(),   # outcome nuisance
    ml_m     = make_xgb(),   # treatment nuisance
    n_folds  = 5,
)
dml_1.fit()

print("\n===== RUN 1: Average Daily Twitter Sentiment (DoubleML PLR) =====")
print(f"Treatment: twitter_sent  |  Outcome: return")
print(f"N (complete cases): {len(df1):,}")
print(dml_1.summary)


# ===========================================================================
# RUN 2 — Negative tweet count as standalone treatment (Teti et al. 2019)
#
# Treatment: twitter_neg_count (count of negative tweets per day)
# Per Teti et al.: polarity-broken counts are significant; total count is not.
# Using twitter_neg_count directly sidesteps neutral-tweet dilution in the
# Bloomberg index and avoids constructing an interaction variable.
# Outcome:   return
# Confounders: twitter_count and twitter_sent both stay in to control for
#   total tweet volume and overall sentiment direction independently.
# ===========================================================================

confounders_2 = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_sent", "twitter_count", "news_sent", "rsi_30", "ma_50",
    "lag1", "lag2", "lag3", "lag5", "lag7",
]

df2 = (
    long[["return", "twitter_neg_count"] + confounders_2]
    .dropna()
    .reset_index(drop=True)
)

data_2 = dml.DoubleMLData(
    df2,
    y_col  = "return",
    d_cols = "twitter_neg_count",
    x_cols = confounders_2,
)

dml_2 = dml.DoubleMLPLR(
    data_2,
    ml_l     = make_xgb(),
    ml_m     = make_xgb(),
    n_folds  = 5,
)
dml_2.fit()

print("\n===== RUN 2: Negative Tweet Count (DoubleML PLR) =====")
print(f"Treatment: twitter_neg_count  |  Outcome: return")
print(f"N (complete cases): {len(df2):,}")
print(dml_2.summary)
