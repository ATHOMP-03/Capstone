"""
px_high_analysis.py
Replicates ml_analysis.py with px_high as the outcome variable.

If negative sentiment drives px_high down, it confirms that sentiment
is suppressing the intraday price ceiling, not just the close-to-open return.

Run 1: twitter_sent as treatment, px_high as outcome
Run 2: neg_tweet_vol as treatment, px_high as outcome
"""

import numpy as np
import pandas as pd
import doubleml as dml
from xgboost import XGBRegressor
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
long = pd.read_csv(ROOT / "data" / "processed" / "panel_long.csv", parse_dates=["date"])

def make_xgb():
    return XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        device           = "cuda",
        tree_method      = "hist",
        random_state     = 42,
        n_jobs           = -1,
    )


# ===========================================================================
# RUN 1 — Twitter sentiment as treatment, px_high as outcome
# return excluded from confounders (derived from px_open/px_close, not px_high)
# ===========================================================================

confounders_1 = [
    "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_count", "news_sent", "rsi_30", "ma_50",
    "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7",
]

df1 = (
    long[["px_high", "twitter_sent"] + confounders_1]
    .dropna()
    .reset_index(drop=True)
)

dml_1 = dml.DoubleMLPLR(
    dml.DoubleMLData(df1, y_col="px_high", d_cols="twitter_sent", x_cols=confounders_1),
    ml_l    = make_xgb(),
    ml_m    = make_xgb(),
    n_folds = 5,
    n_rep   = 20,
)
dml_1.fit()
dml_1.bootstrap(method="normal", n_rep_boot=1000)

print("\n===== RUN 1: Twitter Sentiment -> px_high (DoubleML PLR) =====")
print(f"N (complete cases): {len(df1):,}")
print(dml_1.summary)
print(dml_1.confint(level=0.95))


# ===========================================================================
# RUN 2 — Negative tweet count as treatment, px_high as outcome (Teti et al. 2019)
# ===========================================================================

confounders_2 = [
    "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_sent", "twitter_count", "news_sent", "rsi_30", "ma_50",
    "lag1", "lag2", "lag3", "lag5", "lag7",
]

df2 = (
    long[["px_high", "twitter_neg_count"] + confounders_2]
    .dropna()
    .reset_index(drop=True)
)

dml_2 = dml.DoubleMLPLR(
    dml.DoubleMLData(df2, y_col="px_high", d_cols="twitter_neg_count", x_cols=confounders_2),
    ml_l    = make_xgb(),
    ml_m    = make_xgb(),
    n_folds = 5,
    n_rep   = 20,
)
dml_2.fit()
dml_2.bootstrap(method="normal", n_rep_boot=1000)

print("\n===== RUN 2: Negative Tweet Count -> px_high (DoubleML PLR) =====")
print(f"N (complete cases): {len(df2):,}")
print(dml_2.summary)
print(dml_2.confint(level=0.95))
