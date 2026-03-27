"""
news_sent.py
Replicates ml_analysis.py with news_sent as the treatment variable.

If the same sign pattern holds (negative ATE for continuous sentiment,
positive ATE for negative-day volume), tweet volume is not the driver —
it's Option 3: content and volume have opposing effects regardless of channel.

Run 1: news_sent as continuous treatment
Run 2: neg_news_vol = twitter_count * I(news_sent < 0) as continuous treatment
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
# RUN 1 — News sentiment as continuous treatment
# news_sent moves to treatment; twitter_sent moves into confounder set
# ===========================================================================

confounders_1 = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_sent", "twitter_count", "rsi_30", "ma_50",
    "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7",
]

df1 = (
    long[["return", "news_sent"] + confounders_1]
    .dropna()
    .reset_index(drop=True)
)

dml_1 = dml.DoubleMLPLR(
    dml.DoubleMLData(df1, y_col="return", d_cols="news_sent", x_cols=confounders_1),
    ml_l    = make_xgb(),
    ml_m    = make_xgb(),
    n_folds = 5,
    n_rep   = 20,
)
dml_1.fit()
dml_1.bootstrap(method="normal", n_rep_boot=1000)

print("\n===== RUN 1: News Sentiment (DoubleML PLR) =====")
print(f"N (complete cases): {len(df1):,}")
print(dml_1.summary)
print(dml_1.confint(level=0.95))


# ===========================================================================
# RUN 2 — Negative news volume as continuous treatment
# neg_news_vol = twitter_count * I(news_sent < 0)
# news_sent moves to confounder set (determines treatment eligibility)
# ===========================================================================

long["neg_news_vol"] = long["twitter_count"] * (long["news_sent"] < 0).astype(int)

confounders_2 = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_sent", "news_sent", "rsi_30", "ma_50",
    "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7",
]

df2 = (
    long[["return", "neg_news_vol"] + confounders_2]
    .dropna()
    .reset_index(drop=True)
)

dml_2 = dml.DoubleMLPLR(
    dml.DoubleMLData(df2, y_col="return", d_cols="neg_news_vol", x_cols=confounders_2),
    ml_l    = make_xgb(),
    ml_m    = make_xgb(),
    n_folds = 5,
    n_rep   = 20,
)
dml_2.fit()
dml_2.bootstrap(method="normal", n_rep_boot=1000)

print("\n===== RUN 2: Negative News Volume (DoubleML PLR) =====")
print(f"N (complete cases): {len(df2):,}")
print(dml_2.summary)
print(dml_2.confint(level=0.95))
