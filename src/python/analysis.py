"""
analysis.py
Python equivalent of src/r/analysis.R

Fixed effects regression analysis of Twitter sentiment on stock returns.
Uses linearmodels.PanelOLS (equivalent to plm/feols).

Models:
  1. Simple FE regression
  2. FE with full confounder matrix
  3. Continuous negative sentiment treatment
  4. twitter_neg_count as standalone treatment (Teti et al. 2019)

Robustness checks:
  - Placebo: does twitter_sent predict yesterday's return?
  - No-reversal: lagged sentiment -> return (Gu & Kurov 2020)
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pathlib import Path

np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "panel_long.csv"

long = pd.read_csv(IN_FILE, parse_dates=["date"])
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

# linearmodels requires a MultiIndex of (entity, time)
long = long.set_index(["ticker", "date"])


def run_fe(formula, data, label, note=""):
    model  = PanelOLS.from_formula(formula + " + EntityEffects", data=data)
    result = model.fit(cov_type="heteroskedastic")
    print(f"\n{'='*60}")
    print(f"  {label}")
    if note:
        print(f"  {note}")
    print('='*60)
    print(result.summary.tables[1])   # coefficients table only
    return result


# ===========================================================================
# MODEL 1 — Simple FE
# ===========================================================================
m1 = run_fe("return ~ twitter_sent", long, "MODEL 1: Simple FE")


# ===========================================================================
# MODEL 2 — FE with confounder matrix
# ===========================================================================
confounders = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "news_sent", "rsi_30", "ma_50", "twitter_neg_count",
]
m2 = run_fe(
    f"return ~ twitter_sent + {' + '.join(confounders)}",
    long,
    "MODEL 2: FE with confounders",
)


# ===========================================================================
# MODEL 3 — Continuous negative sentiment treatment
# ===========================================================================
long["neg_twitter_sent"] = long["twitter_sent"].clip(upper=0)

m3 = run_fe("return ~ neg_twitter_sent", long, "MODEL 3: Continuous negative sentiment")


# ===========================================================================
# MODEL 4 — twitter_neg_count as standalone treatment (Teti et al. 2019)
# Polarity-broken tweet counts are significant; total count is not.
# Using twitter_neg_count sidesteps neutral-tweet dilution in Bloomberg index.
# ===========================================================================
m4 = run_fe(
    "return ~ twitter_neg_count",
    long,
    "MODEL 4: Negative tweet count as treatment",
    note="Per Teti et al. (2019): polarity-broken counts outperform total count.",
)


# ===========================================================================
# PLACEBO TEST
# Regress yesterday's return (lag1) on today's twitter_sent.
# Coefficient should be insignificant — if not, reverse causality is present.
# ===========================================================================
placebo = run_fe(
    "lag1 ~ twitter_sent",
    long,
    "PLACEBO: Does twitter_sent predict yesterday's return?",
    note="Coefficient should be insignificant under a causal story.",
)


# ===========================================================================
# NO-REVERSAL TEST — Gu & Kurov (2020)
# Regress return on multiple lags of twitter_sent.
# Only sent_lag1 should be significant; lags 2-7 should be near zero.
# ===========================================================================
long["sent_lag1"] = long.groupby(level="ticker")["twitter_sent"].shift(1)
long["sent_lag2"] = long.groupby(level="ticker")["twitter_sent"].shift(2)
long["sent_lag3"] = long.groupby(level="ticker")["twitter_sent"].shift(3)
long["sent_lag5"] = long.groupby(level="ticker")["twitter_sent"].shift(5)
long["sent_lag7"] = long.groupby(level="ticker")["twitter_sent"].shift(7)

no_reversal = run_fe(
    "return ~ sent_lag1 + sent_lag2 + sent_lag3 + sent_lag5 + sent_lag7",
    long,
    "NO-REVERSAL TEST: Lagged sentiment -> return",
    note="Only sent_lag1 should be significant under a causal interpretation.",
)
