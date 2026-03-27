"""
clean_data.py
Loads Bloomberg25MAR.xlsx, pivots to long format, removes NaNs,
computes daily returns and lags, and saves to data/processed/panel_long.csv.

Output columns (one row per ticker x trading day):
  date, ticker, px_open, px_close, px_high, px_low, mkt_cap, total_equity,
  debt_to_equity, volume, twitter_sent, twitter_count, news_sent,
  rsi_30, ma_50, twitter_neg_count, return, lag1, lag2, lag3, lag5, lag7
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
RAW_FILE  = ROOT / "data" / "raw" / "Bloomberg25MAR.xlsx"
OUT_FILE  = ROOT / "data" / "processed" / "panel_long.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# BLOCK 1 — LOAD & PIVOT
# ===========================================================================

raw = pd.read_excel(RAW_FILE, header=None)

# Row 3 (0-indexed): ticker names, one per 14-column block
# Row 5 (0-indexed): Bloomberg field codes (PX_OPEN, PX_OFFICIAL_CLOSE, ...)
h1 = raw.iloc[3].astype(str).str.strip().replace("nan", np.nan)   # tickers
h3 = raw.iloc[5].astype(str).str.strip().replace("nan", np.nan)   # field codes

# Fill ticker name across its 14 sub-columns
h1 = pd.Series(h1).ffill()

# Keep col 0 (dates) + all cols that have a real field code in row 5
keep_mask = [True] + [pd.notna(h3.iloc[i]) and h3.iloc[i] != "Dates"
                      for i in range(1, len(h3))]
keep_idx  = [i for i, k in enumerate(keep_mask) if k]

raw = raw.iloc[:, keep_idx].reset_index(drop=True)
h1  = h1.iloc[keep_idx].reset_index(drop=True)
h3  = h3.iloc[keep_idx].reset_index(drop=True)

# Data starts at row 6 (rows 0-5 are metadata/headers)
dat = raw.iloc[6:].reset_index(drop=True).copy()

# Build column names as "TICKER__FIELD"
col_names = ["date"] + [
    f"{ticker}__{field}"
    for ticker, field in zip(h1.iloc[1:], h3.iloc[1:])
]
seen = {}
unique_names = []
for name in col_names:
    if name in seen:
        seen[name] += 1
        unique_names.append(f"{name}__dup__{seen[name]}")
    else:
        seen[name] = 0
        unique_names.append(name)
dat.columns = unique_names

# Convert Excel serial date numbers to proper dates
dat["date"] = pd.to_datetime(
    pd.to_numeric(dat["date"], errors="coerce"),
    unit="D",
    origin="1899-12-30"
).dt.normalize()

# Pivot wide → long, then spread fields back to columns
long = dat.melt(id_vars="date", var_name="ticker__field", value_name="value")
long[["ticker", "field"]] = long["ticker__field"].str.split("__", n=1, expand=True)
long = long.drop(columns="ticker__field")

long = long.pivot_table(
    index=["date", "ticker"],
    columns="field",
    values="value",
    aggfunc="first"
).reset_index()
long.columns.name = None

# Rename Bloomberg field codes to snake_case
long = long.rename(columns={
    "PX_OPEN":                      "px_open",
    "PX_OFFICIAL_CLOSE":            "px_close",
    "PX_HIGH":                      "px_high",
    "PX_LOW":                       "px_low",
    "CUR_MKT_CAP":                  "mkt_cap",
    "TOTAL_EQUITY":                 "total_equity",
    "TOT_DEBT_TO_TOT_EQY":         "debt_to_equity",
    "PX_VOLUME":                    "volume",
    "TWITTER_SENTIMENT_DAILY_AVG":  "twitter_sent",
    "TWITTER_PUBLICATION_COUNT":    "twitter_count",
    "NEWS_SENTIMENT_DAILY_AVG":     "news_sent",
    "RSI_30D":                      "rsi_30",
    "MOV_AVG_50D":                  "ma_50",
    "TWITTER_NEG_SENTIMENT_COUNT":  "twitter_neg_count",
})

long = long.sort_values(["ticker", "date"]).reset_index(drop=True)


# ===========================================================================
# BLOCK 2 — REMOVE NaNs
# ===========================================================================

BLOOMBERG_NA = ["#N/A N/A", "#N/A", "#N/A Field Not Applicable"]

numeric_cols = [
    "px_open", "px_close", "px_high", "px_low", "mkt_cap", "total_equity",
    "debt_to_equity", "volume", "twitter_sent", "twitter_count", "news_sent",
    "rsi_30", "ma_50", "twitter_neg_count"
]

for col in numeric_cols:
    if col in long.columns:
        long[col] = long[col].replace(BLOOMBERG_NA, np.nan)
        long[col] = pd.to_numeric(long[col], errors="coerce")

# Drop non-trading days (weekends/holidays are all-NA for prices)
long = long.dropna(subset=["px_open", "px_close"]).reset_index(drop=True)


# ===========================================================================
# BLOCK 3 — DAILY RETURN + LAGS
# ===========================================================================

long["return"] = long["px_close"] - long["px_open"]

for n in [1, 2, 3, 5, 7]:
    long[f"lag{n}"] = long.groupby("ticker")["return"].shift(n)


# ===========================================================================
# SAVE
# ===========================================================================

long.to_csv(OUT_FILE, index=False)
print(f"Saved {len(long):,} rows x {long.shape[1]} cols → {OUT_FILE}")
print(long.dtypes)
print(long.head(10).to_string())
