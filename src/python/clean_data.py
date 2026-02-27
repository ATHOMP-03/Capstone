"""
clean_data.py
Replicates the data_prep, remove_nans, and returns chunks from
'Sentiment Analysis v1.Rmd', but in Python/pandas.

Output: data/processed/panel_long.csv
  one row per (ticker x trading day), ready for OLS with company FE.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (mirrors the `source` variable at the top of the Rmd)
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]   # Desktop/Capstone/
RAW_FILE  = ROOT / "data" / "raw" / "2025basic.xlsx"
OUT_FILE  = ROOT / "data" / "processed" / "panel_long.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# BLOCK 1 — LOAD & PIVOT  (mirrors `data_prep` chunk in the Rmd)
# ===========================================================================

# R: raw = read_excel(source, sheet=1, col_names=FALSE)
# Read with no header so we control all three header rows manually.
raw = pd.read_excel(RAW_FILE, header=None)

# R: h1 = raw[1, ] %>% unlist ... as.character()   <- tickers (row 0 in Python)
# R: h3 = raw[3, ] %>% unlist ... as.character()   <- field codes (row 2 in Python)
# Bloomberg exports company names only once per 3-column block; the rest are NaN.
h1 = raw.iloc[0].astype(str).str.strip().replace("nan", np.nan)   # tickers
h3 = raw.iloc[2].astype(str).str.strip().replace("nan", np.nan)   # PX_OPEN / PX_LAST / TWITTER_...

# R: h1 = tidyr::fill(tibble(x = h1), x, .direction = "down")$x
# Fill the ticker name rightward across its 3 sub-columns so every column
# knows which company it belongs to.
h1 = pd.Series(h1).ffill()

# R: keep_cols = c(1, which(!is.na(h3) & seq_along(h3) != 1))
# Keep column 0 (dates) plus every column that has a real field code in row 2.
# This drops the two extra NaN spacer columns Bloomberg inserts per company.
keep_mask  = [True] + [pd.notna(h3.iloc[i]) and i != 0 for i in range(1, len(h3))]
keep_idx   = [i for i, k in enumerate(keep_mask) if k]

raw  = raw.iloc[:, keep_idx].reset_index(drop=True)
h1   = h1.iloc[keep_idx].reset_index(drop=True)
h3   = h3.iloc[keep_idx].reset_index(drop=True)

# R: dat = raw[-c(1,2,3), ]   <- strip the three header rows, keep data only
dat = raw.iloc[3:].reset_index(drop=True).copy()

# R: names(dat)[1] = "date"
# R: names(dat) = c("date", map2_chr(h1[-1], h3[-1], ~ paste0(.x, "__", .y)))
# Build column names as "TICKER__FIELD" to enable a clean split later.
col_names = ["date"] + [
    f"{ticker}__{field}"
    for ticker, field in zip(h1.iloc[1:], h3.iloc[1:])
]
# R: names(dat) = make.unique(names(dat), sep = "__dup__")
# Deduplicate any names that collide (rare, but Bloomberg can export dupes).
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

# R: dat = dat %>% mutate(date = as.Date(as.numeric(date), origin = "1899-12-30"))
# Bloomberg / Excel stores dates as integer serial numbers counted from 1900-01-00,
# which in practice means origin = 1899-12-30.
dat["date"] = pd.to_datetime(
    pd.to_numeric(dat["date"], errors="coerce"),
    unit="D",
    origin="1899-12-30"
).dt.normalize()                                   # strip time component

# R: pivot_longer(cols=-date, names_to=c("ticker","field"), names_sep="__", values_to="value")
#    then pivot_wider(names_from=field, values_from=value)
# Python equivalent: melt wide → long, then pivot field values back to columns.
long = dat.melt(id_vars="date", var_name="ticker__field", value_name="value")
long[["ticker", "field"]] = long["ticker__field"].str.split("__", n=1, expand=True)
long = long.drop(columns="ticker__field")

long = long.pivot_table(
    index=["date", "ticker"],
    columns="field",
    values="value",
    aggfunc="first"      # each (date, ticker, field) should be unique
).reset_index()
long.columns.name = None

# R: transmute(date, ticker, PX_OPEN, PX_CLOSE=PX_LAST, twitter_sent=TWITTER_SENTIMENT_DAILY_AVG)
# Rename to match the variable names used in the rest of the Rmd.
long = long.rename(columns={
    "PX_LAST":                      "PX_CLOSE",
    "TWITTER_SENTIMENT_DAILY_AVG":  "twitter_sent",
})
long = long[["date", "ticker", "PX_OPEN", "PX_CLOSE", "twitter_sent"]]

# R: arrange(ticker, date)
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)


# ===========================================================================
# BLOCK 2 — REMOVE NaNs  (mirrors `remove nans` chunk in the Rmd)
# ===========================================================================

# R: mutate(across(everything(), ~ na_if(as.character(.x), "#N/A N/A")))
# R: mutate(across(everything(), ~ na_if(.x, "#N/A")))
# Bloomberg exports missing values as the literal strings "#N/A N/A" and "#N/A".
# Replace them with real NaN before numeric conversion.
BLOOMBERG_NA = ["#N/A N/A", "#N/A", "#N/A Field Not Applicable"]
for col in ["PX_OPEN", "PX_CLOSE", "twitter_sent"]:
    long[col] = long[col].replace(BLOOMBERG_NA, np.nan)

# R: mutate(across(c(PX_OPEN, PX_CLOSE, twitter_sent), ~ as.numeric(.x)))
long["PX_OPEN"]       = pd.to_numeric(long["PX_OPEN"],       errors="coerce")
long["PX_CLOSE"]      = pd.to_numeric(long["PX_CLOSE"],      errors="coerce")
long["twitter_sent"]  = pd.to_numeric(long["twitter_sent"],  errors="coerce")

# R: long = long %>% filter(!is.na(PX_OPEN), !is.na(PX_CLOSE))
# Drop non-trading days — weekends/holidays show up as all-NA rows for prices.
long = long.dropna(subset=["PX_OPEN", "PX_CLOSE"]).reset_index(drop=True)


# ===========================================================================
# BLOCK 3 — DAILY RETURN + LAGS  (mirrors `returns` chunk in the Rmd)
# ===========================================================================

# R: mutate(return = PX_CLOSE - PX_OPEN)
# Open-to-close dollar return — the dependent variable for the FE regression.
long["return"] = long["PX_CLOSE"] - long["PX_OPEN"]

# R: group_by(ticker) %>% mutate(lag1=lag(return,1), lag2=lag(return,2), ...)
# Compute lagged returns *within* each company (group_by ticker before shifting)
# so lags never bleed across companies. Matches the 1/2/3/5/7-day lags in the Rmd.
for n in [1, 2, 3, 5, 7]:
    long[f"lag{n}"] = long.groupby("ticker")["return"].shift(n)


# ===========================================================================
# SAVE
# ===========================================================================

long.to_csv(OUT_FILE, index=False)
print(f"Saved {len(long):,} rows x {long.shape[1]} cols → {OUT_FILE}")
print(long.dtypes)
print(long.head(10).to_string())
