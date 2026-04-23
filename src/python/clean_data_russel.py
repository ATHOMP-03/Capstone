"""
clean_data_russel.py
Loads Bloomberg Russell 3000 data from paired .xlsx files (one per year),
merges the price and sentiment files for each year, concatenates across years,
and saves to data/processed/russel_panel.csv.

FILE NAMING CONVENTION (place all files in data/raw/):
    bloomberg_russell_YYYY_prices.xlsx    ← PX_OPEN, PX_CLOSE, PX_HIGH, PX_LOW, PX_VOLUME
    bloomberg_russell_YYYY_sentiment.xlsx ← bid-ask spread, Twitter/news sentiment, RSI, etc.

Both files for the same year must contain the same set of tickers.
The merge key is (date, ticker). Rows present in prices but missing from sentiment
(or vice versa) are retained with NaN in the missing columns.

Output columns (one row per ticker x trading day):
  date, ticker, px_open, px_close, px_high, px_low, volume,
  mkt_cap, total_equity, debt_to_equity,
  twitter_sent, twitter_count, twitter_neg_count, twitter_pos_count,
  news_sent, news_pos_count, news_neg_count,
  rsi_30, ma_50, bid_ask_spread, vol_30d,
  twitter_neu_count (derived),
  return, lag1, lag2, lag3, lag5, lag7
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "russel_panel.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BLOOMBERG_NA = ["#N/A N/A", "#N/A", "#N/A Field Not Applicable"]

FIELD_RENAME = {
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
    "AVERAGE_BID_ASK_SPREAD":       "bid_ask_spread",
    "TWITTER_POS_SENTIMENT_COUNT":  "twitter_pos_count",
    "VOLATILITY_30D":               "vol_30d",
    "NEWS_POS_SENTIMENT_COUNT":     "news_pos_count",
    "NEWS_NEG_SENTIMENT_COUNT":     "news_neg_count",
}

NUMERIC_COLS = list(FIELD_RENAME.values())


# ===========================================================================
# PARSE — one Bloomberg wide-format file → tidy long DataFrame
# ===========================================================================

def parse_bloomberg_sheet(filepath: Path) -> pd.DataFrame:
    """
    Parse a single Bloomberg wide-format export into a tidy long DataFrame.
    Expected layout:
      Row 3 (0-indexed): ticker names, one per field-block (forward-filled)
      Row 5 (0-indexed): Bloomberg field codes
      Row 6+:            date column + data values
    """
    if filepath.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(filepath, header=None, engine="openpyxl")
    else:
        raw = pd.read_csv(filepath, header=None)

    h1 = raw.iloc[3].astype(str).str.strip().replace("nan", np.nan)  # tickers
    h3 = raw.iloc[5].astype(str).str.strip().replace("nan", np.nan)  # field codes

    h1 = pd.Series(h1).ffill()

    keep_mask = [True] + [pd.notna(h3.iloc[i]) and h3.iloc[i] != "Dates"
                          for i in range(1, len(h3))]
    keep_idx  = [i for i, k in enumerate(keep_mask) if k]

    raw = raw.iloc[:, keep_idx].reset_index(drop=True)
    h1  = h1.iloc[keep_idx].reset_index(drop=True)
    h3  = h3.iloc[keep_idx].reset_index(drop=True)

    dat = raw.iloc[6:].reset_index(drop=True).copy()

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

    # Smart date parsing — datetime objects and Excel serial integers
    date_raw = dat.iloc[:, 0]
    if pd.api.types.is_datetime64_any_dtype(date_raw) or hasattr(date_raw.dropna().iloc[0], "year"):
        dat["date"] = pd.to_datetime(date_raw, errors="coerce").dt.normalize()
    else:
        dat["date"] = pd.to_datetime(
            pd.to_numeric(date_raw, errors="coerce"),
            unit="D", origin="1899-12-30"
        ).dt.normalize()

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

    return long


# ===========================================================================
# DISCOVER FILES — group prices/sentiment pairs by year
# Handles both single-year files (bloomberg_russell_YYYY_prices.xlsx) and
# multi-year files (bloomberg_russell_YYYY_YYYY_prices.xlsx). Multi-year
# price files are registered for every year they span; they are parsed once
# and then filtered by year at merge time.
# ===========================================================================

prices_files    = sorted(RAW_DIR.glob("bloomberg_russell_*_prices.xlsx"))
sentiment_files = sorted(RAW_DIR.glob("bloomberg_russell_*_sentiment.xlsx"))

single_year_re = re.compile(r"bloomberg_russell_(\d{4})_prices\.xlsx$")
multi_year_re  = re.compile(r"bloomberg_russell_(\d{4})_(\d{4})_prices\.xlsx$")
sent_year_re   = re.compile(r"bloomberg_russell_(\d{4})_sentiment\.xlsx$")

# Map year string → prices file path (multi-year files register for each year they cover)
prices_by_year: dict[str, Path] = {}
for f in prices_files:
    m = multi_year_re.search(f.name)
    if m:
        start_yr, end_yr = int(m.group(1)), int(m.group(2))
        for yr in range(start_yr, end_yr + 1):
            prices_by_year[str(yr)] = f
        print(f"  Multi-year prices file: {f.name}  → years {start_yr}–{end_yr}")
    else:
        m = single_year_re.search(f.name)
        if m:
            prices_by_year[m.group(1)] = f

sentiment_by_year: dict[str, Path] = {}
for f in sentiment_files:
    m = sent_year_re.search(f.name)
    if m:
        sentiment_by_year[m.group(1)] = f

all_years = sorted(set(prices_by_year) | set(sentiment_by_year))

if not all_years:
    raise FileNotFoundError(
        f"No files found in {RAW_DIR}.\n"
        "Expected naming:\n"
        "  bloomberg_russell_YYYY_prices.xlsx          (single year)\n"
        "  bloomberg_russell_YYYY_YYYY_prices.xlsx     (multi-year span)\n"
        "  bloomberg_russell_YYYY_sentiment.xlsx"
    )

print(f"Years detected: {all_years}")
for yr in all_years:
    p = prices_by_year.get(yr, "MISSING")
    s = sentiment_by_year.get(yr, "MISSING")
    print(f"  {yr}  prices={p if isinstance(p, str) else p.name}"
          f"  sentiment={s if isinstance(s, str) else s.name}")

# Cache parsed prices DataFrames so multi-year files are only read from disk once
_prices_cache: dict[Path, pd.DataFrame] = {}


# ===========================================================================
# MERGE — prices + sentiment per year, then concat across years
# ===========================================================================

yearly_panels = []

for yr in all_years:
    p_file = prices_by_year.get(yr)
    s_file = sentiment_by_year.get(yr)

    if p_file is None:
        print(f"  [{yr}] WARNING: no prices file — skipping year.")
        continue
    if s_file is None:
        print(f"  [{yr}] WARNING: no sentiment file — skipping year.")
        continue

    # Parse prices — use cache so multi-year files are only read once
    if p_file not in _prices_cache:
        print(f"  [{yr}] Parsing {p_file.name} ...", end=" ")
        _prices_cache[p_file] = parse_bloomberg_sheet(p_file)
        print(f"{len(_prices_cache[p_file]):,} rows")
    else:
        print(f"  [{yr}] Reusing cached {p_file.name}")

    # Filter to just this calendar year so multi-year files don't bleed across years
    prices_all = _prices_cache[p_file]
    prices = prices_all[prices_all["date"].dt.year == int(yr)].copy()
    print(f"         → {len(prices):,} rows after filtering to {yr}")

    print(f"  [{yr}] Parsing {s_file.name} ...", end=" ")
    sentiment = parse_bloomberg_sheet(s_file)
    print(f"{len(sentiment):,} rows")

    # Outer join: keep all date×ticker combinations present in either file
    merged = prices.merge(sentiment, on=["date", "ticker"], how="outer",
                          suffixes=("_prices", "_sentiment"))

    # Resolve any column name collisions from suffixes (should be none if the
    # files have disjoint fields, but guard against accidental overlap)
    for col in list(merged.columns):
        if col.endswith("_prices"):
            base = col[:-7]
            alt  = base + "_sentiment"
            if alt in merged.columns:
                # Both files contained the same field — prices takes precedence
                merged[base] = merged[col].combine_first(merged[alt])
                merged = merged.drop(columns=[col, alt])
            else:
                merged = merged.rename(columns={col: base})
        elif col.endswith("_sentiment") and col[:-10] not in merged.columns:
            merged = merged.rename(columns={col: col[:-10]})

    yearly_panels.append(merged)

if not yearly_panels:
    raise RuntimeError("No year panels were assembled. Check that prices and sentiment "
                       "files exist for at least one year.")

long = pd.concat(yearly_panels, ignore_index=True)


# ===========================================================================
# BLOCK 2 — RENAME & CLEAN
# ===========================================================================

long = long.rename(columns=FIELD_RENAME)
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

for col in NUMERIC_COLS:
    if col in long.columns:
        long[col] = long[col].replace(BLOOMBERG_NA, np.nan)
        long[col] = pd.to_numeric(long[col], errors="coerce")

# Drop non-trading days (all-NA prices)
long = long.dropna(subset=["px_open", "px_close"]).reset_index(drop=True)


# ===========================================================================
# BLOCK 3 — DERIVED FIELDS
# ===========================================================================

if {"twitter_pos_count", "twitter_neg_count", "twitter_count"}.issubset(long.columns):
    long["twitter_neu_count"] = (
        long["twitter_count"] - long["twitter_pos_count"] - long["twitter_neg_count"]
    ).clip(lower=0)

long["return"] = long["px_close"] - long["px_open"]

for n in [1, 2, 3, 5, 7]:
    long[f"lag{n}"] = long.groupby("ticker")["return"].shift(n)


# ===========================================================================
# SAVE
# ===========================================================================

long.to_csv(OUT_FILE, index=False)
print(f"\nSaved {len(long):,} rows x {long.shape[1]} cols → {OUT_FILE}")
print(f"Date range: {long['date'].min().date()} → {long['date'].max().date()}")
print(f"Tickers: {long['ticker'].nunique()}")
print(f"Columns: {list(long.columns)}")
