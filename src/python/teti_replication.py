"""
teti_replication.py
Replication of Teti, Dallocchio & Aniasi (2019)
"The relationship between twitter and stock prices. Evidence from the US technology industry"
Technological Forecasting & Social Change 149, 119747.

Key methodological choices carried over from the original:
  - Dependent variable: percentage return (px_close - px_open) / px_open * 100
  - Baxter-King bandpass filter applied to price series to remove trend + high-freq noise
  - SET 1: Simple pooled OLS (heteroskedasticity-robust HC3 errors)
  - SET 2: Panel FE with binary treatment (high vs. low social-media coverage)
           Interaction: twitter_sent_lag_k x group_dummy, k = 0..4
  - SET 3: Custom polarity-weighted sentiment index (ts_b) — replicates the
           "third set of regressions" that strips neutral-tweet dilution from
           the Bloomberg composite score

Data requirements:
  A) Available in panel_long.csv:
     twitter_sent, twitter_neg_count, news_sent, volume, mkt_cap, px_open,
     px_close, px_high, px_low

  B) Requires extended Bloomberg pull (see bloomberg_pull_extended.py):
     twitter_pos_count  — TWITTER_POS_SENTIMENT_COUNT
     twitter_neu_count  — imputed as twitter_count - twitter_pos_count - twitter_neg_count
     market_return      — SPX Index (or CCMP Index) PX_LAST

  Note on group dummy: Bloomberg no longer provides TWITTER_FOLLOWERS.
  The high/low social-media coverage split uses a market-cap median split
  as the permanent proxy. Large-cap firms have materially greater social
  media reach, making this a reasonable structural proxy.

  When B columns are absent the script falls back:
     - Model 2 of SET 1 is skipped (needs pos/neu counts)
     - ts_b falls back to the amplified Bloomberg composite

Usage:
    python src/python/teti_replication.py
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from statsmodels.tsa.filters.bk_filter import bkfilter
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
IN_FILE     = ROOT / "data" / "processed" / "panel_long.csv"
EXT_FILE    = ROOT / "data" / "processed" / "panel_long_extended.csv"
OUT_DIR     = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data (prefer extended file if available)
# ---------------------------------------------------------------------------
src = EXT_FILE if EXT_FILE.exists() else IN_FILE
long = pd.read_csv(src, parse_dates=["date"])
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(long):,} rows x {long.shape[1]} cols from {src.name}")

# ---------------------------------------------------------------------------
# Check which extended columns are present
# ---------------------------------------------------------------------------
HAS_POS_NEU    = {"twitter_pos_count", "twitter_neu_count"}.issubset(long.columns)
HAS_MARKET_RET = "market_return" in long.columns

if not HAS_POS_NEU:
    print("\n[WARNING] twitter_pos_count / twitter_neu_count not found.")
    print("  SET 1 Model 2 and SET 3 will be skipped or approximated.")
    print("  Pull TWITTER_POS_SENTIMENT_COUNT via bloomberg_pull_extended.py\n")

if not HAS_MARKET_RET:
    print("[WARNING] market_return not found.")
    print("  Market index control will be omitted from SET 1 OLS models.\n")


# ===========================================================================
# BLOCK 1 — VARIABLE CONSTRUCTION
# ===========================================================================

# Percentage return (Teti et al. dependent variable)
long["pct_return"] = (long["px_close"] - long["px_open"]) / long["px_open"] * 100

# Volume in millions (to reduce coefficient magnitude, matches Teti scale)
long["volume_mil"] = long["volume"] / 1e6


# ------ Baxter-King filter ------------------------------------------------
# Applied to the closing price series within each ticker.
# Removes trend (frequencies below 1/90 trading days) and high-freq noise
# (frequencies above 1/5 trading days), isolating the cyclical component.
# K=12 truncation points (loses 12 obs from each end of each series).
# After filtering, take first difference to recover the filtered return.
# Note: for already-stationary daily returns the filter has modest impact;
#       we apply it to price levels per Teti et al.
BK_LOW  = 5    # minimum cycle length in trading days
BK_HIGH = 90   # maximum cycle length in trading days
BK_K    = 12   # number of leads/lags in approximation

def apply_bk_to_price(series: pd.Series) -> pd.Series:
    """Apply Baxter-King filter to a price series; return filtered first-diff."""
    arr = series.values.astype(float)
    # Need at least 2*K+1 non-NaN observations
    mask = ~np.isnan(arr)
    if mask.sum() < 2 * BK_K + 1:
        return pd.Series(np.nan, index=series.index)
    filtered = np.full_like(arr, np.nan)
    idx_valid = np.where(mask)[0]
    filtered_vals = bkfilter(arr[mask], low=BK_LOW, high=BK_HIGH, K=BK_K)
    # bkfilter returns NaN at ends; map back to original index
    filtered[idx_valid] = filtered_vals
    # First difference of filtered price = filtered return
    diff = np.diff(filtered, prepend=np.nan)
    return pd.Series(diff, index=series.index)

long["bk_return"] = (
    long.groupby("ticker")["px_close"]
    .transform(apply_bk_to_price)
)

# ------ Sentiment lags (for SET 2, lags 0-4) ------------------------------
for k in range(5):
    long[f"ts_lag_{k}"] = long.groupby("ticker")["twitter_sent"].shift(k)

# ------ Sentiment lags for BK-filtered index if pos/neu available ----------
if HAS_POS_NEU:
    # ts_b: polarity-weighted index that strips neutral-tweet dilution
    # Formula: (pos - neg) / (pos + neg + neu)
    # Neutral count is included in the denominator but not numerator,
    # so it acts as a volume weight that dilutes extremes less than the
    # Bloomberg composite (which uses confidence-weighted neutral scores).
    denom = (long["twitter_pos_count"] + long["twitter_neu_count"] + long["twitter_neg_count"])
    long["ts_b"] = (long["twitter_pos_count"] - long["twitter_neg_count"]) / denom.replace(0, np.nan)
else:
    # Fallback: approximate ts_b by clipping the composite sentiment
    # This removes the neutral-tweet dilution by amplifying the existing index
    # ts_b_approx ∈ [-1, 1], scaled to give more distance from zero
    # Not equivalent to the paper's formula but directionally consistent
    long["ts_b"] = long["twitter_sent"].apply(lambda x: np.sign(x) * (x ** 2) if pd.notna(x) else np.nan)

for k in range(5):
    long[f"tsb_lag_{k}"] = long.groupby("ticker")["ts_b"].shift(k)

# ------ Group dummy (high vs. low social-media coverage) ------------------
# Bloomberg no longer provides TWITTER_FOLLOWERS. Group dummy is permanently
# defined via market-cap median split: large-cap firms have substantially
# greater social media reach and serve as the high-coverage study group.
ticker_med_cap = long.groupby("ticker")["mkt_cap"].median()
high_cap_tickers = ticker_med_cap[ticker_med_cap >= ticker_med_cap.median()].index
long["group"] = long["ticker"].isin(high_cap_tickers).astype(int)
group_note = "group_dummy = 1 if ticker median mkt_cap >= cross-sectional median (market-cap proxy for social-media reach)"
print(f"Group dummy: {long['group'].sum():,} obs in high-coverage group, "
      f"{(long['group'] == 0).sum():,} in low-coverage group.")


# ===========================================================================
# HELPERS
# ===========================================================================

def stars(pval: float) -> str:
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def print_result(label: str, result, params: list = None):
    """Print a clean summary of an OLS/FE result for selected params."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print('='*65)
    params = params or list(result.params.index)
    for p in params:
        if p in result.params:
            coef = result.params[p]
            se   = result.bse[p]
            pval = result.pvalues[p]
            print(f"  {p:<40s}  {coef:>10.6f}  ({se:.6f}) {stars(pval)}")
    try:
        n = int(result.nobs)
    except Exception:
        n = "?"
    try:
        r2 = f"{result.rsquared:.4f}"
    except Exception:
        r2 = "?"
    print(f"  N = {n},  R² = {r2}")


def build_latex_table(rows: list, caption: str, label: str, notes: str = "") -> str:
    """
    rows: list of dicts with keys 'label', 'coef', 'se', 'pval'
    Returns a LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\hline\hline",
        r"Variable & Coefficient & (SE) \\",
        r"\hline",
    ]
    for r in rows:
        coef_str = f"{r['coef']:.6f}{stars(r['pval'])}"
        se_str   = f"({r['se']:.6f})"
        lines.append(f"{r['label'].replace('_', r'_')} & {coef_str} & {se_str} \\\\")
    if notes:
        ncols = 3
        lines.append(r"\hline\hline")
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize \\textit{{Notes:}} {notes}}} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ===========================================================================
# SET 1 — POOLED OLS (Table 3 equivalent)
#
# Teti et al. run simple OLS without firm FE in their first set.
# Controls: market index return + volume.
# Heteroskedasticity-robust standard errors (HC3).
# ===========================================================================

print("\n" + "#"*65)
print("  SET 1: Pooled OLS Models")
print("#"*65)

base_controls = " + volume_mil"
if HAS_MARKET_RET:
    base_controls += " + market_return"

# Model 1: Twitter sentiment
formula_1 = f"pct_return ~ twitter_sent{base_controls}"
m1 = smf.ols(formula_1, data=long).fit(cov_type="HC3")
print_result("SET 1 Model 1: twitter_sent → pct_return", m1, ["twitter_sent"])

# Model 2: Polarity-broken tweet counts (needs pos/neu data)
if HAS_POS_NEU:
    formula_2 = f"pct_return ~ twitter_pos_count + twitter_neu_count + twitter_neg_count"
    m2 = smf.ols(formula_2, data=long.dropna(subset=["twitter_pos_count","twitter_neu_count","twitter_neg_count"])).fit(cov_type="HC3")
    print_result("SET 1 Model 2: polarity-broken tweet counts → pct_return", m2,
                 ["twitter_pos_count", "twitter_neu_count", "twitter_neg_count"])
else:
    m2 = None
    print("\n[SKIPPED] SET 1 Model 2: requires twitter_pos_count and twitter_neu_count")

# Model 3: News sentiment
formula_3 = f"pct_return ~ news_sent{base_controls}"
m3 = smf.ols(formula_3, data=long).fit(cov_type="HC3")
print_result("SET 1 Model 3: news_sent → pct_return", m3, ["news_sent"])

# Model 4: Twitter + news together
formula_4 = f"pct_return ~ twitter_sent + news_sent{base_controls}"
m4 = smf.ols(formula_4, data=long).fit(cov_type="HC3")
print_result("SET 1 Model 4: twitter_sent + news_sent → pct_return", m4, ["twitter_sent", "news_sent"])


# ===========================================================================
# SET 2 — FIXED EFFECTS WITH BINARY TREATMENT (Table 4 equivalent)
#
# Specification (for each lag k = 0, 1, 2, 3, 4):
#
#   pct_return_{ct} = β0 + β1*ts_lag_{k,ct}
#                   + β2*group_{ct}
#                   + β3*(ts_lag_{k,ct} × group_{ct})
#                   + β4*volume_mil_{ct}
#                   + EntityEffects + ε_{ct}
#
# β3 is the key parameter: marginal effect of sentiment for high-coverage firms
# (study group) relative to low-coverage firms (control group).
# Significant β3 indicates that social-media coverage amplifies price-sentiment link.
# ===========================================================================

print("\n" + "#"*65)
print(f"  SET 2: Panel FE with Binary Treatment")
print(f"  ({group_note})")
print("#"*65)

panel = long.copy()
panel = panel.set_index(["ticker", "date"])

set2_results = {}

for k in range(5):
    lag_col = f"ts_lag_{k}"
    inter_col = f"inter_{k}"
    panel[inter_col] = panel[lag_col] * panel["group"]

    sub = panel[[lag_col, inter_col, "group", "volume_mil", "pct_return"]].dropna()
    if len(sub) < 100:
        print(f"  Lag {k}: insufficient data, skipping.")
        continue

    formula = f"pct_return ~ {lag_col} + group + {inter_col} + volume_mil + EntityEffects"
    try:
        mod = PanelOLS.from_formula(formula, data=sub)
        res = mod.fit(cov_type="heteroskedastic")
        set2_results[k] = res
        print_result(f"SET 2 Lag {k}: ts_lag_{k} + group_dummy interaction → pct_return",
                     res, [lag_col, "group", inter_col])
    except Exception as e:
        print(f"  Lag {k} failed: {e}")


# ===========================================================================
# SET 3 — CUSTOM POLARITY-WEIGHTED INDEX ts_b (Table 5 equivalent)
#
# Same structure as SET 2 but replaces twitter_sent with ts_b.
# ts_b = (pos - neg) / (pos + neg + neu), which gives more weight to extreme
# sentiment by excluding neutral tweets from the numerator.
# Teti et al. found ts_b produces a stronger and earlier effect (1-day lag
# is significant) vs. Bloomberg composite (3-day lag significant).
# ===========================================================================

print("\n" + "#"*65)
print("  SET 3: Custom Polarity-Weighted Index (ts_b)")
if not HAS_POS_NEU:
    print("  [APPROXIMATION] Using amplified Bloomberg composite (ts_b ≈ sign(s)*s²)")
    print("  Pull pos/neu counts for exact replication.")
print("#"*65)

set3_results = {}

for k in range(5):
    lag_col   = f"tsb_lag_{k}"
    inter_col = f"interb_{k}"
    panel[inter_col] = panel[lag_col] * panel["group"]

    sub = panel[[lag_col, inter_col, "group", "volume_mil", "pct_return"]].dropna()
    if len(sub) < 100:
        print(f"  Lag {k}: insufficient data, skipping.")
        continue

    formula = f"pct_return ~ {lag_col} + group + {inter_col} + volume_mil + EntityEffects"
    try:
        mod = PanelOLS.from_formula(formula, data=sub)
        res = mod.fit(cov_type="heteroskedastic")
        set3_results[k] = res
        print_result(f"SET 3 Lag {k}: tsb_lag_{k} + group_dummy interaction → pct_return",
                     res, [lag_col, "group", inter_col])
    except Exception as e:
        print(f"  Lag {k} failed: {e}")


# ===========================================================================
# EXPORT — LaTeX tables
# ===========================================================================

def coef_stars(result, param: str) -> tuple:
    coef = result.params[param]
    try:
        se   = result.bse[param]
        pval = result.pvalues[param]
    except Exception:
        se, pval = np.nan, 1.0
    return coef, se, pval


# SET 1 table
set1_rows = []
for label_str, model, param in [
    ("twitter\\_sent (M1)", m1, "twitter_sent"),
    ("news\\_sent (M3)", m3, "news_sent"),
    ("twitter\\_sent (M4)", m4, "twitter_sent"),
    ("news\\_sent (M4)", m4, "news_sent"),
]:
    if model is None:
        continue
    c, s, p = coef_stars(model, param)
    set1_rows.append({"label": label_str, "coef": c, "se": s, "pval": p})

if set1_rows:
    tex_set1 = build_latex_table(
        set1_rows,
        caption="Teti et al. (2019) Replication --- SET 1: Pooled OLS",
        label="tab:teti_set1",
        notes=(
            "HC3-robust SEs. Dependent variable: percentage open-to-close return. "
            "Controls include trading volume and (if available) market index return. "
            "*** p$<$0.01, ** p$<$0.05, * p$<$0.1."
        ),
    )
    out_path = OUT_DIR / "teti_set1.tex"
    out_path.write_text(tex_set1)
    print(f"\nSaved {out_path}")

# SET 2 summary table (key coefficient: interaction term at each lag)
set2_rows = []
for k, res in set2_results.items():
    inter_col = f"inter_{k}"
    if inter_col in res.params:
        c, s, p = coef_stars(res, inter_col)
        set2_rows.append({"label": f"ts\\_lag\\_{k} $\\times$ group", "coef": c, "se": s, "pval": p})

if set2_rows:
    tex_set2 = build_latex_table(
        set2_rows,
        caption="Teti et al. (2019) Replication --- SET 2: FE with Binary Treatment (Interaction $\\beta_3$)",
        label="tab:teti_set2",
        notes=(
            "Heteroskedasticity-robust SEs. Entity FE. Dependent variable: percentage return. "
            "Interaction term captures marginal effect for high-coverage firms. "
            f"{group_note}. *** p$<$0.01, ** p$<$0.05, * p$<$0.1."
        ),
    )
    out_path = OUT_DIR / "teti_set2.tex"
    out_path.write_text(tex_set2)
    print(f"Saved {out_path}")

# SET 3 summary table
set3_rows = []
for k, res in set3_results.items():
    inter_col = f"interb_{k}"
    if inter_col in res.params:
        c, s, p = coef_stars(res, inter_col)
        label = "ts\\_b" if HAS_POS_NEU else "tsb\\_approx"
        set3_rows.append({"label": f"{label}\\_lag\\_{k} $\\times$ group", "coef": c, "se": s, "pval": p})

if set3_rows:
    ts_b_note = (
        "ts\\_b = (pos - neg)/(pos + neg + neu) per Teti et al."
        if HAS_POS_NEU else
        "ts\\_b approximated as sign(s)·s² (neutral-tweet dilution proxy; exact formula requires TWITTER\\_POS\\_SENTIMENT\\_COUNT)."
    )
    tex_set3 = build_latex_table(
        set3_rows,
        caption="Teti et al. (2019) Replication --- SET 3: Custom Polarity-Weighted Index (ts\\_b)",
        label="tab:teti_set3",
        notes=(
            f"Heteroskedasticity-robust SEs. Entity FE. {ts_b_note}. *** p$<$0.01, ** p$<$0.05, * p$<$0.1."
        ),
    )
    out_path = OUT_DIR / "teti_set3.tex"
    out_path.write_text(tex_set3)
    print(f"Saved {out_path}")

print("\nTeti et al. replication complete.")
