"""
gu_kurov_temporal.py
Extended Gu & Kurov (2020) replication — temporal heterogeneity analysis.

Runs the three core Gu & Kurov tests across rolling time windows to track
how Twitter's role in predicting returns has changed over time (2016–2026).

Tests per window:
  1. FMB return predictability  — twitter_sent_lag1 coefficient (Table 2 spec)
  2. No-reversal test           — twitter_sent_lag1..5 jointly (Table 3 spec)
  3. Long-short strategy        — top/bottom decile daily L-S return (Table 6 spec)

Outputs (saved to data/processed/):
  gk_temporal_monthly.csv
  gk_temporal_quarterly.csv
  gk_temporal_annual.csv

Each row = one time window. Columns = estimator, SE, p-value for every
coefficient + long-short performance statistics.

Usage:
    python src/python/gu_kurov_temporal.py
"""

import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels.panel import FamaMacBeth

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
IN_FILE  = ROOT / "data" / "processed" / "panel_long.csv"
EXT_FILE = ROOT / "data" / "processed" / "panel_long_extended.csv"
OUT_DIR  = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
src  = EXT_FILE if EXT_FILE.exists() else IN_FILE
long = pd.read_csv(src, parse_dates=["date"])
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(long):,} rows x {long.shape[1]} cols from {src.name}")

HAS_SPREAD = "bid_ask_spread" in long.columns

# ===========================================================================
# BLOCK 1 — VARIABLE CONSTRUCTION
# ===========================================================================

long["px_open_next"] = long.groupby("ticker")["px_open"].shift(-1)
long["return_oo"] = (
    (long["px_open_next"] - long["px_open"]) / long["px_open"] * 100
)

log_H = np.log(long["px_high"].clip(lower=1e-8))
log_L = np.log(long["px_low"].clip(lower=1e-8))
log_O = np.log(long["px_open"].clip(lower=1e-8))
log_C = np.log(long["px_close"].clip(lower=1e-8))
long["vol_rs"] = (
    (log_H - log_C) * (log_H - log_O) + (log_L - log_C) * (log_L - log_O)
).clip(lower=0) * 100

mean_vol        = long.groupby("ticker")["volume"].transform("mean")
long["abnorm_vol"]  = ((long["volume"] - mean_vol) / mean_vol) * 100
long["log_mkt_cap"] = np.log(long["mkt_cap"].clip(lower=1e-8))

control_base = ["return_oo", "abnorm_vol", "vol_rs", "log_mkt_cap"]
if HAS_SPREAD:
    control_base.append("bid_ask_spread")

for var in control_base:
    for k in range(1, 6):
        long[f"{var}_lag{k}"] = long.groupby("ticker")[var].shift(k)

for k in range(1, 6):
    long[f"twitter_sent_lag{k}"] = long.groupby("ticker")["twitter_sent"].shift(k)

long["news_sent_lag1"] = long.groupby("ticker")["news_sent"].shift(1)

# ===========================================================================
# BLOCK 2 — FAMA-FRENCH RISK ADJUSTMENT (full-sample betas)
# Betas estimated on the full sample for stability; residuals used in windows.
# ===========================================================================

HAS_ADJ_RETURN = False
try:
    import statsmodels.api as sm
    import pandas_datareader.data as web

    ff3  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                           start=long["date"].min(), end=long["date"].max())[0]
    mom  = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                           start=long["date"].min(), end=long["date"].max())[0]
    ff3.index = pd.to_datetime(ff3.index, format="%Y%m%d")
    mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
    ff_factors = ff3.join(mom, how="inner") / 100
    ff_factors.columns = [c.strip() for c in ff_factors.columns]

    merged    = long.merge(
        ff_factors.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb",
                                   "HML": "hml", "Mom": "mom"}),
        left_on="date", right_index=True, how="left"
    )
    residuals = np.full(len(merged), np.nan)
    for ticker, grp in merged.groupby("ticker"):
        sub = grp[["return_oo", "mkt_rf", "smb", "hml", "mom"]].dropna()
        if len(sub) < 30:
            continue
        X = sm.add_constant(sub[["mkt_rf", "smb", "hml", "mom"]])
        residuals[sub.index] = sm.OLS(sub["return_oo"], X).fit().resid.values
    long["return_oo_adj"] = residuals

    for k in range(1, 6):
        long[f"return_oo_adj_lag{k}"] = long.groupby("ticker")["return_oo_adj"].shift(k)

    HAS_ADJ_RETURN = long["return_oo_adj"].notna().sum() > 1000
    print(f"Risk-adjusted returns: {long['return_oo_adj'].notna().sum():,} obs")
except Exception as e:
    print(f"[WARNING] FF download/adjustment failed: {e}\nUsing raw returns.")


# ===========================================================================
# HELPERS
# ===========================================================================

DEFAULT_OTH_VARS = (
    ["return_oo_adj"] if HAS_ADJ_RETURN else ["return_oo"]
) + ["abnorm_vol", "vol_rs", "log_mkt_cap"] + (["bid_ask_spread"] if HAS_SPREAD else [])

DEP_VAR = "return_oo_adj" if HAS_ADJ_RETURN else "return_oo"


def stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fit_fmb(dep_var, treatment_cols, data, oth_vars=None,
            n_lags=5, bandwidth=5, min_cs=30, label=None):
    """
    Fama-MacBeth regression. Returns the fitted result object or None.
    Drops dates with < min_cs cross-sectional observations and dates where
    any regressor has zero cross-sectional variance.
    """
    if oth_vars is None:
        oth_vars = DEFAULT_OTH_VARS
    controls = [f"{v}_lag{k}" for v in oth_vars for k in range(1, n_lags + 1)]
    formula  = dep_var + " ~ " + " + ".join(treatment_cols + controls)

    try:
        panel = data.set_index(["ticker", "date"]) if "ticker" in data.columns else data

        # Drop dates with insufficient cross-sectional depth
        cs_counts   = panel.groupby(level="date").size()
        valid_dates = set(cs_counts[cs_counts >= min_cs].index)

        # Drop dates where any regressor has zero cross-sectional variance
        all_regs = treatment_cols + controls
        for col in all_regs:
            if col in panel.columns:
                cs_var = panel.groupby(level="date")[col].std()
                valid_dates -= set(cs_var[cs_var.isna() | (cs_var == 0)].index)

        panel = panel[panel.index.get_level_values("date").isin(valid_dates)]
        if panel.index.get_level_values("date").nunique() < 10:
            return None

        mod = FamaMacBeth.from_formula(formula, data=panel)
        try:
            return mod.fit(cov_type="kernel", bandwidth=bandwidth)
        except (ZeroDivisionError, Exception):
            try:
                return mod.fit(cov_type="robust")
            except Exception:
                return None
    except Exception:
        return None


def extract_fmb(res, param: str) -> dict:
    """Extract coef, se, pval from a fitted FMB result for one parameter."""
    if res is None or param not in res.params.index:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "stars": ""}
    c  = float(res.params[param])
    s  = float(res.std_errors[param])
    pv = float(res.pvalues[param])
    return {"coef": c, "se": s, "pval": pv, "stars": stars(pv)}


def long_short_stats(data: pd.DataFrame) -> dict:
    """
    Compute L-S strategy statistics for a subset of the panel.
    Uses cross-sectional percentile rank (robust to ties).
    """
    df = data[["date", "ticker", "twitter_sent", "return_oo"]].dropna()
    df = df.copy()
    df["pct_rank"] = df.groupby("date")["twitter_sent"].rank(pct=True)
    df["side"] = df["pct_rank"].apply(
        lambda r: "long" if r >= 0.9 else ("short" if r <= 0.1 else None)
    )
    df = df.dropna(subset=["side", "return_oo"])

    daily = (
        df.groupby(["date", "side"])["return_oo"]
        .mean()
        .unstack("side")
        .dropna(subset=["long", "short"])
    )
    if daily.empty:
        return {"ls_mean_daily": np.nan, "ls_ann_return": np.nan,
                "ls_sharpe": np.nan, "ls_win_rate": np.nan, "ls_n_days": 0}

    daily["ls"] = daily["long"] - daily["short"]
    mu    = daily["ls"].mean()
    sigma = daily["ls"].std()
    return {
        "ls_mean_daily": mu,
        "ls_ann_return": mu * 252,
        "ls_sharpe":     (mu / sigma) * np.sqrt(252) if sigma > 0 else np.nan,
        "ls_win_rate":   (daily["ls"] > 0).mean(),
        "ls_n_days":     len(daily),
    }


# ===========================================================================
# WINDOW RUNNER
# ===========================================================================

NR_LAGS = [1, 2, 3, 4, 5]


def run_window(label: str, start: pd.Timestamp, end: pd.Timestamp,
               bandwidth: int, min_cs: int) -> dict:
    """
    Run all three Gu & Kurov tests on a date-filtered slice of the panel.
    Returns a flat dict of all estimators, SEs, p-values, and strategy stats.
    """
    sub = long[(long["date"] >= start) & (long["date"] <= end)].copy()
    n_dates = sub["date"].nunique()
    n_obs   = len(sub)

    row = {
        "window_label": label,
        "window_start": start.date(),
        "window_end":   end.date(),
        "n_dates":      n_dates,
        "n_obs":        n_obs,
        "dep_var":      DEP_VAR,
    }

    if n_dates < 10:
        # Fill NaN for all estimators and return
        for col in (
            ["fmb_coef", "fmb_se", "fmb_pval", "fmb_stars"]
            + [f"nr_lag{k}_{s}" for k in NR_LAGS for s in ["coef", "se", "pval", "stars"]]
            + ["ls_mean_daily", "ls_ann_return", "ls_sharpe", "ls_win_rate", "ls_n_days"]
        ):
            row[col] = np.nan if col != "fmb_stars" else ""
        row["ls_n_days"] = 0
        return row

    # ── 1. FMB return predictability (Table 2 spec) ─────────────────────────
    fmb_res = fit_fmb(DEP_VAR, ["twitter_sent_lag1"], sub,
                      oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    fmb_est = extract_fmb(fmb_res, "twitter_sent_lag1")
    row["fmb_coef"]  = fmb_est["coef"]
    row["fmb_se"]    = fmb_est["se"]
    row["fmb_pval"]  = fmb_est["pval"]
    row["fmb_stars"] = fmb_est["stars"]

    # ── 2. No-reversal test (Table 3 spec) ──────────────────────────────────
    nr_treatments = [f"twitter_sent_lag{k}" for k in NR_LAGS]
    nr_res = fit_fmb(DEP_VAR, nr_treatments, sub,
                     oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    for k in NR_LAGS:
        est = extract_fmb(nr_res, f"twitter_sent_lag{k}")
        row[f"nr_lag{k}_coef"]  = est["coef"]
        row[f"nr_lag{k}_se"]    = est["se"]
        row[f"nr_lag{k}_pval"]  = est["pval"]
        row[f"nr_lag{k}_stars"] = est["stars"]

    # ── 3. Long-short strategy (Table 6 spec) ───────────────────────────────
    ls = long_short_stats(sub)
    row.update(ls)

    return row


# ===========================================================================
# WINDOW GENERATORS
# ===========================================================================

def monthly_windows(start_year=2016, end=(2026, 2)):
    windows = []
    y, m = start_year, 1
    while (y, m) <= end:
        s = pd.Timestamp(y, m, 1)
        last_day = (s + pd.offsets.MonthEnd(0)).date()
        e = pd.Timestamp(last_day)
        label = s.strftime("%Y-%m")
        windows.append((label, s, e))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return windows


def quarterly_windows(start_year=2016, end=(2026, 1)):
    windows = []
    for y in range(start_year, end[0] + 1):
        for q in range(1, 5):
            q_start_month = (q - 1) * 3 + 1
            q_end_month   = q * 3
            s = pd.Timestamp(y, q_start_month, 1)
            last_day = (pd.Timestamp(y, q_end_month, 1) + pd.offsets.MonthEnd(0)).date()
            e = pd.Timestamp(last_day)
            if (y, q_start_month) > (end[0], (end[1] - 1) // 3 * 3 + 1):
                break
            label = f"{y}-Q{q}"
            windows.append((label, s, e))
        else:
            continue
        break
    return windows


def annual_windows(start_year=2016, end_year=2026):
    windows = []
    for y in range(start_year, end_year + 1):
        s = pd.Timestamp(y, 1, 1)
        e = pd.Timestamp(y, 2, 28) if y == 2026 else pd.Timestamp(y, 12, 31)
        label = str(y) if y < 2026 else "2026 (Jan-Feb)"
        windows.append((label, s, e))
    return windows


# ===========================================================================
# RUN ALL THREE FREQUENCIES
# ===========================================================================

FREQ_CONFIG = [
    ("monthly",   monthly_windows(),   3, 15),   # bandwidth=3, min_cs=15
    ("quarterly", quarterly_windows(), 4, 20),   # bandwidth=4, min_cs=20
    ("annual",    annual_windows(),    5, 30),   # bandwidth=5, min_cs=30
]

results = {}

for freq, windows, bw, mcs in FREQ_CONFIG:
    print(f"\n{'='*65}")
    print(f"  Running {freq.upper()} windows  "
          f"(bandwidth={bw}, min_cs={mcs}, n_windows={len(windows)})")
    print("="*65)
    rows = []
    for label, start, end in windows:
        row = run_window(label, start, end, bandwidth=bw, min_cs=mcs)
        rows.append(row)
        fmb_c = f"{row['fmb_coef']:>9.4f}" if not np.isnan(row.get("fmb_coef", np.nan)) else "     n/a"
        nr1_c = f"{row['nr_lag1_coef']:>9.4f}" if not np.isnan(row.get("nr_lag1_coef", np.nan)) else "     n/a"
        print(f"  {label:<16}  FMB={fmb_c}{row.get('fmb_stars',''):3s}  "
              f"NR_lag1={nr1_c}{row.get('nr_lag1_stars',''):3s}  "
              f"Sharpe={row.get('ls_sharpe', np.nan):>5.2f}  "
              f"N_dates={row.get('n_dates',0):>3d}")
    df = pd.DataFrame(rows)
    results[freq] = df
    out_path = OUT_DIR / f"gk_temporal_{freq}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {out_path}")

# ===========================================================================
# SUMMARY PRINT
# ===========================================================================

print("\n\n" + "="*65)
print("  SUMMARY — Annual FMB coefficient on twitter_sent_lag1")
print("="*65)
ann = results["annual"]
print(f"{'Year':<20} {'FMB coef':>10} {'SE':>10} {'p-val':>8} {'Sharpe':>8}")
print("-"*60)
for _, r in ann.iterrows():
    coef  = f"{r['fmb_coef']:>10.4f}" if not pd.isna(r["fmb_coef"]) else "       n/a"
    se    = f"{r['fmb_se']:>10.4f}"   if not pd.isna(r["fmb_se"])   else "       n/a"
    pval  = f"{r['fmb_pval']:>8.4f}"  if not pd.isna(r["fmb_pval"]) else "     n/a"
    sharpe = f"{r['ls_sharpe']:>8.2f}" if not pd.isna(r["ls_sharpe"]) else "     n/a"
    print(f"  {r['window_label']:<18} {coef} {se} {pval} {sharpe}")

print("\nDone. Output files:")
for freq in ["monthly", "quarterly", "annual"]:
    print(f"  data/processed/gk_temporal_{freq}.csv")
