"""
gu_kurov_replication.py
Replication of Gu & Kurov (2020)
"Informational role of social media: Evidence from Twitter sentiment"
Journal of Banking and Finance 121, 105969.

Key methodological choices carried over from the original:
  - Dependent variable: open-to-open holding-period return (pct, in %)
  - Method: Fama-MacBeth (1973) cross-sectional daily regressions with
            Newey-West (kernel) standard errors, consistent with the paper
  - Regressors: Twitter_i,t-1 + 5 lags of [return, abnorm_vol, RS_vol,
                log(mkt_cap), bid_ask_spread]
  - No-reversal test: includes 5 lags of Twitter sentiment simultaneously
  - Twitter vs. news sentiment comparison
  - Risk-adjusted returns via Fama-French-Carhart 4-factor model
    (downloaded from Kenneth French's data library using pandas_datareader)

Data requirements:
  A) Available in panel_long.csv:
     px_open, px_close, px_high, px_low, mkt_cap, volume,
     twitter_sent, news_sent

  B) Requires extended Bloomberg pull (see bloomberg_pull_extended.py):
     bid_ask_spread  — BID_ASK_SPREAD_DAILY_AVG

  Note: Bloomberg no longer provides analyst coverage counts. Table 4
  (analyst coverage heterogeneity) from the original paper is omitted.

  C) Automatically downloaded at runtime (requires internet):
     Fama-French-Carhart 4 daily factors from Kenneth French's data library
     via pandas_datareader. If download fails, raw returns are used.

Usage:
    python src/python/gu_kurov_replication.py
"""

import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import FamaMacBeth
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "panel_long.csv"
EXT_FILE = ROOT / "data" / "processed" / "panel_long_extended.csv"
OUT_DIR  = ROOT / "output"
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
HAS_SPREAD = "bid_ask_spread" in long.columns

if not HAS_SPREAD:
    print("\n[WARNING] bid_ask_spread not found.")
    print("  Bid-ask spread control will be omitted. Regressions are otherwise identical.")
    print("  Pull BID_ASK_SPREAD_DAILY_AVG via bloomberg_pull_extended.py for exact replication.\n")


# ===========================================================================
# BLOCK 1 — VARIABLE CONSTRUCTION
# ===========================================================================

long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

# ------ Open-to-open return -----------------------------------------------
# Return_i,t = (px_open_{t+1} - px_open_t) / px_open_t * 100
# This aligns with Gu & Kurov: you observe Twitter sentiment released at
# 9:20am on day t and trade at the 9:30am open, holding until open on t+1.
long["px_open_next"] = long.groupby("ticker")["px_open"].shift(-1)
long["return_oo"] = (
    (long["px_open_next"] - long["px_open"]) / long["px_open"] * 100
)

# ------ Rogers-Satchell (1991) realized volatility ------------------------
# Vol_i,t = (ln H - ln C)(ln H - ln O) + (ln L - ln C)(ln L - ln O)
# Clipped at 0 since floating-point errors can produce tiny negatives.
log_H = np.log(long["px_high"].clip(lower=1e-8))
log_L = np.log(long["px_low"].clip(lower=1e-8))
log_O = np.log(long["px_open"].clip(lower=1e-8))
log_C = np.log(long["px_close"].clip(lower=1e-8))
long["vol_rs"] = (
    (log_H - log_C) * (log_H - log_O) + (log_L - log_C) * (log_L - log_O)
).clip(lower=0) * 100  # expressed in % to match scale of other controls

# ------ Abnormal trading volume -------------------------------------------
# (volume_t - mean_volume_i) / mean_volume_i * 100
mean_vol = long.groupby("ticker")["volume"].transform("mean")
long["abnorm_vol"] = ((long["volume"] - mean_vol) / mean_vol) * 100

# ------ Firm size: log(market cap) ----------------------------------------
long["log_mkt_cap"] = np.log(long["mkt_cap"].clip(lower=1e-8))

# ------ Lagged Twitter sentiment (key predictor) --------------------------
# Twitter_i,t-1 in Gu & Kurov notation = twitter_sent shifted by 1 trading day
long["twitter_sent_lag1"] = long.groupby("ticker")["twitter_sent"].shift(1)

# ------ 5-day rolling lags of all control variables -----------------------
control_base = ["return_oo", "abnorm_vol", "vol_rs", "log_mkt_cap"]
if HAS_SPREAD:
    control_base.append("bid_ask_spread")

for var in control_base:
    for k in range(1, 6):
        long[f"{var}_lag{k}"] = long.groupby("ticker")[var].shift(k)

# ------ Additional lags of Twitter sentiment (for no-reversal test) -------
for k in range(1, 6):
    long[f"twitter_sent_lag{k}"] = long.groupby("ticker")["twitter_sent"].shift(k)

# ------ Lagged news sentiment (for Twitter vs. news comparison) -----------
long["news_sent_lag1"] = long.groupby("ticker")["news_sent"].shift(1)


# ===========================================================================
# BLOCK 2 — FAMA-FRENCH-CARHART RISK ADJUSTMENT (optional)
# ===========================================================================
# Download daily Fama-French 4-factor returns from Ken French's data library.
# If unavailable (no internet or package issues), skip and use raw returns.

ff_factors = None
try:
    import pandas_datareader.data as web
    start_date = long["date"].min()
    end_date   = long["date"].max()

    ff3   = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start=start_date, end=end_date)[0]
    mom   = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                            start=start_date, end=end_date)[0]
    ff3.index   = pd.to_datetime(ff3.index, format="%Y%m%d")
    mom.index   = pd.to_datetime(mom.index, format="%Y%m%d")
    ff_factors  = ff3.join(mom, how="inner") / 100  # convert pct to decimal
    ff_factors.columns = [c.strip() for c in ff_factors.columns]
    print(f"Downloaded Fama-French-Carhart factors: {ff_factors.shape[0]} days, "
          f"columns: {list(ff_factors.columns)}")
except Exception as e:
    print(f"[WARNING] Fama-French factor download failed: {e}")
    print("  Risk-adjusted returns will not be computed. Raw returns used throughout.\n")

def compute_risk_adjusted_return(df: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """
    For each ticker, regress return_oo on the 4 Fama-French-Carhart factors
    and return the residuals (abnormal returns).
    """
    if factors is None:
        return df["return_oo"]

    # Merge factor returns onto our panel by date
    df = df.merge(
        factors[["Mkt-RF", "SMB", "HML", "Mom   "]].rename(
            columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "Mom   ": "mom"}
        ),
        left_on="date", right_index=True, how="left"
    )

    residuals = np.full(len(df), np.nan)
    for ticker, grp in df.groupby("ticker"):
        sub = grp[["return_oo", "mkt_rf", "smb", "hml", "mom"]].dropna()
        if len(sub) < 30:
            continue
        X = sm_add_const(sub[["mkt_rf", "smb", "hml", "mom"]])
        y = sub["return_oo"]
        try:
            import statsmodels.api as sm
            ols_res = sm.OLS(y, X).fit()
            residuals[sub.index] = ols_res.resid.values
        except Exception:
            pass

    return pd.Series(residuals, index=df.index)

try:
    import statsmodels.api as sm
    sm_add_const = sm.add_constant
    long["return_oo_adj"] = compute_risk_adjusted_return(long.copy(), ff_factors)
    HAS_ADJ_RETURN = long["return_oo_adj"].notna().sum() > 0
    if HAS_ADJ_RETURN:
        print(f"Risk-adjusted returns computed: {long['return_oo_adj'].notna().sum():,} non-missing obs.")
except Exception as e:
    HAS_ADJ_RETURN = False
    print(f"[WARNING] Risk adjustment failed: {e}")


# ===========================================================================
# BLOCK 3 — SET PANEL INDEX AND CONTROL STRING
# ===========================================================================

panel = long.set_index(["ticker", "date"])

# Build the control formula fragment (5 lags × each base control)
def controls_formula(base_vars: list) -> str:
    terms = []
    for var in base_vars:
        for k in range(1, 6):
            terms.append(f"{var}_lag{k}")
    return " + ".join(terms)

control_vars = ["return_oo", "abnorm_vol", "vol_rs", "log_mkt_cap"]
if HAS_SPREAD:
    control_vars.append("bid_ask_spread")

ctrl_str = controls_formula(control_vars)


# ===========================================================================
# HELPERS
# ===========================================================================

def stars(pval: float) -> str:
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def fit_fmb(formula: str, data: pd.DataFrame, label: str, bandwidth: int = 5):
    """Fit a Fama-MacBeth regression and print the key results."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print('='*65)
    try:
        mod = FamaMacBeth.from_formula(formula, data=data)
        res = mod.fit(cov_type="kernel", bandwidth=bandwidth)
        # Print main Twitter-related coefficients
        for p in res.params.index:
            if "twitter" in p.lower() or "news" in p.lower():
                c    = res.params[p]
                se   = res.std_errors[p]
                pval = res.pvalues[p]
                tstat = res.tstats[p]
                print(f"  {p:<45s}  {c:>10.6f}  ({se:.6f})  t={tstat:.2f}  {stars(pval)}")
        n_obs = res.nobs if hasattr(res, "nobs") else "?"
        print(f"  N = {n_obs}")
        return res
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def fmb_coef(res, param: str):
    """Return (coef, se, pval) or (nan, nan, 1)."""
    try:
        return float(res.params[param]), float(res.std_errors[param]), float(res.pvalues[param])
    except Exception:
        return np.nan, np.nan, 1.0


def multi_col_latex(col_models: list, col_labels: list, row_params: list,
                    caption: str, label_str: str, notes: str = "") -> str:
    """Build multi-column Fama-MacBeth LaTeX table."""
    ncols = len(col_models)
    col_fmt = "l" + "c" * ncols

    def fmt_cell(res, param, is_se=False):
        if res is None or param not in res.params.index:
            return "---"
        c, s, p = fmb_coef(res, param)
        if is_se:
            return f"({s:.6f})"
        return f"{c:.6f}{stars(p)}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label_str}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\hline\hline",
        " & " + " & ".join(f"({i+1}) {col_labels[i]}" for i in range(ncols)) + r" \\",
        r"\hline",
    ]
    for param in row_params:
        p_lbl = param.replace("_", r"\_")
        lines.append(p_lbl + " & " + " & ".join(fmt_cell(m, param) for m in col_models) + r" \\")
        lines.append("& " + " & ".join(fmt_cell(m, param, is_se=True) for m in col_models) + r" \\")
    lines += [
        r"\hline",
        "Estimator & " + " & ".join(["Fama-MacBeth"] * ncols) + r" \\",
        "SE & " + " & ".join(["Newey-West (5 lags)"] * ncols) + r" \\",
        r"\hline\hline",
    ]
    if notes:
        lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\footnotesize \\textit{{Notes:}} {notes}}} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ===========================================================================
# TABLE 2 EQUIVALENT — Predicting returns with Twitter sentiment
#
# Eq. (3): Return_i,t = a + b*Twitter_i,t-1 + Σ controls + ε
#
# Panels:
#   A — all firms, equal-weighted (raw return_oo)
#   A_adj — all firms, equal-weighted (risk-adjusted return_oo_adj)
# ===========================================================================

print("\n" + "#"*65)
print("  TABLE 2: Predicting Returns with Twitter Sentiment (Fama-MacBeth)")
print("#"*65)

base_formula = f"return_oo ~ twitter_sent_lag1 + {ctrl_str}"
t2_raw = fit_fmb(base_formula, panel, "TABLE 2 Col (1): Raw return, equal-weighted")

t2_adj = None
if HAS_ADJ_RETURN:
    panel_adj = panel.copy()
    panel_adj["return_oo_adj_lag1"] = panel_adj.groupby(level="ticker")["return_oo_adj"].shift(1)
    adj_ctrl_str = controls_formula(["return_oo_adj"] + control_vars[1:])
    adj_formula = f"return_oo_adj ~ twitter_sent_lag1 + {adj_ctrl_str}"
    t2_adj = fit_fmb(adj_formula, panel_adj, "TABLE 2 Col (2): Risk-adjusted return, equal-weighted")


# ===========================================================================
# TABLE 3 EQUIVALENT — No-reversal test
#
# Includes 5 lags of Twitter sentiment simultaneously.
# Under the information hypothesis, only lag 1 should matter;
# lags 2-5 should be near zero and insignificant.
# Significant lags 2-5 would suggest momentum or reverse causality.
# ===========================================================================

print("\n" + "#"*65)
print("  TABLE 3: No-Reversal Test (5 Lags of Twitter Sentiment)")
print("#"*65)

multi_lag_formula = (
    f"return_oo ~ twitter_sent_lag1 + twitter_sent_lag2 + twitter_sent_lag3 "
    f"+ twitter_sent_lag4 + twitter_sent_lag5 + {ctrl_str}"
)
t3_raw = fit_fmb(multi_lag_formula, panel, "TABLE 3 Col (1): 5 Twitter lags, raw return")

t3_adj = None
if HAS_ADJ_RETURN:
    multi_lag_adj = multi_lag_formula.replace("return_oo ~", "return_oo_adj ~")
    t3_adj = fit_fmb(multi_lag_adj, panel_adj, "TABLE 3 Col (2): 5 Twitter lags, risk-adjusted")


# ===========================================================================
# TABLE 5 EQUIVALENT — Twitter sentiment vs. news sentiment
#
# Adds news_sent_lag1 as additional regressor.
# Tests whether Twitter provides incremental predictive power beyond news.
# Gu & Kurov: both remain significant and largely independent (corr ≈ 0.20).
# ===========================================================================

print("\n" + "#"*65)
print("  TABLE 5: Twitter Sentiment vs. News Sentiment (Fama-MacBeth)")
print("#"*65)

# Col (1): news sentiment only
news_formula = f"return_oo ~ news_sent_lag1 + {ctrl_str}"
t5_news = fit_fmb(news_formula, panel, "TABLE 5 Col (1): news_sent only")

# Col (2): both Twitter and news
both_formula = f"return_oo ~ twitter_sent_lag1 + news_sent_lag1 + {ctrl_str}"
t5_both = fit_fmb(both_formula, panel, "TABLE 5 Col (2): twitter_sent + news_sent")

# Contemporaneous correlation
corr = long[["twitter_sent", "news_sent"]].corr().iloc[0, 1]
print(f"\n  Contemporaneous corr(twitter_sent, news_sent) = {corr:.3f}")
print("  Gu & Kurov report ≈ 0.20; higher values may indicate Bloomberg data overlap.")


# ===========================================================================
# TABLE 6 EQUIVALENT — Long-short trading strategy
#
# On each day:
#   Long portfolio  = firms in top decile of Twitter sentiment (released t-0 morning)
#   Short portfolio = firms in bottom decile
#   Holding period  = 24h (open t to open t+1)
#   Strategy return = mean(long portfolio return_oo) - mean(short portfolio return_oo)
# ===========================================================================

print("\n" + "#"*65)
print("  TABLE 6: Long-Short Trading Strategy Based on Twitter Sentiment")
print("#"*65)

# Use same-day twitter_sent as the signal (released before open, per Gu & Kurov)
daily_pct = long.groupby("date", group_keys=False).apply(
    lambda grp: grp.assign(
        decile=pd.qcut(grp["twitter_sent"], q=10, labels=False, duplicates="drop")
    )
)
long_port  = daily_pct[daily_pct["decile"] == 9]["return_oo"].dropna()   # top decile
short_port = daily_pct[daily_pct["decile"] == 0]["return_oo"].dropna()   # bottom decile

daily_ls = (
    daily_pct[daily_pct["decile"].isin([0, 9])]
    .groupby(["date", "decile"])["return_oo"]
    .mean()
    .unstack("decile")
    .rename(columns={0: "short", 9: "long"})
    .dropna()
)
daily_ls["ls_return"] = daily_ls["long"] - daily_ls["short"]

n_days     = len(daily_ls)
mean_daily = daily_ls["ls_return"].mean()
std_daily  = daily_ls["ls_return"].std()
ann_return = mean_daily * 252
sharpe     = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
win_rate   = (daily_ls["ls_return"] > 0).mean()

print(f"  Trading days:          {n_days}")
print(f"  Mean daily L-S return: {mean_daily:.4f}%")
print(f"  Annualized return:     {ann_return:.2f}%")
print(f"  Annualized Sharpe:     {sharpe:.2f}  (Gu & Kurov report 3.17)")
print(f"  Win rate:              {win_rate:.1%}")


# ===========================================================================
# EXPORT — LaTeX tables
# ===========================================================================

# Table 2: Return predictability
t2_params = ["twitter_sent_lag1"]
tex_t2 = multi_col_latex(
    [t2_raw, t2_adj] if t2_adj else [t2_raw],
    ["Raw Return", "Risk-Adj Return"] if t2_adj else ["Raw Return"],
    t2_params,
    caption="Gu \\& Kurov (2020) Replication --- Table 2: Predicting Returns with Twitter Sentiment",
    label_str="tab:gk_t2",
    notes=(
        "Fama-MacBeth regressions. Newey-West SEs (5 lags). Dependent variable: "
        "open-to-open holding period return (pct). Controls include 5 lags of return, "
        "abnormal volume, Rogers-Satchell volatility, log(mkt\\_cap)"
        + (", bid-ask spread" if HAS_SPREAD else " [bid-ask spread omitted — pull from Bloomberg]")
        + ". *** p$<$0.01, ** p$<$0.05, * p$<$0.1."
    ),
)
(OUT_DIR / "gk_t2.tex").write_text(tex_t2)
print(f"\nSaved {OUT_DIR / 'gk_t2.tex'}")

# Table 3: No-reversal test
t3_params = [f"twitter_sent_lag{k}" for k in range(1, 6)]
tex_t3 = multi_col_latex(
    [t3_raw, t3_adj] if t3_adj else [t3_raw],
    ["Raw Return", "Risk-Adj Return"] if t3_adj else ["Raw Return"],
    t3_params,
    caption="Gu \\& Kurov (2020) Replication --- Table 3: No-Reversal Test",
    label_str="tab:gk_t3",
    notes=(
        "Fama-MacBeth regressions. Newey-West SEs (5 lags). "
        "Under the information hypothesis, only lag 1 should be significant. "
        "Significant lags 2--5 suggest momentum or reverse causality. "
        "*** p$<$0.01, ** p$<$0.05, * p$<$0.1."
    ),
)
(OUT_DIR / "gk_t3.tex").write_text(tex_t3)
print(f"Saved {OUT_DIR / 'gk_t3.tex'}")

# Table 5: Twitter vs. news
t5_params = ["twitter_sent_lag1", "news_sent_lag1"]
tex_t5 = multi_col_latex(
    [t5_news, t5_both],
    ["News Only", "Twitter + News"],
    t5_params,
    caption="Gu \\& Kurov (2020) Replication --- Table 5: Twitter vs. News Sentiment",
    label_str="tab:gk_t5",
    notes=(
        "Fama-MacBeth regressions. Newey-West SEs (5 lags). "
        "Both sentiment types use a 1-day lag. "
        "Incremental R$^2$ from including Twitter beyond news tests for orthogonal information content. "
        "*** p$<$0.01, ** p$<$0.05, * p$<$0.1."
    ),
)
(OUT_DIR / "gk_t5.tex").write_text(tex_t5)
print(f"Saved {OUT_DIR / 'gk_t5.tex'}")

# Table 6: Trading strategy summary
ls_rows = [
    ("Mean daily L-S return (\\%)", f"{mean_daily:.4f}", ""),
    ("Annualized return (\\%)",      f"{ann_return:.2f}", ""),
    ("Annualized Sharpe ratio",       f"{sharpe:.2f}",    "(Gu \\& Kurov: 3.17)"),
    ("Win rate",                      f"{win_rate:.1%}",  ""),
    ("Trading days (N)",              f"{n_days}",        ""),
]
ls_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Gu \& Kurov (2020) Replication --- Table 6: Long-Short Strategy Performance}",
    r"\label{tab:gk_t6}",
    r"\begin{tabular}{lcc}",
    r"\hline\hline",
    r"Statistic & Value & Note \\",
    r"\hline",
]
for stat, val, note in ls_rows:
    ls_lines.append(f"{stat} & {val} & {note} \\\\")
ls_lines += [
    r"\hline\hline",
    r"\multicolumn{3}{l}{\footnotesize \textit{Notes:} Long (short) portfolio = top (bottom) decile daily Twitter sentiment. 24h holding period.} \\",
    r"\end{tabular}",
    r"\end{table}",
]
(OUT_DIR / "gk_t6.tex").write_text("\n".join(ls_lines))
print(f"Saved {OUT_DIR / 'gk_t6.tex'}")

print("\nGu & Kurov replication complete.")
