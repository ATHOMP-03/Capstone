"""
russel_replication.py
Russell 3000 Twitter sentiment analysis — three studies using Gu & Kurov (2020) methodology.

Study 1:  Original period replication  (2016-2017)
Study 2:  Modern period replication    (2024-2025)
Study 3:  Long-term temporal analysis  (2016-2026, monthly / quarterly / annual)

Input:  data/processed/russel_panel.csv

Output — LaTeX tables (output/):
  russel_2016_t2.tex   russel_2016_t3.tex   russel_2016_t5.tex   russel_2016_t6.tex
  russel_2024_t2.tex   russel_2024_t3.tex   russel_2024_t5.tex   russel_2024_t6.tex
  russel_temporal_fig.tex  russel_temporal_sumstats.tex
  russel_temporal_fmb.tex  russel_temporal_ls.tex

Output — Figures (output/figures/):
  russel_temporal_annual.png
  russel_temporal_quarterly.png
  russel_temporal_monthly.png

Output — CSVs (data/processed/):
  russel_temporal_monthly.csv
  russel_temporal_quarterly.csv
  russel_temporal_annual.csv

Usage:
    python src/python/russel_replication.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from linearmodels.panel import FamaMacBeth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "processed" / "russel_panel.csv"
OUT_DIR = ROOT / "data" / "processed"
TEX_OUT = ROOT / "output"
FIG_DIR = ROOT / "output" / "figures"
for d in [OUT_DIR, TEX_OUT, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FF_CUTOFF = pd.Timestamp("2026-02-28")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
long = pd.read_csv(IN_FILE, parse_dates=["date"])
long = long[long["date"] <= FF_CUTOFF].reset_index(drop=True)
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(long):,} rows x {long.shape[1]} cols")
print(f"Date range: {long['date'].min().date()} to {long['date'].max().date()}")
print(f"Tickers: {long['ticker'].nunique():,}")

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

mean_vol = long.groupby("ticker")["volume"].transform("mean")
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
    long[f"news_sent_lag{k}"]    = long.groupby("ticker")["news_sent"].shift(k)

print("Variable construction complete.")
print(f"  return_oo: {long['return_oo'].notna().sum():,} obs")


# ===========================================================================
# BLOCK 2 — FAMA-FRENCH RISK ADJUSTMENT (full-sample betas)
# ===========================================================================

HAS_ADJ_RETURN = False
try:
    import statsmodels.api as sm
    import pandas_datareader.data as web

    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                          start=long["date"].min(), end=long["date"].max())[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                          start=long["date"].min(), end=long["date"].max())[0]
    ff3.index = pd.to_datetime(ff3.index, format="%Y%m%d")
    mom.index = pd.to_datetime(mom.index, format="%Y%m%d")
    ff = ff3.join(mom, how="inner") / 100
    ff.columns = [c.strip() for c in ff.columns]
    print(f"FF factors downloaded: {ff.shape[0]} days, cols: {list(ff.columns)}")

    merged = long.merge(
        ff.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb",
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

DEP_VAR = "return_oo_adj" if HAS_ADJ_RETURN else "return_oo"
DEFAULT_OTH_VARS = (
    [DEP_VAR] + ["abnorm_vol", "vol_rs", "log_mkt_cap"]
    + (["bid_ask_spread"] if HAS_SPREAD else [])
)
print(f"Dependent variable: {DEP_VAR}")
print(f"Confounders:        {DEFAULT_OTH_VARS}")


# ===========================================================================
# BLOCK 3 — HELPERS
# ===========================================================================

NR_LAGS = [1, 2, 3, 4, 5]


def stars(p) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt(v, decimals=4) -> str:
    return f"{v:.{decimals}f}" if not pd.isna(v) else "---"


def fit_fmb(dep_var, treatment_cols, data, oth_vars=None,
            n_lags=5, bandwidth=5, min_cs=30):
    """
    Fama-MacBeth regression. Drops dates with < min_cs observations and
    zero-variance regressors. Returns fitted result or None.
    """
    if oth_vars is None:
        oth_vars = DEFAULT_OTH_VARS
    controls = [f"{v}_lag{k}" for v in oth_vars for k in range(1, n_lags + 1)]
    formula  = dep_var + " ~ " + " + ".join(treatment_cols + controls)
    try:
        panel = data.set_index(["ticker", "date"]) if "ticker" in data.columns else data
        cs    = panel.groupby(level="date").size()
        valid = set(cs[cs >= min_cs].index)
        for col in treatment_cols + controls:
            if col in panel.columns:
                cv = panel.groupby(level="date")[col].std()
                valid -= set(cv[cv.isna() | (cv == 0)].index)
        panel = panel[panel.index.get_level_values("date").isin(valid)]
        if panel.index.get_level_values("date").nunique() < 10:
            return None
        mod = FamaMacBeth.from_formula(formula, data=panel)
        try:
            return mod.fit(cov_type="kernel", bandwidth=bandwidth)
        except Exception:
            try:    return mod.fit(cov_type="robust")
            except: return None
    except Exception:
        return None


def extract_fmb(res, param) -> dict:
    if res is None or param not in res.params.index:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "stars": ""}
    c  = float(res.params[param])
    s  = float(res.std_errors[param])
    pv = float(res.pvalues[param])
    return {"coef": c, "se": s, "pval": pv, "stars": stars(pv)}


def long_short_stats(data) -> dict:
    df = data[["date", "ticker", "twitter_sent", "return_oo"]].dropna().copy()
    df["pct_rank"] = df.groupby("date")["twitter_sent"].rank(pct=True)
    df["side"]     = df["pct_rank"].apply(
        lambda r: "long" if r >= 0.9 else ("short" if r <= 0.1 else None)
    )
    df = df.dropna(subset=["side", "return_oo"])
    daily = (df.groupby(["date", "side"])["return_oo"]
               .mean().unstack("side").dropna(subset=["long", "short"]))
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
# BLOCK 4 — LATEX TABLE BUILDERS (Studies 1 & 2)
# ===========================================================================

def _ctrl_note():
    base = r"5 lags each of return, abnorm\_vol, vol\_rs, log\_mkt\_cap"
    return base + (r", bid\_ask\_spread" if HAS_SPREAD else "")


def latex_t2(est_raw, est_adj, period_label, prefix):
    """Table 2: return predictability — raw and risk-adjusted columns."""
    def cell(est):
        if pd.isna(est["coef"]): return "---", "---"
        return f"${fmt(est['coef'])}{est['stars']}$", f"$({fmt(est['se'])})$"

    c1, s1 = cell(est_raw)
    c2, s2 = cell(est_adj)
    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Return Predictability ({period_label}) --- Russell 3000}}\n"
        f"\\label{{tab:{prefix}_t2}}\n"
        r"\begin{tabular}{lcc}" + "\n"
        r"\hline\hline" + "\n"
        r" & (1) Raw Return & (2) Risk-Adj \\" + "\n"
        r"\hline" + "\n"
        f"twitter\\_sent\\_lag1 & {c1} & {c2} \\\\\n"
        f"                     & {s1} & {s2} \\\\\n"
        r"\hline" + "\n"
        r"Estimator & Fama-MacBeth & Fama-MacBeth \\" + "\n"
        r"SE & Newey-West (5) & Newey-West (5) \\" + "\n"
        r"\hline\hline" + "\n"
        r"\multicolumn{3}{p{0.9\linewidth}}{\footnotesize \textit{Notes:} "
        r"Fama-MacBeth with Newey-West SEs (5 lags). Open-to-open return. "
        r"Controls: " + _ctrl_note() + r". "
        r"Risk-Adj = Fama-French-Carhart 4-factor residual. "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$.}\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    (TEX_OUT / f"{prefix}_t2.tex").write_text(tex)
    print(f"    Saved {prefix}_t2.tex")


def latex_t3(nr_res, period_label, prefix):
    """Table 3: no-reversal test — 5 sentiment lags, risk-adjusted only."""
    rows = []
    for k in NR_LAGS:
        est = extract_fmb(nr_res, f"twitter_sent_lag{k}")
        c = f"${fmt(est['coef'])}{est['stars']}$" if not pd.isna(est["coef"]) else "---"
        s = f"$({fmt(est['se'])})$"               if not pd.isna(est["coef"]) else "---"
        rows.append(f"twitter\\_sent\\_lag{k} & {c} \\\\\n               & {s} \\\\")
    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{No-Reversal Test ({period_label}) --- Russell 3000}}\n"
        f"\\label{{tab:{prefix}_t3}}\n"
        r"\begin{tabular}{lc}" + "\n"
        r"\hline\hline" + "\n"
        r" & Risk-Adj \\" + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        + r"\hline" + "\n"
        r"Estimator & Fama-MacBeth \\" + "\n"
        r"SE & Newey-West (5) \\" + "\n"
        r"\hline\hline" + "\n"
        r"\multicolumn{2}{p{0.85\linewidth}}{\footnotesize \textit{Notes:} "
        r"All 5 sentiment lags entered jointly. Pass criterion: lag 1 significant, "
        r"lags 2--5 insignificant (Gu \& Kurov, 2020). "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$.}\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    (TEX_OUT / f"{prefix}_t3.tex").write_text(tex)
    print(f"    Saved {prefix}_t3.tex")


def latex_t5(res_news, res_joint, period_label, prefix):
    """Table 5: Twitter vs. news sentiment — two columns."""
    def cell(res, param):
        est = extract_fmb(res, param)
        if pd.isna(est["coef"]): return "---", "---"
        return f"${fmt(est['coef'])}{est['stars']}$", f"$({fmt(est['se'])})$"

    ne_c, ne_s = cell(res_news,  "news_sent_lag1")
    tw_c, tw_s = cell(res_joint, "twitter_sent_lag1")
    nj_c, nj_s = cell(res_joint, "news_sent_lag1")

    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Twitter vs.\\ News Sentiment ({period_label}) --- Russell 3000}}\n"
        f"\\label{{tab:{prefix}_t5}}\n"
        r"\begin{tabular}{lcc}" + "\n"
        r"\hline\hline" + "\n"
        r" & (1) News Only & (2) Twitter + News \\" + "\n"
        r"\hline" + "\n"
        f"twitter\\_sent\\_lag1 & --- & {tw_c} \\\\\n"
        f"                     & --- & {tw_s} \\\\\n"
        f"news\\_sent\\_lag1    & {ne_c} & {nj_c} \\\\\n"
        f"                     & {ne_s} & {nj_s} \\\\\n"
        r"\hline" + "\n"
        r"Estimator & Fama-MacBeth & Fama-MacBeth \\" + "\n"
        r"SE & Newey-West (5) & Newey-West (5) \\" + "\n"
        r"\hline\hline" + "\n"
        r"\multicolumn{3}{p{0.9\linewidth}}{\footnotesize \textit{Notes:} "
        r"Both sentiments use 1-day lag. Risk-adjusted open-to-open return. "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$.}\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    (TEX_OUT / f"{prefix}_t5.tex").write_text(tex)
    print(f"    Saved {prefix}_t5.tex")


def latex_t6(ls, period_label, prefix):
    """Table 6: long-short strategy statistics."""
    n_days = int(ls["ls_n_days"]) if ls["ls_n_days"] > 0 else "---"
    tex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Long-Short Strategy ({period_label}) --- Russell 3000}}\n"
        f"\\label{{tab:{prefix}_t6}}\n"
        r"\begin{tabular}{lc}" + "\n"
        r"\hline\hline" + "\n"
        r"Statistic & Value \\" + "\n"
        r"\hline" + "\n"
        f"Mean daily L-S return (\\%) & {fmt(ls['ls_mean_daily'])} \\\\\n"
        f"Annualized return (\\%)     & {fmt(ls['ls_ann_return'], 2)} \\\\\n"
        f"Annualized Sharpe           & {fmt(ls['ls_sharpe'],     2)} \\\\\n"
        f"Win rate                    & {fmt(ls['ls_win_rate'],   3)} \\\\\n"
        f"Trading days                & {n_days} \\\\\n"
        r"\hline\hline" + "\n"
        r"\multicolumn{2}{p{0.75\linewidth}}{\footnotesize \textit{Notes:} "
        r"Long (short) = top (bottom) decile daily Twitter sentiment. "
        r"24-hour hold. Before transaction costs. "
        r"Gu \& Kurov (2020) report Sharpe of 3.17 over Jan 2015--Feb 2017.}\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    (TEX_OUT / f"{prefix}_t6.tex").write_text(tex)
    print(f"    Saved {prefix}_t6.tex")


def run_replication(sub, period_label, prefix, bandwidth=5, min_cs=30):
    """Run all four replication tables (T2, T3, T5, T6) for a data subset."""
    print(f"\n{'='*65}")
    print(f"  STUDY: {period_label}  "
          f"({sub['date'].nunique()} dates, "
          f"{sub['ticker'].nunique():,} tickers, "
          f"N={len(sub):,})")
    print("="*65)

    raw_oth = (["return_oo", "abnorm_vol", "vol_rs", "log_mkt_cap"]
               + (["bid_ask_spread"] if HAS_SPREAD else []))

    # Table 2: return predictability
    print("  Table 2 — Return Predictability")
    res_raw = fit_fmb("return_oo", ["twitter_sent_lag1"], sub,
                      oth_vars=raw_oth, bandwidth=bandwidth, min_cs=min_cs)
    res_adj = fit_fmb(DEP_VAR, ["twitter_sent_lag1"], sub,
                      oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    est_raw = extract_fmb(res_raw, "twitter_sent_lag1")
    est_adj = extract_fmb(res_adj, "twitter_sent_lag1")
    latex_t2(est_raw, est_adj, period_label, prefix)
    print(f"    Raw: {fmt(est_raw['coef'])}{est_raw['stars']}  "
          f"Adj: {fmt(est_adj['coef'])}{est_adj['stars']}")

    # Table 3: no-reversal
    print("  Table 3 — No-Reversal Test")
    nr_res = fit_fmb(DEP_VAR, [f"twitter_sent_lag{k}" for k in NR_LAGS], sub,
                     oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    latex_t3(nr_res, period_label, prefix)

    # Table 5: Twitter vs. news
    print("  Table 5 — Twitter vs. News Sentiment")
    res_news  = fit_fmb(DEP_VAR, ["news_sent_lag1"], sub,
                        oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    res_joint = fit_fmb(DEP_VAR, ["twitter_sent_lag1", "news_sent_lag1"], sub,
                        oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    latex_t5(res_news, res_joint, period_label, prefix)

    # Table 6: long-short
    print("  Table 6 — Long-Short Strategy")
    ls = long_short_stats(sub)
    latex_t6(ls, period_label, prefix)
    print(f"    Sharpe: {fmt(ls['ls_sharpe'], 2)}  "
          f"Ann. return: {fmt(ls['ls_ann_return'], 2)}%  "
          f"Win rate: {fmt(ls['ls_win_rate'], 3)}")


# ===========================================================================
# STUDIES 1 & 2 — PERIOD REPLICATIONS
# ===========================================================================

sub_2016 = long[
    (long["date"] >= pd.Timestamp("2016-01-01")) &
    (long["date"] <= pd.Timestamp("2017-12-31"))
].copy()
run_replication(sub_2016, "2016--2017", "russel_2016")

sub_2024 = long[
    (long["date"] >= pd.Timestamp("2024-01-01")) &
    (long["date"] <= pd.Timestamp("2025-12-31"))
].copy()
run_replication(sub_2024, "2024--2025", "russel_2024")


# ===========================================================================
# BLOCK 5 — TEMPORAL HELPERS (Study 3)
# ===========================================================================

def run_window(label, start, end, bandwidth, min_cs):
    sub     = long[(long["date"] >= start) & (long["date"] <= end)].copy()
    n_dates = sub["date"].nunique()
    row     = {
        "window_label": label,
        "window_start": start.date(),
        "window_end":   end.date(),
        "n_dates":      n_dates,
        "n_obs":        len(sub),
        "dep_var":      DEP_VAR,
    }
    nan_cols = (
        ["fmb_coef", "fmb_se", "fmb_pval", "fmb_stars"]
        + [f"nr_lag{k}_{s}" for k in NR_LAGS for s in ["coef", "se", "pval", "stars"]]
        + ["ls_mean_daily", "ls_ann_return", "ls_sharpe", "ls_win_rate", "ls_n_days"]
    )
    if n_dates < 10:
        for c in nan_cols:
            row[c] = "" if "stars" in c else (0 if c == "ls_n_days" else np.nan)
        return row

    # FMB return predictability
    res = fit_fmb(DEP_VAR, ["twitter_sent_lag1"], sub,
                  oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    est = extract_fmb(res, "twitter_sent_lag1")
    row.update({"fmb_coef": est["coef"], "fmb_se": est["se"],
                "fmb_pval": est["pval"], "fmb_stars": est["stars"]})

    # No-reversal test
    nr_res = fit_fmb(DEP_VAR, [f"twitter_sent_lag{k}" for k in NR_LAGS], sub,
                     oth_vars=DEFAULT_OTH_VARS, bandwidth=bandwidth, min_cs=min_cs)
    for k in NR_LAGS:
        est = extract_fmb(nr_res, f"twitter_sent_lag{k}")
        row.update({f"nr_lag{k}_coef": est["coef"], f"nr_lag{k}_se": est["se"],
                    f"nr_lag{k}_pval": est["pval"], f"nr_lag{k}_stars": est["stars"]})

    # Long-short
    row.update(long_short_stats(sub))
    return row


def monthly_windows(start_year=2016, end=(2026, 2)):
    windows, y, m = [], start_year, 1
    while (y, m) <= end:
        s = pd.Timestamp(y, m, 1)
        e = s + pd.offsets.MonthEnd(0)
        windows.append((s.strftime("%Y-%m"), s, e))
        m += 1
        if m > 12: m, y = 1, y + 1
    return windows


def quarterly_windows(start_year=2016, end=(2026, 1)):
    windows = []
    for y in range(start_year, end[0] + 1):
        for q in range(1, 5):
            qm = (q - 1) * 3 + 1
            s  = pd.Timestamp(y, qm, 1)
            e  = pd.Timestamp(y, qm + 2, 1) + pd.offsets.MonthEnd(0)
            if s > pd.Timestamp(end[0], (end[1] - 1) // 3 * 3 + 1, 1):
                return windows
            windows.append((f"{y}-Q{q}", s, e))
    return windows


def annual_windows(start_year=2016, end_year=2026):
    windows = []
    for y in range(start_year, end_year + 1):
        s = pd.Timestamp(y, 1, 1)
        e = pd.Timestamp(y, 2, 28) if y == 2026 else pd.Timestamp(y, 12, 31)
        windows.append((str(y) if y < 2026 else "2026 (Jan-Feb)", s, e))
    return windows


# ===========================================================================
# STUDY 3 — LONG-TERM TEMPORAL ANALYSIS
# ===========================================================================

FREQ_CONFIG = [
    ("monthly",   monthly_windows(),   3, 15),
    ("quarterly", quarterly_windows(), 4, 20),
    ("annual",    annual_windows(),    5, 30),
]

temporal_results = {}

for freq, windows, bw, mcs in FREQ_CONFIG:
    print(f"\n{'='*65}")
    print(f"  TEMPORAL {freq.upper()}  "
          f"(bandwidth={bw}, min_cs={mcs}, n_windows={len(windows)})")
    print("="*65)
    rows = []
    for label, start, end in windows:
        row = run_window(label, start, end, bandwidth=bw, min_cs=mcs)
        rows.append(row)
        c  = f"{row['fmb_coef']:>8.4f}" if not np.isnan(row.get("fmb_coef", np.nan)) else "     n/a"
        sh = f"{row['ls_sharpe']:>5.2f}" if not np.isnan(row.get("ls_sharpe", np.nan)) else "  n/a"
        print(f"  {label:<16}  FMB={c}{row.get('fmb_stars',''):3s}  "
              f"Sharpe={sh}  n_dates={row.get('n_dates', 0):>3d}")
    df = pd.DataFrame(rows)
    temporal_results[freq] = df
    df.to_csv(OUT_DIR / f"russel_temporal_{freq}.csv", index=False)
    print(f"\n  Saved russel_temporal_{freq}.csv")

monthly_df   = temporal_results["monthly"]
quarterly_df = temporal_results["quarterly"]
annual_df    = temporal_results["annual"]

# Summary — annual
print(f"\n\n{'='*65}")
print("  SUMMARY — Annual FMB coefficient on twitter_sent_lag1")
print("="*65)
print(f"{'Year':<20} {'FMB coef':>10} {'SE':>10} {'p-val':>8} {'Sharpe':>8}")
print("-"*60)
for _, r in annual_df.iterrows():
    coef   = f"{r['fmb_coef']:>10.4f}" if not pd.isna(r["fmb_coef"]) else "       n/a"
    se     = f"{r['fmb_se']:>10.4f}"   if not pd.isna(r["fmb_se"])   else "       n/a"
    pval   = f"{r['fmb_pval']:>8.4f}"  if not pd.isna(r["fmb_pval"]) else "     n/a"
    sharpe = f"{r['ls_sharpe']:>8.2f}" if not pd.isna(r["ls_sharpe"]) else "     n/a"
    print(f"  {r['window_label']:<18} {coef} {se} {pval} {sharpe}")


# ===========================================================================
# TEMPORAL PLOTS — three separate two-panel figures
# ===========================================================================

def prep_plot(df):
    d = df.copy()
    d["t"] = pd.to_datetime(d["window_start"])
    return d.dropna(subset=["fmb_coef", "fmb_se", "fmb_pval"]).sort_values("t")


def plot_temporal(df, freq_label, color, lw, marker, out_name):
    d = prep_plot(df)
    if d.empty:
        print(f"  [SKIP] No valid data for {freq_label}.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.subplots_adjust(hspace=0.25)

    t, coef, se, pval = d["t"], d["fmb_coef"], d["fmb_se"], d["fmb_pval"]

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax1.plot(t, coef, color=color, linewidth=lw,
             marker=marker, markersize=5 if marker else 0, alpha=0.9)
    ax1.fill_between(t, coef - se, coef + se, color=color, alpha=0.15)
    ax1.set_ylabel("FMB coefficient\n(twitter_sent_lag1)", fontsize=10)
    ax1.set_title(
        f"Twitter Sentiment Predictability — {freq_label} Windows\n"
        f"Gu & Kurov Specification, Russell 3000",
        fontsize=11, fontweight="bold"
    )
    ax1.tick_params(axis="x", labelsize=9)

    ax2.axhline(0.10, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(0.05, color="#444444", linewidth=0.8, linestyle=":",  alpha=0.6)
    t_left = t.iloc[0] - pd.Timedelta(days=10)
    ax2.text(t_left, 0.105, "p = 0.10", fontsize=8, color="#555555",
             va="bottom", ha="right")
    ax2.text(t_left, 0.055, "p = 0.05", fontsize=8, color="#555555",
             va="bottom", ha="right")
    ax2.plot(t, pval, color=color, linewidth=lw,
             marker=marker, markersize=5 if marker else 0, alpha=0.9)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("p-value", fontsize=10)
    ax2.set_xlabel("Time", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    save_path = FIG_DIR / out_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


print("\n\nGenerating temporal figures...")
plot_temporal(annual_df,    "Annual",    "#111111", 2.0, "o",  "russel_temporal_annual.png")
plot_temporal(quarterly_df, "Quarterly", "#444444", 1.5, "s",  "russel_temporal_quarterly.png")
plot_temporal(monthly_df,   "Monthly",   "#888888", 0.9, None, "russel_temporal_monthly.png")


# ===========================================================================
# TEMPORAL LATEX EXPORTS
# ===========================================================================

def _st(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# 1. Figure environments
FREQ_FIGS = [
    ("annual",    "russel_temporal_annual.png",
     "Annual windows (one observation per calendar year). "
     r"Bandwidth = 5 lags; minimum cross-section $N = 30$."),
    ("quarterly", "russel_temporal_quarterly.png",
     "Quarterly windows. "
     r"Bandwidth = 4 lags; minimum cross-section $N = 20$."),
    ("monthly",   "russel_temporal_monthly.png",
     "Monthly windows. "
     r"Bandwidth = 3 lags; minimum cross-section $N = 15$."),
]
tex_fig_parts = []
for freq, fname, note in FREQ_FIGS:
    tex_fig_parts.append(
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\includegraphics[width=\\linewidth]{{figures/{fname}}}\n"
        r"\caption{Twitter Sentiment Predictability Over Time --- "
        f"Gu \\& Kurov Specification, Russell 3000 ({freq.capitalize()} windows). "
        r"Top panel: Fama-MacBeth coefficient on \texttt{twitter\_sent\_lag1} "
        r"with $\pm$1 SE band. Bottom panel: $p$-value; dashed lines "
        r"at $p = 0.10$ and $p = 0.05$. "
        + note + r"}" + "\n"
        f"\\label{{fig:russel_temporal_{freq}}}\n"
        r"\end{figure}"
    )
(TEX_OUT / "russel_temporal_fig.tex").write_text("\n\n".join(tex_fig_parts))
print("Saved russel_temporal_fig.tex  (3 figure environments)")

# 2. Summary statistics
sum_vars = {
    r"return\_oo":      "return_oo",
    r"twitter\_sent":   "twitter_sent",
    r"news\_sent":      "news_sent",
    r"vol\_rs":         "vol_rs",
    r"abnorm\_vol":     "abnorm_vol",
    r"log\_mkt\_cap":  "log_mkt_cap",
}
if HAS_SPREAD:       sum_vars[r"bid\_ask\_spread"] = "bid_ask_spread"
if HAS_ADJ_RETURN:   sum_vars[r"return\_oo\_adj"] = "return_oo_adj"

sum_rows = []
for label, col in sum_vars.items():
    if col not in long.columns: continue
    s = long[col].dropna()
    sum_rows.append(
        f"  {label} & {fmt(s.mean())} & {fmt(s.std())} & "
        f"{fmt(s.min())} & {fmt(s.max())} & {len(s):,} \\\\"
    )
tex_sumstats = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Summary Statistics --- Russell 3000 Panel (2016--2026)}" + "\n"
    r"\label{tab:russel_sumstats}" + "\n"
    r"\begin{tabular}{lrrrrr}" + "\n"
    r"\hline\hline" + "\n"
    r"Variable & Mean & SD & Min & Max & $N$ \\" + "\n"
    r"\hline" + "\n"
    + "\n".join(sum_rows) + "\n"
    + r"\hline\hline" + "\n"
    + r"\multicolumn{6}{p{0.95\linewidth}}{\footnotesize \textit{Notes:} "
    + r"Daily firm-level panel. return\_oo = open-to-open return (\%). "
    + r"vol\_rs = Rogers-Satchell realized volatility. "
    + r"abnorm\_vol = abnormal volume (\% deviation from ticker mean). "
    + r"log\_mkt\_cap = log market capitalization. "
    + (r"return\_oo\_adj = Fama-French-Carhart 4-factor residual." if HAS_ADJ_RETURN else "")
    + r"} \\" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
(TEX_OUT / "russel_temporal_sumstats.tex").write_text(tex_sumstats)
print("Saved russel_temporal_sumstats.tex")

# 3. FMB regression table (annual)
ann = annual_df.copy()
fmb_rows = []
for _, r in ann.iterrows():
    sig = _st(r["fmb_pval"])
    n   = f"{int(r['n_obs']):,}" if not pd.isna(r["n_obs"]) else "---"
    fmb_rows.append(
        f"  {r['window_label']} & ${fmt(r['fmb_coef'])}{sig}$ "
        f"& $({fmt(r['fmb_se'])})$ & ${fmt(r['fmb_pval'])}$ & {n} \\\\"
    )
tex_fmb = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{FMB Return Predictability by Year --- Russell 3000}" + "\n"
    r"\label{tab:russel_fmb_annual}" + "\n"
    r"\begin{tabular}{lcccc}" + "\n"
    r"\hline\hline" + "\n"
    r"Year & Coef & (SE) & $p$-value & $N$ \\" + "\n"
    r"\hline" + "\n"
    + "\n".join(fmb_rows) + "\n"
    + r"\hline\hline" + "\n"
    + r"\multicolumn{5}{p{0.9\linewidth}}{\footnotesize \textit{Notes:} "
    + r"Fama-MacBeth regression with Newey-West SEs (bandwidth = 5). "
    + f"Dependent variable: \\texttt{{{DEP_VAR}}}. "
    + r"Treatment: \texttt{twitter\_sent\_lag1}. "
    + r"Controls: " + _ctrl_note() + r" (5 lags each). "
    + r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$.} \\" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
(TEX_OUT / "russel_temporal_fmb.tex").write_text(tex_fmb)
print("Saved russel_temporal_fmb.tex")

# 4. Long-short strategy table (annual)
ls_rows = []
for _, r in ann.iterrows():
    n_d = int(r["ls_n_days"]) if not pd.isna(r["ls_n_days"]) else "---"
    ls_rows.append(
        f"  {r['window_label']} "
        f"& {fmt(r['ls_mean_daily'], 4)} "
        f"& {fmt(r['ls_ann_return'], 2)} "
        f"& {fmt(r['ls_sharpe'],     2)} "
        f"& {fmt(r['ls_win_rate'],   3)} "
        f"& {n_d} \\\\"
    )
tex_ls = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Long-Short Strategy Performance by Year --- Russell 3000}" + "\n"
    r"\label{tab:russel_ls_annual}" + "\n"
    r"\begin{tabular}{lccccc}" + "\n"
    r"\hline\hline" + "\n"
    r"Year & Mean Daily (\%) & Ann. Return (\%) & Sharpe & Win Rate & Days \\" + "\n"
    r"\hline" + "\n"
    + "\n".join(ls_rows) + "\n"
    + r"\hline\hline" + "\n"
    + r"\multicolumn{6}{p{0.95\linewidth}}{\footnotesize \textit{Notes:} "
    + r"Daily long-short portfolio. Long (short) = top (bottom) decile of contemporaneous "
    + r"Twitter sentiment. 24-hour holding period (open-to-open). "
    + r"Sharpe annualized assuming 252 trading days. Before transaction costs. "
    + r"Gu \& Kurov (2020) report Sharpe = 3.17 over Jan 2015--Feb 2017.} \\" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
(TEX_OUT / "russel_temporal_ls.tex").write_text(tex_ls)
print("Saved russel_temporal_ls.tex")

print(f"\nAll outputs written.")
print("  LaTeX → output/")
print("  Figures → output/figures/")
print("  CSVs → data/processed/")
