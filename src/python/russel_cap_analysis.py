"""
russel_cap_analysis.py
Russell 3000 — market-cap heterogeneity in Twitter sentiment effects.

Three analyses:
  1. Pooled OLS with twitter_sent × log_mkt_cap interaction term (firm FE).
     5 columns: full-sample baseline, full-sample with interaction, then
     large / mid / small cap subsamples separately.

  2. Temporal FMB by cap group — annual and quarterly windows.
     Each figure has three lines (large / mid / small) for the FMB coefficient
     on twitter_sent_lag1 (top panel) and its p-value (bottom panel).

  3. Long-short strategy by cap group — annual table with one column per
     cap group plus a full-sample column.

Cap classification (fixed over the sample period per ticker):
  Large : average mkt_cap >  $10 billion  (mkt_cap > 10,000 in millions)
  Mid   : average mkt_cap   $2B–$10B      (2,000 <= mkt_cap <= 10,000)
  Small : average mkt_cap <  $2 billion   (mkt_cap < 2,000)

Outputs — LaTeX (output/):
  russel_cap_ols.tex
  russel_cap_fmb_annual.tex
  russel_cap_ls_annual.tex
  russel_cap_temporal_fig.tex        (figure environments for both plots)

Outputs — Figures (output/figures/):
  russel_cap_temporal_annual.png
  russel_cap_temporal_quarterly.png

Outputs — CSVs (data/processed/):
  russel_cap_temporal_annual.csv
  russel_cap_temporal_quarterly.csv

Usage:
    python src/python/russel_cap_analysis.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
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

# Cap thresholds — millions of USD
CAP_LARGE = 10_000   # > $10B
CAP_MID   =  2_000   # $2B–$10B
# Small = below CAP_MID

CAP_COLORS  = {"large": "#1f77b4", "mid": "#ff7f0e", "small": "#2ca02c"}
CAP_MARKERS = {"large": "o",       "mid": "s",       "small": "^"}
CAP_LABELS  = {"large": "Large (>\\$10B)", "mid": "Mid (\\$2B–\\$10B)", "small": "Small (<\\$2B)"}
CAP_ORDER   = ["large", "mid", "small"]


# ===========================================================================
# BLOCK 1 — LOAD & VARIABLE CONSTRUCTION
# ===========================================================================

long = pd.read_csv(IN_FILE, parse_dates=["date"])
long = long[long["date"] <= FF_CUTOFF].reset_index(drop=True)
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(long):,} rows  |  "
      f"{long['ticker'].nunique():,} tickers  |  "
      f"{long['date'].min().date()} → {long['date'].max().date()}")

HAS_SPREAD = "bid_ask_spread" in long.columns

# Open-to-open return
long["px_open_next"] = long.groupby("ticker")["px_open"].shift(-1)
long["return_oo"]    = (long["px_open_next"] - long["px_open"]) / long["px_open"] * 100

# Rogers-Satchell realized volatility
lH = np.log(long["px_high"].clip(lower=1e-8))
lL = np.log(long["px_low"].clip(lower=1e-8))
lO = np.log(long["px_open"].clip(lower=1e-8))
lC = np.log(long["px_close"].clip(lower=1e-8))
long["vol_rs"] = ((lH - lC) * (lH - lO) + (lL - lC) * (lL - lO)).clip(lower=0) * 100

# Abnormal volume and log market cap
mean_vol            = long.groupby("ticker")["volume"].transform("mean")
long["abnorm_vol"]  = ((long["volume"] - mean_vol) / mean_vol) * 100
long["log_mkt_cap"] = np.log(long["mkt_cap"].clip(lower=1e-8))

# Lags (controls and sentiment)
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


# ===========================================================================
# BLOCK 2 — FAMA-FRENCH RISK ADJUSTMENT
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

    merged    = long.merge(
        ff.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "Mom": "mom"}),
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
    print(f"[WARNING] FF adjustment failed: {e}  →  using raw returns.")

DEP_VAR       = "return_oo_adj" if HAS_ADJ_RETURN else "return_oo"
DEFAULT_CTRL  = (
    ([DEP_VAR] if HAS_ADJ_RETURN else ["return_oo"])
    + ["abnorm_vol", "vol_rs", "log_mkt_cap"]
    + (["bid_ask_spread"] if HAS_SPREAD else [])
)


# ===========================================================================
# BLOCK 3 — CAP GROUP ASSIGNMENT
# Fixed classification: each ticker gets one group based on its time-averaged
# mkt_cap. This prevents firms from jumping buckets within the panel and
# keeps the subgroup comparisons clean.
# ===========================================================================

avg_cap = long.groupby("ticker")["mkt_cap"].mean()
cap_map = pd.cut(
    avg_cap,
    bins=[0, CAP_MID, CAP_LARGE, np.inf],
    labels=["small", "mid", "large"]
)
long["cap_group"] = long["ticker"].map(cap_map)

# Report composition
print("\nCap group composition:")
for g in CAP_ORDER:
    n_tickers = (avg_cap.index.isin(cap_map[cap_map == g].index)).sum()
    n_obs     = (long["cap_group"] == g).sum()
    print(f"  {g:6s}  {n_tickers:4d} tickers  {n_obs:>10,} obs")


# ===========================================================================
# HELPERS
# ===========================================================================

def stars(p) -> str:
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt(v, d=4) -> str:
    return f"{v:.{d}f}" if (v is not None and not pd.isna(v)) else "---"


def _ctrl_note() -> str:
    base = r"5 lags each of return, abnorm\_vol, vol\_rs, log\_mkt\_cap"
    return base + (r", bid\_ask\_spread" if HAS_SPREAD else "")


# ---------------------------------------------------------------------------
# FMB helper (same pattern as russel_replication.py)
# ---------------------------------------------------------------------------
from linearmodels.panel import FamaMacBeth

def fit_fmb(dep, treatment_cols, data, oth_vars=None,
            n_lags=5, bandwidth=5, min_cs=20):
    if oth_vars is None:
        oth_vars = DEFAULT_CTRL
    controls = [f"{v}_lag{k}" for v in oth_vars for k in range(1, n_lags + 1)]
    formula  = dep + " ~ " + " + ".join(treatment_cols + controls)
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
    df = df.dropna(subset=["side"])
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
# ANALYSIS 1 — POOLED OLS WITH INTERACTION TERM (firm FE via PanelOLS)
#
# Models:
#   (1) Full sample, twitter_sent only (baseline — no interaction)
#   (2) Full sample, twitter_sent + twitter_sent:log_mkt_cap (interaction)
#   (3) Large cap subset, twitter_sent only
#   (4) Mid   cap subset, twitter_sent only
#   (5) Small cap subset, twitter_sent only
# ===========================================================================

print(f"\n{'='*65}")
print("  ANALYSIS 1 — Panel OLS with interaction term")
print("="*65)

from linearmodels.panel import PanelOLS

CTRL_VARS = ["log_mkt_cap", "abnorm_vol", "vol_rs"] + (["bid_ask_spread"] if HAS_SPREAD else [])
ctrl_str  = " + ".join(CTRL_VARS)

def run_panel_ols(data, formula_rhs, label=""):
    """Fit PanelOLS with entity effects; return result or None."""
    try:
        panel = data.set_index(["ticker", "date"])
        mod   = PanelOLS.from_formula(
            f"{DEP_VAR} ~ {formula_rhs} + EntityEffects",
            data=panel, drop_absorbed=True
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        print(f"  [{label}]  N={res.nobs:,}  "
              f"twitter_sent coef={res.params.get('twitter_sent', np.nan):.4f}")
        return res
    except Exception as ex:
        print(f"  [{label}]  FAILED: {ex}")
        return None


# Build formulas
f_base        = f"twitter_sent + {ctrl_str}"
f_interaction = f"twitter_sent * log_mkt_cap + {' + '.join([v for v in CTRL_VARS if v != 'log_mkt_cap'])} + abnorm_vol + vol_rs"
# Note: twitter_sent * log_mkt_cap expands to main effects + interaction in patsy

ols_results = {
    "full_base":        run_panel_ols(long,                             f_base,        "Full — baseline"),
    "full_interaction": run_panel_ols(long,                             f_interaction, "Full — interaction"),
    "large":            run_panel_ols(long[long["cap_group"] == "large"], f_base,      "Large cap"),
    "mid":              run_panel_ols(long[long["cap_group"] == "mid"],   f_base,      "Mid cap"),
    "small":            run_panel_ols(long[long["cap_group"] == "small"], f_base,      "Small cap"),
}


def _ols_cell(res, param):
    """Return (coef_str, se_str) for one parameter from a PanelOLS result."""
    if res is None or param not in res.params.index:
        return "---", ""
    c  = res.params[param]
    s  = res.std_errors[param]
    pv = res.pvalues[param]
    return f"${fmt(c)}{stars(pv)}$", f"$({fmt(s)})$"


# LaTeX OLS table
_params = [
    ("twitter\\_sent",               "twitter_sent"),
    ("log\\_mkt\\_cap",              "log_mkt_cap"),
    ("twitter\\_sent $\\times$ log\\_mkt\\_cap", "twitter_sent:log_mkt_cap"),
]
_cols   = ["full_base", "full_interaction", "large", "mid", "small"]
_heads  = ["(1) Full", "(2) Full+Int.", "(3) Large", "(4) Mid", "(5) Small"]

rows = []
for label_tex, param_key in _params:
    coef_row = label_tex
    se_row   = ""
    c_cells, s_cells = [], []
    for col in _cols:
        c, s = _ols_cell(ols_results[col], param_key)
        c_cells.append(c)
        s_cells.append(s)
    rows.append(" & ".join([label_tex] + c_cells) + " \\\\")
    rows.append(" & ".join([""]         + s_cells) + " \\\\")
    rows.append(r"\addlinespace[0.5ex]")

# Observation counts
n_row = "Observations"
for col in _cols:
    r = ols_results[col]
    n_row += f" & {int(r.nobs):,}" if r is not None else " & ---"
n_row += " \\\\"

# R-squared
r2_row = "$R^2$ (within)"
for col in _cols:
    r = ols_results[col]
    r2_row += f" & {fmt(r.rsquared, 3)}" if r is not None else " & ---"
r2_row += " \\\\"

ncols = len(_cols) + 1
col_spec = "l" + "c" * len(_cols)
head_str = " & ".join([""] + _heads)

tex_ols = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Panel OLS: Twitter Sentiment $\times$ Market Cap --- Russell 3000}" + "\n"
    r"\label{tab:russel_cap_ols}" + "\n"
    f"\\begin{{tabular}}{{{col_spec}}}\n"
    r"\hline\hline" + "\n"
    f"{head_str} \\\\\n"
    r"\hline" + "\n"
    r"\addlinespace[1ex]" + "\n"
    + "\n".join(rows) + "\n"
    r"\midrule" + "\n"
    r"\addlinespace[0.5ex]" + "\n"
    + "Firm FE" + " & Yes" * len(_cols) + " \\\\\n"
    + r"\addlinespace[0.5ex]" + "\n"
    + n_row + "\n"
    + r"\addlinespace[0.5ex]" + "\n"
    + r2_row + "\n"
    r"\hline\hline" + "\n"
    f"\\multicolumn{{{ncols}}}{{p{{0.95\\linewidth}}}}{{\\footnotesize \\textit{{Notes:}} "
    f"Panel OLS with firm fixed effects. Clustered SEs (by entity) in parentheses. "
    f"Dependent variable: \\texttt{{{DEP_VAR}}}. "
    f"Columns (3)--(5) restrict to cap group defined by average \\texttt{{mkt\\_cap}}: "
    f"Large $>\\$10$B, Mid \\$2B--\\$10B, Small $<\\$2$B. "
    r"Controls: log\_mkt\_cap, abnorm\_vol, vol\_rs"
    + (r", bid\_ask\_spread" if HAS_SPREAD else "")
    + r". *** $p<0.01$, ** $p<0.05$, * $p<0.1$.}}" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table}"
)
(TEX_OUT / "russel_cap_ols.tex").write_text(tex_ols)
print("  Saved russel_cap_ols.tex")


# ===========================================================================
# ANALYSIS 2 — TEMPORAL FMB BY CAP GROUP (annual + quarterly)
# ===========================================================================

print(f"\n{'='*65}")
print("  ANALYSIS 2 — Temporal FMB by cap group")
print("="*65)

NR_LAGS = [1, 2, 3, 4, 5]


def run_window_by_cap(label, start, end, bandwidth, min_cs):
    """
    Run FMB for full sample and each cap group within a date window.
    Returns a dict with columns prefixed by cap group name.
    """
    sub = long[(long["date"] >= start) & (long["date"] <= end)].copy()
    row = {
        "window_label": label,
        "window_start": start.date(),
        "window_end":   end.date(),
        "n_dates":      sub["date"].nunique(),
        "n_obs":        len(sub),
    }

    if sub["date"].nunique() < 10:
        for grp in CAP_ORDER + ["full"]:
            for k in ["coef", "se", "pval", "stars"]:
                row[f"{grp}_fmb_{k}"] = np.nan if k != "stars" else ""
            row[f"{grp}_ls_sharpe"]   = np.nan
            row[f"{grp}_ls_n_days"]   = 0
        return row

    for grp_name, grp_data in [("full", sub)] + [(g, sub[sub["cap_group"] == g]) for g in CAP_ORDER]:
        if len(grp_data) < 100:
            for k in ["coef", "se", "pval", "stars"]:
                row[f"{grp_name}_fmb_{k}"] = np.nan if k != "stars" else ""
            row[f"{grp_name}_ls_sharpe"] = np.nan
            row[f"{grp_name}_ls_n_days"] = 0
            continue

        res = fit_fmb(DEP_VAR, ["twitter_sent_lag1"], grp_data,
                      bandwidth=bandwidth, min_cs=min_cs)
        est = extract_fmb(res, "twitter_sent_lag1")
        row[f"{grp_name}_fmb_coef"]  = est["coef"]
        row[f"{grp_name}_fmb_se"]    = est["se"]
        row[f"{grp_name}_fmb_pval"]  = est["pval"]
        row[f"{grp_name}_fmb_stars"] = est["stars"]

        ls = long_short_stats(grp_data)
        row[f"{grp_name}_ls_sharpe"] = ls["ls_sharpe"]
        row[f"{grp_name}_ls_n_days"] = ls["ls_n_days"]

    return row


def annual_windows():
    windows = []
    for y in range(2016, 2027):
        s = pd.Timestamp(y, 1, 1)
        e = pd.Timestamp(y, 2, 28) if y == 2026 else pd.Timestamp(y, 12, 31)
        windows.append((str(y) if y < 2026 else "2026 (Jan-Feb)", s, e))
    return windows


def quarterly_windows():
    windows = []
    for y in range(2016, 2027):
        for q in range(1, 5):
            qm = (q - 1) * 3 + 1
            s  = pd.Timestamp(y, qm, 1)
            e  = (pd.Timestamp(y, qm + 2, 1) + pd.offsets.MonthEnd(0))
            if s > pd.Timestamp(2026, 1, 1):
                break
            windows.append((f"{y}-Q{q}", s, e))
        else:
            continue
        break
    return windows


FREQ_CONFIG = [
    ("annual",    annual_windows(),    5, 20),
    ("quarterly", quarterly_windows(), 4, 15),
]

temporal_by_cap = {}

for freq, windows, bw, mcs in FREQ_CONFIG:
    print(f"\n  {freq.upper()}  (bandwidth={bw}, min_cs={mcs})")
    rows = []
    for label, start, end in windows:
        row = run_window_by_cap(label, start, end, bandwidth=bw, min_cs=mcs)
        rows.append(row)
        parts = [f"  {label:<16}"]
        for g in CAP_ORDER:
            c = row.get(f"{g}_fmb_coef", np.nan)
            s = row.get(f"{g}_fmb_stars", "")
            parts.append(f"{g}={c:>8.4f}{s:3s}" if not pd.isna(c) else f"{g}=     n/a   ")
        print("".join(parts))
    df = pd.DataFrame(rows)
    temporal_by_cap[freq] = df
    df.to_csv(OUT_DIR / f"russel_cap_temporal_{freq}.csv", index=False)
    print(f"  Saved russel_cap_temporal_{freq}.csv")


# ===========================================================================
# TEMPORAL PLOTS — one figure per frequency, three lines per panel
# ===========================================================================

def plot_temporal_by_cap(df, freq_label, out_name):
    d = df.copy()
    d["t"] = pd.to_datetime(d["window_start"])
    d = d.sort_values("t")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.subplots_adjust(hspace=0.25)

    lw = 1.8 if "annual" in out_name else 1.2

    for g in CAP_ORDER:
        col_c = f"{g}_fmb_coef"
        col_s = f"{g}_fmb_se"
        col_p = f"{g}_fmb_pval"
        sub   = d.dropna(subset=[col_c, col_s, col_p])
        if sub.empty:
            continue
        t, coef, se, pval = sub["t"], sub[col_c], sub[col_s], sub[col_p]
        color  = CAP_COLORS[g]
        marker = CAP_MARKERS[g]

        ax1.plot(t, coef, color=color, linewidth=lw,
                 marker=marker, markersize=5, alpha=0.9, label=g.capitalize())
        ax1.fill_between(t, coef - se, coef + se, color=color, alpha=0.10)

        ax2.plot(t, pval, color=color, linewidth=lw,
                 marker=marker, markersize=5, alpha=0.9, label=g.capitalize())

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax1.set_ylabel("FMB coefficient\n(twitter\_sent\_lag1)", fontsize=10)
    ax1.set_title(
        f"Twitter Sentiment Predictability by Market Cap — {freq_label} Windows\n"
        f"Russell 3000  |  Gu & Kurov Specification",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(title="Cap Group", fontsize=9, title_fontsize=9,
               loc="upper right", framealpha=0.8)
    ax1.tick_params(axis="x", labelsize=9)

    ax2.axhline(0.10, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(0.05, color="#444444", linewidth=0.8, linestyle=":",  alpha=0.6)
    t_vals = d["t"].dropna()
    if not t_vals.empty:
        t_left = t_vals.iloc[0] - pd.Timedelta(days=10)
        ax2.text(t_left, 0.105, "p = 0.10", fontsize=8, color="#555555",
                 va="bottom", ha="right")
        ax2.text(t_left, 0.055, "p = 0.05", fontsize=8, color="#555555",
                 va="bottom", ha="right")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("p-value", fontsize=10)
    ax2.set_xlabel("Time", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis="x", labelsize=9)
    ax2.legend(title="Cap Group", fontsize=9, title_fontsize=9,
               loc="upper right", framealpha=0.8)

    fig.tight_layout()
    save_path = FIG_DIR / out_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


print("\n\nGenerating temporal figures...")
plot_temporal_by_cap(temporal_by_cap["annual"],    "Annual",    "russel_cap_temporal_annual.png")
plot_temporal_by_cap(temporal_by_cap["quarterly"], "Quarterly", "russel_cap_temporal_quarterly.png")


# ===========================================================================
# ANALYSIS 3 — LONG-SHORT STRATEGY BY CAP GROUP (annual table)
# ===========================================================================

print(f"\n{'='*65}")
print("  ANALYSIS 3 — Long-short strategy by cap group")
print("="*65)

ann_df = temporal_by_cap["annual"]

ls_rows = []
for _, r in ann_df.iterrows():
    cells = [r["window_label"]]
    for grp in CAP_ORDER + ["full"]:
        sh = r.get(f"{grp}_ls_sharpe", np.nan)
        nd = r.get(f"{grp}_ls_n_days", 0)
        cells.append(fmt(sh, 2) if not pd.isna(sh) else "---")
    ls_rows.append(" & ".join(cells) + " \\\\")

tex_ls = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{Long-Short Strategy Sharpe Ratio by Year and Cap Group --- Russell 3000}" + "\n"
    r"\label{tab:russel_cap_ls_annual}" + "\n"
    r"\begin{tabular}{lcccc}" + "\n"
    r"\hline\hline" + "\n"
    r"Year & Large & Mid & Small & Full \\" + "\n"
    r"\hline" + "\n"
    + "\n".join(ls_rows) + "\n"
    + r"\hline\hline" + "\n"
    + r"\multicolumn{5}{p{0.9\linewidth}}{\footnotesize \textit{Notes:} "
    + r"Annualized Sharpe ratio of daily long-short portfolio (long top decile / short bottom decile "
    + r"of contemporaneous Twitter sentiment) within each cap group. "
    + r"24-hour holding period (open-to-open). Before transaction costs. "
    + r"Cap groups defined by average market cap over 2016--2026: "
    + r"Large $>$\$10B, Mid \$2B--\$10B, Small $<$\$2B. "
    + r"Deciles computed within cap group on each date.} \\" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
(TEX_OUT / "russel_cap_ls_annual.tex").write_text(tex_ls)
print("  Saved russel_cap_ls_annual.tex")

# Annual FMB table (all three groups side-by-side)
fmb_rows = []
for _, r in ann_df.iterrows():
    cells = [r["window_label"]]
    for grp in CAP_ORDER:
        c  = r.get(f"{grp}_fmb_coef",  np.nan)
        se = r.get(f"{grp}_fmb_se",    np.nan)
        st = r.get(f"{grp}_fmb_stars", "")
        if pd.isna(c):
            cells.append("---")
        else:
            cells.append(f"${fmt(c)}{st}$")
    fmb_rows.append(" & ".join(cells) + " \\\\")

tex_fmb = (
    r"\begin{table}[htbp]" + "\n"
    r"\centering" + "\n"
    r"\caption{FMB Return Predictability by Year and Cap Group --- Russell 3000}" + "\n"
    r"\label{tab:russel_cap_fmb_annual}" + "\n"
    r"\begin{tabular}{lccc}" + "\n"
    r"\hline\hline" + "\n"
    r"Year & Large (>{\$}10B) & Mid ({\$}2B--10B) & Small (<{\$}2B) \\" + "\n"
    r"\hline" + "\n"
    + "\n".join(fmb_rows) + "\n"
    + r"\hline\hline" + "\n"
    + r"\multicolumn{4}{p{0.9\linewidth}}{\footnotesize \textit{Notes:} "
    + r"Fama-MacBeth coefficient on \texttt{twitter\_sent\_lag1} by cap group and year. "
    + f"Dependent variable: \\texttt{{{DEP_VAR}}}. "
    + r"Controls: " + _ctrl_note() + r" (5 lags each). "
    + r"Newey-West SEs. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.} \\" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
(TEX_OUT / "russel_cap_fmb_annual.tex").write_text(tex_fmb)
print("  Saved russel_cap_fmb_annual.tex")

# Figure environments for both temporal plots
FREQ_FIGS = [
    ("annual",    "russel_cap_temporal_annual.png",
     r"Annual windows. Bandwidth = 5 lags; minimum cross-section $N = 20$ per group."),
    ("quarterly", "russel_cap_temporal_quarterly.png",
     r"Quarterly windows. Bandwidth = 4 lags; minimum cross-section $N = 15$ per group."),
]
tex_fig_parts = []
for freq, fname, note in FREQ_FIGS:
    tex_fig_parts.append(
        r"\begin{figure}[htbp]" + "\n"
        r"\centering" + "\n"
        f"\\includegraphics[width=\\linewidth]{{figures/{fname}}}\n"
        r"\caption{Twitter Sentiment Predictability by Market Cap --- Russell 3000 "
        f"({freq.capitalize()} windows). "
        r"Top panel: Fama-MacBeth coefficient on \texttt{twitter\_sent\_lag1} "
        r"with $\pm$1 SE band, separately for large (blue), mid (orange), and small (green) "
        r"cap firms. Bottom panel: $p$-values; dashed lines at $p = 0.10$ and $p = 0.05$. "
        + note + r"}" + "\n"
        f"\\label{{fig:russel_cap_temporal_{freq}}}\n"
        r"\end{figure}"
    )
(TEX_OUT / "russel_cap_temporal_fig.tex").write_text("\n\n".join(tex_fig_parts))
print("  Saved russel_cap_temporal_fig.tex")

print(f"\nAll outputs written.")
print("  LaTeX  → output/")
print("  Figures → output/figures/")
print("  CSVs   → data/processed/")
