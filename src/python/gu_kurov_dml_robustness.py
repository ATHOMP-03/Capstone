"""
gu_kurov_dml_robustness.py
DoubleML PLR robustness checks for the Gu & Kurov (2020) replication.

Structure:
  MAIN ATE   — current twitter_sent / news_sent → return_oo
  PART 1     — lagged treatments → current return_oo  (reverse causality, univariate)
  NO-REVERSAL — all twitter_sent lags jointly in one PLR model  (Gu & Kurov Table 3)
  PART 2     — current treatments → return_oo_lead{n}  (impact persistence)

Treatment channels (Gu & Kurov):
  twitter_sent — Bloomberg composite Twitter sentiment
  news_sent    — Bloomberg composite news sentiment

Outcome: return_oo = open-to-open return, %
  return_{i,t} = (px_open_{t+1} - px_open_t) / px_open_t * 100
  Rationale: investor observes Twitter activity pre-open, executes at open, holds overnight.

Confounders (Gu & Kurov specification):
  Rogers-Satchell (1991) realized volatility, abnormal trading volume,
  log(market cap), bid-ask spread (if available), total equity, D/E ratio,
  RSI, 50-day MA, total tweet count.

Usage:
    python src/python/gu_kurov_dml_robustness.py
"""

import warnings
import numpy as np
import pandas as pd
import doubleml as dml
from xgboost import XGBRegressor
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]
IN_FILE  = ROOT / "data" / "processed" / "panel_long.csv"
EXT_FILE = ROOT / "data" / "processed" / "panel_long_extended.csv"
OUT_DIR  = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data (prefer extended file if available)
# ---------------------------------------------------------------------------
src  = EXT_FILE if EXT_FILE.exists() else IN_FILE
long = pd.read_csv(src, parse_dates=["date"])
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Loaded {len(long):,} rows x {long.shape[1]} cols from {src.name}")

HAS_SPREAD = "bid_ask_spread" in long.columns
if not HAS_SPREAD:
    print("[INFO] bid_ask_spread not found — omitted from confounder set.\n")


# ===========================================================================
# VARIABLE CONSTRUCTION  (mirrors gu_kurov_replication.py)
# ===========================================================================

# Open-to-open return
long["px_open_next"] = long.groupby("ticker")["px_open"].shift(-1)
long["return_oo"] = (long["px_open_next"] - long["px_open"]) / long["px_open"] * 100

# Rogers-Satchell (1991) realized volatility
log_H = np.log(long["px_high"].clip(lower=1e-8))
log_L = np.log(long["px_low"].clip(lower=1e-8))
log_O = np.log(long["px_open"].clip(lower=1e-8))
log_C = np.log(long["px_close"].clip(lower=1e-8))
long["vol_rs"] = (
    (log_H - log_C) * (log_H - log_O) + (log_L - log_C) * (log_L - log_O)
).clip(lower=0) * 100

# Abnormal trading volume
mean_vol = long.groupby("ticker")["volume"].transform("mean")
long["abnorm_vol"] = ((long["volume"] - mean_vol) / mean_vol) * 100

# Log market cap
long["log_mkt_cap"] = np.log(long["mkt_cap"].clip(lower=1e-8))


# ===========================================================================
# CONFIGURATION
# ===========================================================================

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

LAGS = [1, 2, 3, 5, 7]

# Gu & Kurov control set — excludes both treatment channels
GK_CONFOUNDERS = [
    "vol_rs", "abnorm_vol", "log_mkt_cap",
    "total_equity", "debt_to_equity",
    "twitter_count", "rsi_30", "ma_50",
]
if HAS_SPREAD:
    GK_CONFOUNDERS.append("bid_ask_spread")

TREATMENTS = [
    ("twitter_sent", "Twitter sentiment"),
    ("news_sent",    "News sentiment"),
]


# ===========================================================================
# HELPERS  (identical structure to teti_dml_robustness.py)
# ===========================================================================

def run_plr(df, y_col, d_col, x_cols):
    data  = dml.DoubleMLData(
        df[[y_col, d_col] + x_cols].dropna().reset_index(drop=True),
        y_col=y_col, d_cols=d_col, x_cols=x_cols,
    )
    model = dml.DoubleMLPLR(data, ml_l=make_xgb(), ml_m=make_xgb(), n_folds=5, n_rep=20)
    model.fit()
    model.bootstrap(method="normal", n_rep_boot=1000)
    coef     = model.summary["coef"].iloc[0]
    pval     = model.summary["P>|t|"].iloc[0]
    ci_lower = model.confint(level=0.95)["2.5 %"].iloc[0]
    ci_upper = model.confint(level=0.95)["97.5 %"].iloc[0]
    return coef, pval, ci_lower, ci_upper


def run_plr_multi(df, y_col, d_cols, x_cols):
    keep  = [y_col] + d_cols + x_cols
    data  = dml.DoubleMLData(
        df[keep].dropna().reset_index(drop=True),
        y_col=y_col, d_cols=d_cols, x_cols=x_cols,
    )
    model = dml.DoubleMLPLR(data, ml_l=make_xgb(), ml_m=make_xgb(), n_folds=5, n_rep=20)
    model.fit()
    model.bootstrap(method="normal", n_rep_boot=1000)
    return model


def print_table_header(title):
    print(f"\n\n{'='*70}")
    print(f"  {title}")
    print('='*70)
    print(f"{'Treatment':<30} {'Coef':>10} {'P-value':>10} {'CI Lower':>12} {'CI Upper':>12}")
    print("-" * 78)


def print_row(label, coef, pval, ci_lower, ci_upper):
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    print(
        f"{label:<30} {coef:>10.6f} {pval:>10.4f} "
        f"{ci_lower:>12.6f} {ci_upper:>12.6f}  {sig}"
    )


def to_latex(rows, caption, label, period_col=""):
    period_hdr = f" & {period_col.title()}" if period_col else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llcccc}" if period_col else r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        f"Treatment{period_hdr} & Coefficient & p-value & 95\\% CI Lower & 95\\% CI Upper \\\\",
        r"\hline",
    ]
    for r in rows:
        def stars(p):
            if p < 0.01: return "^{***}"
            if p < 0.05: return "^{**}"
            if p < 0.10: return "^{*}"
            return ""
        period_cell = f" & {int(r['period'])}" if period_col else ""
        lbl = str(r["label"]).replace("_", r"\_")
        lines.append(
            f"{lbl}{period_cell} & ${r['coef']:.6f}{stars(r['pval'])}$ "
            f"& {r['pval']:.4f} & {r['ci_lower']:.6f} & {r['ci_upper']:.6f} \\\\"
        )
    spread_note = " Bid-ask spread included." if HAS_SPREAD else " Bid-ask spread unavailable; omitted from confounders."
    lines += [
        r"\hline",
        r"\multicolumn{5}{l}{\footnotesize DoubleML PLR, XGBoost (GPU), 5-fold, 20 reps, 1000 bootstrap draws.} \\",
        f"\\multicolumn{{5}}{{l}}{{\\footnotesize Confounders: RS vol, abnorm vol, log mkt cap, D/E, RSI, MA50, tweet count.{spread_note}}} \\\\",
        r"\multicolumn{5}{l}{\footnotesize $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.} \\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# MAIN ATE — current treatment → return_oo
# ===========================================================================

print_table_header("MAIN ATE: Treatment → return_oo  (Gu & Kurov 2020)")
main_rows = []

for col, desc in TREATMENTS:
    coef, pval, ci_lower, ci_upper = run_plr(long, "return_oo", col, GK_CONFOUNDERS)
    print_row(desc, coef, pval, ci_lower, ci_upper)
    main_rows.append({"label": desc, "coef": coef, "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper})

tex = to_latex(
    main_rows,
    caption="Gu \\& Kurov (2020) Replication --- DoubleML PLR Main ATE Estimates",
    label="tab:gk_main_ate",
)
(OUT_DIR / "gk_main_ate.tex").write_text(tex)
print("\nSaved: output/gk_main_ate.tex")


# ===========================================================================
# PART 1 — Lagged treatments → current return_oo  (reverse causality)
# ===========================================================================

print_table_header("PART 1: Lagged treatment → return_oo  (reverse causality, univariate)")

for col, _ in TREATMENTS:
    for lag_n in LAGS:
        long[f"{col}_lag{lag_n}"] = long.groupby("ticker")[col].shift(lag_n)

part1_rows = []

for lag_n in LAGS:
    print(f"\n  --- Lag {lag_n} ---")
    for col, desc in TREATMENTS:
        lag_col = f"{col}_lag{lag_n}"
        coef, pval, ci_lower, ci_upper = run_plr(long, "return_oo", lag_col, GK_CONFOUNDERS)
        print_row(f"lag{lag_n} | {desc}", coef, pval, ci_lower, ci_upper)
        part1_rows.append({
            "label": desc, "period": lag_n, "coef": coef,
            "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper,
        })

tex = to_latex(
    part1_rows,
    caption="Gu \\& Kurov (2020) --- Lagged Sentiment and Current Open-to-Open Return (Reverse Causality Check)",
    label="tab:gk_lag_decay",
    period_col="lag",
)
(OUT_DIR / "gk_lag_decay.tex").write_text(tex)
print("\nSaved: output/gk_lag_decay.tex")


# ===========================================================================
# NO-REVERSAL TEST — Gu & Kurov (2020) simultaneous specification
#
# Replicates the logic of Gu & Kurov Table 3 under the DoubleML framework.
# All sentiment lags enter jointly; each coefficient is estimated conditional
# on the others, ruling out correlated-lag confounding.
#
# Pass: lag 1 significant, lags 2-7 insignificant → information absorbed at t+1.
# Fail: lags 3-7 persist → price-driven sentiment (return → tweets, not tweets → return).
# ===========================================================================

print_table_header("NO-REVERSAL TEST: all sentiment lags simultaneously (Gu & Kurov 2020)")

for col, desc in TREATMENTS:
    lag_cols = [f"{col}_lag{n}" for n in LAGS]

    nr_model   = run_plr_multi(long, "return_oo", lag_cols, GK_CONFOUNDERS)
    nr_summary = nr_model.summary
    nr_confint = nr_model.confint(level=0.95)

    print(f"\n  --- {desc} ---")
    verdicts = {}
    for lag_col in lag_cols:
        coef     = nr_summary.loc[lag_col, "coef"]
        pval     = nr_summary.loc[lag_col, "P>|t|"]
        ci_lower = nr_confint.loc[lag_col, "2.5 %"]
        ci_upper = nr_confint.loc[lag_col, "97.5 %"]
        verdicts[lag_col] = pval < 0.05
        print_row(lag_col, coef, pval, ci_lower, ci_upper)

    lag1_col  = f"{col}_lag1"
    later_sig = [c for c in lag_cols if c != lag1_col and verdicts.get(c, False)]
    print(f"\n  VERDICT ({desc}):")
    if verdicts.get(lag1_col) and not later_sig:
        print("  PASS — lag 1 significant, lags 2-7 insignificant.")
        print("         Consistent with Gu & Kurov (2020): permanent information effect.")
    elif not verdicts.get(lag1_col):
        print("  INCONCLUSIVE — lag 1 not significant.")
    else:
        print(f"  FAIL — significant persistence at: {', '.join(later_sig)}")
        print("         Suggests momentum or reverse causality. Interpret main ATE with caution.")


# ===========================================================================
# PART 2 — Current treatments → future return_oo  (impact persistence)
#
# Under Gu & Kurov, the Twitter signal should be absorbed within 1-2 trading
# days. Lead effects beyond day 3 would imply drift — inconsistent with the
# semi-strong efficient markets hypothesis.
# ===========================================================================

print_table_header("PART 2: Current treatment → return_oo_lead{n}  (impact persistence)")

for lead_n in LAGS:
    long[f"return_oo_lead{lead_n}"] = long.groupby("ticker")["return_oo"].shift(-lead_n)

part2_rows = []

for lead_n in LAGS:
    outcome = f"return_oo_lead{lead_n}"
    print(f"\n  --- Lead {lead_n} ---")
    for col, desc in TREATMENTS:
        coef, pval, ci_lower, ci_upper = run_plr(long, outcome, col, GK_CONFOUNDERS)
        print_row(f"lead{lead_n} | {desc}", coef, pval, ci_lower, ci_upper)
        part2_rows.append({
            "label": desc, "period": lead_n, "coef": coef,
            "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper,
        })

tex = to_latex(
    part2_rows,
    caption="Gu \\& Kurov (2020) --- Current Sentiment and Future Open-to-Open Return (Impact Persistence)",
    label="tab:gk_impact_persistence",
    period_col="lead",
)
(OUT_DIR / "gk_impact_persistence.tex").write_text(tex)
print("\nSaved: output/gk_impact_persistence.tex")

print("\n\nDone. All Gu & Kurov DoubleML robustness tables written to output/")
