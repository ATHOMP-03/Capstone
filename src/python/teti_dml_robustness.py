"""
teti_dml_robustness.py
DoubleML PLR robustness checks for the Teti et al. (2019) replication.

Structure:
  MAIN ATE   — current twitter_sent / twitter_neg_count / ts_b → pct_return
  PART 1     — lagged treatments → current pct_return  (reverse causality, univariate)
  NO-REVERSAL — all twitter_sent lags jointly in one PLR model
  PART 2     — current treatments → pct_return_lead{n}  (impact persistence)

DoubleML PLR (Robinson 1988): partials out nonlinear nuisance with XGBoost so
the treatment coefficient is interpreted causally under the conditional unconfoundedness
assumption, i.e., E[Y(d) | X] is consistently estimated.

Treatment channels (Teti):
  twitter_sent      — Bloomberg composite Twitter sentiment score
  twitter_neg_count — raw count of negative tweets (Teti SET 2/3 key predictor)
  ts_b              — polarity-weighted index (pos-neg)/(pos+neg+neu); falls
                      back to sign(s)*s^2 if pos/neu counts unavailable

Outcome: pct_return = (px_close - px_open) / px_open * 100

Usage:
    python src/python/teti_dml_robustness.py
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

HAS_POS_NEU = {"twitter_pos_count", "twitter_neu_count"}.issubset(long.columns)
if not HAS_POS_NEU:
    print("[INFO] twitter_pos_count / twitter_neu_count unavailable — ts_b uses sign(s)*s^2 fallback.\n")


# ===========================================================================
# VARIABLE CONSTRUCTION
# ===========================================================================

long["pct_return"] = (long["px_close"] - long["px_open"]) / long["px_open"] * 100

if HAS_POS_NEU:
    denom = (
        long["twitter_pos_count"] + long["twitter_neu_count"] + long["twitter_neg_count"]
    )
    long["ts_b"] = (
        (long["twitter_pos_count"] - long["twitter_neg_count"]) / denom.replace(0, np.nan)
    )
else:
    long["ts_b"] = long["twitter_sent"].apply(
        lambda x: np.sign(x) * (x ** 2) if pd.notna(x) else np.nan
    )


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

# Excludes all three treatment channels; news_sent is a structural control
# (correlated with Twitter activity but not the treatment of interest here)
TETI_CONFOUNDERS = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_count", "rsi_30", "ma_50", "news_sent",
]

TREATMENTS = [
    ("twitter_sent",      "Twitter sentiment (composite)"),
    ("twitter_neg_count", "Negative tweet count"),
    ("ts_b",              "Polarity-weighted index (ts_b)"),
]


# ===========================================================================
# HELPERS
# ===========================================================================

def run_plr(df, y_col, d_col, x_cols):
    """Fit DoubleML PLR; return (coef, pval, ci_lower, ci_upper)."""
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
    """Fit DoubleML PLR with multiple simultaneous treatment columns."""
    keep = [y_col] + d_cols + x_cols
    data = dml.DoubleMLData(
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
    """Minimal publication-grade LaTeX tabular."""
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
    lines += [
        r"\hline",
        r"\multicolumn{5}{l}{\footnotesize DoubleML PLR, XGBoost (GPU), 5-fold, 20 reps, 1000 bootstrap draws.} \\",
        r"\multicolumn{5}{l}{\footnotesize $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.} \\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# MAIN ATE — current treatment → current pct_return
# ===========================================================================

print_table_header("MAIN ATE: Treatment → pct_return  (Teti et al. 2019)")
main_rows = []

for col, desc in TREATMENTS:
    if col not in long.columns:
        print(f"  [SKIP] {desc} — column not found")
        continue
    coef, pval, ci_lower, ci_upper = run_plr(long, "pct_return", col, TETI_CONFOUNDERS)
    print_row(desc, coef, pval, ci_lower, ci_upper)
    main_rows.append({"label": desc, "coef": coef, "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper})

tex = to_latex(
    main_rows,
    caption="Teti et al. (2019) Replication --- DoubleML PLR Main ATE Estimates",
    label="tab:teti_main_ate",
)
(OUT_DIR / "teti_main_ate.tex").write_text(tex)
print("\nSaved: output/teti_main_ate.tex")


# ===========================================================================
# PART 1 — Lagged treatments → current pct_return  (reverse causality check)
# ===========================================================================

print_table_header("PART 1: Lagged treatment → pct_return  (reverse causality, univariate)")

for col, desc in TREATMENTS:
    if col not in long.columns:
        continue
    for lag_n in LAGS:
        lag_col = f"{col}_lag{lag_n}"
        long[lag_col] = long.groupby("ticker")[col].shift(lag_n)

part1_rows = []

for lag_n in LAGS:
    print(f"\n  --- Lag {lag_n} ---")
    for col, desc in TREATMENTS:
        if col not in long.columns:
            continue
        lag_col = f"{col}_lag{lag_n}"
        coef, pval, ci_lower, ci_upper = run_plr(long, "pct_return", lag_col, TETI_CONFOUNDERS)
        print_row(f"lag{lag_n} | {desc[:25]}", coef, pval, ci_lower, ci_upper)
        part1_rows.append({
            "label": desc, "period": lag_n, "coef": coef,
            "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper,
        })

tex = to_latex(
    part1_rows,
    caption="Teti et al. --- Lagged Sentiment and Current Return (Reverse Causality Check)",
    label="tab:teti_lag_decay",
    period_col="lag",
)
(OUT_DIR / "teti_lag_decay.tex").write_text(tex)
print("\nSaved: output/teti_lag_decay.tex")


# ===========================================================================
# NO-REVERSAL TEST — all twitter_sent lags jointly (Gu & Kurov 2020 procedure)
#
# Each lag's coefficient is estimated conditional on the others.
# Pass: lag 1 significant; lags 2-7 insignificant → permanent information effect.
# Fail: lags 3-7 persist → momentum or reverse causality.
# ===========================================================================

print_table_header("NO-REVERSAL TEST: all twitter_sent lags simultaneously")

for col, desc in [("twitter_sent", "Twitter sentiment"), ("ts_b", "Polarity-weighted ts_b")]:
    if col not in long.columns:
        continue
    lag_cols = [f"{col}_lag{n}" for n in LAGS if f"{col}_lag{n}" in long.columns]

    nr_model = run_plr_multi(long, "pct_return", lag_cols, TETI_CONFOUNDERS)
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
        print("         Consistent with permanent information effect (no feedback loop).")
    elif not verdicts.get(lag1_col):
        print("  INCONCLUSIVE — lag 1 not significant.")
    else:
        print(f"  FAIL — significant persistence at: {', '.join(later_sig)}")
        print("         Suggests momentum or reverse causality. Interpret main ATE with caution.")


# ===========================================================================
# PART 2 — Current treatments → future pct_return  (impact persistence)
#
# Hypothesis: Twitter sentiment effect decays by lead 2-3, consistent with
# rapid tweet cycle absorption under semi-strong efficiency.
# ===========================================================================

print_table_header("PART 2: Current treatment → pct_return_lead{n}  (impact persistence)")

for lead_n in LAGS:
    long[f"pct_return_lead{lead_n}"] = long.groupby("ticker")["pct_return"].shift(-lead_n)

part2_rows = []

for lead_n in LAGS:
    outcome = f"pct_return_lead{lead_n}"
    print(f"\n  --- Lead {lead_n} ---")
    for col, desc in TREATMENTS:
        if col not in long.columns:
            continue
        coef, pval, ci_lower, ci_upper = run_plr(long, outcome, col, TETI_CONFOUNDERS)
        print_row(f"lead{lead_n} | {desc[:25]}", coef, pval, ci_lower, ci_upper)
        part2_rows.append({
            "label": desc, "period": lead_n, "coef": coef,
            "pval": pval, "ci_lower": ci_lower, "ci_upper": ci_upper,
        })

tex = to_latex(
    part2_rows,
    caption="Teti et al. --- Current Sentiment and Future Return (Impact Persistence)",
    label="tab:teti_impact_persistence",
    period_col="lead",
)
(OUT_DIR / "teti_impact_persistence.tex").write_text(tex)
print("\nSaved: output/teti_impact_persistence.tex")

print("\n\nDone. All Teti DoubleML robustness tables written to output/")
