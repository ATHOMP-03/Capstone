"""
lag_cates.py
Two complementary analyses for causal identification and impact persistence.

PART 1 — Lagged sentiment -> current return  (reverse causality check)
  For each lag n: shift twitter_sent/news_sent back n days, regress on
  today's return. Tests whether old tweets still move today's price.
  Used for the no-reversal test (Gu & Kurov 2020): only lag 1 should
  be significant. Persistence at lags 3-7 signals reverse causality.

PART 2 — Current sentiment -> future return  (research question)
  For each horizon n: shift return forward n days, regress today's
  twitter_sent/news_sent on that future return. Tests how long the
  price impact of today's tweets lasts. Under efficient markets the
  effect decays quickly; slow decay is a meaningful finding.
  Hypothesis: tweet effects decay faster than news effects.
"""

import numpy as np
import pandas as pd
import doubleml as dml
from xgboost import XGBRegressor
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
long = pd.read_csv(ROOT / "data" / "processed" / "panel_long.csv", parse_dates=["date"])
long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

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

# Confounders: excludes current twitter_sent and news_sent (both are treatments
# at different lags — including current values would introduce endogeneity)
BASE_CONFOUNDERS = [
    "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
    "volume", "twitter_count", "rsi_30", "ma_50",
    "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7",
]

# ===========================================================================
# PART 1 — Lagged sentiment -> current return  (reverse causality check)
# ===========================================================================

print("\n" + "="*70)
print("  PART 1: Lagged sentiment -> current return (reverse causality check)")
print("="*70)

results = []

for lag_n in LAGS:
    # Create lagged treatment variables within each ticker
    long[f"twitter_sent_lag{lag_n}"] = (
        long.groupby("ticker")["twitter_sent"].shift(lag_n)
    )
    long[f"news_sent_lag{lag_n}"] = (
        long.groupby("ticker")["news_sent"].shift(lag_n)
    )

    for treatment, label in [
        (f"twitter_sent_lag{lag_n}", "twitter_sent"),
        (f"news_sent_lag{lag_n}",   "news_sent"),
    ]:
        df = (
            long[["return", treatment] + BASE_CONFOUNDERS]
            .dropna()
            .reset_index(drop=True)
        )

        model = dml.DoubleMLPLR(
            dml.DoubleMLData(df, y_col="return", d_cols=treatment, x_cols=BASE_CONFOUNDERS),
            ml_l    = make_xgb(),
            ml_m    = make_xgb(),
            n_folds = 5,
            n_rep   = 20,
        )
        model.fit()
        model.bootstrap(method="normal", n_rep_boot=1000)

        summary  = model.summary
        confint  = model.confint(level=0.95)
        coef     = summary["coef"].iloc[0]
        pval     = summary["P>|t|"].iloc[0]
        ci_lower = confint["2.5 %"].iloc[0]
        ci_upper = confint["97.5 %"].iloc[0]

        print(f"\n  lag={lag_n} | treatment={label}")
        print(f"  coef={coef:.6f}  p={pval:.4f}  95% CI=[{ci_lower:.6f}, {ci_upper:.6f}]")

        results.append({
            "treatment": label,
            "lag":       lag_n,
            "coef":      coef,
            "p_value":   pval,
            "ci_lower":  ci_lower,
            "ci_upper":  ci_upper,
        })

# ===========================================================================
# DECAY TABLE — side-by-side comparison of twitter vs news effect by lag
# (univariate: each lag estimated in its own model)
# ===========================================================================

results_df = pd.DataFrame(results)

print("\n\n===== LAG DECAY TABLE (univariate) =====")
print(f"{'Lag':<6} {'Treatment':<15} {'Coef':>10} {'P-value':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
print("-" * 72)
for _, row in results_df.sort_values(["lag", "treatment"]).iterrows():
    sig = "*" if row["p_value"] < 0.05 else ""
    print(
        f"{int(row['lag']):<6} {row['treatment']:<15} "
        f"{row['coef']:>10.6f} {row['p_value']:>10.4f} "
        f"{row['ci_lower']:>14.6f} {row['ci_upper']:>14.6f}  {sig}"
    )
print("  * p < 0.05")


# ===========================================================================
# NO-REVERSAL TEST — Gu & Kurov (2020) simultaneous specification
#
# Runs all sentiment lags jointly in a single DoubleML model so each lag's
# coefficient is estimated conditional on the others. This is the key
# difference from the univariate loop above: correlated lags can't
# individually absorb each other's variation.
#
# Pass criteria:
#   - lag 1 significant (p < 0.05) — sentiment has an immediate effect
#   - lags 2-7 insignificant — effect is permanent/absorbed, not a feedback loop
#
# Fail signal: lags 3-7 remain significant → suggests momentum or reverse
# causality (price moves driving sentiment rather than the reverse).
# ===========================================================================

print("\n\n===== NO-REVERSAL TEST (simultaneous, Gu & Kurov 2020) =====")

for channel, label in [
    ("twitter_sent", "Twitter sentiment"),
    ("news_sent",    "News sentiment"),
]:
    lag_cols = [f"{channel}_lag{n}" for n in LAGS]

    df_nr = (
        long[["return"] + lag_cols + BASE_CONFOUNDERS]
        .dropna()
        .reset_index(drop=True)
    )

    nr_model = dml.DoubleMLPLR(
        dml.DoubleMLData(df_nr, y_col="return", d_cols=lag_cols, x_cols=BASE_CONFOUNDERS),
        ml_l    = make_xgb(),
        ml_m    = make_xgb(),
        n_folds = 5,
        n_rep   = 20,
    )
    nr_model.fit()
    nr_model.bootstrap(method="normal", n_rep_boot=1000)

    nr_summary = nr_model.summary
    nr_confint = nr_model.confint(level=0.95)

    print(f"\n  --- {label} ---")
    print(f"  {'Lag':<18} {'Coef':>10} {'P-value':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}  Sig")
    print(f"  {'-'*70}")

    verdicts = {}
    for col in lag_cols:
        coef     = nr_summary.loc[col, "coef"]
        pval     = nr_summary.loc[col, "P>|t|"]
        ci_lower = nr_confint.loc[col, "2.5 %"]
        ci_upper = nr_confint.loc[col, "97.5 %"]
        sig      = "*" if pval < 0.05 else ""
        verdicts[col] = pval < 0.05
        print(
            f"  {col:<18} {coef:>10.6f} {pval:>10.4f} "
            f"{ci_lower:>14.6f} {ci_upper:>14.6f}  {sig}"
        )

    # Formal verdict
    lag1_col  = f"{channel}_lag1"
    later_sig = [c for c in lag_cols if c != lag1_col and verdicts[c]]
    lag1_sig  = verdicts[lag1_col]

    print(f"\n  VERDICT ({label}):")
    if lag1_sig and not later_sig:
        print(f"  PASS — lag 1 significant, lags 2-7 insignificant.")
        print(f"         Consistent with permanent information effect, not feedback.")
    elif not lag1_sig:
        print(f"  INCONCLUSIVE — lag 1 not significant.")
    else:
        print(f"  FAIL — significant persistence at: {', '.join(later_sig)}")
        print(f"         Suggests momentum or reverse causality. Interpret main results with caution.")


# ===========================================================================
# PART 2 — Current sentiment -> future return  (research question)
#
# Creates lead variables of return (return at t+n) and regresses today's
# sentiment on each. Shows how long the price impact of today's tweets lasts.
# Confounders stay the same — they're all measured today alongside sentiment.
#
# Hypothesis: twitter_sent effect decays faster than news_sent effect,
# consistent with shorter tweet cycles vs. longer news cycles.
# ===========================================================================

print("\n\n" + "="*70)
print("  PART 2: Current sentiment -> future return (impact persistence)")
print("="*70)

# Create lead return variables within each ticker
for lead_n in LAGS:
    long[f"return_lead{lead_n}"] = (
        long.groupby("ticker")["return"].shift(-lead_n)
    )

lead_results = []

for lead_n in LAGS:
    outcome = f"return_lead{lead_n}"

    for treatment, label in [
        ("twitter_sent", "twitter_sent"),
        ("news_sent",    "news_sent"),
    ]:
        df = (
            long[[outcome, treatment] + BASE_CONFOUNDERS]
            .dropna()
            .reset_index(drop=True)
        )

        model = dml.DoubleMLPLR(
            dml.DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=BASE_CONFOUNDERS),
            ml_l    = make_xgb(),
            ml_m    = make_xgb(),
            n_folds = 5,
            n_rep   = 20,
        )
        model.fit()
        model.bootstrap(method="normal", n_rep_boot=1000)

        summary  = model.summary
        confint  = model.confint(level=0.95)
        coef     = summary["coef"].iloc[0]
        pval     = summary["P>|t|"].iloc[0]
        ci_lower = confint["2.5 %"].iloc[0]
        ci_upper = confint["97.5 %"].iloc[0]

        print(f"\n  lead={lead_n} | treatment={label}")
        print(f"  coef={coef:.6f}  p={pval:.4f}  95% CI=[{ci_lower:.6f}, {ci_upper:.6f}]")

        lead_results.append({
            "treatment": label,
            "lead":      lead_n,
            "coef":      coef,
            "p_value":   pval,
            "ci_lower":  ci_lower,
            "ci_upper":  ci_upper,
        })

# Impact persistence table
lead_df = pd.DataFrame(lead_results)

print("\n\n===== IMPACT PERSISTENCE TABLE =====")
print("  How long does today's sentiment affect future returns?")
print(f"{'Lead':<6} {'Treatment':<15} {'Coef':>10} {'P-value':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
print("-" * 72)
for _, row in lead_df.sort_values(["lead", "treatment"]).iterrows():
    sig = "*" if row["p_value"] < 0.05 else ""
    print(
        f"{int(row['lead']):<6} {row['treatment']:<15} "
        f"{row['coef']:>10.6f} {row['p_value']:>10.4f} "
        f"{row['ci_lower']:>14.6f} {row['ci_upper']:>14.6f}  {sig}"
    )
print("  * p < 0.05")
print("\n  Interpretation: if twitter_sent becomes insignificant by lead 2-3")
print("  but news_sent remains significant to lead 5-7, tweet cycles are")
print("  shorter than news cycles, consistent with the core hypothesis.")
