# analysis.R
# Fixed effects regression analysis of Twitter sentiment on stock returns.
#
# Models:
#   1. Simple FE regression (plm)
#   2. FE regression with full confounder matrix (feols)
#   3. FE regression with continuous negative sentiment treatment (feols)
#   4. FE regression with twitter_neg_count as standalone treatment (Teti et al.)
#
# Robustness checks:
#   - Placebo test: does twitter_sent predict yesterday's return? (it shouldn't)
#   - No-reversal test: lagged sentiment -> return (Gu & Kurov 2020)

library(readr)
library(dplyr)
library(plm)
library(fixest)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ROOT    = normalizePath(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "../.."))
IN_FILE = file.path(ROOT, "data", "processed", "panel_long.csv")

long = read_csv(IN_FILE, show_col_types = FALSE)
long = long %>% mutate(date = as.Date(date))


# ===========================================================================
# MODEL 1 — Simple fixed effects regression (plm)
# Baseline within-firm estimate of Twitter sentiment on same-day return.
# Company FE absorbs time-invariant firm characteristics.
# ===========================================================================

model_1 = plm(
  return ~ twitter_sent,
  data  = long,
  index = c("ticker"),
  model = "within"
)

cat("\n===== MODEL 1: Simple FE (plm) =====\n")
print(summary(model_1))

# feols equivalent of model_1 — used only for the combined etable below
model_1_feols = feols(return ~ twitter_sent | ticker, data = long, vcov = "hetero")


# ===========================================================================
# MODEL 2 — FE regression with confounder matrix (feols)
# Controls for firm-level fundamentals and technical indicators to reduce
# omitted variable bias. Excludes px_open and px_close (collinear with
# return by construction) and excludes twitter_count and lags per design.
# ===========================================================================

confounders = c(
  "px_high", "px_low",
  "mkt_cap", "total_equity", "debt_to_equity", "volume",
  "news_sent", "rsi_30", "ma_50", "twitter_neg_count"
)

f2 = as.formula(
  paste("return ~ twitter_sent +", paste(confounders, collapse = " + "), "| ticker")
)

model_2 = feols(f2, data = long, vcov = "hetero")

cat("\n===== MODEL 2: FE with confounders (feols) =====\n")
print(summary(model_2))


# ===========================================================================
# MODEL 3 — Continuous negative sentiment treatment (feols)
# neg_twitter_sent = pmin(twitter_sent, 0): equals the raw sentiment score
# on negative days, 0 on neutral/positive days. The coefficient captures
# how each additional unit of negativity shifts returns, holding firm FE.
# ===========================================================================

long = long %>%
  mutate(neg_twitter_sent = pmin(twitter_sent, 0, na.rm = TRUE))

model_3 = feols(
  return ~ neg_twitter_sent | ticker,
  data  = long,
  vcov  = "hetero"
)

cat("\n===== MODEL 3: Continuous negative sentiment treatment (feols) =====\n")
print(summary(model_3))


# ===========================================================================
# MODEL 4 — twitter_neg_count as standalone treatment (feols)
# Per Teti et al. (2019): polarity-broken tweet counts (positive vs negative)
# are significant predictors; total count is not. Using twitter_neg_count
# directly sidesteps the neutral-tweet dilution problem in the Bloomberg index.
# ===========================================================================

model_4 = feols(
  return ~ twitter_neg_count | ticker,
  data  = long,
  vcov  = "hetero"
)

cat("\n===== MODEL 4: Negative Tweet Count as Treatment (feols) =====\n")
print(summary(model_4))


# ===========================================================================
# PLACEBO TEST — Pre-period reverse causality check
# Regress yesterday's return (lag1) on today's twitter_sent.
# A causal story requires sentiment predicts forward, not backward.
# Significant coefficient here signals reverse causality in the data.
# ===========================================================================

placebo = feols(lag1 ~ twitter_sent | ticker, data = long, vcov = "hetero")

cat("\n===== PLACEBO: Does twitter_sent predict yesterday's return? =====\n")
cat("(Coefficient should be insignificant — if not, reverse causality is present)\n")
print(summary(placebo))


# ===========================================================================
# NO-REVERSAL TEST — Gu & Kurov (2020) falsification
# Regress return on multiple lags of twitter_sent simultaneously.
# A causal information effect appears only at lag 1 and dissipates.
# Persistent significance at lags 3-7 would indicate momentum/feedback.
# ===========================================================================

long = long %>%
  arrange(ticker, date) %>%
  group_by(ticker) %>%
  mutate(
    sent_lag1 = lag(twitter_sent, 1),
    sent_lag2 = lag(twitter_sent, 2),
    sent_lag3 = lag(twitter_sent, 3),
    sent_lag5 = lag(twitter_sent, 5),
    sent_lag7 = lag(twitter_sent, 7)
  ) %>%
  ungroup()

no_reversal = feols(
  return ~ sent_lag1 + sent_lag2 + sent_lag3 + sent_lag5 + sent_lag7 | ticker,
  data  = long,
  vcov  = "hetero"
)

cat("\n===== NO-REVERSAL TEST: Lagged sentiment -> return =====\n")
cat("(Only sent_lag1 should be significant; lags 2-7 should be near zero)\n")
print(summary(no_reversal))


# ===========================================================================
# COMBINED TABLE — document-ready output (copy/paste into paper)
# All three models side by side. model_1_feols is the feols re-fit of
# model_1 (identical estimates) so all three can sit in one etable.
# ===========================================================================

cat("\n===== COMBINED TABLE (copy-ready) =====\n")
etable(
  model_1_feols, model_2, model_3, model_4,
  title      = "Effect of Twitter Sentiment on Stock Returns",
  headers    = c("(1) Simple FE", "(2) Confounders", "(3) Neg. Sentiment", "(4) Neg. Tweet Count"),
  depvar     = FALSE,
  se.below   = TRUE,
  digits     = 4,
  fitstat    = c("n", "r2", "wr2"),
  notes      = "Heteroskedasticity-robust SE in parentheses. Company fixed effects in all models. Model 3 treatment is pmin(twitter_sent, 0). Model 4 treatment is twitter_neg_count per Teti et al. (2019)."
)

cat("\n===== ROBUSTNESS TABLE (copy-ready) =====\n")
etable(
  placebo, no_reversal,
  title   = "Robustness Checks",
  headers = c("Placebo", "No-Reversal"),
  depvar  = FALSE,
  se.below = TRUE,
  digits  = 4,
  fitstat = c("n", "r2", "wr2"),
  notes   = "Placebo: lag1 ~ twitter_sent | ticker. No-reversal: return ~ sent_lag1 + ... + sent_lag7 | ticker. Only sent_lag1 should be significant under a causal interpretation."
)
