# ml_analysis.R
# Doubly-robust ATE estimation via DoubleML Partially Linear Regression (PLR).
# Both treatments are continuous, so DoubleMLPLR is used in place of the
# binary AIPW package. Cross-fitted ranger (random forest) nuisance models
# provide the same doubly-robust guarantee as AIPW for continuous treatment.
#
# Run 1: Average daily Twitter sentiment (twitter_sent, uniform on [-1, 1])
# Run 2: Negative tweet volume (twitter_count * I(twitter_sent < 0))
#
# Confounders in both runs: all variables except px_open, px_close,
# identifiers (date, ticker), outcome (return), and the treatment variable.

library(readr)
library(dplyr)
library(DoubleML)
library(mlr3)
library(mlr3learners)

set.seed(42)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ROOT    = normalizePath(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "../.."))
IN_FILE = file.path(ROOT, "data", "processed", "panel_long.csv")

long = read_csv(IN_FILE, show_col_types = FALSE)
long = long %>% mutate(date = as.Date(date))


# ===========================================================================
# RUN 1 — Average daily Twitter sentiment as continuous treatment
#
# Treatment: twitter_sent (continuous, uniform [-1, 1])
# Outcome:   return (open-to-close price change)
# Confounders: px_high, px_low, mkt_cap, total_equity, debt_to_equity,
#              volume, twitter_count, news_sent, rsi_30, ma_50,
#              twitter_neg_count, lag1-lag7
# ===========================================================================

confounders_1 = c(
  "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
  "volume", "twitter_count", "news_sent", "rsi_30", "ma_50",
  "twitter_neg_count", "lag1", "lag2", "lag3", "lag5", "lag7"
)

df1 = long %>%
  select(return, twitter_sent, all_of(confounders_1)) %>%
  drop_na()

data_1 = DoubleMLData$new(
  data   = as.data.frame(df1),
  y_col  = "return",
  d_cols = "twitter_sent",
  x_cols = confounders_1
)

# ranger learners for outcome (ml_l) and treatment (ml_m) nuisance models
ml_l_1 = lrn("regr.ranger", num.trees = 500)
ml_m_1 = lrn("regr.ranger", num.trees = 500)

dml_1 = DoubleMLPLR$new(data_1, ml_l_1, ml_m_1, n_folds = 5)
dml_1$fit()

cat("\n===== RUN 1: Average Daily Twitter Sentiment (DoubleML PLR) =====\n")
cat("Treatment: twitter_sent | Outcome: return\n")
cat(sprintf("N (complete cases): %d\n", nrow(df1)))
print(dml_1$summary())


# ===========================================================================
# RUN 2 — Negative tweet count as standalone treatment (Teti et al. 2019)
#
# Treatment: twitter_neg_count (count of negative tweets per day)
# Per Teti et al.: polarity-broken counts are significant; total count is not.
# Using twitter_neg_count directly sidesteps neutral-tweet dilution in the
# Bloomberg index and avoids constructing an interaction variable.
# Outcome:   return
# Confounders: twitter_count and twitter_sent both stay in to control for
#   total tweet volume and overall sentiment direction independently.
# ===========================================================================

confounders_2 = c(
  "px_high", "px_low", "mkt_cap", "total_equity", "debt_to_equity",
  "volume", "twitter_sent", "twitter_count", "news_sent", "rsi_30", "ma_50",
  "lag1", "lag2", "lag3", "lag5", "lag7"
)

df2 = long %>%
  select(return, twitter_neg_count, all_of(confounders_2)) %>%
  drop_na()

data_2 = DoubleMLData$new(
  data   = as.data.frame(df2),
  y_col  = "return",
  d_cols = "twitter_neg_count",
  x_cols = confounders_2
)

ml_l_2 = lrn("regr.ranger", num.trees = 500)
ml_m_2 = lrn("regr.ranger", num.trees = 500)

dml_2 = DoubleMLPLR$new(data_2, ml_l_2, ml_m_2, n_folds = 5)
dml_2$fit()

cat("\n===== RUN 2: Negative Tweet Count (DoubleML PLR) =====\n")
cat("Treatment: twitter_neg_count  |  Outcome: return\n")
cat(sprintf("N (complete cases): %d\n", nrow(df2)))
print(dml_2$summary())
