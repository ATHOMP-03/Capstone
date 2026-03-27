# clean_data.R
# R equivalent of src/python/clean_data.py
#
# Loads Bloomberg25MAR.xlsx, pivots to long format, removes NaNs,
# computes daily returns and lags, and saves to data/processed/panel_long.csv.
#
# Output columns (one row per ticker x trading day):
#   date, ticker, px_open, px_close, px_high, px_low, mkt_cap, total_equity,
#   debt_to_equity, volume, twitter_sent, twitter_count, news_sent,
#   rsi_30, ma_50, twitter_neg_count, return, lag1, lag2, lag3, lag5, lag7

library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     <- normalizePath(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "../.."))
RAW_FILE <- file.path(ROOT, "data", "raw", "Bloomberg25MAR.xlsx")
OUT_FILE <- file.path(ROOT, "data", "processed", "panel_long.csv")

dir.create(dirname(OUT_FILE), showWarnings = FALSE, recursive = TRUE)


# ===========================================================================
# BLOCK 1 — LOAD & PIVOT
# ===========================================================================

raw <- read_excel(RAW_FILE, sheet = 1, col_names = FALSE)

# Row 4 (1-indexed) = row 3 (0-indexed): ticker names, one per 14-column block
# Row 6 (1-indexed) = row 5 (0-indexed): Bloomberg field codes
h1 <- raw[4, ] %>% unlist(use.names = FALSE) %>% as.character()   # tickers
h3 <- raw[6, ] %>% unlist(use.names = FALSE) %>% as.character()   # field codes

h1 <- na_if(str_trim(h1), "NA")
h3 <- na_if(str_trim(h3), "NA")
h3 <- na_if(h3, "Dates")

# Fill ticker name across its 14 sub-columns
h1 <- tidyr::fill(tibble(x = h1), x, .direction = "down")$x

# Keep col 1 (dates) + all cols that have a real field code in row 6
keep_cols <- c(1, which(!is.na(h3) & seq_along(h3) != 1))

raw <- raw[, keep_cols]
h1  <- h1[keep_cols]
h3  <- h3[keep_cols]

# Data starts at row 7 (rows 1-6 are metadata/headers)
dat <- raw[-c(1, 2, 3, 4, 5, 6), ]

names(dat)[1] <- "date"
names(dat)    <- c("date", map2_chr(h1[-1], h3[-1], ~ paste0(.x, "__", .y)))
names(dat)    <- make.unique(names(dat), sep = "__dup__")

# Convert Excel serial date numbers to proper dates
dat <- dat %>%
  mutate(date = as.Date(as.numeric(date), origin = "1899-12-30"))

# Pivot wide → long, then spread fields back to columns
long <- dat %>%
  pivot_longer(
    cols      = -date,
    names_to  = c("ticker", "field"),
    names_sep = "__",
    values_to = "value"
  ) %>%
  pivot_wider(names_from = field, values_from = value)

# Rename Bloomberg field codes to snake_case
long <- long %>%
  rename(
    px_open         = PX_OPEN,
    px_close        = PX_OFFICIAL_CLOSE,
    px_high         = PX_HIGH,
    px_low          = PX_LOW,
    mkt_cap         = CUR_MKT_CAP,
    total_equity    = TOTAL_EQUITY,
    debt_to_equity  = TOT_DEBT_TO_TOT_EQY,
    volume          = PX_VOLUME,
    twitter_sent    = TWITTER_SENTIMENT_DAILY_AVG,
    twitter_count   = TWITTER_PUBLICATION_COUNT,
    news_sent       = NEWS_SENTIMENT_DAILY_AVG,
    rsi_30          = RSI_30D,
    ma_50           = MOV_AVG_50D,
    twitter_neg_count = TWITTER_NEG_SENTIMENT_COUNT
  ) %>%
  arrange(ticker, date)


# ===========================================================================
# BLOCK 2 — REMOVE NaNs
# ===========================================================================

BLOOMBERG_NA <- c("#N/A N/A", "#N/A", "#N/A Field Not Applicable")

numeric_cols <- c(
  "px_open", "px_close", "px_high", "px_low", "mkt_cap", "total_equity",
  "debt_to_equity", "volume", "twitter_sent", "twitter_count", "news_sent",
  "rsi_30", "ma_50", "twitter_neg_count"
)

long <- long %>%
  mutate(across(all_of(numeric_cols), ~ na_if(as.character(.x), BLOOMBERG_NA[1]))) %>%
  mutate(across(all_of(numeric_cols), ~ na_if(.x, BLOOMBERG_NA[2]))) %>%
  mutate(across(all_of(numeric_cols), ~ na_if(.x, BLOOMBERG_NA[3]))) %>%
  mutate(across(all_of(numeric_cols), ~ as.numeric(.x)))

# Drop non-trading days (weekends/holidays are all-NA for prices)
long <- long %>%
  filter(!is.na(px_open), !is.na(px_close))


# ===========================================================================
# BLOCK 3 — DAILY RETURN + LAGS
# ===========================================================================

long <- long %>%
  mutate(return = px_close - px_open)

long <- long %>%
  arrange(ticker, date) %>%
  group_by(ticker) %>%
  mutate(
    lag1 = lag(return, 1),
    lag2 = lag(return, 2),
    lag3 = lag(return, 3),
    lag5 = lag(return, 5),
    lag7 = lag(return, 7)
  ) %>%
  ungroup()


# ===========================================================================
# SAVE
# ===========================================================================

write.csv(long, OUT_FILE, row.names = FALSE)
cat(sprintf("Saved %d rows x %d cols -> %s\n", nrow(long), ncol(long), OUT_FILE))
print(str(long))
print(head(long, 10))


