# CLAUDE.MD -- Ashley Thompson Capstone Project

## Project
- Topic: Causal effect of social media sentiment on stock price movements
- Languages: Python (data wrangling, sentiment analysis), R (causal inference, fixest)
- Methods: OLS with fixed effects (panel data), AIPW planned for later
- Data: Bloomberg terminal CSV exports, daily stock-level, dependent variable is open-to-close price change

## Folder Structure
- data/raw/ -- original Bloomberg CSVs (never modify)
- data/processed/ -- cleaned and merged datasets
- notebooks/ -- Jupyter notebooks for exploration
- src/python/ -- Python scripts (data cleaning, sentiment analysis)
- src/r/ -- R scripts (01_clean.R, 02_merge.R, 03_analysis.R)
- output/ -- figures, tables, and results

## Commands
- python src/python/clean_data.py   # data cleaning
- Rscript src/r/03_analysis.R       # estimation

## Conventions
- snake_case variable names in both Python and R
- NA for missing values (never 999)
- Raw data is read-only; all transformations produce new files in data/processed/
- Commit code only; raw data stays local (.gitignore excludes CSVs)

## Future Plans
- Build custom sentiment analysis tool (beyond Twitter/Bloomberg sentiment)
- Implement AIPW for more robust causal estimates
