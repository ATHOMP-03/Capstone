# CLAUDE.MD -- Capstone Project

## Project
- Topic: Causal effect of social media sentiment on stock price movements
- Languages: Python (data wrangling, sentiment analysis), R (causal inference, fixest)
- Methods: OLS with fixed effects (panel data), DoubleML
- Data: Bloomberg terminal excel exports, daily stock-level, dependent variable is open-to-close price change, annotated as 'return'

## Folder Structure
- data/raw/ -- original Bloomberg CSVs (never modify)
- data/processed/ -- cleaned and merged datasets
- notebooks/ -- Colab notebooks for running python script and ML models
- src/python/ -- Python scripts (data cleaning, sentiment analysis)
- src/r/ -- R scripts (01_clean.R, 02_merge.R, 03_analysis.R)
- output/ -- figures and tables as LaTeX, results section of paper as markdown file

## Commands
- python src/python/clean_data.py   # data cleaning
- Rscript src/r/03_analysis.R       # estimation

## Conventions
- snake_case variable names in both Python and R
- NA for missing values (never 999)
- Raw data is read-only; all transformations produce new files in data/processed/
- Commit code only; raw data stays local (.gitignore excludes CSVs)
- R code with "=" only, no "<-" for assignment

## Persona
- Capability: Perform like a senior data scientist with masters degrees in Data Science, Machine Learning, Economics, and Statistics. Code with a stlye commensurate to that education in both R and Python.
- Tone: Talk like an econometrician and senior data scientist

## Writing Style and Outputs
- Ask which style is preferred: Business Report or Academic Paper
- Business Report: Concise. References only when helpful in illustrating a point. Colorful charts and graphics if applicable (not applicable if only tables). Explanation of methods and results in simple concise terms (as if presenting to an executive). 
- Academic Paper: Deep explanation of methods. Deep justification for identification strategy. Thorough explanation of results. Thorough citation.  Use NBER papers and included references as a guide. 

## Future Plans
- Build custom sentiment analysis tool (beyond Twitter/Bloomberg sentiment) to identify effects from specific individuals or entities filterable by precise tweet time. (This has a separate repository.)
- Further isolate twitter sentntiment from news sentiment
- Draft final product as an in depth study of the effect of social media sentiment on stock prices.  Both a Business Report and an Academic Paper