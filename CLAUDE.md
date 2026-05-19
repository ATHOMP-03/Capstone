# CLAUDE.MD -- Capstone Project

## Project
- Topic: Causal effect of social media sentiment on stock price movements
- Language: Python (all analysis; run as Colab notebooks in notebooks/)
- Methods: FE-OLS (linearmodels/pyfixest), DoubleML PLR (XGBoost nuisance), Fama-MacBeth
- Data: Bloomberg terminal Excel exports, daily stock-level; dependent variable is open-to-close price change (`return`)

## Folder Structure
- data/raw/           -- original Bloomberg CSVs/XLSX (never modify; gitignored)
- data/processed/     -- cleaned panel CSVs produced by src/python/clean_data*.py
- notebooks/          -- Colab notebooks (canonical execution; one per analysis script)
- src/python/         -- Python scripts (data cleaning, FE-OLS, DoubleML, robustness)
- output/             -- LaTeX tables, figures, draft sections, Overleaf package
- output/overleaf_upload/ -- self-contained Overleaf package (master: capstone_final_v2.tex)
- references/         -- PDFs and notes (gitignored)

## Commands
- python src/python/clean_data.py          # primary S&P 500 panel cleaning
- python src/python/clean_data_russel.py   # Russell 3000 panel cleaning
- python src/python/ml_analysis.py         # DoubleML primary spec
- python src/python/analysis.py            # FE-OLS specs + placebo
- # All other analyses: see notebooks/ for Colab equivalents

## Conventions
- snake_case variable names in Python
- NA for missing values (never 999)
- Raw data is read-only; all transformations produce new files in data/processed/
- Commit code only; raw data stays local (.gitignore excludes CSVs/XLSX)

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