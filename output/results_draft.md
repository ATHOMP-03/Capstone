# Results

## Analytical Approach

This study seeks to estimate the causal effect of social media sentiment on stock prices using two complimentary identification strategies.  

The first employs a firm-level fixed effects regresssion using `pyfixest`. This allows for the control of time-invariate factors within each firm, to account shocks to individual firms which may skew results. This strategy assumes that confounders influence the outcome in some linear manner, making it the simpler of the two models.

The second strategy employs a doubly debiaed machine learning estimator - DoubleML Partial Linear Regression (`DoubleMLPLR`) combined with `XGBoost`, a nuisance learner.  This straegy allows XGBoost to identify connections between the treatment and dependent variables across the matrix of confounders to generate predictive models, then using DoubleMLPLR to identify a coefficient, producing a treatment effect. For this initial set of estimations, I opted for 5-fold cross fitting done over 20 repetitions, with 1000 bootstrap draws to arrive at an estimator. 

With both methods, data consisted of daily panel data pulled from Bloomberg.  Sentiment ratings were provided using Bloomberg's proprietary tool as well.

---

## Fixed Effects Results\

Table 1 presents the four FE-OLS specifications. None of the models produced statistically significant results for Twitter sentiment's impact on stock price. This indicates that interactions between tweets and traders do not influence behavior more than other factors.  However, each model provides interesting insights into the relationship between social media and share price.

Model 1 is the baseline regression of return on `twitter_sent` witout confounders. The estimated coefficient is –0.370, indicating that a one-unit increase in Twitter sentiment is associated with a 0.37 cent decrease in the same-day return. While the sign is consistent with expected findings, the result is not statistically significant.  Also, the R-squared (0.002) tells us that sentiment explains little of what's driving share prices.

Model 2 adds the full confounder matrix: intraday price range (px_high, px_low), market capitalization, total equity, debt-to-equity ratio, trading volume, news sentiment, RSI (30-day), and the 50-day moving average. The `twitter_sent` coefficient's significance improved slightly, however, it is still insignificant. However, `news_sent` is substantial and significant. RSI was significant as well.  This tells us that news sentiment and RSI are stronger predictors of returns than is Twitter sentiment, at least in an OLS framework.

Models 3 and 4 were intended to check other approaches to see if they yielded better retulst. Model 3 uses a negative sentiment measure `neg_twitter_sent`, which is zero on positive-sentiment days and equals the raw score on negative-sentiment days. The coefficient was –0.072, smaller in magnitude than Model 1 and not statistically significant. This suggests that negative sentiment does not have particularly more effect than the entire spectrum of sentiment scores. 

Model 4 borrowed from a previous study and used tweet counts (`twitter_neg_count`) as the treatment. The coefficient is 0.000533, indicating that to some extent any press is good press, but not in a statistically significant way.


---

## DoubleML Results - Sentiment

Table 2 presents the DoubleMLPLR estimates, amd the contrast with the FE results is stark. 

Run 1, using `twitter_sent` as the continuous treatment, yielded a coefficient of –0.924 with strong significance. This indicates that changes in twitter sentiment negatively influence share prices, regardless of the direction of sentiment. So maybe all twitter coverage is bad press. However, in Run 2, which uses `twitter_neg_count` as the treatment to determine if tweet volume played a role, the DoubleML estimate is +0.003987. This positive, significant coefficient indicates that a higher count of negative tweets is associated with higher same-day returns.  This was unexpected, and means that potentially tweet volume and tweet sentiment are not necessarily connected.  My interpretation is that when a company is focused upon, the total number of tweets for the day increases, also increasing the amount of negative comments, but not impacting sentiment across the day. 


Table 3 shows results from applying the DoubleML framework to news sentiment. Run 1 shows that when using `news_sent` as treatment, we get a confident estimator of –0.759.  This result is consistent with the run of FE Model 2 (which showed news sentiment as the strongest predictor of retun), but adds additional credibility.  My second run for news sentiment was incorrectly set up, and is being removed as it is essentially the same as the run negative tweet counts.  This is from a limitation in Bloomberg's data, wherein they cannot collect a count of news mentions across a day. 

---

## DoubleML Results - Intraday Price High

Table 4 shows results for the eamination of `px_high` (highest sale price of the day) as the outcome, while leaving the confounder set the same as in other runs. Here Run 1, `twitter_sent ~ px_high` yielded +1.610. This positive coefficient means positive Twitter sentiment raises the intraday price ceiling - a result in direct contrast with the negative `twitter_sent ~ return`, which was negative. Then in Run 2, `twitter_neg_count ~ px_high` produced a significant estimator of –0.002082.  Again, this is the opposite direction of `twitter_neg_count ~ px_high` from the previous runs.  This indicates that either px_high is being influenced differently than return, or that factors impacting highs resolve themeselves over the course of the trading day.

---

## Robustness Checks

**Placebo test.** To test whether today's Twitter sentiment predicts yesterday's return 
I regress `lag1` (prior-day return) on `twitter_sent` within firms. If there were no reverse causality, the coefficient should be statistically indistinguishable from zero, however estimated coefficient is +2.348, making it both large and highly significant. This is a serious concern bbecause it indicates that Twitter sentiment is more likely caused by returns, than the otehr way around.

**No-reversal test.** I estimate all sentiment lags jointly (lags 1, 2, 3, 5, and 7) to test whether only lag 1 is significant. This would be expected if sentiment reflects news that is fully absorbed on day 1. None of the lag coefficients are statistically significant, and the signs alternate. This result is inconclusive, and the lack of significance at lag 1 suggests that any causal sentiment signal is likely too weak to survive multicollinear estimation against lags 2–7.

**Lag decay.** Table 5 estimates each lag separately in a univariate DoubleML specification. The results fail the no-reversal test comprehensively, meaning all lagged sentiment variables — at lags 1, 2, 3, 5, and 7 — are highly significant (p < 0.001) and negative for both Twitter and news sentiment. Because there is no decay pattern, this means that rises in price are likely due to factors other than tweets or news (being the result of more permanent information or performance), or that reverse causality is in effect and price changes are driving social media sentiment.

**Impact persistence.** Table 6 estimates the effect of current sentiment on future returns. Under the hypothesis that tweet effects decay quickly (within 2–3 days) while news effects persist, `twitter_sent` should be most significant at lag 1 and insignificant thereafter. The results show it insignificant at every lag.  Also, while for `news_sent` there are significant estimators, the signs are inconsistent.  Taken together, this tells me that while twitter sentiment is likely moot in the eyes of traders, news varies based on how the specific content of the news matters to traders.  Likely, important information is conveyed that does not tip the sentiment scales highly but is of value for determining tradability of stocks (eg, exposure to a new regulation in a given industry). 

---

## Summary and Next Steps

While the DoubleML results are interesting, the conflicting signals and failure of the reverse causality tests are deeply concerning.  The current evidence points towards price driving tweets, and not the otehr way around.  This seems to contradict past studies where teams have found an effect from tweets.  I will review their methods and see if I can replicate them in this case.  It may also be the case, that traders no longer use twitter as much as before, or that its role as an information source has declined.  I can look at differences across different time periods as well to see if that possibility holds.  

In any case, the current evidence indicates that twitter holds little influecne over daily price changes, and that the news is a stronger indicator of price fluctuations.  
