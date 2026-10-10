# 📊 Classical ML

Fast, explainable, interview-critical — most production ML still lives here.

Three projects exploring the classical algorithm toolkit: comparing every major model head-to-head, handling severely imbalanced real-world data, and forecasting time series without leaking the future.

**Status: In progress 🔄** — 1 of 3 complete

## Projects

### ⚔️ 1. Algorithm Wars — *Complete ✅*
*8 classical algorithm families (9 models), 3 deliberately different datasets*

The usual tutorial answer is "XGBoost wins." That's incomplete. This project runs Logistic Regression, Decision Tree, Random Forest, SVM (RBF + Linear), KNN, Gradient Boosting, XGBoost, and Naive Bayes across three datasets with fundamentally different characteristics — and watches the leaderboard reorder itself each time.

| Dataset | Winner | Biggest surprise |
|---|---|---|
| Breast Cancer (569 rows, numeric) | Logistic Regression — 98.07% | Linear models won a "non-linear" dataset |
| Adult Income (48k rows, mixed types) | XGBoost — 87.23% | SVM's fit time went 0.018s → 145s |
| BBC News (TF-IDF, sparse) | Linear SVM — 98.47% | XGBoost dropped from #1 to #8 |

**The finding:** There is no overall winner. "Best algorithm" is a property of the dataset, not the algorithm.

**Stack:** scikit-learn · XGBoost · pandas · matplotlib · seaborn

---

### 💳 2. Card Fraud Detective — *Not started ⬜*
*Catching fraud in a severely imbalanced dataset*

Fraud is ~0.17% of transactions, so accuracy is meaningless. PR-AUC, cost-based threshold selection, probability calibration, and an honest comparison of class weights vs resampling vs doing nothing — on a time-ordered split. Reuses the Algorithm Wars harness.

**Planned stack:** LightGBM · CatBoost · XGBoost · SHAP · Optuna

---

### 📈 3. Signal Forecaster — *Not started ⬜*
*Time series forecasting without leaking the future*

Walk-forward validation, a seasonal-naive baseline, classical models (ETS/ARIMA) vs gradient boosting on lag features vs a zero-shot foundation forecasting model — ideally on sensor data from my own ESP32 hardware.

**Planned stack:** pandas · statsforecast · LightGBM

*(A house-price regression project, "Zillow Killer", was planned here and dropped in the October 2026 roadmap update — it overlapped too much with Titanic and Algorithm Wars.)*

---

## What this repo demonstrates

- Choosing algorithms based on data characteristics rather than reputation
- Fair model comparison — identical preprocessing, identical CV folds, timing alongside accuracy
- Understanding *why* an algorithm wins or loses on a given dataset, not just *that* it does
- Class imbalance handling, explainability, and leakage-free time series validation — the parts of classical ML that matter in production

---

*Projects 2.1–2.3 of an ongoing roadmap building AI projects from the ground up — see the [hub repo](https://github.com/Altster/AI-Projects) for the full roadmap*
