# 📊 Classical ML

Fast, explainable, interview-critical — most production ML still lives here.

Three projects exploring the classical algorithm toolkit: comparing every major model head-to-head, regression with careful feature engineering, and handling severely imbalanced real-world data.

**Status: In progress 🔄** — 1 of 3 complete

## Projects

### ⚔️ 1. Algorithm Wars — *Complete ✅*
*8 classical models, 3 deliberately different datasets*

The usual tutorial answer is "XGBoost wins." That's incomplete. This project runs Logistic Regression, Decision Tree, Random Forest, SVM (RBF + Linear), KNN, Gradient Boosting, XGBoost, and Naive Bayes across three datasets with fundamentally different characteristics — and watches the leaderboard reorder itself each time.

| Dataset | Winner | Biggest surprise |
|---|---|---|
| Breast Cancer (569 rows, numeric) | Logistic Regression — 98.07% | Linear models won a "non-linear" dataset |
| Adult Income (48k rows, mixed types) | XGBoost — 87.23% | SVM's fit time went 0.018s → 145s |
| BBC News (TF-IDF, sparse) | Linear SVM — 98.47% | XGBoost dropped from #1 to #8 |

**The finding:** There is no overall winner. "Best algorithm" is a property of the dataset, not the algorithm.

**Stack:** scikit-learn · XGBoost · pandas · matplotlib · seaborn

---

### 🏠 2. Zillow Killer — *In progress 🔄*
*House price regression with obsessive feature engineering*

Linear and polynomial regression, L1/L2/Elastic Net regularization, feature importance analysis, and boosting vs linear model comparison.

**Planned stack:** scikit-learn · pandas · XGBoost · SHAP

---

### 💳 3. Card Fraud Detective — *Not started ⬜*
*Catching fraud in a severely imbalanced dataset*

XGBoost / LightGBM / CatBoost comparison, SHAP explainability, class imbalance strategies, and precision-recall tradeoffs where accuracy is a misleading metric.

**Planned stack:** XGBoost · LightGBM · SHAP · Optuna

---

## What this repo demonstrates

- Choosing algorithms based on data characteristics rather than reputation
- Fair model comparison — identical preprocessing, identical CV folds, timing alongside accuracy
- Understanding *why* an algorithm wins or loses on a given dataset, not just *that* it does
- Regularization, class imbalance handling, and explainability — the parts of classical ML that matter in production

---

*Part of an ongoing roadmap building AI projects from the ground up*
