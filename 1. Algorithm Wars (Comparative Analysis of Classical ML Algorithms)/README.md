# ⚔️ Algorithm Wars
Pitting 8 classical ML algorithms against each other across 3 deliberately different datasets — to find out *when* each one actually wins, not just *which* scores highest.

---

## The question this project answers

Every ML tutorial ends with "XGBoost wins." But XGBoost doesn't always win. The real skill isn't knowing which algorithm is best in general — it's knowing which algorithm is best *for your specific data*.

This project runs the same 9 algorithms across 3 datasets with fundamentally different characteristics and watches the leaderboard reorder itself each time. The reordering is the lesson.

---

## Decision Boundaries

Represents the boundaries that the algorithms drew for classification. The representation may be inconsistent from reality because of possible information loss from PCA. The boundaries shown are for Dataset-A

![Decision Boundaries](statistics/Decision-Boundaries(A).png)

## Results

Note: The accuracies and times may wary accross runs, but the rankings and main lessons remain the same

### Dataset A — Breast Cancer Wisconsin
*569 samples · 30 numeric features · binary classification*

| Rank | Model | Accuracy | Fit Time |
|---|---|---|---|
| 1 | Logistic Regression | 98.07% | 0.026s |
| 2 | SVM (RBF) | 97.71% | 0.017s |
| 3 | SVM (Linear) | 96.84% | 0.015s |
| 4 | XGBoost | 96.48% | 0.122s |
| 5 | K-Nearest Neighbors | 96.31% | 0.009s |
| 5 | Gradient Boosting | 96.31% | 0.537s |
| 7 | Random Forest | 95.60% | 0.252s |
| 8 | Decision Tree | 93.15% | 0.031s |
| 9 | Gaussian Naive Bayes | 92.80% | 0.012s |

**Key finding:** Linear models dominated a dataset described as "non-linear." At 30 dimensions, a linear hyperplane has enough degrees of freedom to separate complex-looking distributions. *"Non-linear in 2D" ≠ "non-linear in 30D."*

---

### Dataset B — Adult Census Income
*~48,000 samples · mixed numeric + categorical features · binary classification*

| Rank | Model | Accuracy | Fit Time |
|---|---|---|---|
| 1 | XGBoost | 87.23% | 1.22s |
| 2 | Gradient Boosting | 86.65% | 22.73s |
| 3 | SVM (RBF) | 85.55% | **144.71s** |
| 4 | Random Forest | 85.31% | 8.18s |
| 5 | SVM (Linear) | 85.09% | 1.43s |
| 6 | Logistic Regression | 85.08% | 1.28s |
| 7 | K-Nearest Neighbors | 83.29% | 0.21s |
| 8 | Decision Tree | 81.61% | 1.20s |
| 9 | Gaussian Naive Bayes | 60.68% | 0.339s |

**Key finding:** SVM (RBF) went from 0.018s to 145s — an 8,000× slowdown — while dropping from #2 to #3. Its training complexity scales O(n²) to O(n³), making it impractical at 48k samples. XGBoost won by a clear margin. Gaussian Naive Bayes collapsed to 60.68% — its normality assumption is fundamentally violated by one-hot encoded categorical features.

---

### Dataset C — BBC News
*2,225 articles · 5 topic categories · TF-IDF features (high-dimensional, sparse)*

| Rank | Model | Accuracy | Fit Time |
|---|---|---|---|
| 1 | SVM (Linear) | 98.47% | 2.36s |
| 2 | SVM (RBF) | 98.16% | 10.60s |
| 3 | Logistic Regression | 98.11% | 2.73s |
| 4 | Multinomial Naive Bayes | 97.30% | 4.24s |
| 5 | K-Nearest Neighbors | 96.40% | 2.24s |
| 6 | Random Forest | 96.13% | 4.18s |
| 7 | Gradient Boosting | 95.87% | **72.16s** |
| 8 | XGBoost | 94.97% | **70.23s** |
| 9 | Decision Tree | 83.60% | 3.08s |

**Key finding:** XGBoost went from #1 on Adult to #8 on BBC — the biggest swing of any model. Trees split on one feature at a time; with 20,000 TF-IDF features, no single word is informative enough to justify the computation. Multinomial Naive Bayes scored 97.3% despite its independence assumption being provably wrong for text — the word-level signal is strong enough that word relationships don't need to be modeled.

---

## Cross-dataset ranking summary

| Model | Cancer | Adult | BBC | Most notable |
|---|---|---|---|---|
| Logistic Regression | #1 | #6 | #3 | Never collapses — most consistent |
| SVM (RBF) | #2 | #3 | #2 | Stable accuracy, unstable training time |
| SVM (Linear) | #3 | #5 | #1 | Best on text, average elsewhere |
| XGBoost | #4 | #1 | #8 | Biggest swing — best on tabular, worst on text |
| KNN | #5 | #7 | #5 | Middling everywhere |
| Gradient Boosting | #6 | #2 | #7 | Strong on large tabular, slow everywhere |
| Random Forest | #7 | #4 | #6 | Moderate, never leads |
| Decision Tree | #8 | #8 | #9 | Consistently worst — never use alone |
| Naive Bayes | #9 | #9 | #4 | Entirely determined by assumption fit |

---

## Decision framework

Derived from actual observed results, not theory:

| Situation | Reach for |
|---|---|
| Small data, all numeric | Logistic Regression — fast, interpretable, rarely beaten at this scale |
| Large data, mixed types, tabular | XGBoost — best accuracy, reasonable training time |
| High-dimensional sparse (text) | Linear SVM or Logistic Regression |
| Quick baseline in 30 seconds | Logistic Regression (tabular) or Multinomial Naive Bayes (text) |
| Need interpretability | Logistic Regression — coefficients show direction and magnitude per feature |
| Almost never | Single Decision Tree — always outperformed by Random Forest with no compensating benefit |
| Watch out for | SVM (RBF) on large data — accuracy holds up but training time explodes |

**The core lesson:** There is no overall winner. "Best algorithm" is a property of the dataset, not the algorithm. Logistic Regression is the best *default starting point* — never catastrophic, always fast, interpretable. Switch to XGBoost for large structured data, Linear SVM for text.

---

## Tech stack

- **scikit-learn** — all classical ML algorithms, preprocessing pipelines, cross-validation
- **XGBoost** — gradient boosting baseline
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualizations
- **HuggingFace `datasets`** — BBC News dataset
