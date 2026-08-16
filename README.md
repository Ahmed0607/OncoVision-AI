# OncoVision AI: Explainable Breast Cancer Diagnostics with XGBoost & Bayesian Optimization

## 📌 Project Overview & Objective

**OncoVision AI** is an end-to-end clinical decision-support pipeline designed to classify breast tumor biopsies as **Malignant (0)** or **Benign (1)** using the Wisconsin Breast Cancer dataset.

The primary objective is to maximize diagnostic sensitivity and classification confidence using modern gradient boosting (**XGBoost**) while eliminating the "black-box" nature of tree-based ensembles through **SHAP (Shapley Additive exPlanations)** and **2D PCA Decision Boundary Mapping**.

---

## 🔬 Core Methodology & Statistical Concepts

### 1. Stratified K-Fold Cross-Validation

* **What it is:** A cross-validation technique where each fold contains approximately the same percentage of samples of each target class as the complete set.
* **Why it is essential here:** The dataset contains an inherent class imbalance (~62.7% Benign vs. ~37.3% Malignant). Standard K-Fold risks creating validation folds with disproportionately few malignant cases, causing unstable gradient updates. `StratifiedKFold(n_splits=5)` guarantees robust, variance-controlled metric estimation across all splits.

### 2. Correlation Heatmap Analysis

* **What it is:** A bivariate matrix displaying Pearson correlation coefficients ($r$) between morphological cell features.
* **Clinical & Mathematical Importance:** Features such as `radius_mean`, `perimeter_mean`, and `area_mean` exhibit extreme collinearity ($r > 0.95$). Identifying multicollinearity prevents tree splitting redundancy, reduces model complexity, and stabilizes feature attribution weights during post-hoc explainability.

### 3. Confusion Matrix Breakdown

The confusion matrix tracks the clinical reliability of the model on the 114 unseen test samples:

$$\begin{array}{c\|cc} & \textbf{Pred: Malignant (0)} & \textbf{Pred: Benign (1)} \\ \hline \textbf{True: Malignant (0)} & 38\text{ (True Neg/Pos)} & 4\text{ (False Alarm)} \\ \textbf{True: Benign (1)} & 1\text{ (Critical Miss)} & 71\text{ (True Neg/Pos)} \\ \end{array}$$

* **True Positives/Negatives:** High diagonal density confirms high precision across both classes.
* **False Negatives vs False Positives:** In oncology, minimizing False Negatives (predicting Benign when cancer is present) is vital. A high ROC-AUC threshold ensures high recall for malignant cases.

---

## ⚙️ Hyperparameter Optimization: Optuna vs RandomizedSearchCV

### What is Optuna & Why Use It?

* **What it is:** A next-generation hyperparameter optimization framework powered by Bayesian optimization (Tree-structured Parzen Estimators - TPE).
* **Why it was used over Grid/Random Search:** Standard `RandomizedSearchCV` evaluates parameter combinations independently at random. Optuna constructs a probabilistic model of the objective function, learning from previous trials to actively sample regions of the parameter space most likely to yield higher ROC-AUC scores.
* **How it was implemented:** We defined an objective function over continuous and discrete search spaces (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `alpha`, `lambda`) across 50 trials using 5-fold Stratified CV.

### 📊 Performance Comparison

| Metric | RandomizedSearchCV (25 Iterations) | Optuna Bayesian Search (50 Trials) | Improvement |
| --- | --- | --- | --- |
| **Best CV ROC-AUC** | `~0.9920` | **`0.9954`** | $+0.34\%$ |
| **Test Set Accuracy** | `95.61%` (109/114) | **`96.49%`** (110/114) | $+0.88\%$ |
| **Test Set ROC-AUC** | `0.9927` | **`0.9947`** | $+0.20\%$ |

---

## 🔍 Explainable AI (XAI) with SHAP

Tree-based ensembles require interpretability before clinical adoption. We employ **SHAP (Shapley Additive exPlanations)** rooted in cooperative game theory.

```
       [Global Feature Impact]                     [Local Patient Explanation]
       
  Feature            SHAP Value              Baseline (E[f(x)]) ──────► Final Prediction f(x)
  ───────────────────────────────                     ▲                     │
  concave_points_worst  ████████                      │ + Perimeter (+0.35) │
  radius_worst          ██████                        │ + Texture   (+0.12) │
  perimeter_worst       █████                         │ - Smoothness(-0.08) ▼
  area_worst            ████                                          [Malignant]

```

### 1. SHAP Summary Plot (Global Explainability)

* **What it is:** A beeswarm plot aggregating SHAP values for every feature across all test instances.
* **Why it is useful:** It ranks overall feature importance while illustrating the directional impact of each feature. For instance, high values of `concave points_mean` and `perimeter_worst` push predictions toward Malignant.

### 2. SHAP Waterfall Plot (Local Patient Prediction)

* **How it works:** Begins at the expected base value $E[f(x)]$ (average model output across the training population) and adds/subtracts the contribution ($\phi_i$) of each individual measurement for a specific patient:

$$f(x) = E[f(x)] + \sum_{i=1}^{M} \phi_i$$

* **Why it is useful in clinical practice:** It provides an interpretable breakdown for a doctor, explaining *why* a particular biopsy was flagged as malignant (e.g., "Patient #0 was flagged primarily due to elevated `perimeter_worst` (+0.42) and high `concavity_mean` (+0.18)").

---

## 🚀 Getting Started & Execution

1. **Install dependencies:**
```bash
pip install -q xgboost seaborn scikit-learn optuna shap matplotlib pandas numpy

```


2. **Run the pipeline in Colab / Jupyter:**
Open `notebooks/cancer_classification.ipynb` and run all cells to reproduce data preprocessing, Optuna tuning, metric evaluation, and SHAP visual outputs.
