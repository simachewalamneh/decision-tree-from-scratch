# Decision Trees (DT) — Concept Presentation Note

> **Capsule summary:** A Decision Tree is a supervised learning model that predicts a target by asking a sequence of yes/no questions about the input features, recursively splitting the data into purer and purer subsets until it can make a confident prediction at each leaf. The whole algorithm is really just one idea repeated: *"what question splits this data best right now?"*

---

## 1. What is a Decision Tree?

A Decision Tree is a **hierarchical, rule-based model** shaped like an upside-down tree:

- **Root node** — the full dataset, before any question is asked.
- **Internal (decision) nodes** — each asks a single test on one feature (e.g. `Age < 30?`, `Color == Red?`).
- **Branches** — the possible answers to that test.
- **Leaf nodes** — terminal nodes that hold the final prediction (a class label or a numeric value).

**How it's built (high level):**
1. Start with all training data at the root.
2. Search across every feature and every possible split point for the one that best separates the target variable.
3. Split the data into child nodes using that rule.
4. Repeat recursively on each child (this is why it's called a **greedy, recursive partitioning** algorithm).
5. Stop when a stopping rule is hit (max depth, min samples per leaf, node is pure, no more information gain, etc.).

**Why it matters:**
- It's **interpretable** — you can read the path from root to leaf as a set of human-readable rules.
- It makes **no assumption about linearity** or feature distribution.
- It naturally handles **mixed data types** (numeric + categorical) and **non-linear interactions**.
- It's the building block for ensembles like Random Forests, Gradient Boosted Trees (XGBoost/LightGBM), and Extra Trees.

---

## 2. The Core Idea: Splitting = Reducing Impurity

At every node, the tree asks: *"Which feature, and which threshold/category grouping, gives me the purest possible children?"*

"Purity" means: after the split, how homogeneous (concentrated in one class, or low-variance) are the resulting subsets, compared to before the split?

This is formalized as **Information Gain**:

```
Gain(split) = Impurity(parent) − [ weighted average of Impurity(children) ]
```

The feature/threshold combination that **maximizes gain** (or equivalently minimizes weighted child impurity) is chosen at each node.

---

## 3. Splitting Criteria for Different Feature Types

### 3.1 Numerical (continuous) features

Numeric features are handled with **threshold splits**: `feature ≤ t` vs `feature > t`.

- The algorithm **sorts** the unique values of the feature.
- It considers candidate thresholds — typically the **midpoints between consecutive sorted values**.
- For each candidate threshold, it computes the impurity reduction if the data were split there.
- The threshold with the **best gain** is kept as the node's decision rule.

This means a numeric feature can be **reused multiple times** at different depths of the tree with different thresholds (e.g. `Income ≤ 50k` near the root, `Income ≤ 20k` further down).

### 3.2 Categorical features

Categorical features are handled differently depending on cardinality:

| Strategy | Description | When used |
|---|---|---|
| **One-vs-rest (binary)** | `feature == category_A` vs "everything else" | Common in CART, works for any cardinality |
| **Subset splitting** | Partition the *set* of categories into two groups (e.g. {Red, Blue} vs {Green}) that best separate the target | CART for low/medium cardinality categorical features |
| **One-hot encoding pre-processing** | Convert each category into its own binary column, then treat each as a binary numeric split | Common in scikit-learn (which doesn't natively support multi-way categorical splits) |
| **Multi-way split** | One branch per category value (native to ID3/C4.5) | Classic ID3/C4.5 algorithms |

**Key trade-off with categorical features:** high-cardinality categorical variables (e.g. "zip code" with 10,000 values) create a huge search space of possible subset splits, which is computationally expensive and prone to **overfitting** (the tree can "memorize" rare categories). This is why one-hot encoding + binary splits, or **target/mean encoding**, are common practical workarounds.

---

## 4. Splitting Criteria — Formulas and Trade-offs

### 4.1 For Classification

| Criterion | Formula | Behavior |
|---|---|---|
| **Gini Impurity** | `Gini = 1 − Σ pᵢ²` | Measures probability of misclassifying a randomly chosen element if labeled according to the class distribution in the node. Faster to compute (no logarithm). |
| **Entropy / Information Gain** | `Entropy = − Σ pᵢ log₂(pᵢ)` | Measures disorder/uncertainty. Information Gain = entropy reduction from parent to children. |
| **Log Loss (Cross-Entropy)** | Similar to entropy, used in probabilistic settings | Sensitive to probability calibration, used in gradient boosting more than plain DT. |

**Trade-offs:**
- **Gini vs Entropy**: In practice they usually pick very similar splits (their curves are shaped almost identically). Gini is **computationally cheaper** (no log), so it's the scikit-learn default. Entropy is slightly more sensitive to changes in class probabilities (more "peaked" near balanced splits), which can occasionally favor more balanced splits.
- **Information Gain has a bias toward high-cardinality features** — a feature with many distinct values can create many small, pure nodes and look artificially "informative." This is why **C4.5 introduced Gain Ratio**, which normalizes information gain by the intrinsic information of the split (penalizing features that fragment the data into many branches).
- **Gini/Entropy vs Misclassification Error**: Misclassification error is a valid impurity measure but is **less sensitive** to changes in node purity (it's piecewise linear, not strictly concave), so it often fails to distinguish between two candidate splits that Gini/Entropy would clearly rank — that's why it's rarely used for growing trees (though it's still used for **pruning**).

### 4.2 For Regression

| Criterion | Formula | Behavior |
|---|---|---|
| **Variance Reduction / MSE** | `MSE = (1/n) Σ (yᵢ − ȳ)²` | Splits to minimize the variance of target values within each child node. |
| **MAE (Mean Absolute Error)** | `MAE = (1/n) Σ \|yᵢ − ȳ\|` | More robust to outliers, but less commonly used because it's harder to optimize (not smooth) and computationally heavier. |
| **Friedman MSE** | Adjusted MSE used in Gradient Boosted Trees | Corrects for the improvement score to better guide boosting-specific splits. |

**Trade-off:** MSE-based splitting pulls leaf predictions toward the **mean**, which is optimal under squared error but **sensitive to outliers**. MAE-based splitting targets the **median**, more robust to outliers, but slower to compute and can produce less smooth trees.

### 4.3 General Trade-off Summary

| Concern | Gini | Entropy | MSE (regression) |
|---|---|---|---|
| Speed | Fastest (no log) | Slower (log computation) | Fast |
| Sensitivity to class imbalance | Moderate | Slightly higher | N/A |
| Bias toward high-cardinality splits | Yes | Yes (more so) | Yes |
| Outlier sensitivity | N/A | N/A | High (use MAE if this matters) |
| Interpretability of the metric | Intuitive (misclassification-like) | Information-theoretic | Straightforward (variance) |

---

## 5. Decision Trees for Classification vs Regression

| Aspect | Classification Tree | Regression Tree |
|---|---|---|
| **Target type** | Discrete class labels | Continuous numeric value |
| **Splitting criterion** | Gini, Entropy / Information Gain | Variance reduction (MSE), MAE |
| **Leaf prediction** | Majority class (or class probability distribution) in that leaf | Mean (or median) of target values in that leaf |
| **Output** | A class label + optionally class probabilities | A single continuous number |
| **Common algorithm names** | ID3, C4.5, CART (classification mode) | CART (regression mode) |
| **Evaluation metrics** | Accuracy, F1, AUC, log-loss | RMSE, MAE, R² |

**Practically identical mechanics** — both grow the tree the same recursive way; only the **impurity/loss function** and **leaf prediction rule** change. This is exactly why libraries like scikit-learn expose `DecisionTreeClassifier` and `DecisionTreeRegressor` as near-mirror APIs.

---

## 6. Strengths, Weaknesses & Overfitting Control

**Strengths**
- Highly interpretable (can be visualized and read as rules).
- Handles non-linear relationships and feature interactions without manual engineering.
- Requires minimal data preprocessing (no scaling/normalization needed).
- Naturally handles missing values (with surrogate splits, depending on implementation).

**Weaknesses**
- **High variance / prone to overfitting** — a fully grown tree can memorize training data (deep trees → low bias, high variance).
- **Unstable** — small changes in the data can produce a very different tree structure.
- **Greedy, not globally optimal** — each split is locally optimal, not guaranteed to yield the best overall tree.
- **Biased toward features with many levels** (especially with Information Gain).

**Common overfitting controls (regularization):**
- `max_depth`, `min_samples_split`, `min_samples_leaf`
- `max_features` (limit features considered per split)
- **Pruning** (pre-pruning = stopping early; post-pruning = growing full tree then trimming back using cost-complexity pruning, `ccp_alpha` in scikit-learn)
- Ensembling (Random Forests, Gradient Boosting) — trades away some interpretability for much better generalization.

---

## 7. One-Line Takeaways (for slide bullets)

- **DT = recursive rule-based partitioning of data to reduce impurity/variance.**
- **Numeric features → threshold splits (`≤ t`); categorical features → subset/one-hot splits.**
- **Gini is faster, Entropy is more information-theoretic — both usually agree; both favor high-cardinality features (fix: Gain Ratio).**
- **Classification uses Gini/Entropy + majority-vote leaves; Regression uses variance/MSE + mean-value leaves.**
- **Single trees overfit easily — pruning and depth limits (or ensembling) are essential in practice.**
