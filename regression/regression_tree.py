"""
Decision Tree Regressor — implemented from scratch, following the
"Decision trees for regression" section of Chapter 9 in Luis Serrano's
"Grokking Machine Learning".

As the book states: "the algorithm used for training a regression decision
tree is very similar to the one used for training a classification decision
tree. The only difference is that for classification trees, we used
accuracy, Gini index, or entropy, and for regression trees, we use the mean
square error (MSE)." Leaf predictions become the AVERAGE label instead of
the majority label.

No scikit-learn (or any other library) is used for the tree logic itself.
Only numpy is used for basic array math.

Supports:
  - Continuous (numerical) targets
  - Numerical features  -> split by "is feature <= cutoff?" (all midpoints tried,
                            exactly as in the book's age-cutoff example)
  - Categorical features -> split by "is feature == category?" (same as classifier)
  - Splitting criterion: mean squared error (MSE)
  - The same four stopping criteria as the classifier:
        1. max_depth
        2. min_samples_split
        3. min_samples_leaf
        4. min_impurity_decrease (here: minimum MSE decrease)
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Regression impurity: mean squared error around the average prediction
# --------------------------------------------------------------------------- #

def mse_of_set(labels):
    """
    MSE of a set of continuous labels when predicting their own average
    (i.e. how spread out the labels are around their mean). 0 = all labels
    identical. This plays the same role Gini/entropy played for classification.
    """
    labels = np.asarray(labels, dtype=float)
    if len(labels) == 0:
        return 0.0
    avg = labels.mean()
    return float(np.mean((labels - avg) ** 2))


def weighted_mse(left_labels, right_labels):
    """Weighted average MSE of a split, weighted by branch size —
    same weighting rule as the classifier (Figure 9.14)."""
    n_left, n_right = len(left_labels), len(right_labels)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left / n_total) * mse_of_set(left_labels) + \
           (n_right / n_total) * mse_of_set(right_labels)


# --------------------------------------------------------------------------- #
# Tree node (identical structure to the classifier's Node)
# --------------------------------------------------------------------------- #

class RegressionNode:
    def __init__(self, depth):
        self.depth = depth
        self.is_leaf = True
        self.prediction = None      # AVERAGE label (not majority, unlike classifier)
        self.feature_index = None
        self.feature_type = None    # "numerical" or "categorical"
        self.threshold = None
        self.category = None
        self.left = None
        self.right = None
        self.n_samples = 0
        self.mse_value = None


# --------------------------------------------------------------------------- #
# Decision Tree Regressor
# --------------------------------------------------------------------------- #

class DecisionTreeRegressor:
    """
    Same stopping-criteria parameters as DecisionTreeClassifier, but the
    splitting metric is fixed to MSE (the book's regression criterion) and
    leaf predictions are the average label instead of the majority label.
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, feature_types=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_types = feature_types
        self.root_ = None
        self.n_features_ = None

    # ---- fitting --------------------------------------------------------- #

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]

        if self.feature_types is None:
            self.feature_types = []
            for j in range(self.n_features_):
                col = X[:, j]
                try:
                    col.astype(float)
                    self.feature_types.append("numerical")
                except (ValueError, TypeError):
                    self.feature_types.append("categorical")

        self.root_ = self._build_node(X, y, depth=0)
        return self

    def _build_node(self, X, y, depth):
        node = RegressionNode(depth)
        node.n_samples = len(y)
        node.prediction = float(np.mean(y)) if len(y) > 0 else 0.0   # AVERAGE, not majority
        node.mse_value = mse_of_set(y)

        # ---- Stopping conditions (same four as the classifier) ------------
        # 1. Node is already "pure" (all labels identical) -> nothing to split.
        if node.mse_value == 0:
            return node
        # 2. Max depth reached.
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        # 3. Not enough samples in this node to justify splitting.
        if len(y) < self.min_samples_split:
            return node

        best = self._best_split(X, y)
        if best is None:
            return node

        feat_idx, feat_type, split_value, left_mask, gain = best

        # 4. Minimum MSE-decrease (equivalent of min_impurity_decrease).
        if gain < self.min_impurity_decrease:
            return node

        node.is_leaf = False
        node.feature_index = feat_idx
        node.feature_type = feat_type
        if feat_type == "numerical":
            node.threshold = split_value
        else:
            node.category = split_value

        node.left = self._build_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_node(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """
        Same search as the classifier's _best_split, except the quality
        metric is MSE decrease instead of impurity decrease. For numerical
        features, every midpoint between consecutive sorted unique values
        is tried as a cutoff (exactly the book's age-cutoff procedure in
        Table 9.7).
        """
        parent_mse = mse_of_set(y)
        best_gain = -np.inf
        best = None
        n_samples = len(y)

        for feat_idx in range(self.n_features_):
            col = X[:, feat_idx]
            feat_type = self.feature_types[feat_idx]

            if feat_type == "numerical":
                col_f = col.astype(float)
                uniq = np.unique(col_f)
                if len(uniq) < 2:
                    continue
                cutoffs = (uniq[:-1] + uniq[1:]) / 2.0
                for cutoff in cutoffs:
                    left_mask = col_f <= cutoff
                    n_left, n_right = left_mask.sum(), n_samples - left_mask.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    w_mse = weighted_mse(y[left_mask], y[~left_mask])
                    gain = parent_mse - w_mse
                    if gain > best_gain:
                        best_gain = gain
                        best = (feat_idx, "numerical", cutoff, left_mask.copy(), gain)
            else:
                for category in np.unique(col):
                    left_mask = (col == category)
                    n_left, n_right = left_mask.sum(), n_samples - left_mask.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    w_mse = weighted_mse(y[left_mask], y[~left_mask])
                    gain = parent_mse - w_mse
                    if gain > best_gain:
                        best_gain = gain
                        best = (feat_idx, "categorical", category, left_mask.copy(), gain)

        return best

    # ---- prediction -------------------------------------------------------#

    def predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            if node.feature_type == "numerical":
                go_left = float(x[node.feature_index]) <= node.threshold
            else:
                go_left = x[node.feature_index] == node.category
            node = node.left if go_left else node.right
        return node.prediction

    def predict(self, X):
        X = np.asarray(X, dtype=object)
        return np.array([self.predict_one(x) for x in X], dtype=float)

    # ---- evaluation metrics (the regression analogue of evaluate()) ------ #

    def evaluate(self, X, y):
        """
        Regression evaluation metrics -- the regression counterparts of
        the classifier's accuracy/precision/recall/F1:
          MSE  - mean squared error (average squared distance from truth)
          RMSE - root MSE (same units as the target, easier to interpret)
          MAE  - mean absolute error (less sensitive to outliers than MSE)
          R^2  - fraction of variance explained (1.0 = perfect, 0.0 = no
                 better than always predicting the mean, can go negative)
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        errors = y_true - y_pred
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def print_evaluation(self, X, y):
        results = self.evaluate(X, y)
        print(f"MSE:  {results['mse']:.3f}")
        print(f"RMSE: {results['rmse']:.3f}")
        print(f"MAE:  {results['mae']:.3f}")
        print(f"R^2:  {results['r2']:.3f}")
        return results

    # ---- inspection ------------------------------------------------------ #

    def print_tree(self, feature_names=None, node=None, indent=""):
        if node is None:
            node = self.root_
        name = lambda i: feature_names[i] if feature_names else f"feature[{i}]"
        if node.is_leaf:
            print(f"{indent}Leaf: predict={node.prediction:.3f}  "
                  f"(n={node.n_samples}, mse={node.mse_value:.3f})")
        else:
            q = (f"{name(node.feature_index)} <= {node.threshold:.3f}?"
                 if node.feature_type == "numerical"
                 else f"{name(node.feature_index)} == {node.category!r}?")
            print(f"{indent}[{q}]  (n={node.n_samples}, mse={node.mse_value:.3f})")
            print(f"{indent}├─ Yes:")
            self.print_tree(feature_names, node.left, indent + "│   ")
            print(f"{indent}└─ No:")
            self.print_tree(feature_names, node.right, indent + "    ")

    def depth(self, node=None):
        if node is None:
            node = self.root_
        if node.is_leaf:
            return node.depth
        return max(self.depth(node.left), self.depth(node.right))

    def n_leaves(self, node=None):
        if node is None:
            node = self.root_
        if node.is_leaf:
            return 1
        return self.n_leaves(node.left) + self.n_leaves(node.right)
