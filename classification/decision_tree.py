from __future__ import annotations
import numpy as np
from collections import Counter

 # Impurity / quality metrics (Section "First step to build the model")
 
def gini_impurity(labels):
    """Gini = 1 - sum(p_i^2). 0 = pure set, closer to 1 = mixed set."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels):
    """Entropy = -sum(p_i * log2(p_i)). 0 = pure set."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def accuracy_of_majority_prediction(labels):
    n = len(labels)
    if n == 0:
        return 1.0
    most_common_count = Counter(labels).most_common(1)[0][1]
    return most_common_count / n
#[(most_common_label, most_common_count)] = Counter(labels).most_common(1)

def impurity(labels, criterion):
 
    if criterion == "gini":
        return gini_impurity(labels)
    elif criterion == "entropy":
        return entropy(labels)
    elif criterion == "accuracy":
        return 1.0 - accuracy_of_majority_prediction(labels)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def weighted_impurity(left_labels, right_labels, criterion):
    n_left, n_right = len(left_labels), len(right_labels)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left / n_total) * impurity(left_labels, criterion) + \
           (n_right / n_total) * impurity(right_labels, criterion)


# Tree node

class Node:
    """A node is either a leaf (prediction set) or a decision node
    (feature + question set, with left/right children)."""

    def __init__(self, depth):
        self.depth = depth
        self.is_leaf = True
        self.prediction = None      # majority label, set for every node
        self.feature_index = None   # which feature this node splits on
        self.feature_type = None    # "numerical" or "categorical"
        self.threshold = None       # cutoff, for numerical features
        self.category = None        # category value, for categorical features
        self.left = None            # branch for "yes" / condition true
        self.right = None           # branch for "no"  / condition false
        self.n_samples = 0
        self.impurity_value = None

    def question_str(self):
        if self.feature_type == "numerical":
            return f"feature[{self.feature_index}] <= {self.threshold:.3f} ?"
        else:
            return f"feature[{self.feature_index}] == {self.category!r} ?"
# Decision Tree Classifier

class DecisionTreeClassifier:

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 feature_types=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_types = feature_types
        self.root_ = None
        self.n_features_ = None

    # ---- fitting --------------------------------------------------------- #

    def fit(self, X, y):  # start a training
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]

        if self.feature_types is None: #
            self.feature_types = [
                "numerical" if np.issubdtype(np.array(X[:, j], dtype=float).dtype, np.number)
                else "categorical"
                for j in range(self.n_features_)
            ]
            # try/except fallback for genuinely non-numeric columns
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

    def _build_node(self, X, y, depth): #reccursively construct a tree
        node = Node(depth)
        node.n_samples = len(y)
        node.prediction = Counter(y).most_common(1)[0][0]
        node.impurity_value = impurity(y, self.criterion)  
        # ---- Stopping conditions (checked BEFORE searching for a split) ---
        # 1. Node is already pure -> nothing to gain from splitting.
        if node.impurity_value == 0:
            return node
        # 2. Max depth reached.
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        # 3. Not enough samples in this node to justify splitting.
        if len(y) < self.min_samples_split:
            return node

        best = self._best_split(X, y)  # Searches for the best feature/threshold
        if best is None:
            return node  # no split improved things enough / respected min_samples_leaf

        feat_idx, feat_type, split_value, left_mask, gain = best

        # 4. Minimum impurity decrease.
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

    def _best_split(self, X, y):  # Searches for the best feature/threshold
        parent_impurity = impurity(y, self.criterion)   
        best_gain = -np.inf   # wNegative infinity
        best = None  # (feat_idx, feat_type, split_value, left_mask, gain)

        n_samples = len(y)

        for feat_idx in range(self.n_features_):
            col = X[:, feat_idx]
            feat_type = self.feature_types[feat_idx]

            if feat_type == "numerical":
                col_f = col.astype(float)
                uniq = np.unique(col_f)
                if len(uniq) < 2:
                    continue
                # candidate cutoffs = midpoints between consecutive unique values
                cutoffs = (uniq[:-1] + uniq[1:]) / 2.0
                for cutoff in cutoffs:
                    left_mask = col_f <= cutoff
                    n_left, n_right = left_mask.sum(), n_samples - left_mask.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    w_imp = weighted_impurity(y[left_mask], y[~left_mask], self.criterion)
                    gain = parent_impurity - w_imp # if I use this exact cutoff, how much cleaner do the resulting two groups get, compared to not splitting at all.
                    if gain > best_gain:
                        best_gain = gain
                        best = (feat_idx, "numerical", cutoff, left_mask.copy(), gain)

            else:  # categorical -> one binary question per category (Fig 9.18)
                for category in np.unique(col):
                    left_mask = (col == category)
                    n_left, n_right = left_mask.sum(), n_samples - left_mask.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    w_imp = weighted_impurity(y[left_mask], y[~left_mask], self.criterion)
                    gain = parent_impurity - w_imp
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
        return np.array([self.predict_one(x) for x in X])

    def score(self, X, y): #calculate accuracy 
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))

    # ---- evaluation metrics ------------------------------------------------ #

    def evaluate(self, X, y, average="macro"):
       
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        classes = sorted(set(y_true) | set(y_pred))

        # confusion matrix: rows = true class, columns = predicted class
        idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[idx[t], idx[p]] += 1

        per_class = {}
        for c in classes:
            i = idx[c]
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()  # number of true samples of this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_class[c] = {"precision": precision, "recall": recall,
                             "f1": f1, "support": int(support)}

        accuracy = float(np.mean(y_pred == y_true))
        total = sum(v["support"] for v in per_class.values())
        if average == "weighted":
            avg_precision = sum(v["precision"] * v["support"] for v in per_class.values()) / total
            avg_recall = sum(v["recall"] * v["support"] for v in per_class.values()) / total
            avg_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total
        else:  # macro
            n = len(per_class)
            avg_precision = sum(v["precision"] for v in per_class.values()) / n
            avg_recall = sum(v["recall"] for v in per_class.values()) / n
            avg_f1 = sum(v["f1"] for v in per_class.values()) / n

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classes": classes,
            "per_class": per_class,
            f"{average}_precision": avg_precision,
            f"{average}_recall": avg_recall,
            f"{average}_f1": avg_f1,
        }

    def print_evaluation(self, X, y, average="macro"):
        """Pretty-print the output of evaluate()."""
        results = self.evaluate(X, y, average=average)
        classes = results["classes"]
        cm = results["confusion_matrix"]

        print(f"Accuracy: {results['accuracy']:.3f}\n")

        print("Confusion matrix (rows=true, columns=predicted):")
        header = "        " + "".join(f"{str(c):>10s}" for c in classes)
        print(header)
        for i, c in enumerate(classes):
            row = "".join(f"{cm[i, j]:>10d}" for j in range(len(classes)))
            print(f"{str(c):>8s}{row}")

        print(f"\n{'Class':>10s}{'Precision':>12s}{'Recall':>10s}{'F1':>10s}{'Support':>10s}")
        for c in classes:
            m = results["per_class"][c]
            print(f"{str(c):>10s}{m['precision']:>12.3f}{m['recall']:>10.3f}"
                  f"{m['f1']:>10.3f}{m['support']:>10d}")

        print(f"\n{average.capitalize()} avg  "
              f"precision={results[f'{average}_precision']:.3f}  "
              f"recall={results[f'{average}_recall']:.3f}  "
              f"f1={results[f'{average}_f1']:.3f}")
        return results

    # ---- inspection ------------------------------------------------------ #

    def print_tree(self, feature_names=None, node=None, indent=""):
        if node is None:
            node = self.root_
        name = lambda i: feature_names[i] if feature_names else f"feature[{i}]"
        if node.is_leaf:
            print(f"{indent}Leaf: predict={node.prediction}  "
                  f"(n={node.n_samples}, impurity={node.impurity_value:.3f})")
        else:
            q = (f"{name(node.feature_index)} <= {node.threshold:.3f}?"
                 if node.feature_type == "numerical"
                 else f"{name(node.feature_index)} == {node.category!r}?")
            print(f"{indent}[{q}]  (n={node.n_samples}, impurity={node.impurity_value:.3f})")
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
