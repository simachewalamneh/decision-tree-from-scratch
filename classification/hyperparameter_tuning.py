"""
From-scratch k-fold cross-validation and grid search for
DecisionTreeClassifier hyperparameters -- no scikit-learn's
GridSearchCV/cross_val_score, same "from scratch" spirit as the rest
of this repo. decision_tree.py itself is NOT modified by this file.

Proper procedure (unlike the informal sweeps in wine_demo.py, which
picked hyperparameters by eyeballing TEST-set accuracy -- a form of
leakage):

  1. Split off a test set ONCE, set it aside untouched.
  2. Split the remaining training data into k folds.
  3. For every candidate hyperparameter combination, train k times
     (holding out a different fold each time as validation), and
     average the k validation scores.
  4. Pick whichever combination has the best AVERAGE cross-validation
     score -- the test set is never touched for this decision.
  5. Retrain once on the FULL training set using the winning
     hyperparameters, and evaluate ONCE on the test set. That's the
     only number reported as "test accuracy".
"""

import itertools
import numpy as np
from decision_tree import DecisionTreeClassifier


def stratified_k_fold_indices(y, k=5, seed=42):
    """
    Split sample indices into k folds, keeping each class's proportion
    roughly equal across folds (plain random folds can accidentally
    leave a class out of a fold entirely on small/imbalanced data).
    Returns a list of k arrays of indices (the fold's VALIDATION indices).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    fold_indices = [[] for _ in range(k)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        # distribute this class's indices round-robin across the k folds
        for i, idx in enumerate(cls_idx):
            fold_indices[i % k].append(idx)

    return [np.array(sorted(f)) for f in fold_indices]


def cross_validate(param_dict, X, y, k=5, seed=42):
    """
    Train/evaluate k times with the given hyperparameters, holding out
    a different fold each time. Returns (mean_accuracy, std_accuracy).
    """
    folds = stratified_k_fold_indices(y, k=k, seed=seed)
    n = len(y)
    all_idx = np.arange(n)
    scores = []

    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx)
        clf = DecisionTreeClassifier(**param_dict)
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[val_idx], y[val_idx]))

    return float(np.mean(scores)), float(np.std(scores))


def grid_search(param_grid, X, y, k=5, seed=42, verbose=True):
    """
    param_grid: dict of {param_name: [candidate values]}, e.g.
        {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5],
         "min_samples_leaf": [1, 3, 5]}

    Tries every combination (Cartesian product), cross-validates each
    with cross_validate(), and returns (best_params, all_results),
    where all_results is a list of (params, mean_acc, std_acc) sorted
    best-first. NEVER touches a test set -- X, y here should be the
    TRAINING data only.
    """
    keys = list(param_grid.keys())
    value_lists = [param_grid[k_] for k_ in keys]
    results = []

    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        mean_acc, std_acc = cross_validate(params, X, y, k=k, seed=seed)
        results.append((params, mean_acc, std_acc))
        if verbose:
            param_str = ", ".join(f"{k_}={v}" for k_, v in params.items())
            print(f"  {param_str:60s}  CV acc = {mean_acc:.3f} ± {std_acc:.3f}")

    results.sort(key=lambda r: r[1], reverse=True)
    best_params = results[0][0]
    return best_params, results
