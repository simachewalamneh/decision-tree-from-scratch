"""
Proper hyperparameter tuning for the Wine dataset: k-fold cross-validation
on the TRAINING set only to pick hyperparameters, then a single, honest
evaluation on the untouched test set.

(For why this matters -- and what NOT to do -- see the "Doing
hyperparameter tuning properly" section of the README: picking
hyperparameters by eyeballing test-set accuracy across several values,
then reporting that same test set's score, is a form of leakage. This
script only performs the proper procedure.)

Same DecisionTreeClassifier from decision_tree.py, unchanged.
"""

import numpy as np
from decision_tree import DecisionTreeClassifier
from hyperparameter_tuning import grid_search, cross_validate
from sklearn.datasets import load_wine

RNG_SEED = 42


def manual_train_test_split(X, y, test_ratio=0.2, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)} "
          f"(test set will be touched exactly ONCE, at the very end)\n")

    # ---------------------------------------------------------------- #
    # (a) Cross-validation grid search on TRAINING data only
    # ---------------------------------------------------------------- #
    print("=" * 70)
    print("(a) Cross-validation grid search (5-fold, training set only)")
    print("=" * 70)
    # Grid kept small deliberately -- an expanded grid (adding "accuracy" as
    # a criterion and sweeping min_samples_split) was tested separately and
    # took 7.8x longer (61.0s vs 7.8s) for an IDENTICAL result (same best
    # combination's effective settings, same 0.914 final test accuracy).
    # See HYPERPARAMETER_SEARCH_COMPARISON.md for the full comparison.
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 3, 5],
    }
    best_params, all_results = grid_search(param_grid, X_train, y_train, k=5, seed=RNG_SEED)

    print(f"\n  Best hyperparameters found via CV: {best_params}")
    print(f"  Their CV accuracy: {all_results[0][1]:.3f} ± {all_results[0][2]:.3f}")
    print(f"  (chosen from {len(all_results)} combinations, using ONLY training data)\n")

    # ---------------------------------------------------------------- #
    # (b) The ONE honest evaluation -- test set touched for the first time
    # ---------------------------------------------------------------- #
    print("=" * 70)
    print("(b) Final, single evaluation on the untouched test set")
    print("=" * 70)
    final_clf = DecisionTreeClassifier(**best_params)
    final_clf.fit(X_train, y_train)
    final_test_acc = final_clf.score(X_test, y_test)
    print(f"  Retrained on full training set with: {best_params}")
    print(f"  Test accuracy (first and only time touching the test set): {final_test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # (c) Print the actual tree for the CV-selected configuration
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("(c) Tree structure for the CV-selected hyperparameters")
    print("=" * 70)
    final_clf.print_tree(feature_names=list(data.feature_names))

    # ---------------------------------------------------------------- #
    # (d) Full evaluation: confusion matrix, precision, recall, F1
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("(d) Full evaluation on the test set (not just accuracy)")
    print("=" * 70)
    class_name_map = {i: name for i, name in enumerate(data.target_names)}
    y_test_named = np.array([class_name_map[v] for v in y_test])
    y_train_named = np.array([class_name_map[v] for v in y_train])
    named_clf = DecisionTreeClassifier(**best_params)
    named_clf.fit(X_train, y_train_named)
    named_clf.print_evaluation(X_test, y_test_named, average="macro")


if __name__ == "__main__":
    main()
