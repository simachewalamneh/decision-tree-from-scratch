"""
Demo: train the from-scratch DecisionTreeClassifier on a real-world
multiclass dataset (the UCI Wine dataset — 3 classes, 13 numerical
chemical-analysis features, 178 samples).

We use sklearn ONLY to load the dataset (sklearn.datasets.load_wine),
never for the tree/model itself — that is 100% implemented in
decision_tree.py with no external ML library.
"""

import numpy as np
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_wine  # data loading only, not modeling

RNG_SEED = 42


def manual_train_test_split(X, y, test_ratio=0.2, seed=RNG_SEED):
    """A simple from-scratch stratified-ish shuffle split (no sklearn)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names
    print(f"Dataset: Wine (UCI) — {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes: {list(class_names)}\n")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}\n")

    # ---------------------------------------------------------------- #
    # 1. Compare the three splitting criteria (accuracy / gini / entropy)
    # ---------------------------------------------------------------- #
    print("=" * 70)
    print("Comparing splitting criteria (max_depth=4, min_samples_leaf=2)")
    print("=" * 70)
    for criterion in ["accuracy", "gini", "entropy"]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=4,
                                      min_samples_split=4, min_samples_leaf=2)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print(f"criterion={criterion:8s}  depth={clf.depth()}  leaves={clf.n_leaves()}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # 2. Show the effect of stopping criteria (max_depth) on over/underfitting
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Effect of max_depth (stopping criterion) on train vs test accuracy")
    print("=" * 70)
    for depth in [1, 2, 3, 4, 6, None]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=depth,
                                      min_samples_split=4, min_samples_leaf=1)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        depth_label = depth if depth is not None else "None (unbounded)"
        print(f"max_depth={str(depth_label):18s} actual_depth={clf.depth():2d}  "
              f"leaves={clf.n_leaves():3d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # 3. Effect of min_samples_leaf (a different stopping criterion)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Effect of min_samples_leaf (stopping criterion) at max_depth=None")
    print("=" * 70)
    for msl in [1, 2, 5, 10]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=None,
                                      min_samples_split=2 * msl, min_samples_leaf=msl)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print(f"min_samples_leaf={msl:3d}  actual_depth={clf.depth():2d}  leaves={clf.n_leaves():3d}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # 4. Print the actual tree for the best-looking configuration
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Final tree (criterion=gini, max_depth=3, min_samples_leaf=3)")
    print("=" * 70)
    final_clf = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                        min_samples_split=6, min_samples_leaf=3)
    final_clf.fit(X_train, y_train)
    final_clf.print_tree(feature_names=feature_names)
    print(f"\nFinal test accuracy: {final_clf.score(X_test, y_test):.3f}")

    # ---------------------------------------------------------------- #
    # 5. Full evaluation: confusion matrix, precision, recall, F1
    #    (accuracy alone hides which classes get confused with each
    #    other -- see README section "Why not just accuracy?")
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Full evaluation on the test set (not just accuracy)")
    print("=" * 70)
    class_name_map = {i: name for i, name in enumerate(class_names)}
    y_test_named = np.array([class_name_map[v] for v in y_test])
    y_train_named = np.array([class_name_map[v] for v in y_train])
    named_clf = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                        min_samples_split=6, min_samples_leaf=3)
    named_clf.fit(X_train, y_train_named)
    named_clf.print_evaluation(X_test, y_test_named, average="macro")


if __name__ == "__main__":
    main()
