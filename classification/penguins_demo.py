"""
Demo: train the from-scratch DecisionTreeClassifier on a real-world
dataset that is BOTH multiclass AND has a genuine mix of numerical and
categorical features -- neither wine_demo.py (multiclass, numeric-only)
nor titanic_demo.py (mixed features, binary) covers this combination on
its own.

Dataset: Palmer Penguins (via seaborn, sourced from the public
seaborn-data repository -- data loading only, not modeling).

Target: species (Adelie / Chinstrap / Gentoo) -- 3 classes.

Features:
  Categorical: island (Torgersen / Biscoe / Dream), sex
  Numerical:   bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

Same DecisionTreeClassifier from decision_tree.py, completely unchanged.
"""

import numpy as np
import seaborn as sns  # data loading only, not modeling
from decision_tree import DecisionTreeClassifier

RNG_SEED = 42


def manual_train_test_split(X, y, test_ratio=0.2, seed=RNG_SEED):
    """Same from-scratch shuffle split used in wine_demo.py / titanic_demo.py."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    df = sns.load_dataset("penguins").dropna()  # drop rows with missing measurements/sex

    feature_names = ["island", "bill_length_mm", "bill_depth_mm",
                      "flipper_length_mm", "body_mass_g", "sex"]
    feature_types = ["categorical", "numerical", "numerical",
                      "numerical", "numerical", "categorical"]

    X = df[feature_names].to_numpy(dtype=object)
    y = df["species"].to_numpy()
    class_names = sorted(df["species"].unique())

    print(f"Dataset: Palmer Penguins — {X.shape[0]} samples, {X.shape[1]} features "
          f"(mixed numeric + categorical), {len(class_names)} classes: {class_names}")
    print(f"Feature types: {dict(zip(feature_names, feature_types))}\n")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}\n")

    # ---------------------------------------------------------------- #
    # 1. Compare the three splitting criteria
    # ---------------------------------------------------------------- #
    print("=" * 70)
    print("Comparing splitting criteria (max_depth=4, min_samples_leaf=2)")
    print("=" * 70)
    for criterion in ["accuracy", "gini", "entropy"]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=4,
                                      min_samples_split=4, min_samples_leaf=2,
                                      feature_types=feature_types)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print(f"criterion={criterion:8s}  depth={clf.depth()}  leaves={clf.n_leaves()}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # 2. Effect of max_depth
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Effect of max_depth (stopping criterion) on train vs test accuracy")
    print("=" * 70)
    for depth in [1, 2, 3, 4, 6, None]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=depth,
                                      min_samples_split=4, min_samples_leaf=1,
                                      feature_types=feature_types)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        depth_label = depth if depth is not None else "None (unbounded)"
        print(f"max_depth={str(depth_label):18s} actual_depth={clf.depth():2d}  "
              f"leaves={clf.n_leaves():3d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # ---------------------------------------------------------------- #
    # 3. Final tree
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Final tree (criterion=gini, max_depth=3, min_samples_leaf=5)")
    print("=" * 70)
    final_clf = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                        min_samples_split=10, min_samples_leaf=5,
                                        feature_types=feature_types)
    final_clf.fit(X_train, y_train)
    final_clf.print_tree(feature_names=feature_names)
    print(f"\nFinal test accuracy: {final_clf.score(X_test, y_test):.3f}")

    # ---------------------------------------------------------------- #
    # 4. Full evaluation (3-class confusion matrix + per-class P/R/F1)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Full evaluation on the test set (not just accuracy)")
    print("=" * 70)
    final_clf.print_evaluation(X_test, y_test, average="macro")


if __name__ == "__main__":
    main()
