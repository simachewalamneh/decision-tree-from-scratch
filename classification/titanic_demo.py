
import numpy as np
import seaborn as sns  # data loading only, not modeling
from decision_tree import DecisionTreeClassifier

RNG_SEED = 42


def manual_train_test_split(X, y, test_ratio=0.2, seed=RNG_SEED):
    """Same from-scratch shuffle split used in wine_demo.py."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    df = sns.load_dataset("titanic")
    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[cols + ["survived"]].dropna()  # drop rows with missing age/embarked

    feature_names = cols
    # pclass, sex, embarked are categorical; age, sibsp, parch, fare are numerical
    feature_types = ["categorical", "categorical", "numerical",
                      "numerical", "numerical", "numerical", "categorical"]

    X = df[cols].to_numpy(dtype=object)
    # keep pclass as a genuine category (not compared numerically), so cast to str
    X[:, 0] = X[:, 0].astype(str)
    y = df["survived"].to_numpy()
    class_names = ["died", "survived"]

    print(f"Dataset: Titanic passengers — {X.shape[0]} samples, {X.shape[1]} features "
          f"(mixed numeric + categorical), 2 classes: {class_names}")
    print(f"Feature types: {dict(zip(feature_names, feature_types))}\n")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}\n")

    # 1. Compare the three splitting criteria
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

    # 2. Effect of max_depth
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

        # 3. Final tree -- this is where categorical splits actually show up
    print("\n" + "=" * 70)
    print("Final tree (criterion=gini, max_depth=3, min_samples_leaf=5)")
    print("=" * 70)
    final_clf = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                        min_samples_split=10, min_samples_leaf=5,
                                        feature_types=feature_types)
    final_clf.fit(X_train, y_train)
    final_clf.print_tree(feature_names=feature_names)
    print(f"\nFinal test accuracy: {final_clf.score(X_test, y_test):.3f}")

    # 4. Full evaluation
    print("\n" + "=" * 70)
    print("Full evaluation on the test set (not just accuracy)")
    print("=" * 70)
    class_name_map = {0: "died", 1: "survived"}
    y_test_named = np.array([class_name_map[v] for v in y_test])
    y_train_named = np.array([class_name_map[v] for v in y_train])
    named_clf = DecisionTreeClassifier(criterion="gini", max_depth=3,
                                        min_samples_split=10, min_samples_leaf=5,
                                        feature_types=feature_types)
    named_clf.fit(X_train, y_train_named)
    named_clf.print_evaluation(X_test, y_test_named, average="macro")


if __name__ == "__main__":
    main()
