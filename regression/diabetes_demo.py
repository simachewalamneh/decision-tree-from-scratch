import numpy as np
from regression_tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes  # data loading only, not modeling

RNG_SEED = 42
def manual_train_test_split(X, y, test_ratio=0.2, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def main():
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"Dataset: Diabetes (real-world) -- {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Target: disease progression one year after baseline "
          f"(range {y.min():.0f}-{y.max():.0f}, mean {y.mean():.1f})\n")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}\n")

    # 1. Baseline: predicting the training mean for everyone (R^2 = 0)
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mse = float(np.mean((y_test - baseline_pred) ** 2))
    print(f"Baseline (always predict the mean) test MSE: {baseline_mse:.1f}\n")

    # 2. Effect of max_depth (stopping criterion) on train vs test error
    print("=" * 70)
    print("Effect of max_depth (stopping criterion) on train vs test performance")
    print("=" * 70)
    for depth in [1, 2, 3, 4, 6, None]:
        reg = DecisionTreeRegressor(max_depth=depth, min_samples_split=6, min_samples_leaf=3)
        reg.fit(X_train, y_train)
        train_metrics = reg.evaluate(X_train, y_train)
        test_metrics = reg.evaluate(X_test, y_test)
        depth_label = depth if depth is not None else "None (unbounded)"
        print(f"max_depth={str(depth_label):18s} actual_depth={reg.depth():2d}  leaves={reg.n_leaves():3d}  "
              f"train_MSE={train_metrics['mse']:7.1f}  test_MSE={test_metrics['mse']:7.1f}  "
              f"test_R2={test_metrics['r2']:.3f}")

    # 3. Effect of min_samples_leaf
    print("\n" + "=" * 70)
    print("Effect of min_samples_leaf (stopping criterion) at max_depth=4")
    print("=" * 70)
    for msl in [1, 3, 5, 10, 20]:
        reg = DecisionTreeRegressor(max_depth=4, min_samples_split=2 * msl, min_samples_leaf=msl)
        reg.fit(X_train, y_train)
        train_metrics = reg.evaluate(X_train, y_train)
        test_metrics = reg.evaluate(X_test, y_test)
        print(f"min_samples_leaf={msl:3d}  leaves={reg.n_leaves():3d}  "
              f"train_MSE={train_metrics['mse']:7.1f}  test_MSE={test_metrics['mse']:7.1f}  "
              f"test_R2={test_metrics['r2']:.3f}")

    # 4. Final model + full evaluation
    print("\n" + "=" * 70)
    print("Final tree (max_depth=3, min_samples_leaf=10)")
    print("=" * 70)
    final_reg = DecisionTreeRegressor(max_depth=3, min_samples_split=20, min_samples_leaf=10)
    final_reg.fit(X_train, y_train)
    final_reg.print_tree(feature_names=feature_names)

    print("\nFull evaluation on the test set:")
    final_reg.print_evaluation(X_test, y_test)

if __name__ == "__main__":
    main()
