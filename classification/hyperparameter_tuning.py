import itertools
import numpy as np
from decision_tree import DecisionTreeClassifier

def stratified_k_fold_indices(y, k=5, seed=42):
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
