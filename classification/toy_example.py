import numpy as np
from decision_tree import (
    DecisionTreeClassifier, gini_impurity, entropy,
    accuracy_of_majority_prediction, weighted_impurity,
)

data = [
    ("Sunny", 85, False, "No"),
    ("Sunny", 80, True,  "No"),
    ("Overcast", 83, False, "Yes"),
    ("Rainy", 70, False, "Yes"),
    ("Rainy", 68, False, "Yes"),
    ("Rainy", 65, True,  "No"),
    ("Overcast", 64, True,  "Yes"),
    ("Sunny", 72, False, "No"),
    ("Sunny", 69, False, "Yes"),
    ("Rainy", 75, False, "Yes"),
    ("Sunny", 75, True,  "Yes"),
    ("Overcast", 72, True,  "Yes"),
    ("Overcast", 81, False, "Yes"),
    ("Rainy", 71, True,  "No"),
]
X = np.array([[r[0], r[1], r[2]] for r in data], dtype=object)
y = np.array([r[3] for r in data])
labels = list(y)

print("=" * 60)
print("Step 1: impurity of the whole dataset")
print("=" * 60)
print(f"Gini:     {gini_impurity(labels):.3f}")
print(f"Entropy:  {entropy(labels):.3f}")
print(f"Accuracy (majority baseline): {accuracy_of_majority_prediction(labels):.3f}")

print("\n" + "=" * 60)
print("Step 2: weighted Gini of a few candidate splits")
print("=" * 60)
candidates = {
    "Outlook == Sunny?":    [row[3] for row in data if row[0] == "Sunny"],
    "Outlook == Overcast?": [row[3] for row in data if row[0] == "Overcast"],
    "Windy == True?":       [row[3] for row in data if row[2] is True],
}
for name, left in candidates.items():
    condition_map = {
        "Outlook == Sunny?": lambda r: r[0] == "Sunny",
        "Outlook == Overcast?": lambda r: r[0] == "Overcast",
        "Windy == True?": lambda r: r[2] is True,
    }
    cond = condition_map[name]
    left = [row[3] for row in data if cond(row)]
    right = [row[3] for row in data if not cond(row)]
    print(f"{name:24s} weighted Gini = {weighted_impurity(left, right, 'gini'):.3f}")

print("\n" + "=" * 60)
print("Step 3: full tree (criterion=gini, max_depth=3)")
print("=" * 60)
clf = DecisionTreeClassifier(
    criterion="gini", max_depth=3, min_samples_split=2, min_samples_leaf=1,
    feature_types=["categorical", "numerical", "categorical"],
)
clf.fit(X, y)
clf.print_tree(feature_names=["Outlook", "Temperature", "Windy"])
print(f"\nTraining accuracy: {clf.score(X, y):.3f}")

print("\n" + "=" * 60)
print("Step 4: predict a new sample")
print("=" * 60)
sample = ("Rainy", 66, False)
pred = clf.predict_one(np.array(sample, dtype=object))
print(f"Sample {sample} -> predicted PlayOutside = {pred}")

print("\n" + "=" * 60)
print("Step 5: testing accuracy (leave-one-out cross-validation)")
print("=" * 60)
correct = 0
for i in range(len(y)):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_test_point = X[i]
    y_test_point = y[i]

    loo_clf = DecisionTreeClassifier(
        criterion="gini", max_depth=3, min_samples_split=2, min_samples_leaf=1,
        feature_types=["categorical", "numerical", "categorical"],
    )
    loo_clf.fit(X_train, y_train)
    pred = loo_clf.predict_one(X_test_point)
    is_correct = pred == y_test_point
    correct += is_correct
    print(f"  held out row {i:2d}: true={y_test_point:3s} predicted={pred:3s} "
          f"{'✓' if is_correct else '✗'}")

loo_accuracy = correct / len(y)
print(f"\nLeave-one-out testing accuracy: {loo_accuracy:.3f} ({correct}/{len(y)})")
print(f"(Training accuracy on the full set, for comparison, was {clf.score(X, y):.3f} "
      f"-- higher, as expected, since that tree saw every point during training.)")
