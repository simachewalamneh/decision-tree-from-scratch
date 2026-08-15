"""
Reproduces the book's own regression walkthrough exactly: Table 9.6/9.7
from Chapter 9 -- 8 users, their age, and how many days/week they engage
with an app. This lets you verify the from-scratch code against numbers
that are already in the book itself.
"""

import numpy as np
from regression_tree import DecisionTreeRegressor, mse_of_set, weighted_mse

# Table 9.6 from the book
ages = [10, 20, 30, 40, 50, 60, 70, 80]
engagement = [7, 5, 7, 1, 2, 1, 5, 4]

X = np.array([[a] for a in ages], dtype=object)
y = np.array(engagement, dtype=float)

print("=" * 60)
print("Step 1: MSE of the whole dataset (predicting the average for everyone)")
print("=" * 60)
print(f"Average label: {y.mean():.3f}")
print(f"MSE: {mse_of_set(y):.3f}")

print("\n" + "=" * 60)
print("Step 2: weighted MSE for every candidate age cutoff (Table 9.7)")
print("=" * 60)
print(f"{'Cutoff':>8s}{'Left labels':>22s}{'Right labels':>20s}{'MSE':>10s}")
for cutoff in [0, 15, 25, 35, 45, 55, 65, 75, 100]:
    # Book's convention: cutoff splits by "age < cutoff"
    left = [e for a, e in zip(ages, engagement) if a < cutoff]
    right = [e for a, e in zip(ages, engagement) if a >= cutoff]
    w = weighted_mse(left, right)
    print(f"{cutoff:>8d}{str(left):>22s}{str(right):>20s}{w:>10.3f}")
print("\n(Book reports the minimum at cutoff=35, MSE=1.983 -- matches above.)")

print("\n" + "=" * 60)
print("Step 3: full regression tree (max_depth=2)")
print("=" * 60)
reg = DecisionTreeRegressor(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                             feature_types=["numerical"])
reg.fit(X, y)
reg.print_tree(feature_names=["Age"])

print("\n" + "=" * 60)
print("Step 4: predict a new user's engagement")
print("=" * 60)
for age in [18, 42, 68]:
    pred = reg.predict_one(np.array([age], dtype=object))
    print(f"Age {age:3d} -> predicted engagement: {pred:.3f} days/week")

print("\n" + "=" * 60)
print("Step 5: testing accuracy (leave-one-out cross-validation)")
print("=" * 60)
# Same reasoning as the classification toy example: only 8 samples, so a
# single train/test split isn't reliable. LOO trains on 7 and tests on the
# 1 left out, repeated for every point.
sq_errors = []
for i in range(len(y)):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    loo_reg = DecisionTreeRegressor(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                                     feature_types=["numerical"])
    loo_reg.fit(X_train, y_train)
    pred = loo_reg.predict_one(X[i])
    true = y[i]
    sq_err = (true - pred) ** 2
    sq_errors.append(sq_err)
    print(f"  held out age={ages[i]:3d}: true={true:.1f}  predicted={pred:.3f}  "
          f"squared_error={sq_err:.3f}")

loo_mse = float(np.mean(sq_errors))
print(f"\nLeave-one-out testing MSE: {loo_mse:.3f}")
print(f"(Training MSE on the full tree, for comparison, was "
      f"{reg.evaluate(X, y)['mse']:.3f} -- lower, since that tree saw every point.)")
