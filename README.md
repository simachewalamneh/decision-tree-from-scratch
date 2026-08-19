# Decision Trees — Concept Walkthrough and From-Scratch Implementation

Based on Chapter 9 ("Splitting data by asking questions: Decision trees") of
Luis Serrano's *Grokking Machine Learning*, with a fully worked toy example
and the actual implementation code it maps to.

## Repo structure

The two problem domains are kept in separate folders — everything
classification-related is self-contained under `classification/`, and
everything regression-related under `regression/`. Each demo script
imports its module from the same folder, so nothing needs to be run from
the repo root.

```
decision-tree-from-scratch/
├── README.md
├── requirements.txt
├── app.py                       # interactive GUI (Streamlit) -- dataset/hyperparameter picker + visual tree
├── tree_plot.py                 # shared matplotlib tree-diagram renderer, used by app.py
├── classification/
│   ├── decision_tree.py         # from-scratch DecisionTreeClassifier
│   ├── toy_example.py           # reproduces every classification number in this README
│   ├── wine_unTune_demo.py      # real multiclass dataset (UCI Wine, numeric-only)
│   ├── titanic_demo.py          # real dataset with mixed numeric + categorical features (binary)
│   ├── penguins_demo.py         # real dataset: multiclass AND mixed numeric + categorical
│   ├── hyperparameter_tuning.py # from-scratch k-fold CV + grid search (does not modify decision_tree.py)
│   ├── wine_tuning_demo.py      # proper CV tuning vs. the old "eyeball the test set" approach
│   ├── HYPERPARAMETER_SEARCH_COMPARISON.md  # small vs. large grid: same result, 7.8x slower
│   └── outputs/
│       ├── outputs_classification_toy.txt
│       ├── outputs_classification_wine_untune.txt
│       ├── outputs_classification_titanic.txt
│       ├── outputs_classification_penguins.txt
│       └── outputs_classification_wine_tuning.txt
└── regression/
    ├── regression_tree.py       # from-scratch DecisionTreeRegressor
    ├── toy_regression_example.py # reproduces the book's own Table 9.6/9.7
    ├── diabetes_demo.py         # real-world dataset (Diabetes progression)
    └── outputs/
        ├── outputs_regression_toy.txt
        └── outputs_regression_diabetes.txt
```

```bash
pip install -r requirements.txt

cd classification
python toy_example.py    # classification concept walkthrough
python wine_unTune_demo.py      # real multiclass classification

cd ../regression
python toy_regression_example.py   # regression walkthrough (matches the book exactly)
python diabetes_demo.py            # real regression, with stopping-criteria experiments
```

## How to run (full steps)

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

This creates an isolated Python environment in a `venv/` folder so the
project's packages don't clash with anything else on your system.

### 2. Activate it

Linux/macOS:
```bash
source venv/bin/activate
```

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

Your prompt should now show `(venv)` at the start of the line.

### 3. Install requirements

```bash
pip install -r requirements.txt
```

Installs `numpy` (used by both `decision_tree.py` and `regression_tree.py`)
and `scikit-learn` (used only to load the Wine and Diabetes datasets —
never for the tree models themselves, which are 100% from scratch).

### 4. Run the classification scripts

```bash
cd classification
python toy_example.py    # concept walkthrough — reproduces every number in section 2-7 above
python wine_unTune_demo.py      # real multiclass dataset (UCI Wine), criteria + stopping-criteria comparisons
python titanic_demo.py   # real dataset with mixed numeric + categorical features (binary)
python penguins_demo.py  # real dataset: multiclass AND mixed numeric + categorical
python wine_tuning_demo.py    # proper hyperparameter tuning via cross-validation
```

### 5. Run the regression scripts

```bash
cd ../regression
python toy_regression_example.py   # regression walkthrough — reproduces the book's own Table 9.6/9.7
python diabetes_demo.py            # real-world dataset (Diabetes), stopping-criteria experiments
```

### 6. When you're done

```bash
deactivate
```
Exits the virtual environment back to your normal shell.

### 7. Or, skip the scripts and use the GUI instead

```bash
streamlit run app.py
```

Opens an interactive browser page where you can pick any dataset
(toy/Wine/Titanic/Penguins for classification, toy/Diabetes for
regression), tune `max_depth` / `min_samples_split` / `min_samples_leaf` /
splitting criterion with sliders, and see three views of the result:

- **Tree diagram** — an actual rendered tree (boxes and branches via
  matplotlib), not just the ASCII `print_tree()` output
- **Text tree** — the same ASCII output as the scripts, for direct
  comparison
- **Evaluation metrics** — accuracy/MSE, confusion matrix, precision/
  recall/F1 (classification) or MSE/RMSE/MAE/R² (regression)

`app.py` and `tree_plot.py` only *display* results from the exact same
`decision_tree.py` / `regression_tree.py` used everywhere else in this
repo — no algorithm changes.

---

## 1. What a decision tree is

A decision tree makes a prediction the same way a person does: it asks a
sequence of yes/no questions and follows whichever branch fits, until it
reaches an answer.

```
Outlook == Overcast?
├─ Yes → Play
└─ No  → (ask another question...)
```

**Vocabulary:**

| Term | Meaning |
|---|---|
| Root node | The first question, at the top of the tree |
| Decision node | Any node that still asks a question (has two children) |
| Leaf node | A node with no more questions — it holds the final prediction |
| Branch | The "yes" or "no" edge coming out of a decision node |
| Depth | Number of questions on the longest path from root to leaf |

---

## 2. Toy dataset (3 features)

To keep the concept concrete, here's a small "should I play outside today?"
dataset with three features — one categorical with 3 classes, one numerical,
one categorical binary — and a binary label.

| Outlook | Temperature | Windy | PlayOutside |
|---|---|---|---|
| Sunny | 85 | False | No |
| Sunny | 80 | True | No |
| Overcast | 83 | False | Yes |
| Rainy | 70 | False | Yes |
| Rainy | 68 | False | Yes |
| Rainy | 65 | True | No |
| Overcast | 64 | True | Yes |
| Sunny | 72 | False | No |
| Sunny | 69 | False | Yes |
| Rainy | 75 | False | Yes |
| Sunny | 75 | True | Yes |
| Overcast | 72 | True | Yes |
| Overcast | 81 | False | Yes |
| Rainy | 71 | True | No |

14 samples, 9 `Yes` and 5 `No`. In code, this is just:

```python
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
```

---

## 3. How "pure" is a set? Gini and entropy

Before picking a question, we need a number that says how mixed a set of
labels is.

**Gini impurity:**

```
Gini = 1 - Σ pᵢ²
```

where `pᵢ` is the fraction of samples with label *i*. `Gini = 0` means the
set is pure (all one label); it grows toward `1` as the set gets more mixed.

**Entropy:**

```
Entropy = -Σ pᵢ log₂(pᵢ)
```

Same idea, different formula — `0` for pure, `1` for a 50/50 split of two
classes.

For the whole 14-row dataset (9 Yes, 5 No): `p_Yes = 9/14`, `p_No = 5/14`.

```python
def gini_impurity(labels):
    n = len(labels)
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * np.log2(p)
    return ent
```

Running this on the whole dataset:

```
Gini:     0.459
Entropy:  0.940
Accuracy (majority-class baseline): 0.643   # predicting "Yes" for everyone
```

This is the *before* number. Every candidate split is judged by how much it
lowers this number.

---

## 4. Judging a split: weighted impurity

A split breaks the dataset into a left and right child. Because the two
children are usually different sizes, we don't just average their impurities
— we weight by how many samples land in each branch:

```
weighted_impurity = (n_left / n_total) * impurity(left) + (n_right / n_total) * impurity(right)
```

```python
def weighted_impurity(left_labels, right_labels, criterion):
    n_left, n_right = len(left_labels), len(right_labels)
    n_total = n_left + n_right
    return (n_left / n_total) * impurity(left_labels, criterion) + \
           (n_right / n_total) * impurity(right_labels, criterion)
```

**Categorical features** (`Outlook`, `Windy`): each category becomes its own
yes/no question — e.g. "Is Outlook == Overcast?" splits into
`{Overcast}` vs. `{Sunny, Rainy}`.

**Numerical features** (`Temperature`): every midpoint between consecutive
sorted values becomes a candidate cutoff — e.g. "Is Temperature ≤ 70.5?"

Trying a few candidate splits on our toy data (Gini shown):

| Candidate question | Weighted Gini |
|---|---|
| Outlook == Sunny? | 0.394 |
| **Outlook == Overcast?** | **0.357** |
| Windy == True? | 0.429 |
| Temperature ≤ 70.5? | 0.432 |
| Temperature ≤ 83? | 0.396 |

`Outlook == Overcast?` gives the lowest weighted Gini (0.357), so it wins —
that becomes the root question. This matches exactly what the actual
`_best_split` function in the code picks:

```python
best = clf._best_split(X, y)
# -> (feature_index=0, "categorical", "Overcast", ..., gain=0.102)
```

`gain = parent_impurity - weighted_impurity = 0.459 - 0.357 = 0.102`. This
gain is what the algorithm maximizes at every node (equivalently: it
minimizes weighted impurity, since the parent's impurity is a fixed number
during that node's search).

---

## 5. Building the whole tree: recursion

Once the best question for a node is found, its two branches become two new
leaf nodes containing the corresponding subset of samples. We then treat
each of *those* as a new node and repeat the same search — unless a
stopping condition says not to (see below).

```python
def _build_node(self, X, y, depth):
    node = Node(depth)
    node.prediction = Counter(y).most_common(1)[0][0]   # majority label
    node.impurity_value = impurity(y, self.criterion)

    if node.impurity_value == 0:                 # already pure -> leaf
        return node
    if depth >= self.max_depth:                  # stopping condition: depth
        return node
    if len(y) < self.min_samples_split:           # stopping condition: too few samples
        return node

    best = self._best_split(X, y)
    if best is None or best_gain < self.min_impurity_decrease:  # stopping condition: gain too small
        return node

    node.is_leaf = False
    node.left  = self._build_node(X[left_mask],  y[left_mask],  depth + 1)
    node.right = self._build_node(X[~left_mask], y[~left_mask], depth + 1)
    return node
```

Running this on the toy dataset (`criterion="gini", max_depth=3`) produces:

```
[Outlook == 'Overcast'?]  (n=14, impurity=0.459)
├─ Yes:
│   Leaf: predict=Yes  (n=4, impurity=0.000)
└─ No:
    [Temperature <= 77.500?]  (n=10, impurity=0.500)
    ├─ Yes:
    │   [Temperature <= 66.500?]  (n=8, impurity=0.469)
    │   ├─ Yes:
    │   │   Leaf: predict=No  (n=1, impurity=0.000)
    │   └─ No:
    │       Leaf: predict=Yes  (n=7, impurity=0.408)
    └─ No:
        Leaf: predict=No  (n=2, impurity=0.000)
```

Read in plain English: *If it's overcast, play. Otherwise, if it's
77.5°F or cooler, and warmer than 66.5°F, play; if it's 66.5°F or colder,
don't play; if it's warmer than 77.5°F, don't play.* Training accuracy on
this tiny dataset: **0.857**.

**Testing accuracy on this toy set:** with only 14 rows, a normal
train/test split would test on just 2-3 points — not reliable. Instead,
`toy_example.py` runs **leave-one-out cross-validation**: train on 13
samples, test on the 1 left out, repeat for every sample, then average.
That gives a genuine held-out accuracy of **0.500** — much lower than the
0.857 training accuracy, which is expected: a tree built from only 13
points is unstable, and this gap is itself a live example of why the
stopping criteria in section 6 matter (an unconstrained tree fits its
training data very well but generalizes poorly on small data).

---

## 6. Stopping criteria — why the tree doesn't grow forever

Left unchecked, a tree keeps splitting until every leaf is pure — which
usually means leaves with 1–2 samples that memorize noise instead of
learning a pattern (overfitting). The book gives four stopping conditions;
all four are implemented as constructor parameters:

| Condition | Parameter | What it checks | Why it helps |
|---|---|---|---|
| Minimum gain | `min_impurity_decrease` | Is the best available split's improvement too small? | Skips splits that don't meaningfully help |
| Minimum samples to split | `min_samples_split` | Does this node have enough samples to trust a split decision? | A tiny node's "best feature" is often just noise |
| Minimum samples per leaf | `min_samples_leaf` | Would either resulting child be too small? | Stops leaves that memorize a handful of points |
| Maximum depth | `max_depth` | Has the tree already gone this deep? | Directly caps how complex the tree can get |

```python
def __init__(self, criterion="gini", max_depth=5, min_samples_split=2,
             min_samples_leaf=1, min_impurity_decrease=0.0, feature_types=None):
    ...
```

Each one is checked in `_build_node` / `_best_split` before a split is
allowed to happen — see the four commented checks in the code above.

---

## 7. Making a prediction

To predict a new sample, walk down from the root, following whichever
branch its feature values satisfy, until you hit a leaf — then return that
leaf's stored (majority) label:

```python
def predict_one(self, x):
    node = self.root_
    while not node.is_leaf:
        if node.feature_type == "numerical":
            go_left = float(x[node.feature_index]) <= node.threshold
        else:
            go_left = x[node.feature_index] == node.category
        node = node.left if go_left else node.right
    return node.prediction
```

For example, `("Rainy", 66, False)`: Outlook isn't Overcast → go right.
Temperature 66 ≤ 77.5 → go left. Temperature 66 ≤ 66.5 → go left →
leaf predicts `No`.

## 8. Why not just accuracy? Other evaluation metrics

Accuracy alone can hide real problems, especially with imbalanced classes.
Two more useful tools, implemented in `evaluate()` / `print_evaluation()`:

**Confusion matrix** — a table of true label (rows) vs. predicted label
(columns). The diagonal is correct predictions; everything off-diagonal
shows exactly which classes get confused with which.

**Precision, recall, F1 (per class)**:

```
precision = TP / (TP + FP)   # of everything predicted as this class, how much was right?
recall    = TP / (TP + FN)   # of everything that truly is this class, how much was caught?
F1        = 2 * precision * recall / (precision + recall)
```

For multiclass problems these are computed per class, then combined as a
**macro average** (unweighted mean — treats every class equally) or a
**weighted average** (weighted by how many true samples each class has).

On the Wine test set, for example, one class had perfect recall (1.000)
but lower precision (0.875) — meaning the model never *missed* that class,
but it occasionally mislabeled samples from the other two classes as it.
Accuracy alone (0.943) would never reveal that asymmetry — this is exactly
why a real evaluation reports more than one number.

---

## 8.5 A dataset with both feature types — Titanic

Wine is 100% numerical features, so `wine_unTune_demo.py` never actually
exercises the **categorical** branch of `_best_split` (the "is feature ==
category?" question described in section 4). `titanic_demo.py` uses a
real-world dataset that genuinely mixes both feature types, so both
splitting branches show up in one tree:

| Feature | Type |
|---|---|
| `pclass` (1st/2nd/3rd class) | categorical |
| `sex` | categorical |
| `embarked` (port: S/C/Q) | categorical |
| `age`, `sibsp`, `parch`, `fare` | numerical |

Target: `survived` (0/1), 712 passengers with complete records (rows with
missing `age`/`embarked` dropped). Loaded via `seaborn.load_dataset` for
data access only — same `decision_tree.py`, completely unchanged.

The resulting tree (`max_depth=3, min_samples_leaf=5`) mixes both
question types naturally:

```
[sex == 'female'?]
├─ Yes: [pclass == '3'?] → [fare <= 20.800?] → ...
└─ No:  [pclass == '1'?] → [age <= 53.000?] → ...
```

The root split is `sex == 'female'?` — the tree independently rediscovers
the "women and children first" evacuation pattern directly from the data,
which is a nice concrete example of a categorical split actually mattering
(rather than being a code path that's implemented but never demonstrated).
Test accuracy: **0.789** — lower than Wine's 0.943, which makes sense:
survival on the Titanic is a genuinely noisier, harder problem than
separating three wine cultivars by chemistry. Note that `survived` is
binary (2 classes) — for a dataset that is both multiclass *and* mixed
feature types, see the next section.

## 8.6 Multiclass AND mixed feature types — Palmer Penguins

Wine is multiclass but numeric-only. Titanic is mixed-feature but binary.
`penguins_demo.py` closes that gap with a real dataset that is **both**:
3-class target, genuine mix of categorical and numerical features.

| Feature | Type |
|---|---|
| `island` (Torgersen/Biscoe/Dream) | categorical |
| `sex` | categorical |
| `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` | numerical |

Target: `species` — **Adelie / Chinstrap / Gentoo** (3 classes). 333
penguins with complete records. Loaded via `seaborn.load_dataset` for
data access only — same `decision_tree.py`, completely unchanged.

The final tree (`max_depth=3, min_samples_leaf=5`) mixes numeric cutoffs
with a categorical split, and achieves **0.985 test accuracy**:

```
[flipper_length_mm <= 206.500?]
├─ Yes: [bill_length_mm <= 43.350?] → Adelie / Chinstrap
└─ No:  [island == 'Biscoe'?] → Gentoo / Chinstrap
```

The 3×3 confusion matrix on the test set is nearly diagonal (only 1 of 66
samples misclassified), giving per-class precision/recall of 0.92–1.00
across all three species — a clean example of the full evaluation
machinery (section 8) applied to a genuinely multiclass, mixed-feature
problem in one place.

---

## 8.7 Doing hyperparameter tuning properly

Every earlier demo (`wine_unTune_demo.py`'s `max_depth`/`min_samples_leaf`
sweeps) picked hyperparameters by training several trees and looking at
whichever got the **highest test-set accuracy**. That's an informal
sweep, not real tuning — and it has a real problem: using the test set
to *choose* a hyperparameter, then reporting that same test set's score
as "the" accuracy, is a form of leakage. The reported number is
optimistically biased toward whichever setting happened to do best on
that one specific 35-row test split.

**Proper procedure**, implemented from scratch in
`hyperparameter_tuning.py` (does not modify `decision_tree.py`):

1. Split off a test set once; set it aside untouched.
2. Split the training data into k folds (`stratified_k_fold_indices` —
   keeps each class's proportion roughly equal per fold).
3. For every candidate hyperparameter combination, train k times
   (holding out a different fold each time), average the k validation
   scores — this is `cross_validate()`.
4. `grid_search()` tries every combination in a parameter grid and
   picks whichever has the best **average cross-validation score** —
   the test set is never touched for this decision.
5. Retrain once on the full training set with the winning
   hyperparameters, and evaluate **once** on the test set.

`wine_tuning_demo.py` performs only the proper procedure (a-d):

| Step | Result |
|---|---|
| (a) 5-fold CV grid search (36 combinations — criterion ∈ {gini, entropy}, max_depth ∈ 1-6, min_samples_leaf ∈ {1,3,5} — training data only) | best = `{criterion: entropy, max_depth: 4, min_samples_leaf: 1}`, CV acc=0.923 ± 0.041 |
| (b) Final, single honest test evaluation | test_acc=**0.914** |
| (c) Tree structure for the CV-selected config | printed via `print_tree()` |
| (d) Full evaluation (confusion matrix, precision/recall/F1) | printed via `print_evaluation()` |

For comparison, `wine_unTune_demo.py`'s earlier informal `max_depth` sweep
picked `max_depth=3` by looking directly at test accuracy across 6
values, reporting **0.943** — a number inflated by having used the test
set 6 times to choose. `wine_tuning_demo.py`'s honest number is **0.914**,
lower, because it was never used for selection. That gap is itself the
concrete argument for why the CV procedure matters, even though the
demo script itself no longer runs the biased comparison directly.

One result worth flagging from the full evaluation (d): `class_2` has
perfect precision (1.000) but only 0.625 recall — the CV-winning
`min_samples_leaf=1` setting makes the tree very cautious about that
class, missing 3 of 8 true `class_2` samples. A real trade-off the CV
procedure surfaced, not something to gloss over.

**Why this grid, and not a bigger one:** a 270-combination grid was also
tested — adding `accuracy` as a third criterion and sweeping
`min_samples_split` across 5 values. It took **7.8x longer** (61.0s vs
7.8s) but converged on the exact same result: `accuracy` never
outperformed `gini`/`entropy` here, `min_samples_split` was never the
limiting stopping condition, and the final test accuracy was identical
(0.914). See `HYPERPARAMETER_SEARCH_COMPARISON.md` for the full
comparison — we use the smaller, faster grid going forward since it
reaches the same answer.

---

## 9. From here to the real task

This toy example is deliberately tiny so every number can be checked by
hand — run `toy_example.py` to reproduce every number above from scratch.
The same `decision_tree.py` code, unchanged, is what's used on the real
multiclass dataset (UCI Wine — 178 samples, 13 numerical features, 3
classes) in `wine_unTune_demo.py`, which compares the three splitting criteria and
shows overfitting as `max_depth` grows.

Regression trees reuse this exact same structure: the only change is
swapping impurity (Gini/entropy/accuracy) for **mean squared error** as
the split-quality metric, and each leaf's prediction becomes the
**average label** instead of the majority label. The rest of this README
covers that in the same step-by-step way.

---

## 10. Regression trees

### 10.1 What changes from classification

The book states it directly: *"the algorithm used for training a
regression decision tree is very similar to the one used for training a
classification decision tree. The only difference is that for
classification trees, we used accuracy, Gini index, or entropy, and for
regression trees, we use the mean square error (MSE)."*

| | Classification | Regression |
|---|---|---|
| Splitting metric | accuracy / Gini / entropy | mean squared error (MSE) |
| Leaf prediction | majority label | **average** label |
| Stopping criteria | max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease | **identical** — same four criteria |
| Numerical feature splits | try every cutoff between sorted values | **identical** procedure |

Everything else — the recursive `_build_node` structure, the weighted
combination of left/right children, the four stopping conditions — is
unchanged. This is why `regression_tree.py` mirrors `decision_tree.py`
almost line for line.

### 10.2 MSE — the regression "impurity"

```
MSE(set) = average of (label - mean_of_set)²
```

`MSE = 0` means every label in the set is identical (a "pure" set, the
regression analogue of a pure classification leaf). Just like Gini/entropy,
splits are judged by **weighted average MSE** of the two children:

```python
def mse_of_set(labels):
    avg = labels.mean()
    return float(np.mean((labels - avg) ** 2))

def weighted_mse(left_labels, right_labels):
    n_left, n_right = len(left_labels), len(right_labels)
    n_total = n_left + n_right
    return (n_left / n_total) * mse_of_set(left_labels) + \
           (n_right / n_total) * mse_of_set(right_labels)
```

### 10.3 Worked example — straight from the book (Table 9.6/9.7)

The book's own regression example: 8 users, one feature (age), and how
many days per week they engage with an app.

| Age | Engagement (days/week) |
|---|---|
| 10 | 7 |
| 20 | 5 |
| 30 | 7 |
| 40 | 1 |
| 50 | 2 |
| 60 | 1 |
| 70 | 5 |
| 80 | 4 |

Whole-set MSE (predicting the average, 4.0, for everyone): **5.250**.

Trying every candidate age cutoff and computing weighted MSE for each
(exactly Table 9.7 in the book):

| Cutoff | Left labels | Right labels | Weighted MSE |
|---|---|---|---|
| 15 | {7} | {5,7,1,2,1,5,4} | 3.964 |
| 25 | {7,5} | {7,1,2,1,5,4} | 3.917 |
| **35** | **{7,5,7}** | **{1,2,1,5,4}** | **1.983** |
| 45 | {7,5,7,1} | {2,1,5,4} | 4.250 |
| 55 | {7,5,7,1,2} | {1,5,4} | 4.983 |
| 65 | {7,5,7,1,2,1} | {5,4} | 5.167 |

Cutoff 35 gives the lowest weighted MSE — that's the root question,
matching the book's result exactly. `toy_regression_example.py`
reproduces this table from the code (not by hand) and confirms the same
1.983 minimum.

The resulting tree (`max_depth=2`):

```
[Age <= 35.000?]  (n=8, mse=5.250)
├─ Yes:
│   [Age <= 15.000?]  (n=3, mse=0.889)
│   ├─ Yes:
│   │   Leaf: predict=7.000  (n=1, mse=0.000)
│   └─ No:
│       Leaf: predict=6.000  (n=2, mse=1.000)
└─ No:
    [Age <= 65.000?]  (n=5, mse=2.640)
    ├─ Yes:
    │   Leaf: predict=1.333  (n=3, mse=0.222)
    └─ No:
        Leaf: predict=4.500  (n=2, mse=0.250)
```

This matches the book's own scikit-learn result (Figure 9.26): cutoffs at
35, 15, and 65, with leaf predictions 7, 6, 1.33, 4.5.

**Testing accuracy on this toy set:** same reasoning as the classification
toy example — 8 samples is too small for a normal train/test split, so
`toy_regression_example.py` uses **leave-one-out cross-validation**
instead: train on 7, test on the 1 left out, repeat for all 8. That gives
a genuine held-out MSE of **6.587**, versus a training MSE of only
**0.396** — the tree fits its training data far better than it predicts
unseen users, the same overfitting signature seen in the classification
example.

### 10.4 Regression evaluation metrics

Accuracy/precision/recall don't apply to continuous predictions.
`evaluate()` in `regression_tree.py` instead reports:

```
MSE  = average squared error                (penalizes large errors heavily)
RMSE = sqrt(MSE)                            (same units as the target — easier to interpret)
MAE  = average absolute error                (less sensitive to outliers than MSE)
R²   = 1 - (SS_residual / SS_total)          (fraction of variance explained;
                                               1.0 = perfect, 0.0 = no better
                                               than always predicting the mean)
```

### 10.5 Real-world dataset — Diabetes progression

`diabetes_demo.py` trains the same, unchanged `DecisionTreeRegressor` on
the real-world **Diabetes dataset** (442 patients, 10 numerical features —
age, sex, BMI, blood pressure, six blood serum measurements; target: a
quantitative measure of disease progression one year later). Loaded via
`sklearn.datasets.load_diabetes` for data access only — the regressor
itself is still 100% from scratch.

**Effect of `max_depth` (stopping criterion) — the same overfitting curve
seen in classification:**

| max_depth | train MSE | test MSE | test R² |
|---|---|---|---|
| 1 | 4216.7 | 4651.4 | 0.129 |
| 2 | 3349.3 | 3908.9 | 0.268 |
| 3 | 2926.9 | 3697.6 | 0.308 |
| 4 | 2537.1 | **3487.1** | **0.347** |
| 6 | 1652.8 | 3813.9 | 0.286 |
| unbounded | **737.3** | 4102.2 | 0.232 |

Training error keeps dropping as depth grows unbounded (737.3 — the tree
is memorizing individual patients), but test error gets *worse* past
depth 4. This is concrete evidence for why `max_depth` matters, not just
a theoretical claim.

**Effect of `min_samples_leaf` at `max_depth=4`:**

| min_samples_leaf | test MSE | test R² |
|---|---|---|
| 1 | 3749.3 | 0.298 |
| 3 | 3487.1 | 0.347 |
| **5** | **3011.0** | **0.436** |
| 10 | 3165.4 | 0.407 |
| 20 | 3179.4 | 0.405 |

Forcing each leaf to hold at least 5 patients (instead of 1) noticeably
*improves* test R² — smaller leaves were overfitting to individual
patients' noise.

The final tree (`max_depth=3, min_samples_leaf=10`) splits first on
**BMI**, then on blood serum measurements — which lines up with BMI being
a well-established diabetes risk factor, giving the tree's structure a
sensible real-world interpretation beyond just its accuracy numbers.
Final test-set evaluation: **MSE=3192, RMSE=56.5, MAE=45.7, R²=0.402**.