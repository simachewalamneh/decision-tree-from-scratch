"""
Interactive GUI for the from-scratch decision tree project.

Run with:  streamlit run app.py

Lets you pick a classification or regression dataset, tune the same
stopping-criteria hyperparameters used throughout this repo, and see
BOTH the text output (accuracy/MSE, evaluation metrics) and a real
rendered tree diagram (not just the ASCII print_tree() version).

Imports decision_tree.py / regression_tree.py from the classification/
and regression/ folders unchanged -- this file adds a GUI on top, it
does not modify the underlying algorithm at all.
"""

import sys
import os
import io
import contextlib

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classification"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "regression"))

from decision_tree import DecisionTreeClassifier          # noqa: E402
from regression_tree import DecisionTreeRegressor          # noqa: E402
from tree_plot import plot_tree                             # noqa: E402

st.set_page_config(page_title="Decision Trees From Scratch", layout="wide")


# --------------------------------------------------------------------------- #
# Dataset loaders (mirrors the *_demo.py / toy_*.py scripts in the repo)
# --------------------------------------------------------------------------- #

def load_toy_classification():
    data = [
        ("Sunny", 85, False, "No"), ("Sunny", 80, True, "No"),
        ("Overcast", 83, False, "Yes"), ("Rainy", 70, False, "Yes"),
        ("Rainy", 68, False, "Yes"), ("Rainy", 65, True, "No"),
        ("Overcast", 64, True, "Yes"), ("Sunny", 72, False, "No"),
        ("Sunny", 69, False, "Yes"), ("Rainy", 75, False, "Yes"),
        ("Sunny", 75, True, "Yes"), ("Overcast", 72, True, "Yes"),
        ("Overcast", 81, False, "Yes"), ("Rainy", 71, True, "No"),
    ]
    X = np.array([[r[0], r[1], r[2]] for r in data], dtype=object)
    y = np.array([r[3] for r in data])
    feature_names = ["Outlook", "Temperature", "Windy"]
    feature_types = ["categorical", "numerical", "categorical"]
    return X, y, feature_names, feature_types


def load_wine_dataset():
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data.astype(object), d.target, list(d.feature_names), None  # None -> auto-detect (all numeric)


def load_titanic_dataset():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[cols + ["survived"]].dropna()
    X = df[cols].to_numpy(dtype=object)
    X[:, 0] = X[:, 0].astype(str)
    y = df["survived"].to_numpy()
    feature_types = ["categorical", "categorical", "numerical",
                      "numerical", "numerical", "numerical", "categorical"]
    return X, y, cols, feature_types


def load_penguins_dataset():
    import seaborn as sns
    df = sns.load_dataset("penguins").dropna()
    feature_names = ["island", "bill_length_mm", "bill_depth_mm",
                      "flipper_length_mm", "body_mass_g", "sex"]
    feature_types = ["categorical", "numerical", "numerical",
                      "numerical", "numerical", "categorical"]
    X = df[feature_names].to_numpy(dtype=object)
    y = df["species"].to_numpy()
    return X, y, feature_names, feature_types


def load_toy_regression():
    ages = [10, 20, 30, 40, 50, 60, 70, 80]
    engagement = [7, 5, 7, 1, 2, 1, 5, 4]
    X = np.array([[a] for a in ages], dtype=object)
    y = np.array(engagement, dtype=float)
    return X, y, ["Age"], ["numerical"]


def load_diabetes_dataset():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return d.data.astype(object), d.target, list(d.feature_names), None


CLASSIFICATION_DATASETS = {
    "Toy (Play Outside — 3 features, tiny)": load_toy_classification,
    "Wine (real, numeric-only, multiclass)": load_wine_dataset,
    "Titanic (real, mixed features, binary)": load_titanic_dataset,
    "Penguins (real, mixed features, multiclass)": load_penguins_dataset,
}
REGRESSION_DATASETS = {
    "Toy (Age -> App Engagement, book's own example)": load_toy_regression,
    "Diabetes (real, 10 features, continuous target)": load_diabetes_dataset,
}


def manual_train_test_split(X, y, test_ratio, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

st.sidebar.title("Decision Trees From Scratch")
task = st.sidebar.radio("Task", ["Classification", "Regression"])

if task == "Classification":
    dataset_label = st.sidebar.selectbox("Dataset", list(CLASSIFICATION_DATASETS.keys()))
    criterion = st.sidebar.selectbox("Splitting criterion", ["gini", "entropy", "accuracy"])
else:
    dataset_label = st.sidebar.selectbox("Dataset", list(REGRESSION_DATASETS.keys()))

max_depth_choice = st.sidebar.slider("max_depth", min_value=1, max_value=10, value=3)
unbounded_depth = st.sidebar.checkbox("Unbounded depth (ignore slider)", value=False)
min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=30, value=6)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=15, value=3)
test_ratio = st.sidebar.slider("Test set fraction", min_value=0.1, max_value=0.4, value=0.2)

run = st.sidebar.button("Run", type="primary")


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #

st.title(" Decision Tree From-Scratch")
st.caption("Same decision_tree.py / regression_tree.py used throughout the repo — "
           "this page just adds a GUI on top, no algorithm changes.")

if not run:
    st.info("Set your options in the sidebar and click **Run**.")
    st.stop()

max_depth = None if unbounded_depth else max_depth_choice

if task == "Classification":
    loader = CLASSIFICATION_DATASETS[dataset_label]
    X, y, feature_names, feature_types = loader()
    class_names = sorted(set(y.tolist()))

    st.subheader(f"Dataset: {dataset_label}")
    st.write(f"**{X.shape[0]} samples**, **{X.shape[1]} features**, "
             f"**{len(class_names)} classes**: {class_names}")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio)

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  feature_types=feature_types)
    clf.fit(X_train, y_train)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train accuracy", f"{clf.score(X_train, y_train):.3f}")
    col2.metric("Test accuracy", f"{clf.score(X_test, y_test):.3f}")
    col3.metric("Tree depth", clf.depth())
    col4.metric("Leaves", clf.n_leaves())

    tab1, tab2, tab3 = st.tabs(["🌳 Tree diagram", "📄 Text tree", "📊 Evaluation metrics"])

    with tab1:
        fig = plot_tree(clf.root_, feature_names=feature_names,
                         title=f"{dataset_label} — {criterion}, max_depth={max_depth_choice if not unbounded_depth else '∞'}")
        st.pyplot(fig)

    with tab2:
        text_output, _ = capture_stdout(clf.print_tree, feature_names=feature_names)
        st.code(text_output, language=None)

    with tab3:
        text_output, results = capture_stdout(clf.print_evaluation, X_test, y_test, average="macro")
        st.code(text_output, language=None)

else:
    loader = REGRESSION_DATASETS[dataset_label]
    X, y, feature_names, feature_types = loader()

    st.subheader(f"Dataset: {dataset_label}")
    st.write(f"**{X.shape[0]} samples**, **{X.shape[1]} features**, target range "
             f"[{y.min():.1f}, {y.max():.1f}]")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio)

    reg = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, feature_types=feature_types)
    reg.fit(X_train, y_train)

    test_metrics = reg.evaluate(X_test, y_test)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test MSE", f"{test_metrics['mse']:.2f}")
    col2.metric("Test RMSE", f"{test_metrics['rmse']:.2f}")
    col3.metric("Test R²", f"{test_metrics['r2']:.3f}")
    col4.metric("Tree depth", reg.depth())

    tab1, tab2, tab3 = st.tabs(["🌳 Tree diagram", "📄 Text tree", "📊 Evaluation metrics"])

    with tab1:
        fig = plot_tree(reg.root_, feature_names=feature_names,
                         title=f"{dataset_label} — max_depth={max_depth_choice if not unbounded_depth else '∞'}")
        st.pyplot(fig)

    with tab2:
        text_output, _ = capture_stdout(reg.print_tree, feature_names=feature_names)
        st.code(text_output, language=None)

    with tab3:
        text_output, results = capture_stdout(reg.print_evaluation, X_test, y_test)
        st.code(text_output, language=None)
