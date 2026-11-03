import numpy as np

from tree import FEATURE, LABEL, LEFT, RIGHT, THRESHOLD, compact, is_leaf, make_leaf


def entropy(y):
    """ENTROPY. y is an array of 0s and 1s."""
    raise NotImplementedError


def information_gain(y, y_left, y_right):
    """INFORMATION-GAIN."""
    raise NotImplementedError


def majority_label(y):
    """MAJORITY-LABEL. A tie goes to 0."""
    raise NotImplementedError


def best_split(X, y, min_samples_leaf=1):
    """BEST-SPLIT. Returns (feature, threshold, gain), or None."""
    raise NotImplementedError


def build_tree(X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_gain=0.0):
    """BUILD-TREE. Returns the tree as a NumPy array, in preorder."""
    raise NotImplementedError


def predict_one(tree, row, i=0):
    """PREDICT-ONE. Starts at node i."""
    raise NotImplementedError


def predict(tree, X):
    """PREDICT. Returns an integer array with one label per row of X."""
    raise NotImplementedError


def f1_score(y_true, y_pred):
    """F1-SCORE, with 1 (gamma) as the positive class."""
    raise NotImplementedError


def prune(tree, X, y):
    """PRUNE. Returns a new, smaller tree; the tree passed in is left unchanged."""
    raise NotImplementedError


# The hyperparameters used for the final model. These placeholder values are
# deliberately very poor. Replace all of them with values you have tuned.
HYPERPARAMETERS = {
    "max_depth": 1,
    "min_samples_split": 20000,
    "min_samples_leaf": 5000,
    "min_gain": 0.5,
}


def train_final_model(X, y):
    """Return the tree to be scored on the hidden test set."""
    return build_tree(X, y, **HYPERPARAMETERS)
