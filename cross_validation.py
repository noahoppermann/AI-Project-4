"""k-fold cross-validation, provided for tuning. Do not modify."""
import numpy as np

import decision_tree as dt


def k_fold_indices(n, k, seed=0):
    """A list of k (train_indices, test_indices) pairs."""
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


def cross_val_f1(X, y, k=5, seed=0, **params):
    """The mean held-out F1 over k folds. params are passed on to build_tree."""
    scores = []
    for train, test in k_fold_indices(len(y), k, seed):
        tree = dt.build_tree(X[train], y[train], **params)
        scores.append(dt.f1_score(y[test], dt.predict(tree, X[test])))
    return float(np.mean(scores))


def cross_val_final_model(X, y, k=5, seed=0):
    """The mean held-out F1 over k folds, training with train_final_model."""
    scores = []
    for train, test in k_fold_indices(len(y), k, seed):
        tree = dt.train_final_model(X[train], y[train])
        scores.append(dt.f1_score(y[test], dt.predict(tree, X[test])))
    return float(np.mean(scores))
