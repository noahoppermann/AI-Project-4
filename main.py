import argparse
import os

import numpy as np

import cross_validation as cv
import data
import decision_tree as dt
from tree import FEATURE, LABEL, LEFT, RIGHT, THRESHOLD, is_leaf, tree_depth, tree_size

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {data.GAMMA: "gamma", data.HADRON: "hadron"}


def path(name):
    return os.path.join(HERE, "data", name + ".csv")


def show_tree(tree, features, i=0, indent=""):
    if is_leaf(tree, i):
        print(f"{indent}predict {NAMES[int(tree[i, LABEL])]}")
        return
    name, threshold = features[int(tree[i, FEATURE])], tree[i, THRESHOLD]
    print(f"{indent}if {name} <= {threshold:g}:")
    show_tree(tree, features, int(tree[i, LEFT]), indent + "    ")
    print(f"{indent}else:")
    show_tree(tree, features, int(tree[i, RIGHT]), indent + "    ")


def worksheet(args):
    X, y, features = data.load(path("worksheet"))
    print(f"entropy of all rows: {dt.entropy(y):.4f}\n")
    print("feature  median   gain")
    for f, name in enumerate(features):
        m = float(np.median(X[:, f]))
        gain = dt.information_gain(y, y[X[:, f] <= m], y[X[:, f] > m])
        print(f"{name:8} {m:6.2f} {gain:6.4f}")
    tree = dt.build_tree(X, y)
    print("\nThe tree as an array (feature, threshold, left, right, label):")
    print(tree)
    print()
    show_tree(tree, features)


def report(tree, X, y, cross_validated, k):
    print(f"nodes: {tree_size(tree)}, depth: {tree_depth(tree)}")
    print(f"training F1: {dt.f1_score(y, dt.predict(tree, X)):.4f}")
    print(f"{k}-fold cross-validated F1: {cross_validated:.4f}")


def evaluate(args=None):
    """Evaluate train_final_model, with the values in HYPERPARAMETERS."""
    X, y, _ = data.load(path("train"))
    print(f"HYPERPARAMETERS = {dt.HYPERPARAMETERS}")
    report(dt.train_final_model(X, y), X, y, cv.cross_val_final_model(X, y), 5)


def try_hyperparameters(args):
    """Start from HYPERPARAMETERS, and apply any values given on the command line."""
    X, y, _ = data.load(path("train"))
    params = dict(dt.HYPERPARAMETERS)
    for name in ["max_depth", "min_samples_split", "min_samples_leaf", "min_gain"]:
        if getattr(args, name) is not None:
            params[name] = getattr(args, name)
    print(f"hyperparameters: {params}")
    report(dt.build_tree(X, y, **params), X, y, cv.cross_val_f1(X, y, k=args.k, **params), args.k)


def main():
    parser = argparse.ArgumentParser()
    # With no command, as when clicking Run, evaluate the model.
    parser.set_defaults(fn=evaluate)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("worksheet").set_defaults(fn=worksheet)
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    t = sub.add_parser("try")
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--max-depth", type=int)
    t.add_argument("--min-samples-split", type=int)
    t.add_argument("--min-samples-leaf", type=int)
    t.add_argument("--min-gain", type=float)
    t.set_defaults(fn=try_hyperparameters)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
