"""Reading the data. Do not modify."""
import csv

import numpy as np

GAMMA, HADRON = 1, 0
LABELS = {"gamma": GAMMA, "hadron": HADRON}


def load(path):
    """(X, y, feature names) from a CSV whose last column is the label.

    X is a float array with one row per example and one column per feature.
    y is an integer array holding 1 for gamma and 0 for hadron.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        records = list(reader)
    X = np.array([[float(v) for v in r[:-1]] for r in records], dtype=float)
    y = np.array([LABELS[r[-1]] for r in records], dtype=int)
    return X, y, header[:-1]
