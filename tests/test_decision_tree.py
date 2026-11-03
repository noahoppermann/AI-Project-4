import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("SUBMISSION_DIR"):
    sys.path.insert(0, os.path.abspath(os.environ["SUBMISSION_DIR"]))

import decision_tree as dt  # noqa: E402
import data                 # noqa: E402
from tree import FEATURE, LABEL, LEAF, LEFT, RIGHT, THRESHOLD, tree_depth, tree_size  # noqa: E402

# These tests are a sample. Grading uses more, including the edge cases.
HINT = " Check the Requirements section of the README."


def load(name):
    return data.load(os.path.join(ROOT, "data", name + ".csv"))[:2]


def H(p):
    return 0.0 if p in (0, 1) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def a(*values):
    return np.array(values, dtype=int)


# The worksheet tree, built by hand so the prediction tests do not depend on build_tree.
WORKSHEET_TREE = np.array([[0, 5.5, 1, 2, 1],
                           [LEAF, 0, LEAF, LEAF, 1],
                           [0, 7.5, 3, 4, 0],
                           [LEAF, 0, LEAF, LEAF, 0],
                           [0, 8.5, 5, 6, 0],
                           [LEAF, 0, LEAF, LEAF, 0],
                           [LEAF, 0, LEAF, LEAF, 1]], dtype=float)


class TreeTest(unittest.TestCase):
    longMessage = False

    def close(self, got, want, what):
        self.assertIsInstance(got, (int, float, np.floating, np.integer),
                              f"{what} returned {got!r}; expected a number.")
        self.assertAlmostEqual(float(got), want, places=9,
                               msg=f"{what} returned {got!r}; expected {want!r}." + HINT)

    def same(self, got, want, what):
        self.assertEqual(got, want, f"{what} gave {got!r}; expected {want!r}." + HINT)

    def same_array(self, got, want, what):
        self.assertIsInstance(got, np.ndarray, f"{what} returned {type(got).__name__}; "
                                               f"expected a NumPy array.")
        self.assertTrue(got.shape == want.shape and np.allclose(got, want),
                        f"{what} gave\n{got}\nexpected\n{want}" + HINT)


class TestEntropy(TreeTest):

    def test_entropy_values(self):
        self.close(dt.entropy(a(1, 0)), 1.0, "entropy([1, 0])")
        self.close(dt.entropy(a(1, 1, 1, 0)), H(0.75), "entropy([1, 1, 1, 0])")

    def test_entropy_of_worksheet(self):
        self.close(dt.entropy(load("worksheet")[1]), H(5 / 8), "entropy on the worksheet")

    def test_entropy_of_pure_labels(self):
        self.close(dt.entropy(a(1, 1, 1)), 0.0, "entropy([1, 1, 1])")
        self.close(dt.entropy(a(0, 0)), 0.0, "entropy([0, 0])")

    def test_information_gain(self):
        y = load("worksheet")[1]
        self.close(dt.information_gain(y, y[:4], y[4:]), H(5 / 8) - 0.5 * H(1 / 4),
                   "information_gain for length <= 5.5 on the worksheet")

    def test_majority_label(self):
        self.same(int(dt.majority_label(a(1, 1, 0))), 1, "majority_label([1, 1, 0])")
        self.same(int(dt.majority_label(a(0, 0, 1))), 0, "majority_label([0, 0, 1])")

class TestSplitting(TreeTest):

    def test_best_split_worksheet(self):
        X, y = load("worksheet")
        feature, threshold, gain = dt.best_split(X, y)
        self.same(int(feature), 0, "the feature chosen by best_split on the worksheet")
        self.close(threshold, 5.5, "the threshold chosen by best_split on the worksheet")
        self.close(gain, H(5 / 8) - 0.5 * H(1 / 4), "the gain from best_split on the worksheet")

    def test_second_feature_can_win(self):
        X = np.array([[1, 1.0], [2, 2.0], [1, 3.0], [2, 4.0]], dtype=float)
        self.same(int(dt.best_split(X, a(1, 1, 0, 0))[0]), 1, "the feature chosen by best_split")

class TestTree(TreeTest):

    def test_worksheet_tree(self):
        self.same_array(dt.build_tree(*load("worksheet")), WORKSHEET_TREE,
                        "build_tree on the worksheet")

    def test_tree_is_a_float_array_with_five_columns(self):
        tree = dt.build_tree(*load("worksheet"))
        self.assertIsInstance(tree, np.ndarray, "build_tree must return a NumPy array.")
        self.same(tree.ndim, 2, "the number of array dimensions")
        self.same(tree.shape[1], 5, "the number of columns")
        self.assertTrue(np.issubdtype(tree.dtype, np.floating), "The tree must hold floats.")

    def test_pure_labels_make_one_leaf(self):
        X = np.array([[1.0], [2.0]])
        self.same_array(dt.build_tree(X, a(1, 1)), np.array([[LEAF, 0, LEAF, LEAF, 1]], float),
                        "build_tree with pure labels")

    def test_max_depth(self):
        X, y = load("worksheet")
        self.same_array(dt.build_tree(X, y, max_depth=0),
                        np.array([[LEAF, 0, LEAF, LEAF, 1]], float), "build_tree with max_depth=0")
        tree = dt.build_tree(X, y, max_depth=1)
        self.same(tree_size(tree), 3, "the size of the tree with max_depth=1")
        self.same(int(tree[2, LABEL]), 0, "the label of the right leaf with max_depth=1")

    def test_predict_one(self):
        for row, want in [([5, 0], 1), ([6, 0], 0), ([8, 0], 0), ([9, 0], 1)]:
            self.same(int(dt.predict_one(WORKSHEET_TREE, np.array(row, float))), want,
                      f"predict_one on {row}")

    def test_predict(self):
        X, y = load("worksheet")
        self.same_array(dt.predict(WORKSHEET_TREE, X), y, "predict on the worksheet")


class TestF1(TreeTest):

    def test_f1(self):
        self.close(dt.f1_score(a(1, 1, 1, 0, 0), a(1, 1, 0, 1, 0)), 2 / 3, "f1_score")

class TestPruning(TreeTest):
    # Node 0 splits at 5; its right child splits at 7.
    TREE = np.array([[0, 5, 1, 2, 1],
                     [LEAF, 0, LEAF, LEAF, 1],
                     [0, 7, 3, 4, 0],
                     [LEAF, 0, LEAF, LEAF, 0],
                     [LEAF, 0, LEAF, LEAF, 1]], dtype=float)

    def test_useful_split_is_kept(self):
        X = np.array([[1.0], [6.0], [8.0]])
        self.same_array(dt.prune(self.TREE, X, a(1, 0, 1)), self.TREE,
                        "prune when every split helps")

    def test_harmful_split_is_removed(self):
        X = np.array([[1.0], [2.0], [6.0], [8.0]])
        want = np.array([[0, 5, 1, 2, 1], [LEAF, 0, LEAF, LEAF, 1],
                         [LEAF, 0, LEAF, LEAF, 0]], dtype=float)
        self.same_array(dt.prune(self.TREE, X, a(1, 1, 0, 0)), want,
                        "prune when the node at 7 makes an error")

if __name__ == "__main__":
    unittest.main()
