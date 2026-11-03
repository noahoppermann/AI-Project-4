"""Do not modify.

A tree is a float array with one row per node and five columns. The root is row 0.
A row goes to the left child when row[feature] <= threshold, and to the right
child otherwise. Every node stores the majority label of the training rows that
reached it. In a leaf, columns 0, 2 and 3 (feature, left, right) are all LEAF (-1), and
column 1 (threshold) is 0: [-1, 0, -1, -1, label].
"""
import numpy as np

FEATURE, THRESHOLD, LEFT, RIGHT, LABEL = range(5)
NUM_COLUMNS = 5
LEAF = -1


def is_leaf(tree, i):
    return int(tree[i, FEATURE]) == LEAF


def make_leaf(label):
    """The row for a leaf."""
    return [LEAF, 0.0, LEAF, LEAF, label]


def tree_size(tree):
    """The number of nodes in the tree."""
    return len(tree)


def tree_depth(tree, i=0):
    """The number of decisions on the longest path from node i to a leaf."""
    if is_leaf(tree, i):
        return 0
    return 1 + max(tree_depth(tree, int(tree[i, LEFT])), tree_depth(tree, int(tree[i, RIGHT])))


def compact(tree):
    """A copy without unreachable rows, renumbered in preorder: each node, then
    its whole left subtree, then its whole right subtree."""
    rows = []

    def visit(i):
        at = len(rows)
        rows.append(tree[i].copy())
        if not is_leaf(tree, i):
            rows[at][LEFT] = visit(int(tree[i, LEFT]))
            rows[at][RIGHT] = visit(int(tree[i, RIGHT]))
        return at

    visit(0)
    return np.array(rows, dtype=float)
