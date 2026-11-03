# Assignment 4: Decision Trees

In this assignment you will implement a decision tree classifier that uses entropy to choose its splits, stores the tree in a NumPy array, and classifies telescope images of particle showers. You will then control overfitting by tuning the model's hyperparameters through trial and error, comparing your attempts with cross-validation, until your model reaches the benchmark F1 score on a hidden test set. The placeholder hyperparameters provided in the starter code perform very poorly, so you must replace them.

**To score 100, your model must have F1 ≥ 0.890, and all implementation tests pass.**

## Setup

The assignment uses the same conda environment as the previous assignments. If that environment is already installed, no further setup is required.

```
conda env create -f environment.yml
conda activate ai-search
```

## Files

| File | |
| --- | --- |
| `decision_tree.py` | Your code. |
| `main.py` | Click Run to evaluate your model. Also runs your code on the worksheet. |
| `tests/` | Sample tests. Grading uses more. |
| `environment.yml` | The course environment. |
| `data/` | `worksheet.csv`, small enough to check by hand, and `train.csv` (14,905 examples). The 4,000-example test set is held by the grader. |
| `tree.py`, `data.py`, `cross_validation.py` | The array layout, the data reader and cross-validation. Do not modify. |

## Tasks

1. Implement `entropy`, `information_gain` and `majority_label`.
2. Implement `best_split`.
3. Implement `build_tree`, `predict_one` and `predict`.
4. Implement `f1_score`.
5. Implement `prune`.
6. Open `main.py` and click Run in your editor, or run `python main.py`, to evaluate the placeholder hyperparameters. Compare training F1 with cross-validated F1.
7. Tune the hyperparameters by trial and error, as described under Tuning your model. Record your chosen values in `HYPERPARAMETERS`, and submit to CodeGrade until your model reaches the benchmark.

## The dataset

The data is the MAGIC Gamma Telescope dataset (UCI Machine Learning Repository). A ground-based telescope records the light produced when high-energy particles strike the atmosphere. The examples are Monte Carlo simulations of such events. Each must be classified as a **gamma** ray, which is the signal of scientific interest, or a **hadron**, which is background from cosmic rays.

Each example has 10 numeric features describing the elliptical image recorded by the camera:

| Feature | Meaning |
| --- | --- |
| `length`, `width` | Major and minor axes of the ellipse (mm) |
| `size` | Base-10 logarithm of the total brightness of the image |
| `conc`, `conc1` | Share of the brightness held by the two brightest pixels, and by the brightest pixel |
| `asym` | Distance from the brightest pixel to the centre of the image, along the major axis |
| `m3long`, `m3trans` | Cube root of the third moment along the major and minor axes |
| `alpha` | Angle between the major axis and the direction to the centre of the camera (degrees) |
| `dist` | Distance from the centre of the camera to the centre of the image (mm) |

`data.load` returns `X`, a float array with one row per example and one column per feature, `y`, an integer array holding 1 for gamma and 0 for hadron, and the feature names.

## Background

### Entropy and information gain

Entropy measures how mixed a set of labels is. If a proportion *p* of the labels are 1,

```
H = −p log₂ p − (1 − p) log₂ (1 − p)
```

where a term with a proportion of 0 is defined to be 0. Entropy is 0 for a set with only one label and reaches its maximum, 1 bit, for an even mixture.

A split divides the examples into a left and a right group. Its **information gain** is the entropy before the split minus the weighted average entropy after it, where each group is weighted by its share of the examples. The tree chooses the split with the highest gain.

### Median splits

Every split in this assignment uses the **median**, computed over the rows that reach the current node. For each feature, there is exactly one candidate split: rows with a value less than or equal to the feature's median go left, and all other rows go right. The tree compares the information gain of these candidates, one per feature, and chooses the highest. A feature whose values are mostly identical can leave one side empty, in which case that feature is not a candidate.

### Overfitting

A tree grown until its leaves are pure memorizes the noise in its training data. An unrestricted tree on this dataset reaches a training F1 near 1.0 and a much lower F1 on unseen data. Two families of technique reduce this effect:

- **Pre-pruning** stops growth early, through `max_depth`, `min_samples_split`, `min_samples_leaf` and `min_gain`.
- **Reduced-error pruning** grows a tree, then works upward from the leaves using examples set aside from training. A subtree is replaced by a leaf whenever the leaf makes no more errors than the subtree on those examples.

### F1 score

Gamma events form about 65% of the data, so labelling every example as gamma already achieves an accuracy of 65%. F1 is more informative. With gamma as the positive class,

```
F1 = 2 TP / (2 TP + FP + FN)
```

where TP is the number of gamma events labelled gamma, FP is the number of hadrons labelled gamma, and FN is the number of gamma events labelled hadron.

### Cross-validation

Selecting hyperparameters by their score on the test set would make that score an overestimate. Instead, *k*-fold cross-validation divides the training data into *k* folds. Each fold in turn is held out while a tree is trained on the remaining folds, and the *k* held-out F1 scores are averaged. The hyperparameters with the best average are then used to train on all of the training data. Cross-validation is provided in `cross_validation.py`, and `python main.py try` uses it, so you do not implement it. This estimate is less dependent on one particular division of the data than a single held-out set, at the cost of training *k* trees for each group of hyperparameters.

## The tree array

A tree is a float NumPy array with one row per node and five columns. The root is row 0. `tree.py` names the columns:

| Column | Constant | Decision node | Leaf |
| --- | --- | --- | --- |
| 0 | `FEATURE` | column of `X` to test | `LEAF`−1 |
| 1 | `THRESHOLD` | the median | 0 |
| 2 | `LEFT` | row of the left child | `LEAF`−1 |
| 3 | `RIGHT` | row of the right child | `LEAF`−1 |
| 4 | `LABEL` | majority label at this node | majority label at this node |

In a leaf, columns 0, 2 and 3 are all −1, the value of the constant `LEAF`, and column 1 is 0. A leaf therefore has the form `[-1, 0, -1, -1, label]`.

A row of `X` goes left when `row[feature] <= threshold`. `build_tree` must number the nodes in **preorder**: each node, then its whole left subtree, then its whole right subtree.

Because a child's row number is not known until the subtree before it has been grown, the simplest approach is to collect rows in a Python list and convert it with `np.array(nodes, dtype=float)` at the end. `tree.py` provides `make_leaf(label)`, `is_leaf(tree, i)`, `tree_size(tree)`, `tree_depth(tree)` and `compact(tree)`. The last of these returns a copy in preorder without unreachable rows.

## Pseudocode

```
function ENTROPY(y) returns a number of bits
    if y is empty then return 0
    p ← proportion of y equal to 1
    if p = 0 or p = 1 then return 0
    return −p log₂ p − (1 − p) log₂ (1 − p)

function INFORMATION-GAIN(y, y-left, y-right) returns a number of bits
    return ENTROPY(y) − (|y-left| / |y|) ENTROPY(y-left) − (|y-right| / |y|) ENTROPY(y-right)

function MAJORITY-LABEL(y) returns a label
    if there are more 1s than 0s in y then return 1
    return 0

function BEST-SPLIT(X, y, min-samples-leaf) returns (feature, threshold, gain), or null
    best, best-gain ← null, 0
    for each feature f, in order, do
        t ← the median of column f of X
        y-left ← labels of the rows where X[f] ≤ t
        y-right ← labels of the rows where X[f] > t
        if |y-left| < min-samples-leaf or |y-right| < min-samples-leaf then continue
        g ← INFORMATION-GAIN(y, y-left, y-right)
        if g > best-gain then
            best, best-gain ← (f, t, g), g
    return best

function BUILD-TREE(X, y, max-depth, min-samples-split, min-samples-leaf, min-gain) returns a tree
    nodes ← an empty list
    GROW(X, y, 0)
    return nodes as a float array

    function GROW(X, y, depth) returns a row number
        at ← |nodes|
        append a leaf with label MAJORITY-LABEL(y) to nodes
        if y contains only one label, or depth = max-depth, or |y| < min-samples-split then
            return at
        split ← BEST-SPLIT(X, y, min-samples-leaf)
        if split is null or its gain < min-gain then return at
        (f, t, gain) ← split
        set FEATURE and THRESHOLD of nodes[at] to f and t
        LEFT of nodes[at] ← GROW(rows where X[f] ≤ t, their labels, depth + 1)
        RIGHT of nodes[at] ← GROW(rows where X[f] > t, their labels, depth + 1)
        return at

function PREDICT-ONE(tree, row, i) returns a label
    while node i is not a leaf do
        if row[FEATURE of i] ≤ THRESHOLD of i then i ← LEFT of i
        else i ← RIGHT of i
    return LABEL of i

function PREDICT(tree, X) returns an integer array
    return PREDICT-ONE(tree, row, 0) for each row of X

function F1-SCORE(y-true, y-pred) returns a number
    if TP = 0 then return 0
    return 2 TP / (2 TP + FP + FN)

function PRUNE(tree, X, y) returns a tree
    copy ← a copy of tree
    VISIT(0, X, y)
    return COMPACT(copy)

    function VISIT(i, X, y)
        if node i is a leaf then return
        VISIT(LEFT of i, the rows that go left, their labels)
        VISIT(RIGHT of i, the rows that go right, their labels)
        subtree-errors ← number of rows where PREDICT-ONE(copy, row, i) ≠ its label
        leaf-errors ← number of labels in y that differ from LABEL of i
        if leaf-errors ≤ subtree-errors then replace node i with a leaf with the same label
```

`|y|` means `len(y)`.

## Worked example

`data/worksheet.csv` has eight examples:

| length | width | label |
| --- | --- | --- |
| 2 | 1 | gamma |
| 3 | 2 | gamma |
| 4 | 1 | gamma |
| 5 | 4 | gamma |
| 6 | 5 | hadron |
| 7 | 2 | hadron |
| 8 | 6 | hadron |
| 9 | 5 | gamma |

Compute the following values by hand before you begin implementing, and confirm that your implementation reproduces them:

- `entropy(y)` = −(5/8) log₂(5/8) − (3/8) log₂(3/8) ≈ 0.9544
- The median of `length` is 5.5. The left group has four gamma; the right group has one gamma and three hadrons. The gain is 0.9544 − (4/8)(0) − (4/8)(0.8113) ≈ 0.5488.
- The median of `width` is 3. The left group has three gamma and one hadron; the right group has two of each. The gain is 0.9544 − (4/8)(0.8113) − (4/8)(1) ≈ 0.0488.
- The root therefore tests `length <= 5.5`.

The complete tree has seven nodes:

```
[[ 0.   5.5  1.   2.   1. ]
 [-1.   0.  -1.  -1.   1. ]
 [ 0.   7.5  3.   4.   0. ]
 [-1.   0.  -1.  -1.   0. ]
 [ 0.   8.5  5.   6.   0. ]
 [-1.   0.  -1.  -1.   0. ]
 [-1.   0.  -1.  -1.   1. ]]
```

Node 4 receives one gamma and one hadron, so its stored label is 0 by the tie rule. `python main.py worksheet` prints these values.

## Requirements

The implementation tests check exact values, so follow the pseudocode exactly:

- Use `math.log2` or `np.log2`, so that entropy is measured in bits.
- Compute the median from the rows at the current node, not from the whole dataset, and use `<=` for the left side.
- Consider one split per feature, at its median. Do not search other thresholds.
- Try features in column order, and keep the first on a tie (`>`, not `≥`). Return `None` when no split has positive gain.
- A split must leave at least `min_samples_leaf` rows on each side.
- Stop when `gain < min_gain`; a gain equal to `min_gain` still splits.
- Store trees as float NumPy arrays in the layout above, numbered in preorder.
- `f1_score` treats 1 as the positive class and returns 0 when there are no true positives.
- `prune` must not modify the tree passed to it. On equal errors, it keeps the leaf. A subtree that no rows reach becomes a leaf.
- Call your own functions. In particular, `build_tree` should use `best_split`, and `predict` should use `predict_one`.

## Tuning your model

`HYPERPARAMETERS`, a dictionary near the end of `decision_tree.py`, holds the values used to train your final model. The starter code provides placeholder values that are deliberately very poor, and each of the four can prevent the tree from growing, so all of them must be changed. An unrestricted tree also earns no performance points, because it overfits. Adjust the values by trial and error:

| Hyperparameter | Effect of increasing it |
| --- | --- |
| `max_depth` | Permits a deeper, more complex tree |
| `min_samples_split` | Stops splitting nodes with few examples, giving a simpler tree |
| `min_samples_leaf` | Requires more examples on each side of a split, giving a simpler tree |
| `min_gain` | Rejects splits that provide little information, giving a simpler tree |

To test a combination of values, enter them in `HYPERPARAMETERS`, then open `main.py` and click Run in your editor, or run `python main.py`. This evaluates `train_final_model`, including any pruning you have added.

You may also test a change without editing the file. `python main.py try` starts from `HYPERPARAMETERS` and applies any values you type:

```
python main.py try --max-depth 5 --min-samples-leaf 10
```

Both report training F1 and cross-validated F1. A large gap between them indicates overfitting, while low values of both indicate that the model is too simple. Base your decisions on the cross-validated F1, change your values in response, and repeat. You are not required to automate this process.

CodeGrade uses the values in `HYPERPARAMETERS` when you submit, so make sure that they are your final choice. You may also apply `prune` in `train_final_model`, in which case the examples used for pruning must be set aside from training.

### `train_final_model(X, y)`

CodeGrade calls `train_final_model` with all of `train.csv`, then scores the returned tree on the hidden test set using your `predict`. The function is provided, and you may modify it, subject to these rules:

- It must use your `build_tree`, so that every split is at a median, and return a single tree. Ensembles are not permitted.
- It must finish within 300 seconds.
- Only NumPy and the Python standard library may be used.

## Testing

The tests in `tests/` check the correctness of your tree and functions. They do not measure your model's F1 score or award performance points; CodeGrade grades both correctness and performance. The local tests are a representative sample. CodeGrade uses a more extensive hidden test suite that checks every rule in Requirements, so passing the sample tests does not guarantee full implementation marks. Identify the cases they do not cover and verify your implementation's behaviour on them. Consider, in particular, constant features, ties, hyperparameters at their limits, and nodes that no validation rows reach.

```
pytest                              # the sample tests
pytest -k TestEntropy               # one part: also TestSplitting, TestTree, TestF1, TestPruning
python main.py worksheet            # the worked example
python main.py                      # evaluate your model with HYPERPARAMETERS; the same as clicking Run
python main.py try                  # the same scores for the values in HYPERPARAMETERS, using build_tree only
python main.py try --max-depth 5 --min-samples-leaf 10
```

CodeGrade reports the hidden test score each time you submit. Repeatedly adjusting a model in response to that score would make it an overestimate for the same reason described under Cross-validation, so base your tuning decisions on the cross-validated F1. The cross-validated F1 can differ from the hidden test score.

## Submission

Submit `decision_tree.py` to CodeGrade. You may submit any number of times. For each submission, CodeGrade trains a tree using your `train_final_model` and `HYPERPARAMETERS`, labels the hidden test set using your `predict`, and reports the resulting F1 score.

## Grading

| Component | Points |
| --- | --- |
| Entropy and gain | 10 |
| Median splitting | 10 |
| Tree and prediction | 15 |
| F1 score | 5 |
| Pruning | 10 |
| **Implementation subtotal** | **50** |
| Performance | 50 |
| **Total** | **100** |

Performance is awarded in ten levels of 5 points each, according to the F1 of your final tree on the hidden test set:

| Hidden test F1 at least | 0.845 | 0.850 | 0.855 | 0.860 | 0.865 | 0.870 | 0.875 | 0.880 | 0.885 | **0.890** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Performance points | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | **50** |

Code that special-cases a dataset, or that uses the test set in any way, earns no credit for that part.

## Reference

Bock, R. (2004). MAGIC Gamma Telescope [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
