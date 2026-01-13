from math import log2, inf

from random import shuffle

from collections import Counter, defaultdict

from statistics import mean, stdev


def load_data(path):
    data = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(line.split(","))

    return data


def fill_missing_by_class(data):
    n_attrs = len(data[0]) - 1
    by_class = defaultdict(list)

    for row in data:
        by_class[row[-1]].append(row)

    for rows in by_class.values():
        for i in range(n_attrs):
            vals = [r[i] for r in rows if r[i] != "?"]
            mode = Counter(vals).most_common(1)[0][0]

            for r in rows:
                if r[i] == "?":
                    r[i] = mode


def entropy(rows):
    counts = Counter(r[-1] for r in rows)
    total = len(rows)

    return -sum((c / total) * log2(c / total) for c in counts.values())


def information_gain(rows, attr):
    subsets = defaultdict(list)

    for r in rows:
        subsets[r[attr]].append(r)

    return entropy(rows) - sum((len(s) / len(rows)) * entropy(s) for s in subsets.values())


class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}
        self.majority = None

    def leaf(self):
        return self.label is not None


def majority_class(rows):
    return Counter(r[-1] for r in rows).most_common(1)[0][0]


def id3(rows, attributes, max_depth, min_samples, min_gain, depth=0):
    labels = [r[-1] for r in rows]
    node = Node()
    node.majority = majority_class(rows)

    if len(set(labels)) == 1:
        node.label = labels[0]

        return node

    if not attributes or len(rows) < min_samples or depth >= max_depth:
        node.label = node.majority

        return node

    best_gain, best_attr = max([(information_gain(rows, a), a) for a in attributes])

    if best_gain < min_gain:
        node.label = node.majority

        return node

    node.attribute = best_attr
    subsets = defaultdict(list)

    for r in rows:
        subsets[r[best_attr]].append(r)

    remaining = [a for a in attributes if a != best_attr]

    for val, subset in subsets.items():
        node.children[val] = id3(subset, remaining, max_depth, min_samples, min_gain, depth + 1)

    return node


def predict(node, row):
    while not node.leaf():
        val = row[node.attribute]

        if val not in node.children:
            return node.majority

        node = node.children[val]

    return node.label


def accuracy(tree, data):
    return sum(predict(tree, r) == r[-1] for r in data) / len(data)


def reduced_error_pruning(node, validation_data):
    if node.leaf():
        return

    for child in node.children.values():
        reduced_error_pruning(child, validation_data)

    curr_acc = accuracy(node, validation_data)
    curr_children, curr_attribute, curr_label = node.children, node.attribute, node.label
    node.children, node.attribute, node.label = {}, None, node.majority
    pruned_acc = accuracy(node, validation_data)

    if pruned_acc < curr_acc:
        node.children = curr_children
        node.attribute = curr_attribute
        node.label = curr_label


def stratified_split(data, ratio=0.2):
    classes = defaultdict(list)

    for r in data:
        classes[r[-1]].append(r)

    train, test = [], []

    for rows in classes.values():
        shuffle(rows)
        cut = int(len(rows) * ratio)
        train += rows[cut:]
        test += rows[:cut]

    return train, test


def stratified_fold(data, k=10):
    by_class = defaultdict(list)

    for row in data:
        by_class[row[-1]].append(row)

    folds = [[] for _ in range(k)]

    for rows in by_class.values():
        shuffle(rows)

        for i, row in enumerate(rows):
            folds[i % k].append(row)

    return folds


def evaluate(data, mode):
    shuffle(data)
    train, test = stratified_split(data)
    fill_missing_by_class(train)
    fill_missing_by_class(test)
    attributes = list(range(len(train[0]) - 1))
    mode = mode[0]
    pre, post = mode in "02", mode in "12"
    max_depth = 6 if pre else inf
    min_samples = 5 if pre else 1
    min_gain = 0.01 if pre else 0.0

    if post:
        subtrain, prune_val = stratified_split(train)
        tree = id3(subtrain, attributes, max_depth, min_samples, min_gain)
        reduced_error_pruning(tree, prune_val)

    else:
        tree = id3(train, attributes, max_depth, min_samples, min_gain)

    train_acc = accuracy(tree, train)
    folds = stratified_fold(train)
    fold_accuracies = []

    for i in range(10):
        validation = folds[i]
        training = [r for j, f in enumerate(folds) if j != i for r in f]
        t = id3(training, attributes, max_depth, min_samples, min_gain)

        if post:
            reduced_error_pruning(t, validation)

        fold_accuracies.append(accuracy(t, validation))

    avg_acc = mean(fold_accuracies)
    std_acc = stdev(fold_accuracies)
    test_acc = accuracy(tree, test)
    print("1. Train Set Accuracy:")
    print(f"Accuracy: {100 * train_acc:.2f}%\n")
    print("10-Fold Cross-Validation Results:\n")

    for i, acc in enumerate(fold_accuracies, 1):
        print(f"Accuracy Fold {i}: {100 * acc:.2f}%")

    print(f"\nAverage Accuracy: {100 * avg_acc:.2f}%")
    print(f"Standard Deviation: {100 * std_acc:.2f}%\n")
    print("2. Test Set Accuracy:")
    print(f"Accuracy: {100 * test_acc:.2f}%")


if __name__ == "__main__":
    data = load_data("breast+cancer/breast-cancer.data")
    mode = input().strip()
    evaluate(data, mode)
