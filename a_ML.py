"""
kNN
from __future__ import annotations

from math import dist

from random import shuffle

from statistics import stdev

import matplotlib.pyplot as plt

from collections import defaultdict

from heapq import heappush, heapreplace

Point = tuple[float]


class KDTree:
    def __init__(self, d: int = 3, *points: Point) -> None:
        self.root = self.left = self.right = None
        self.__d = d

        for p in points:
            self.insert(p)

    @property
    def d(self) -> int:
        return self.__d

    def insert(self, p: Point) -> KDTree:
        def dfs(rt, i):
            if rt.root is None:
                rt.root = p

                return

            if p[i % self.d] < rt.root[i % self.d]:
                if rt.left is None:
                    rt.left = KDTree(self.d, p)

                else:
                    dfs(rt.left, i + 1)

            else:
                if rt.right is None:
                    rt.right = KDTree(self.d, p)

                else:
                    dfs(rt.right, i + 1)

        dfs(self, 0)

        return self

    def kNN(self, point: Point, k: int) -> set[Point]:
        best = []

        def search(tree, depth):
            if tree is None or tree.root is None:
                return

            axis = depth % tree.d
            p = tree.root

            if p != point:
                d = dist(point, p)

                if len(best) < k:
                    heappush(best, (-d, p))

                elif d < -best[0][0]:
                    heapreplace(best, (-d, p))

            primary, secondary = (tree.left, tree.right) if point[axis] < p[axis] else (tree.right, tree.left)
            search(primary, depth + 1)

            if len(best) < k or abs(point[axis] - p[axis]) < -best[0][0]:
                search(secondary, depth + 1)

        search(self, 0)

        return {node for (_, node) in best}


class KNNClassifier:
    def __init__(self, k: int, d: int):
        self.k = k
        self.d = d
        self.tree = None
        self.labels = {}

    def fit(self, dataset: list[tuple[Point, str]]):
        self.tree = KDTree(self.d, *[p for (p, _) in dataset])
        self.labels = {p: label for (p, label) in dataset}

    def predict(self, x: Point) -> str:
        neighbors = self.tree.kNN(x, self.k)
        votes = defaultdict(int)

        for n in neighbors:
            votes[self.labels[n]] += 1

        return max(votes, key=votes.get)

    def accuracy(self, dataset: list[tuple[Point, str]]) -> float:
        correct = 0

        for x, true_label in dataset:
            correct += self.predict(x) == true_label

        return correct / len(dataset)


def cross_validate(dataset, k, d):
    shuffle(dataset)
    fold_size = len(dataset) // 10
    accuracies = []
    clf = KNNClassifier(k, d)

    for i in range(10):
        test = dataset[i * fold_size:(i + 1) * fold_size]
        train = dataset[:i * fold_size] + dataset[(i + 1) * fold_size:]
        clf.fit(train)
        accuracies.append(clf.accuracy(test) * 100)

    return accuracies


if __name__ == "__main__":
    dataset = []

    with open(f"iris\\iris.data", "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")
            label = parts.pop()
            features = tuple(map(float, parts))
            dataset.append((features, label))

    k = int(input())
    d = len(dataset[0][0])
    clf = KNNClassifier(k, d)
    clf.fit(dataset)
    train_acc = clf.accuracy(dataset) * 100
    cv_acc = cross_validate(dataset.copy(), k, d)
    cv_mean = sum(cv_acc) / 10
    cv_std = stdev(cv_acc)
    shuffle(dataset)
    split = int(0.8 * len(dataset))
    train = dataset[:split]
    test = dataset[split:]
    clf2 = KNNClassifier(k, d)
    clf2.fit(train)
    test_acc = clf2.accuracy(test) * 100
    print("1. Train Set Accuracy:")
    print(f"Accuracy: {train_acc:.2f}%\n")
    print("2. 10-Fold Cross-Validation Results:\n")

    for i, a in enumerate(cv_acc, 1):
        print(f"Accuracy Fold {i}: {a:.2f}%")

    print(f"Average Accuracy: {cv_mean:.2f}%")
    print(f"Standard Deviation: {cv_std:.2f}%\n")
    print("3. Test Set Accuracy:")
    print(f"Accuracy: {test_acc:.2f}%")
    accuracies = []

    for k in range(20):
        cv = cross_validate(dataset.copy(), k + 1, d)
        accuracies.append(sum(cv) / len(cv))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 21), accuracies, marker='o')
    plt.title("Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("% Accuracy")
    plt.grid()
    plt.show()



NBC
from math import log

from random import shuffle, seed

from statistics import mean, stdev

from collections import defaultdict, Counter


def load_data(path):
    X, y = [], []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            y.append(parts.pop(0))
            X.append(parts)

    return X, y


def stratified_split(X, y):
    class_indices = defaultdict(list)

    for i, label in enumerate(y):
        class_indices[label].append(i)

    train_idx, test_idx = [], []

    for indices in class_indices.values():
        shuffle(indices)
        split = .8 * len(indices)
        train_idx += indices[:split]
        test_idx += indices[split:]

    return [X[i] for i in train_idx], [y[i] for i in train_idx], [X[i] for i in test_idx], [y[i] for i in test_idx]


def fill_missing_by_class(X, y):
    res = [row.copy() for row in X]
    n = len(X[0])

    for cls in {"republican", "democrat"}:
        indices = [i for i, label in enumerate(y) if label == cls]

        for j in range(n):
            values = [X[i][j] for i in indices if X[i][j] != "?"]

            if not values:
                continue

            most_common = Counter(values).most_common(1)[0][0]

            for i in indices:
                if res[i][j] == "?":  # enlightened centrist
                    res[i][j] = most_common

    return res


class NBC:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_values = defaultdict(set)

    def fit(self, X, y):
        n = len(X[0])
        class_counts = Counter(y)
        self.class_priors = {cls: log(count / len(y)) for cls, count in class_counts.items()}

        for j in range(n):
            for row in X:
                self.feature_values[j].add(row[j])

        self.feature_probs = {cls: [defaultdict(float) for _ in range(n)] for cls in class_counts}

        for cls in class_counts:
            indices = {i for i, label in enumerate(y) if label == cls}
            p_C_i = len(indices)

            for j in range(n):
                counts = Counter(X[i][j] for i in indices)
                V = len(self.feature_values[j])

                for value in self.feature_values[j]:
                    prob = (counts[value] + self.lambda_) / (p_C_i + self.lambda_ * V)
                    self.feature_probs[cls][j][value] = log(prob)

    def predict(self, X):
        res = []

        for row in X:
            scores = {}

            for cls, score in self.class_priors.items():
                for j, value in enumerate(row):
                    score += self.feature_probs[cls][j][value]

                scores[cls] = score

            res.append(max(scores, key=scores.get))

        return res


def accuracy(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))

    return correct / len(y_true)


def cross_validation(X, y, k=10):
    indices = list(range(len(y)))
    shuffle(indices)
    folds = [indices[i::k] for i in range(k)]
    accuracies = []

    for i, test_idx in enumerate(folds):
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        nb = NBC(1)
        nb.fit(X_train, y_train)
        accuracies.append(accuracy(y_test, nb.predict(X_test)))

    return accuracies


if __name__ == "__main__":
    mode = int(input().strip())
    X, y = load_data("congressional+voting+records/house-votes-84.data")

    if mode == 1:
        X = fill_missing_by_class(X, y)

    X_train, y_train, X_test, y_test = stratified_split(X, y)
    nb = NBC(1)
    nb.fit(X_train, y_train)
    train_acc = accuracy(y_train, nb.predict(X_train))
    cv_accuracies = cross_validation(X_train, y_train)
    test_acc = accuracy(y_test, nb.predict(X_test))
    print("1. Train Set Accuracy:")
    print(f"Accuracy: {100 * train_acc:.2f}%\n")
    print("10-Fold Cross-Validation Results:\n")

    for i, acc in enumerate(cv_accuracies, 1):
        print(f"Accuracy Fold {i}: {100 * acc:.2f}%")

    print(f"\nAverage Accuracy: {100 * mean(cv_accuracies):.2f}%")
    print(f"Standard Deviation: {100 * stdev(cv_accuracies):.2f}%\n")
    print("2. Test Set Accuracy:")
    print(f"Accuracy: {100 * test_acc:.2f}%")



from collections import defaultdict
import math

class NaiveBayes:
    def fit(self, X, y):
        self.classes = set(y)
        self.priors = {}
        self.likelihoods = {}

        n = len(y)
        for c in self.classes:
            X_c = [x for x, label in zip(X, y) if label == c]
            self.priors[c] = len(X_c) / n

            counts = defaultdict(lambda: defaultdict(int))
            for x in X_c:
                for i, val in enumerate(x):
                    counts[i][val] += 1

            self.likelihoods[c] = counts

    def predict(self, x):
        posteriors = {}
        for c in self.classes:
            log_prob = math.log(self.priors[c])
            for i, val in enumerate(x):
                count = self.likelihoods[c][i][val] + 1
                total = sum(self.likelihoods[c][i].values()) + len(self.likelihoods[c][i])
                log_prob += math.log(count / total)
            posteriors[c] = log_prob

        return max(posteriors, key=posteriors.get)



DT
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



import math
from collections import Counter

def entropy(labels):
    counts = Counter(labels)
    n = len(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def information_gain(X, y, feature_index):
    base = entropy(y)
    subsets = {}

    for xi, yi in zip(X, y):
        subsets.setdefault(xi[feature_index], []).append(yi)

    remainder = sum(
        (len(sub)/len(y)) * entropy(sub)
        for sub in subsets.values()
    )

    return base - remainder



kMeans
from subprocess import run

from math import dist, inf

from numpy import mean, random, savetxt

from random import choice, uniform


def load_normal():
    res = []

    with open("normal.txt", "r") as f:
        for line in f:
            x, y = line.split("\t")
            res.append((float(x), float(y)))

    return res


def load_unbalance():
    res = []

    with open("unbalance.txt", "r") as f:
        for line in f:
            x, y = line.split()
            res.append((float(x), float(y)))

    return res


def load_data(name):
    if name == "normal.txt":
        return load_normal()

    if name == "unbalance.txt":
        return load_unbalance()

    return []


def assign_clusters(X, centroids):
    indices = []

    for x in X:
        distances = [dist(x, c) for c in centroids]
        indices.append(distances.index(min(distances)))

    return indices


def recompute_centroids(X, indices, k):
    clusters = [[] for _ in range(k)]

    for idx, x in zip(indices, X):
        clusters[idx].append(x)

    new_centroids = []

    for cluster in clusters:
        if cluster:
            new_centroids.append(tuple(mean(cluster, axis=0)))

        else:
            new_centroids.append(choice(X))

    return new_centroids


def initialize_centroids(X, k):
    xs = {x for x, _ in X}
    ys = {y for _, y in X}
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    centroids = [(uniform(x_min, x_max), uniform(y_min, y_max)) for _ in range(k)]

    return centroids


def kMeans(X, k, tol=1e-4):
    centroids, indices, shift = initialize_centroids(X, k), [], 1

    while shift > tol:
        indices = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(X, indices, k)

        shift = sum(dist(a, b) for a, b in zip(centroids, new_centroids))
        centroids = new_centroids

    return centroids, indices


def wcss(X, centroids, indices):
    total = 0.0

    for idx, x in zip(indices, X):
        total += dist(x, centroids[idx]) ** 2

    return total


def silhouette_score(X, indices, k):
    clusters = [[] for _ in range(k)]

    for idx, x in zip(indices, X):
        clusters[idx].append(x)

    scores = []

    for i, x in enumerate(X):
        idx = indices[i]
        cluster = clusters[idx]

        a = 0

        if len(cluster) > 1:
            a = sum(dist(x, y) for y in cluster if y != x) / (len(cluster) - 1)

        b = inf

        for j in range(k):
            if j != idx and clusters[j]:
                b = min(b, sum(dist(x, y) for y in clusters[j]) / len(clusters[j]))

        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)

    return sum(scores) / len(scores)


def kmeans_random_restart(X, k, restarts=20):
    best, best_wcss = None, inf

    for _ in range(restarts):
        centroids, indices = kMeans(X, k)
        score = wcss(X, centroids, indices)

        if score < best_wcss:
            best_wcss = score
            best = (centroids, indices)

    return best


def kmeans_plus_plus_init(X, k):
    centroids = [choice(X)]

    for _ in range(k - 1):
        distances = [min(dist(x, c) ** 2 for c in centroids) for x in X]
        total = sum(distances)
        probs = [d / total for d in distances]
        centroids.append(X[random.choice(len(X), p=probs)])

    return centroids


def kmeans_plus_plus(X, k, max_iters=100):
    centroids, indices = kmeans_plus_plus_init(X, k), []

    for _ in range(max_iters):
        indices = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(X, indices, k)

        if centroids == new_centroids:
            break

        centroids = new_centroids

    return centroids, indices


def save_results(centroids, indices, centroids_file, indices_file):
    savetxt(centroids_file, centroids)
    savetxt(indices_file, indices, fmt="%d")


def main():
    ln = input().split()
    file_name = ln[0]
    algorithm = ln[1]
    metric = ln[2]
    k = int(ln[3])

    X = load_data(file_name)

    if algorithm.lower() == "kmeans":
        centroids, indices = kmeans_random_restart(X, k)

    elif algorithm.lower() == "kmeans++":
        centroids, indices = kmeans_plus_plus(X, k)

    else:
        raise ValueError("Unknown algorithm")

    if metric.lower() == "wcss":
        score = wcss(X, centroids, indices)

    elif metric.lower() == "silhouette":
        score = silhouette_score(X, indices, k)

    else:
        raise ValueError("Unknown metric")

    print(f"{metric.upper()} = {score}")
    save_results(centroids, indices, "centroids.txt", "indices.txt")
    run(["python", "plot_clusters.py", file_name, "centroids.txt", "indices.txt"])


main()



import random
import math

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def kmeans(X, k, iters=10):
    centroids = random.sample(X, k)

    for _ in range(iters):
        clusters = {i: [] for i in range(k)}
        for x in X:
            idx = min(range(k), key=lambda i: euclidean(x, centroids[i]))
            clusters[idx].append(x)

        for i in range(k):
            centroids[i] = [
                sum(dim) / len(clusters[i])
                for dim in zip(*clusters[i])
            ]

    return centroids



NN
from math import exp, tanh

from random import uniform, shuffle

DATA = {
    "AND": [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
    "OR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
    "XOR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
}


def sigmoid(x):
    return 1 / (1 + exp(-x))


def d_sigmoid(y):
    return y * (1 - y)


def d_tanh(y):
    return 1 - y * y


def show(func):
    net = NeuralNetwork(layers, neurons, func_id)
    net.train(DATA[func].copy())
    print(f"{func}:")

    for x, _ in DATA[func]:
        print(f"{tuple(x)} -> {net.forward(x):.4f}")

    print()


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = [[uniform(-2.25, 2.25) for _ in range(n_inputs + 1)] for _ in range(n_neurons)]
        self.output = [0] * n_neurons
        self.delta = [0] * n_neurons


class NeuralNetwork:
    def __init__(self, hidden_layers, neurons, activation):
        self.activation = activation
        self.layers = []
        prev = 2

        for _ in range(hidden_layers):
            self.layers.append(Layer(prev, neurons))
            prev = neurons

        self.layers.append(Layer(prev, 1))

    def function(self, x):
        return tanh(x) if self.activation else sigmoid(x)

    def function_d(self, y):
        return d_tanh(y) if self.activation else d_sigmoid(y)

    def forward(self, inputs):
        outputs = None

        for layer in self.layers:
            inputs = inputs + [1]
            outputs = []

            for weights in layer.weights:
                outputs.append(self.function(sum(w * i for w, i in zip(weights, inputs))))

            layer.output = outputs
            inputs = outputs

        return outputs[0]

    def backward(self, target, lr=0.055):
        layer = self.layers[-1]
        error = target - layer.output[0]
        layer.delta[0] = error * self.function_d(layer.output[0])

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j in range(len(layer.output)):
                error = sum(next_layer.weights[k][j] * next_layer.delta[k] for k in range(len(next_layer.weights)))
                layer.delta[j] = error * self.function_d(layer.output[j])

        layer, inputs = self.layers[0], [0, 0, 1]

        for j in range(len(layer.weights)):
            for k in range(3):
                layer.weights[j][k] += lr * layer.delta[j] * inputs[k]

        for i, layer in enumerate(self.layers[1:]):
            inputs = self.layers[i].output + [1]

            for j in range(len(layer.weights)):
                for k in range(len(inputs)):
                    layer.weights[j][k] += lr * layer.delta[j] * inputs[k]

    def train(self, data, epochs=100000):
        for _ in range(epochs):
            shuffle(data)

            for x, y in data:
                self.forward(x), self.backward(y)


bool_func = input().upper()
func_id = bool(int(input()))
layers = int(input())
neurons = int(input())

if bool_func == "ALL":
    for f in ["AND", "OR", "XOR"]:
        show(f)

else:
    show(bool_func)



LR
import math

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        self.w = [0.0] * len(X[0])
        self.b = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for x, yi in zip(X, y):
                z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
                y_hat = self.sigmoid(z)
                error = y_hat - yi

                for i in range(len(self.w)):
                    self.w[i] -= self.lr * error * x[i]
                self.b -= self.lr * error

    def predict(self, x):
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return 1 if self.sigmoid(z) >= 0.5 else 0



Standardization
import math
from collections import Counter

def entropy(labels):
    counts = Counter(labels)
    n = len(labels)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def information_gain(X, y, feature_index):
    base = entropy(y)
    subsets = {}

    for xi, yi in zip(X, y):
        subsets.setdefault(xi[feature_index], []).append(yi)

    remainder = sum(
        (len(sub)/len(y)) * entropy(sub)
        for sub in subsets.values()
    )

    return base - remainder



Confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn

def f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
"""