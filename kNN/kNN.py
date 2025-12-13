from __future__ import annotations

from math import dist

from random import shuffle

from statistics import stdev

import matplotlib.pyplot as plt

from collections import defaultdict

from heapq import heappush, heapreplace

Point = tuple[float]


class KDTree:
    def __init__(self, d: int = 3, *points: tuple[float]) -> None:
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

    def kNN(self, p: Point, k: int) -> set[Point]:
        best = []

        def search(tree, depth):
            if tree is None or tree.root is None:
                return

            axis = depth % tree.d
            point = tree.root

            if point != p:
                d = dist(p, point)

                if len(best) < k:
                    heappush(best, (-d, point))

                else:
                    if d < -best[0][0]:
                        heapreplace(best, (-d, point))

            primary, secondary = (tree.left, tree.right) if p[axis] < point[axis] else (tree.right, tree.left)
            search(primary, depth + 1)

            if len(best) < k or abs(p[axis] - point[axis]) < -best[0][0]:
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
    plt.grid(True)
    plt.show()
