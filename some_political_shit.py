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

    for label, indices in class_indices.items():
        shuffle(indices)
        split = 4 * len(indices) // 5
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
            indices = [i for i, label in enumerate(y) if label == cls]
            k = len(indices)

            for j in range(n):
                counts = Counter(X[i][j] for i in indices)
                V = len(self.feature_values[j])

                for value in self.feature_values[j]:
                    prob = (counts[value] + self.lambda_) / (k + self.lambda_ * V)
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
    seed(420)
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
