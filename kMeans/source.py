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
    xs = {x for x, y in X}
    ys = {y for x, y in X}
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

    for _ in range(1, k):
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
