from random import randint, shuffle, random

from numpy import array, where

from time import time


def mutate(sol):
    for i in range(N):
        if random() < 0.05:
            sol[i] = 1 - sol[i]

    return sol


def validate(sol):
    weight = sum(items[i][0] for i in range(N) if sol[i])
    indices = list(where(array(sol) == 1)[0])
    indices.sort(key=lambda i: items[i][1] / items[i][0])

    for i in indices:
        if weight <= M:
            break

        sol[i] = 0
        weight -= items[i][0]

    return sol


def fitness(sol):
    return sum(items[i][1] for i in range(N) if sol[i])


M, N = map(int, input().split(maxsplit=1))
items, total_mass, total_price = [], 0, 0

for _ in range(N):
    items.append(curr := tuple(map(int, input().split(maxsplit=1))))
    total_mass += curr[0]
    total_price += curr[1]

if total_mass <= M:
    print(total_price)
    exit()

solutions = [validate([randint(0, 1) for _ in range(N)]) for _ in range(50)]
printings = []
t = time()

for _ in range(6000):
    solutions.sort(key=fitness, reverse=True)
    survivors = solutions[:20]
    next_gen = survivors.copy()
    shuffle(survivors)

    for i in range(0, 19, 2):
        j = randint(1, N - 2)
        p0, p1 = survivors[i], survivors[i + 1]
        c0 = mutate(p0[:j] + p1[j:])
        c1 = mutate(p1[:j] + p0[j:])
        next_gen.append(validate(c0)), next_gen.append(validate(c1))

    solutions = next_gen

    if not _ % 600:
        printings.append(fitness(max(solutions, key=fitness)))

t = time() - t
printings.append("")
printings.append(fitness(max(solutions, key=fitness)))
print(f"# TIMES_MS: alg={1000 * t}")

for el in printings:
    print(el)

