from random import randrange, choice

from time import time

from numpy import arange, zeros, nonzero, flatnonzero, random


def is_valid():
    if len(set(cols)) < n:
        return False

    coordinates = [(r, c) for c, r in enumerate(cols)]
    on_main_diags = zeros(2 * n - 1, bool)
    on_sec_diags = zeros(2 * n - 1, bool)

    for r, c in coordinates:
        main_d, sec_d = n - 1 - r + c, r + c

        if on_main_diags[main_d] or on_sec_diags[sec_d]:
            return False

        on_main_diags[main_d] += 1
        on_sec_diags[sec_d] += 1

    return True


n = int(input())

if n == 1:
    print("# TIMES_MS: alg=0.0", [0], sep="\n")
    exit()

if n < 4:
    print(-1)
    exit()

rows = zeros(n, int)
cols = random.randint(0, n, n)
main_diags = zeros(2 * n - 1, int)
sec_diags = zeros(2 * n - 1, int)

for i in range(n):
    cols[i] = randrange(n)

for c, r in enumerate(cols):
    rows[r] += 1
    main_diags[n + c - r - 1] += 1
    sec_diags[r + c] += 1

t = time()
indices = arange(n)

while True:
    conflict_values = rows[cols] + main_diags[n + indices - cols - 1] + sec_diags[cols + indices] - 3
    conflicted = nonzero(conflict_values)[0]

    if not conflicted.size:
        break

    col = choice(conflicted)
    curr_r = cols[col]
    conflicts_for_rows = rows + main_diags[n + col - indices - 1] + sec_diags[indices + col] - 3
    min_conf = conflicts_for_rows.min()
    row = random.choice(flatnonzero(conflicts_for_rows == min_conf))

    if curr_r == row:
        continue

    rows[curr_r] -= 1
    main_diags[n + col - curr_r - 1] -= 1
    sec_diags[curr_r + col] -= 1
    cols[col] = row
    rows[row] += 1
    main_diags[n + col - row - 1] += 1
    sec_diags[row + col] += 1

print(f"# TIMES_MS: alg={(time() - t) * 1000}", cols, sep="\n")
print(is_valid())
