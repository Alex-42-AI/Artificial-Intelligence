from math import sqrt, inf

from time import time

directions = ["up", "down", "left", "right"]
FOUND = -1


def solvable():
    inversions = 0
    flat = sum(grid, [])
    flat.remove(0)

    for i, m in enumerate(flat):
        for k in flat[i + 1:]:
            inversions += m > k

    if n % 2:
        return not inversions % 2

    return (inversions + empty_r) % 2 == wanted_empty % 2


def ida_star():
    global heuristic, path

    def manhattan_delta(old_r, old_c, new_r, new_c):
        w_r, w_c = wanted_positions[grid[old_r][old_c]]

        return abs(new_r - w_r) + abs(new_c - w_c) - abs(old_r - w_r) - abs(old_c - w_c)

    def dfs(last=-1, g=0):
        global heuristic, empty_r, empty_c, wanted_positions

        if not heuristic:
            return FOUND

        next_moves = []

        if empty_r < n - 1:
            new_r, new_c = empty_r + 1, empty_c
            delta_h = manhattan_delta(new_r, new_c, empty_r, empty_c)

            if last:
                next_moves.append((0, 1, new_r, new_c, delta_h))

        if empty_r:
            new_r, new_c = empty_r - 1, empty_c
            delta_h = manhattan_delta(new_r, new_c, empty_r, empty_c)

            if last != 1:
                next_moves.append((1, 0, new_r, new_c, delta_h))

        if empty_c < n - 1:
            new_r, new_c = empty_r, empty_c + 1
            delta_h = manhattan_delta(new_r, new_c, empty_r, empty_c)

            if last != 2:
                next_moves.append((2, 3, new_r, new_c, delta_h))

        if empty_c:
            new_r, new_c = empty_r, empty_c - 1
            delta_h = manhattan_delta(new_r, new_c, empty_r, empty_c)

            if last != 3:
                next_moves.append((3, 2, new_r, new_c, delta_h))

        next_moves = sorted(next_moves, key=lambda s: s[-1])
        f = inf

        for i, j, r, c, d_h in next_moves:
            if (c_f := g + heuristic + d_h) > bound:
                f = min(f, c_f)

                break

            num = grid[r][c]
            grid[empty_r][empty_c], grid[r][c] = num, 0
            heuristic += d_h
            tmp_r, tmp_c = empty_r, empty_c
            empty_r, empty_c = r, c
            path.append(i)
            t = dfs(j, g + 1)

            if t == FOUND:
                return FOUND

            f = min(f, t)
            path.pop()
            empty_r, empty_c = tmp_r, tmp_c
            heuristic -= d_h
            grid[empty_r][empty_c], grid[r][c] = 0, num

        return f

    bound = heuristic

    while True:
        t = dfs()

        if t == FOUND:
            return

        if t <= bound:
            bound += 1

        else:
            bound = t


k = int(input()) + 1
n = round(sqrt(k))
wanted_empty = int(input()) % k
wanted_empty_r, wanted_empty_c = wanted_empty // n, wanted_empty % n
grid = [list(map(int, input().split(maxsplit=n))) for _ in range(n)]
heuristic = 0
empty_r, empty_c, empty = 0, 0, 0
wanted_positions = [(0, 0) for _ in range(k)]

for i in range(k):
    if not (num := grid[r := i // n][c := i % n]):
        empty_r, empty_c = r, c
        empty = r * n + c

        continue

    else:
        wanted_index = num - 1 + (num > wanted_empty)

    wanted_r, wanted_c = wanted_index // n, wanted_index % n
    wanted_positions[num] = (wanted_r, wanted_c)
    heuristic += abs(r - wanted_r) + abs(c - wanted_c)

bound = heuristic

if not solvable():
    print(-1)

else:
    flat_wanted = list(range(1, k))
    path = []
    curr_empty = 0
    flat_wanted.insert(wanted_empty, 0)
    t0 = time()
    ida_star()
    print(time() - t0, len(path), sep="\n")

    for d in path:
        print(directions[d])
