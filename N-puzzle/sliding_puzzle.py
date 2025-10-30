from math import sqrt

directions = ["up", "down", "left", "right"]


def unique_id():
    return hash(tuple(map(tuple, grid)))


def solvable():
    inversions = 0
    flat = sum(grid, start=[])
    flat.remove(0)

    for i, m in enumerate(flat):
        for k in flat[i + 1:]:
            inversions += m > k

    if n % 2:
        return not inversions % 2

    return (inversions + empty_r) % 2 == wanted_empty % 2


def ida_star(last=-1):
    global heuristic, empty_r, empty_c, wanted_positions

    next_moves = {0, 1, 2, 3} - {last}

    if not heuristic:
        return

    if 1 in next_moves and empty_r:  # down
        num = grid[empty_r - 1][empty_c]
        grid[empty_r][empty_c], grid[empty_r - 1][empty_c] = num, 0

        if unique_id() in so_far:
            grid[empty_r][empty_c], grid[empty_r - 1][empty_c] = 0, num

        else:
            so_far.add(unique_id())
            empty_r -= 1
            delta_h = 1 if empty_r + 1 <= wanted_empty_r else -1

            if empty_r < wanted_positions[num][0]:
                delta_h -= 1

            else:
                delta_h += 1

            heuristic += delta_h
            path.append(1)
            ida_star(0)

            if not heuristic:
                return

            heuristic -= delta_h
            so_far.remove(unique_id())
            empty_r += 1
            grid[empty_r][empty_c], grid[empty_r - 1][empty_c] = 0, num

    if 3 in next_moves and empty_c:  # right
        num = grid[empty_r][empty_c - 1]
        grid[empty_r][empty_c], grid[empty_r][empty_c - 1] = num, 0

        if unique_id() in so_far:
            grid[empty_r][empty_c], grid[empty_r][empty_c - 1] = 0, num

        else:
            so_far.add(unique_id())
            empty_c -= 1
            delta_h = 1 if empty_c + 1 <= wanted_empty_c else -1

            if empty_c < wanted_positions[num][1]:
                delta_h -= 1

            else:
                delta_h += 1

            heuristic += delta_h
            path.append(3)
            ida_star(2)

            if not heuristic:
                return

            heuristic -= delta_h
            so_far.remove(unique_id())
            empty_c += 1
            grid[empty_r][empty_c], grid[empty_r][empty_c - 1] = 0, num

    if 0 in next_moves and empty_r < n - 1:  # up
        num = grid[empty_r + 1][empty_c]
        grid[empty_r][empty_c], grid[empty_r + 1][empty_c] = num, 0

        if unique_id() in so_far:
            grid[empty_r][empty_c], grid[empty_r + 1][empty_c] = 0, num

        else:
            so_far.add(unique_id())
            empty_r += 1
            delta_h = 1 if empty_r - 1 >= wanted_empty_r else -1

            if empty_r > wanted_positions[num][0]:
                delta_h -= 1

            else:
                delta_h += 1

            heuristic += delta_h
            path.append(0)
            ida_star(1)

            if not heuristic:
                return

            heuristic -= delta_h
            so_far.remove(unique_id())
            empty_r -= 1
            grid[empty_r][empty_c], grid[empty_r + 1][empty_c] = 0, num

    if 2 in next_moves and empty_c < n - 1:  # left
        num = grid[empty_r][empty_c + 1]
        grid[empty_r][empty_c], grid[empty_r][empty_c + 1] = num, 0

        if unique_id() in so_far:
            grid[empty_r][empty_c], grid[empty_r][empty_c + 1] = 0, num

        else:
            so_far.add(unique_id())
            empty_c += 1
            delta_h = 1 if empty_c - 1 >= wanted_empty_c else -1

            if empty_c > wanted_positions[num][1]:
                delta_h -= 1

            else:
                delta_h += 1

            heuristic += delta_h
            path.append(2)
            ida_star(3)

            if not heuristic:
                return

            heuristic -= delta_h
            so_far.remove(unique_id())
            empty_c -= 1
            grid[empty_r][empty_c], grid[empty_r][empty_c + 1] = 0, num

    path.pop()


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
        wanted_index = wanted_empty

    else:
        wanted_index = num - 1 + (num > wanted_empty)

    wanted_r, wanted_c = wanted_index // n, wanted_index % n
    wanted_positions[num] = (wanted_r, wanted_c)
    heuristic += abs(r - wanted_r) + abs(c - wanted_c)

if not solvable():
    print(-1)

else:
    so_far = set()
    flat_wanted = list(range(1, k))
    path = []
    curr_empty = 0
    flat_wanted.insert(wanted_empty, 0)

    ida_star()
    print(len(path))

    for d in path:
        print(directions[d])
