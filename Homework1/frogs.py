from time import time

def dfs(position: list[int], stack: list[int], index) -> None:
    global mismatch

    if index and position[index - 1] == 1:
        position[index], position[index - 1] = 1, 0
        stack.append(index - 1)
        off = -2 if n == index - 1 else index == n
        mismatch += off
        dfs(position, stack, index - 1)

        if not mismatch:
            return

        position[index - 1], position[index] = 1, 0
        stack.pop()
        mismatch -= off

    if index > 1 and position[index - 2] == 1 and position[index - 1] == -1:
        position[index], position[index - 2] = 1, 0
        stack.append(index - 2)
        off = index == n

        if n == index - 1:
            off = -1

        elif n == index - 2:
            off = -2

        mismatch += off
        dfs(position, stack, index - 2)

        if not mismatch:
            return

        position[index - 2], position[index] = 1, 0
        stack.pop()
        mismatch -= off

    if index < 2 * n and position[index + 1] == -1:
        position[index], position[index + 1] = -1, 0
        stack.append(index + 1)
        off = -2 if n == index + 1 else index == n
        mismatch += off
        dfs(position, stack, index + 1)

        if not mismatch:
            return

        position[index + 1], position[index] = -1, 0
        stack.pop()
        mismatch -= off

    if index < 2 * n - 1 and position[index + 2] == -1 and position[index + 1] == 1:
        position[index], position[index + 2] = -1, 0
        stack.append(index + 2)
        off = index == n

        if n == index + 1:
            off = -1

        elif n == index + 2:
            off = -2

        mismatch += off
        dfs(position, stack, index + 2)

        if not mismatch:
            return

        position[index + 2], position[index] = -1, 0
        stack.pop()
        mismatch -= off

symbols = "_><"
n = int(input())
start = n * [1] + [0] + n * [-1]
mismatch = 2 * n
result = []
_start = start.copy()
t = time()
dfs(_start, result, n)
print(time() - t)
print("".join([symbols[i] for i in start]))
last = n

for idx in result:
    start[last], start[idx] = start[idx], start[last]
    print("".join([symbols[i] for i in start]))
    last = idx

