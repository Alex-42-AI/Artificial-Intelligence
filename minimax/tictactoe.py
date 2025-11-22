def print_board():
    res = "+---+---+---+\n| "
    res += board[0][0] + " | " + board[0][1] + " | " + board[0][2] + " |\n"
    res += "+---+---+---+\n| "
    res += board[1][0] + " | " + board[1][1] + " | " + board[1][2] + " |\n"
    res += "+---+---+---+\n| "
    res += board[2][0] + " | " + board[2][1] + " | " + board[2][2] + " |\n"
    res += "+---+---+---+\n"
    print(res)


def transposed(b):
    return list(map(list, zip(*b)))


def user_won(b):
    l = [human, human, human]

    if l in b:
        return True

    t = transposed(b)

    if l in t:
        return True

    if [b[0][0], b[1][1], b[2][2]] == l:
        return True

    if [b[0][2], b[1][1], b[2][0]] == l:
        return True

    return False


def bot_won(b):
    l = [bot, bot, bot]

    if l in b:
        return True

    t = transposed(b)

    if l in t:
        return True

    if [b[0][0], b[1][1], b[2][2]] == l:
        return True

    if [b[0][2], b[1][1], b[2][0]] == l:
        return True

    return False


def play():
    def minimax(alpha, betta, l=0, player=bot):
        nonlocal x, y

        t = "".join("".join(tmp[i]) for i in range(3))

        if t in memoization:
            return memoization[t]

        if user_won(tmp):
            memoization[t] = (-1, l)

            return memoization[t]

        if bot_won(tmp):
            memoization[t] = (1, -l)

            return memoization[t]

        is_bot, best, best_p = player == bot, (0, 0), None

        for i, p in enumerate(slots):
            tmp[p[0]][p[1]] = player
            slots.pop(i)
            curr = minimax(alpha, betta, l + 1, human if is_bot else bot)
            tmp[p[0]][p[1]] = "_"
            slots.insert(i, p)

            if is_bot:
                if best_p is None or curr > best:
                    best, best_p = curr, p

                alpha = max(alpha, best)

            else:
                if best_p is None or curr < best:
                    best, best_p = curr, p

                betta = min(betta, best)

            if alpha >= betta:
                break

        memoization[t] = best

        if not l:
            x, y = best_p

        return memoization[t]

    tmp = [board[0].copy(), board[1].copy(), board[2].copy()]
    memoization = {}
    slots = [(i, j) for i in range(3) for j in range(3) if tmp[i][j] == "_"]
    x = y = None
    minimax((-2, 0), (2, 0))

    return x, y


mode = input()

if mode.upper() == "JUDGE":
    bot = input()[-1].upper()
    human = "O" if bot == "X" else "X"
    board, terminated = [], True

    for i in range(7):
        curr = input()

        if i % 2:
            curr = curr[2:-2].split(" | ")
            board.append(curr)

            if "_" in curr:
                terminated = False

    if terminated or user_won(board) or bot_won(board):
        print(-1)

    else:
        x, y = play()
        print(x + 1, y + 1)

elif mode.upper() == "GAME":
    fst = input().split()
    snd = input().split()
    first, human = "", ""

    if fst[0].upper() == "FIRST":
        first = fst[1].upper()

    elif fst[0].upper() == "HUMAN":
        human = fst[1].upper()

    if snd[0].upper() == "FIRST":
        first = snd[1].upper()

    elif snd[0].upper() == "HUMAN":
        human = snd[1].upper()

    board = [["_", "_", "_"], ["_", "_", "_"], ["_", "_", "_"]]
    bot = "O" if human == "X" else "X"
    x_winner, y_winner, turn = False, False, 0

    if first == human:
        print_board()

    while turn < 9:
        if turn % 2 == (first == bot):
            x, y = map(int, input().split(maxsplit=1))

            if board[x - 1][y - 1] == "_":
                board[x - 1][y - 1] = human

                if user_won(board):
                    if human == "X":
                        x_winner = True

                    else:
                        y_winner = True

                    break

            else:
                print("Slot taken")
                turn -= 1

        else:
            x, y = play()
            board[x][y] = bot
            print_board()

            if bot_won(board):
                if bot == "X":
                    x_winner = True

                else:
                    y_winner = True

                break

        turn += 1

    if first == human:
        print_board()

    if x_winner:
        print("WINNER: X")

    elif y_winner:
        print("WINNER: O")

    else:
        print("DRAW")
