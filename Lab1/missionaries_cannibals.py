from collections import deque


def is_valid(m, c):
    # Check whether the number of people is within limits
    if m < 0 or c < 0 or m > 3 or c > 3:
        return False

    # Missionaries must not be outnumbered by cannibals
    if m > 0 and c > m:
        return False

    if (3 - m) > 0 and (3 - c) > (3 - m):
        return False

    return True


def missionaries_cannibals():
    # State: (Missionaries Left, Cannibals Left, Boat Position)
    # 0 = Left Bank, 1 = Right Bank

    start = (3, 3, 0)
    goal = (0, 0, 1)

    # Possible boat movements
    moves = [
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
        (1, 1)
    ]

    queue = deque([start])
    parent = {start: None}
    actions = {}

    while queue:
        m, c, boat = queue.popleft()
        current = (m, c, boat)

        if current == goal:
            break

        for dm, dc in moves:

            if boat == 0:
                new_state = (m - dm, c - dc, 1)
                direction = "Left to Right"
            else:
                new_state = (m + dm, c + dc, 0)
                direction = "Right to Left"

            nm, nc, _ = new_state

            if (
                dm + dc >= 1
                and dm + dc <= 2
                and is_valid(nm, nc)
                and new_state not in parent
            ):
                parent[new_state] = current
                actions[new_state] = (
                    f"Move {dm} Missionary(s) and "
                    f"{dc} Cannibal(s) - {direction}"
                )
                queue.append(new_state)

    # Reconstruct path
    if goal not in parent:
        print("No solution found.")
        return

    path = []
    state = goal

    while state is not None:
        path.append(state)
        state = parent[state]

    path.reverse()

    print("\nMissionaries and Cannibals Solution")
    print("--------------------------------------")

    for i, state in enumerate(path):
        if i == 0:
            print(f"Initial State: {state}")
        else:
            print(f"{actions[state]} -> {state}")

    print("--------------------------------------")
    print(f"Total crossings: {len(path) - 1}")
    print("All missionaries and cannibals crossed safely!")


if __name__ == "__main__":
    missionaries_cannibals()
