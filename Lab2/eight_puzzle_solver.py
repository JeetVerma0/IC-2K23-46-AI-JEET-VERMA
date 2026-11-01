from collections import deque

initial_state = (
    1, 2, 3,
    4, 0, 6,
    7, 5, 8
)

goal_state = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)


def print_state(state):
    for i in range(0, 9, 3):
        print(state[i], state[i + 1], state[i + 2])
    print()


def get_neighbors(state):
    neighbors = []

    zero = state.index(0)
    row = zero // 3
    col = zero % 3

    moves = [
        (-3, "Up"),
        (3, "Down"),
        (-1, "Left"),
        (1, "Right")
    ]

    for change, action in moves:

        new_pos = zero + change

        if action == "Up" and row == 0:
            continue
        if action == "Down" and row == 2:
            continue
        if action == "Left" and col == 0:
            continue
        if action == "Right" and col == 2:
            continue

        new_state = list(state)

        new_state[zero], new_state[new_pos] = (
            new_state[new_pos],
            new_state[zero]
        )

        neighbors.append((tuple(new_state), action))

    return neighbors


def solve_puzzle():
    queue = deque([initial_state])

    parent = {initial_state: None}
    actions = {}

    while queue:
        current = queue.popleft()

        if current == goal_state:
            break

        for next_state, action in get_neighbors(current):

            if next_state not in parent:
                parent[next_state] = current
                actions[next_state] = action
                queue.append(next_state)

    if goal_state not in parent:
        print("No solution found.")
        return

    path = []
    state = goal_state

    while state is not None:
        path.append(state)
        state = parent[state]

    path.reverse()

    print("\n8-Puzzle Solution using BFS")
    print("----------------------------")

    for i, state in enumerate(path):
        if i == 0:
            print("Initial State:")
        else:
            print("Move:", actions[state])

        print_state(state)

    print("Goal State Reached!")
    print("Total moves:", len(path) - 1)


if __name__ == "__main__":
    solve_puzzle()
