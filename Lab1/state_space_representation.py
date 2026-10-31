# State Space Representation using 8-Puzzle

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


def get_next_states(state):
    states = []

    zero_position = state.index(0)

    row = zero_position // 3
    column = zero_position % 3

    moves = []

    # Possible movements of the blank tile
    if row > 0:
        moves.append((-3, "Move Up"))

    if row < 2:
        moves.append((3, "Move Down"))

    if column > 0:
        moves.append((-1, "Move Left"))

    if column < 2:
        moves.append((1, "Move Right"))

    for position_change, action in moves:
        new_position = zero_position + position_change

        new_state = list(state)

        new_state[zero_position], new_state[new_position] = (
            new_state[new_position],
            new_state[zero_position]
        )

        states.append((tuple(new_state), action))

    return states


if __name__ == "__main__":

    print("Initial State:")
    print_state(initial_state)

    print("Goal State:")
    print_state(goal_state)

    print("Possible Next States:")

    for state, action in get_next_states(initial_state):
        print(action)
        print_state(state)
