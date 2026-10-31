from collections import deque


def vacuum_cleaner():
    # State: (Vacuum Location, Room A, Room B)
    # 0 = Dirty, 1 = Clean

    initial_state = ('A', 0, 0)

    queue = deque([initial_state])
    visited = {initial_state: None}
    actions = {}

    goal_state = None

    while queue:
        current = queue.popleft()
        location, room_a, room_b = current

        # Goal: Both rooms are clean
        if room_a == 1 and room_b == 1:
            goal_state = current
            break

        next_states = []

        # Suck dirt from current room
        if location == 'A' and room_a == 0:
            next_states.append(
                (('A', 1, room_b), "Clean Room A")
            )

        elif location == 'B' and room_b == 0:
            next_states.append(
                (('B', room_a, 1), "Clean Room B")
            )

        # Move between rooms
        if location == 'A':
            next_states.append(
                (('B', room_a, room_b), "Move Right")
            )
        else:
            next_states.append(
                (('A', room_a, room_b), "Move Left")
            )

        for next_state, action in next_states:
            if next_state not in visited:
                visited[next_state] = current
                actions[next_state] = action
                queue.append(next_state)

    # Reconstruct solution
    path = []
    state = goal_state

    while state is not None:
        path.append(state)
        state = visited[state]

    path.reverse()

    print("\nVacuum Cleaner Solution")
    print("----------------------------")

    for i, state in enumerate(path):
        if i == 0:
            print(f"Initial State: {state}")
        else:
            print(f"{actions[state]} -> {state}")

    print("----------------------------")
    print(f"Total steps: {len(path) - 1}")
    print("Both rooms are clean!")


if __name__ == "__main__":
    vacuum_cleaner()
