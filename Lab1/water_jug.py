from collections import deque
from math import gcd


def water_jug(capacity_a, capacity_b, target):
    if target < 0 or target > max(capacity_a, capacity_b):
        print("Solution not possible.")
        return

    if target != 0 and target % gcd(capacity_a, capacity_b) != 0:
        print("Solution not possible.")
        return

    start = (0, 0)
    queue = deque([start])

    # Store visited states and their previous states
    parent = {start: None}
    actions = {}

    goal = None

    while queue:
        a, b = queue.popleft()
        current = (a, b)

        if a == target or b == target:
            goal = current
            break

        next_states = [
            ((capacity_a, b), "Fill Jug A"),
            ((a, capacity_b), "Fill Jug B"),
            ((0, b), "Empty Jug A"),
            ((a, 0), "Empty Jug B"),
            ((a - min(a, capacity_b - b),
              b + min(a, capacity_b - b)), "Pour A into B"),
            ((a + min(b, capacity_a - a),
              b - min(b, capacity_a - a)), "Pour B into A")
        ]

        for next_state, action in next_states:
            if next_state not in parent:
                parent[next_state] = current
                actions[next_state] = action
                queue.append(next_state)

    if goal is None:
        print("No solution found.")
        return

    # Reconstruct the solution path
    path = []
    state = goal

    while state is not None:
        path.append(state)
        state = parent[state]

    path.reverse()

    print("\nSolution Found!")
    print("-------------------------")

    for i, state in enumerate(path):
        if i == 0:
            print(f"Initial State: {state}")
        else:
            print(f"{actions[state]} -> {state}")

    print("-------------------------")
    print(f"Total steps: {len(path) - 1}")


if __name__ == "__main__":
    jug_a = int(input("Enter capacity of Jug A: "))
    jug_b = int(input("Enter capacity of Jug B: "))
    target = int(input("Enter target amount: "))

    if jug_a <= 0 or jug_b <= 0:
        print("Jug capacities must be positive.")
    else:
        water_jug(jug_a, jug_b, target)