from collections import deque

# Grid representation
# 0 = Free path
# 1 = Obstacle
# S = Starting position
# G = Goal position

grid = [
    ['S', '0', '0', '1', '0'],
    ['1', '1', '0', '1', '0'],
    ['0', '0', '0', '0', '0'],
    ['0', '1', '1', '1', '0'],
    ['0', '0', '0', '0', 'G']
]

rows = len(grid)
cols = len(grid[0])

start = (0, 0)
goal = (4, 4)


def bfs(start, goal):

    queue = deque([start])

    visited = {start}
    parent = {start: None}

    directions = [
        (0, 1),   # Right
        (1, 0),   # Down
        (0, -1),  # Left
        (-1, 0)   # Up
    ]

    while queue:

        current = queue.popleft()

        if current == goal:
            break

        row, col = current

        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc

            if (
                0 <= new_row < rows
                and 0 <= new_col < cols
                and grid[new_row][new_col] != '1'
            ):

                next_position = (new_row, new_col)

                if next_position not in visited:

                    visited.add(next_position)
                    parent[next_position] = current
                    queue.append(next_position)

    if goal not in parent:
        return None

    # Reconstruct shortest path
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path


def display_grid(path):

    result = [row[:] for row in grid]

    for row, col in path:
        if result[row][col] not in ('S', 'G'):
            result[row][col] = '*'

    for row in result:
        print(" ".join(row))


if __name__ == "__main__":

    print("Robot Navigation using BFS")
    print("----------------------------")

    print("\nOriginal Grid:")

    for row in grid:
        print(" ".join(row))

    path = bfs(start, goal)

    if path:

        print("\nShortest Path Found!")
        print("Path:", path)

        print("\nNavigation Grid:")
        display_grid(path)

        print("\nTotal Steps:", len(path) - 1)

    else:
        print("\nNo valid path found.")
