# Maze Solver using Depth-First Search (DFS)

maze = [
    ['S', '0', '1', '0', '0'],
    ['1', '0', '1', '0', '1'],
    ['0', '0', '0', '0', '1'],
    ['0', '1', '1', '0', '0'],
    ['0', '0', '0', '1', 'G']
]

rows = len(maze)
cols = len(maze[0])

start = (0, 0)
goal = (4, 4)

visited = set()
path = []


def dfs(row, col):

    # Check boundaries
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return False

    # Check wall or visited position
    if maze[row][col] == '1' or (row, col) in visited:
        return False

    visited.add((row, col))
    path.append((row, col))

    # Check whether goal is reached
    if (row, col) == goal:
        return True

    # Possible movements: Down, Right, Up, Left
    directions = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1)
    ]

    for dr, dc in directions:
        if dfs(row + dr, col + dc):
            return True

    # Backtrack if no path is found
    path.pop()
    return False


def print_maze():

    for r in range(rows):
        for c in range(cols):

            if (r, c) in path and maze[r][c] not in ('S', 'G'):
                print('*', end=' ')
            else:
                print(maze[r][c], end=' ')

        print()


if __name__ == "__main__":

    print("Maze Solver using DFS")
    print("----------------------")

    print("\nOriginal Maze:")
    for row in maze:
        print(" ".join(row))

    if dfs(start[0], start[1]):

        print("\nPath Found!")
        print("Path:", path)

        print("\nSolved Maze:")
        print_maze()

        print("\nTotal Steps:", len(path) - 1)

    else:
        print("\nNo path found.")
