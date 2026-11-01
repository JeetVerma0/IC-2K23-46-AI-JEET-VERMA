## 1. 8-Puzzle Solver using Breadth-First Search

## Objective
To solve the 8-Puzzle problem using Breadth-First Search (BFS) and find the shortest sequence of moves from an initial state to a goal state.

## Problem Description
The 8-Puzzle consists of a 3 × 3 board containing eight numbered tiles and one empty space represented by `0`.

The objective is to arrange the tiles into the required goal configuration by moving the blank tile.

## Initial State

```text
1 2 3
4 0 6
7 5 8
```

## Goal State

```text
1 2 3
4 5 6
7 8 0
```

## Possible Operations
- Move blank tile Up.
- Move blank tile Down.
- Move blank tile Left.
- Move blank tile Right.

## Algorithm Used
**Breadth-First Search (BFS)**

BFS explores the possible puzzle configurations level by level using a queue. It tracks visited states to avoid repetition and stores parent states to reconstruct the solution path.

## Execution

```bash
python3 eight_puzzle_solver.py
```

## Sample Output

```text
8-Puzzle Solution using BFS
----------------------------
Initial State:
1 2 3
4 0 6
7 5 8

Move: Down
1 2 3
4 5 6
7 0 8

Move: Right
1 2 3
4 5 6
7 8 0

Goal State Reached!
Total moves: 2
```

## Result
The 8-Puzzle problem was successfully solved using BFS. The algorithm reached the goal configuration in two moves.
