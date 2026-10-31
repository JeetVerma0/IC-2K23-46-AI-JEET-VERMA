# Lab 1 – AI Problem Solving

## 1. Water Jug Problem

### Objective
To solve the Water Jug Problem using State Space Search and Breadth-First Search (BFS).

### Problem Description
Given two water jugs with capacities of 4 litres and 3 litres, the objective is to obtain exactly 2 litres of water in either jug.

Initially, both jugs are empty.

### Operations
The following operations can be performed:

- Fill Jug A completely.
- Fill Jug B completely.
- Empty Jug A.
- Empty Jug B.
- Pour water from Jug A into Jug B.
- Pour water from Jug B into Jug A.

### Algorithm Used
**Breadth-First Search (BFS)**

Each state is represented as a pair `(a, b)`, where:
- `a` represents the amount of water in Jug A.
- `b` represents the amount of water in Jug B.

BFS explores all possible states level by level until the target amount is reached. A queue is used to explore states, and a visited-state mechanism prevents revisiting previously explored states.

### Python Implementation
File: `water_jug.py`

### Execution

Run the following command:

```bash
python3 water_jug.py
```

### Sample Input

```text
Enter capacity of Jug A: 4
Enter capacity of Jug B: 3
Enter target amount: 2
```

### Sample Output

```text
Solution Found!
-------------------------
Initial State: (0, 0)
Fill Jug B -> (0, 3)
Pour B into A -> (3, 0)
Fill Jug B -> (3, 3)
Pour B into A -> (4, 2)
-------------------------
Total steps: 4
```

### Result
The Water Jug Problem was successfully solved using Breadth-First Search. The algorithm finds a sequence of valid operations to obtain exactly 2 litres of water in one of the jugs.




## 2 . Vacuum Cleaner Problem

## Objective
To implement a simple AI Vacuum Cleaner Agent using State Space Search and Breadth-First Search (BFS).

## Problem Description
A vacuum cleaner operates in two rooms, A and B. Each room can be either clean or dirty.

The objective is to clean both rooms by performing suitable actions.

## Possible Actions
- Clean the current room (Suck).
- Move Right (from A to B).
- Move Left (from B to A).

## State Representation
Each state is represented as:

`(Location, Room A, Room B)`

Where:
- `A` or `B` represents the vacuum cleaner's current location.
- `0` represents Dirty.
- `1` represents Clean.

Initial State: `('A', 0, 0)`

Goal State: Both rooms must be clean.

## Algorithm Used
Breadth-First Search (BFS).

BFS explores possible states level by level using a queue. Visited-state tracking prevents repeated exploration.

## Execution

```bash
python3 vacuum_cleaner.py
```

## Sample Output

```text
Vacuum Cleaner Solution
----------------------------
Initial State: ('A', 0, 0)
Clean Room A -> ('A', 1, 0)
Move Right -> ('B', 1, 0)
Clean Room B -> ('B', 1, 1)
----------------------------
Total steps: 3
Both rooms are clean!
```

## Result
The vacuum cleaner successfully cleans both rooms using BFS and reaches the goal state in three actions.


## 3. Missionaries and Cannibals Problem

## Objective
To solve the Missionaries and Cannibals Problem using State Space Search and Breadth-First Search (BFS).

## Problem Description
Three missionaries and three cannibals are present on the left bank of a river. They must cross to the right bank using a boat that can carry a maximum of two people.

The number of cannibals must never exceed the number of missionaries on either bank when missionaries are present.

## State Representation

Each state is represented as:

`(M, C, B)`

Where:
- `M` = Number of missionaries on the left bank.
- `C` = Number of cannibals on the left bank.
- `B` = Boat position (0 = Left, 1 = Right).

Initial State: `(3, 3, 0)`

Goal State: `(0, 0, 1)`

## Possible Boat Movements
- 1 Missionary
- 2 Missionaries
- 1 Cannibal
- 2 Cannibals
- 1 Missionary and 1 Cannibal

## Algorithm Used
Breadth-First Search (BFS).

The algorithm explores valid states using a queue and avoids unsafe states where cannibals outnumber missionaries.

## Execution

```bash
python3 missionaries_cannibals.py
```

## Expected Result

The algorithm finds a valid sequence of 11 boat crossings, transferring all six people safely to the opposite bank.

## Result
The problem was successfully solved using BFS while maintaining the safety constraints on both river banks.


## 4. State Space Representation using 8-Puzzle

## Objective
To represent the 8-Puzzle problem as a state space and generate all possible next states by moving the blank tile.

## Problem Description
The 8-Puzzle consists of a 3 × 3 board containing eight numbered tiles and one empty space represented by `0`.

The objective is to understand how a given state can be transformed into other valid states by moving the empty space.

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

A movement is allowed only when the blank tile does not cross the boundary of the 3 × 3 board.

## State Representation
The puzzle is represented using a tuple of nine elements.

The value `0` represents the blank tile.

## Execution

```bash
python3 state_space_representation.py
```

## Sample Output

```text
Initial State:
1 2 3
4 0 6
7 5 8

Goal State:
1 2 3
4 5 6
7 8 0

Possible Next States:
Move Up
1 0 3
4 2 6
7 5 8

Move Down
1 2 3
4 5 6
7 0 8

Move Left
1 2 3
0 4 6
7 5 8

Move Right
1 2 3
4 6 0
7 5 8
```

## Result
The 8-Puzzle was successfully represented using a state-space approach. All four valid successor states were generated from the given initial configuration.
