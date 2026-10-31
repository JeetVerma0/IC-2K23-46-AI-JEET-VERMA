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
