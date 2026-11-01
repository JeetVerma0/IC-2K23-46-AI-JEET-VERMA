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


## 2 — Graph Traversal using BFS and DFS

### Objective

To implement and understand two fundamental graph traversal algorithms: Breadth-First Search (BFS) and Depth-First Search (DFS).

The experiment demonstrates how a graph can be traversed systematically using different searching strategies.

### Problem Description

A graph is a non-linear data structure consisting of vertices (nodes) and edges (connections between nodes).

Graph traversal is the process of visiting all reachable vertices of a graph starting from a selected source vertex.

In this experiment, an undirected graph is represented using an adjacency list. Both BFS and DFS are implemented to traverse the graph starting from vertex A.

### Graph Representation

The graph contains six vertices:

A, B, C, D, E and F.

The connections are:

- A is connected to B and C.
- B is connected to A, D and E.
- C is connected to A and F.
- D is connected to B.
- E is connected to B and F.
- F is connected to C and E.

**Adjacency List:**

```text
A -> B, C
B -> A, D, E
C -> A, F
D -> B
E -> B, F
F -> C, E
```

An adjacency list stores the neighboring vertices of each vertex and provides an efficient way to represent sparse graphs.

### Breadth-First Search (BFS)

**Concept**

Breadth-First Search explores a graph level by level. It visits all neighboring vertices of the current vertex before moving to the next level.

BFS uses a Queue data structure that follows the FIFO (First In, First Out) principle.

**Working Procedure**

- Start from the selected source vertex A.
- Mark A as visited and insert it into the queue.
- Remove the front vertex from the queue and display it.
- Insert all unvisited neighboring vertices into the queue.
- Mark each inserted vertex as visited.
- Repeat until the queue becomes empty.

**BFS Traversal:**

```text
A B C D E F
```

**Data Structure:** Queue (`collections.deque`)

### Depth-First Search (DFS)

**Concept**

Depth-First Search explores one branch of a graph as deeply as possible before backtracking to explore other branches.

DFS can be implemented using recursion or an explicit stack. In this experiment, recursive DFS is used, where the function call stack manages traversal.

**Working Procedure**

- Start from the source vertex A.
- Mark A as visited and display it.
- Visit an unvisited adjacent vertex.
- Continue exploring deeper vertices recursively.
- When no unvisited neighbor remains, backtrack.
- Continue until all reachable vertices have been visited.

**DFS Traversal:**

```text
A B D E F C
```

**Data Structure:** Recursion / Call Stack

The traversal order depends on the order of neighboring vertices in the adjacency list.

### Difference Between BFS and DFS

| Feature | BFS | DFS |
|---|---|---|
| Approach | Level by level | Depth-wise |
| Data structure | Queue | Stack / Recursion |
| Traversal | Explores nearby nodes first | Explores one branch first |
| Shortest path | Yes, in unweighted graphs | Not guaranteed |
| Backtracking | Not explicitly required | Required |
| Memory | Can be higher for wide graphs | Depends on search depth |

### Time and Space Complexity

Let V represent the number of vertices and E represent the number of edges.

| Algorithm | Time Complexity | Auxiliary Space |
|---|---|---|
| BFS | O(V + E) | O(V) |
| DFS | O(V + E) | O(V) |

Both algorithms visit each reachable vertex and examine its adjacency list.

These complexities assume an adjacency-list representation.

### Python Implementation Details

**File:** `graph_traversal.py`

**Language:** Python 3

**Libraries and data structures:**

- `collections.deque` – Implements the queue used by BFS.
- `set` – Stores visited vertices and prevents repeated traversal.
- Recursion – Implements DFS traversal.

No external libraries are required.

### Execution Instructions

Navigate to the project directory and execute:

```bash
python3 lab2/graph_traversal.py
```

### Sample Output

```text
Graph Traversal using BFS and DFS
----------------------------------

Graph:
A -> ['B', 'C']
B -> ['A', 'D', 'E']
C -> ['A', 'F']
D -> ['B']
E -> ['B', 'F']
F -> ['C', 'E']

BFS Traversal: A B C D E F
DFS Traversal: A B D E F C
```

### Applications

**Applications of BFS:**
- Finding shortest paths in unweighted graphs.
- Social network connection analysis.
- Web crawling.
- Network broadcasting.

**Applications of DFS:**
- Maze solving.
- Cycle detection in graphs.
- Topological sorting.
- Connected component analysis.

### Result and Conclusion

The graph was successfully traversed using both Breadth-First Search and Depth-First Search.

BFS explored the graph level by level using a queue, whereas DFS explored deeper vertices using recursion and backtracking.

The experiment demonstrates the difference between breadth-wise and depth-wise graph exploration and the importance of selecting an appropriate traversal algorithm for a given problem.
