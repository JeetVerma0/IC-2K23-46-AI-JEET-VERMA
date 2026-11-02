# Artificial Intelligence Laboratory

This repository contains practical implementations of problems and algorithms studied as part of my Artificial Intelligence laboratory coursework.

The experiments are implemented in Python and cover fundamental AI problem-solving techniques, state-space representation, uninformed search algorithms, graph traversal and pathfinding.

The main objective is to understand how different search algorithms work, how problems can be represented as states, and how these techniques can be applied to solve practical problems.

---

## Laboratory Experiments

### Lab 1 — Introduction to Artificial Intelligence Problem Solving

This lab covers basic AI problems involving state-space representation, possible actions, initial states, goal states and problem-solving strategies.

| Experiment | Description | Technique |
|---|---|---|
| Water Jug Problem | Obtaining a target quantity of water using two jugs of fixed capacities. | BFS |
| Vacuum Cleaner Problem | Cleaning two rooms by moving an agent between different locations. | State Space Search / BFS |
| Missionaries and Cannibals | Transferring missionaries and cannibals across a river while maintaining safety constraints. | BFS |
| State Space Representation using 8-Puzzle | Representing puzzle configurations and generating possible successor states. | State Space Representation |

**Concepts covered:**
- State-space representation
- Initial and goal states
- State transitions and operators
- Breadth-First Search
- Problem formulation
- Search space exploration

[View Lab 1 – Source Code and Documentation](Lab1/README.md)

---

### Lab 2 — Search Algorithms and Navigation

This lab focuses on search techniques used to explore graphs, solve puzzles and navigate through different environments.

| Experiment | Description | Technique |
|---|---|---|
| 8-Puzzle Solver | Finding a sequence of moves to transform an initial puzzle configuration into the goal configuration. | BFS |
| Graph Traversal | Traversing graph vertices using two different searching strategies. | BFS and DFS |
| Maze Solver | Finding a valid path from the starting position to the destination while avoiding obstacles. | DFS |
| Robot Navigation | Finding the shortest path for a robot moving through a grid containing obstacles. | BFS |
| Route Planning | Finding a route between two locations represented as vertices in a graph. | BFS |

**Concepts covered:**
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Graph representation using adjacency lists
- Queue and recursion
- Visited-state tracking
- Path reconstruction
- Backtracking
- Grid-based navigation
- Shortest-path search in unweighted graphs

[View Lab 2 – Source Code and Documentation](Lab2/README.md)

---

## Algorithms Implemented

### Breadth-First Search (BFS)

BFS explores possible states level by level, using a queue to maintain the order of exploration.

It is particularly useful for finding the shortest path in unweighted graphs and exploring state spaces systematically.

**Used in:**
- Water Jug Problem
- Vacuum Cleaner Problem
- Missionaries and Cannibals
- 8-Puzzle Solver
- Graph Traversal
- Robot Navigation
- Route Planning

### Depth-First Search (DFS)

DFS explores one branch as deeply as possible before backtracking to explore other available branches.

It can be implemented using recursion or an explicit stack.

**Used in:**
- Graph Traversal
- Maze Solver

---

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python 3 | Implementing algorithms and solving problems |
| Git | Version control and tracking code changes |
| GitHub | Hosting and maintaining the repository |
| VS Code | Writing and executing Python programs |

### Python Concepts Used

- Functions and conditional statements
- Lists, tuples and dictionaries
- Sets and queue operations
- Recursion
- Loops and iteration
- Graphs and adjacency lists
- Two-dimensional arrays
- Basic algorithm analysis

No external Python packages are required for the current experiments beyond the Python standard library.

---

## Repository Structure

```text
artificial-intelligence/
│
├── README.md
│
├── lab1/
│   ├── README.md
│   ├── water_jug.py
│   ├── vacuum_cleaner.py
│   ├── missionaries_cannibals.py
│   └── state_space_representation.py
│
└── lab2/
    ├── README.md
    ├── eight_puzzle_solver.py
    ├── graph_traversal.py
    ├── maze_solver.py
    ├── robot_navigation.py
    └── route_planning.py
```

---

## Running the Programs

Python 3 is required to execute the programs.

Navigate to the repository's root directory and run an individual experiment using:

```bash
python3 lab1/water_jug.py
```

For experiments in Lab 2:

```bash
python3 lab2/graph_traversal.py
```

Each Python file can be executed independently.

---

## Documentation

Each laboratory directory contains a README file with detailed information about its experiments, including:

- Problem statement and objective
- Theoretical background
- Algorithm and working procedure
- State representation, where applicable
- Time and space complexity
- Python implementation details
- Execution instructions
- Sample input and output
- Result and conclusion

---

## Progress

The repository is being updated as I work through the Artificial Intelligence laboratory experiments.

Currently, Lab 1 and Lab 2 have been added. Additional laboratory experiments will be included as the coursework progresses.
