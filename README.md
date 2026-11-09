# Artificial Intelligence Laboratory

This repository contains practical implementations of problems, algorithms, frameworks, and knowledge representation models studied as part of my Artificial Intelligence laboratory coursework.

The experiments are implemented in Python and span across fundamental AI problem-solving paradigms: state-space representation, uninformed search algorithms, graph traversal and pathfinding, tabular data preparation and feature engineering pipelines, as well as symbolic knowledge representation and automated logical reasoning.

The primary objective is to gain hands-on, foundational experience in representing real-world problems mathematically and symbolically, designing search and reasoning algorithms, and preparing data for intelligent decision-making systems.

---

## Laboratory Experiments

### Lab 1 — Introduction to Artificial Intelligence Problem Solving

This lab covers foundational AI problems involving state-space representation, allowable actions, initial and goal state formulations, and systematic search strategies.

| Experiment | Description | Technique |
|---|---|---|
| Water Jug Problem | Obtaining a target quantity of water using two jugs of fixed capacities. | BFS |
| Vacuum Cleaner Problem | Cleaning two rooms by moving an agent between different locations. | State Space Search / BFS |
| Missionaries and Cannibals | Transferring missionaries and cannibals across a river while maintaining safety constraints. | BFS |
| State Space Representation using 8-Puzzle | Representing puzzle configurations and generating valid successor states. | State Space Representation |

**Concepts covered:**
- State-space representation
- Initial and goal states
- State transitions and operators
- Breadth-First Search (BFS)
- Problem formulation
- Search space exploration

[View Lab 1 – Source Code and Documentation](Lab1/README.md)

---

### Lab 2 — Search Algorithms and Navigation

This lab focuses on graph search techniques used to explore search spaces, solve combinatorial puzzles, and navigate through spatial environments.

| Experiment | Description | Technique |
|---|---|---|
| 8-Puzzle Solver | Finding the shortest sequence of moves to transform an initial puzzle configuration into the goal state. | BFS |
| Graph Traversal | Traversing graph vertices systematically using level-order and depth-order strategies. | BFS and DFS |
| Maze Solver | Finding a valid obstacle-free path from start to destination in a 2D grid. | DFS |
| Robot Navigation | Finding the optimal shortest path for an autonomous agent moving through an obstacle grid. | BFS |
| Route Planning | Finding a connected path between two locations represented as vertices in an unweighted road network graph. | BFS |

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

### Lab 3 — AI Programming Frameworks and Data Preparation

This lab provides hands-on experience using core Python data science and machine learning frameworks (**NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn**) to clean, analyze, encode, scale, and engineer tabular datasets while strictly preventing data leakage.

| Experiment | Description | Technique / Libraries |
|---|---|---|
| Data Preprocessing | Inspecting raw data, handling missing values, encoding categorical variables, and applying feature scaling without test-set leakage. | Pandas, NumPy, Scikit-learn (`StandardScaler`, `MinMaxScaler`) |
| Exploratory Data Analysis & Visualization | Performing univariate, bivariate, and multivariate visual analysis to discover patterns, correlations, and class distributions. | Matplotlib, Seaborn (KDE, Boxplots, Heatmaps, Bar charts) |
| Feature Engineering & Feature Selection | Engineering domain-specific features, enforcing leak-free pipelines, and selecting top predictors using statistical hypothesis tests. | Scikit-learn (ANOVA $F$-test `f_classif`, Mutual Information `mutual_info_classif`) |

**Concepts covered:**
- Missing-value handling and domain-specific imputation
- Categorical encoding (Label Encoding and One-Hot Encoding)
- Feature scaling ($Z$-score Standardization vs. Min-Max Normalization)
- Exploratory Data Analysis (EDA) and distribution profiling
- Leak-free preprocessing pipelines (fitting transformers strictly on training sets)
- Domain feature construction and interaction ratios
- Statistical feature selection (ANOVA $F$-tests and Mutual Information)

[View Lab 3 – Source Code and Documentation](Lab3/README.md)

---

### Lab 4 — Knowledge Representation and Logical Reasoning

This lab explores symbolic AI and classical knowledge representation methodologies, implementing declarative knowledge structures, relational networks, object-centered frame hierarchies, and automated rule-based forward chaining engines.

| Experiment | Description | Technique |
|---|---|---|
| Predicate Representation | Representing domain facts, objects, and relationships using first-order predicates, and deriving conclusions via AND, OR, and NOT logical operations. | First-Order Predicate Logic & Unification Querying |
| Semantic Networks | Constructing associative knowledge graphs with nodes and edges representing `IS-A`, `HAS-A`, and `PART-OF` relations with property inheritance. | Directed Associative Graphs & Inheritance Traversal |
| Frame Representation | Modeling hierarchical entities using frames with slots, default facet values, and local slot overrides. | Minsky Frame Theory & Recursive Slot Resolution |
| Logical Rule Engine | Building a production rule engine that performs forward chaining over facts, university eligibility rules, educational medical rules, and Conceptual Dependency. | Production Systems, Forward Chaining & Conceptual Dependency (CD) |

**Concepts covered:**
- First-Order Predicate Logic, arity, and ground fact assertion
- Closed-World Assumption and logical connectives (AND, OR, NOT)
- Semantic networks, associative links (`IS-A`, `HAS-A`, `PART-OF`), and property inheritance
- Frame structures, slots, fillers, default values, and local property override
- Production rule engines (Working Memory, Rule Base, Inference Engine)
- Forward chaining data-driven inference until fixed-point saturation
- Conceptual Dependency (CD) theory and primitive actions (`ATRANS`)

[View Lab 4 – Source Code and Documentation](Lab4/README.md)

---

## Algorithms and Methods Implemented

### Breadth-First Search (BFS)
Explores search spaces level-by-level using a FIFO queue. Guarantees finding the shortest path in unweighted graphs and state spaces.
- **Used in:** Water Jug Problem, Vacuum Cleaner Problem, Missionaries and Cannibals, 8-Puzzle Solver, Graph Traversal, Robot Navigation, Route Planning.

### Depth-First Search (DFS)
Explores deep along graph branches before backtracking using an explicit stack or recursion.
- **Used in:** Graph Traversal, Maze Solver.

### Tabular Data Preprocessing and Feature Selection
Systematic pipelines to clean, transform, and select high-predictive features from raw tabular datasets without information leakage.
- **Techniques:** Median/Mode Imputation, One-Hot Encoding, $Z$-Score Standardization, Min-Max Normalization, ANOVA $F$-test (`f_classif`), Mutual Information (`mutual_info_classif`).

### First-Order Predicate Inference
Declarative reasoning evaluating exact assertions, wildcard unification queries, and logical truth rules across Conjunction ($\wedge$), Disjunction ($\vee$), and Negation ($\neg$).
- **Used in:** Academic course eligibility, exam clearance, and graduation auditing.

### Semantic Graph Traversal & Property Inheritance
Depth-first upward traversal along taxonomic `IS-A` links to inherit physical and conceptual properties from ancestor nodes without redundant storage.
- **Used in:** University knowledge network modeling students, professors, courses, and departments.

### Frame-Based Recursive Slot Resolution
Object-oriented knowledge representation that resolves slot queries locally first, falling back to parent frames to inherit default attributes.
- **Used in:** Class and instance frames for students, faculty, and academic units.

### Forward Chaining Production Rule Engine
Data-driven inference engine that matches active fact premises against production rules, asserting newly derived conclusions into working memory until saturation.
- **Used in:** University academic prerequisite pipelines, educational medical symptom evaluation, and Conceptual Dependency decomposition.

---

## Technologies and Tools

| Technology / Library | Version / Role | Purpose |
|---|---|---|
| **Python 3** | `>= 3.10` | Core programming language for all algorithms and models. |
| **NumPy** | `2.5.3` | High-performance numerical computing and array operations. |
| **Pandas** | `3.0.6` | Tabular data manipulation, cleaning, aggregation, and encoding. |
| **Matplotlib** | `3.11.2` | Plotting, axes formatting, and figure visualization. |
| **Seaborn** | `0.13.2` | Statistical data visualization (distributions, heatmaps, boxplots). |
| **Scikit-learn** | `1.9.1` | Data preprocessing transformations and statistical feature selection. |
| **Jupyter Notebook** | Interactive Lab 3 execution | Interactive execution and visualization rendering. |
| **Git & GitHub** | Version Control | Source code tracking, documentation, and repository management. |

---

## Repository Structure

```text
Artificial_Intelligence/
├── README.md                          # Root laboratory repository documentation
├── datasets/                          # Benchmark datasets for ML & data preparation
│   ├── Placement_Data_Full_Class.csv
│   └── bank.csv
├── Lab1/                              # Lab 1: Introduction to Problem Solving
│   ├── README.md
│   ├── water_jug.py
│   ├── vacuum_cleaner.py
│   ├── missionaries_cannibals.py
│   └── state_space_representation.py
├── Lab2/                              # Lab 2: Search Algorithms and Navigation
│   ├── README.md
│   ├── eight_puzzle_solver.py
│   ├── graph_traversal.py
│   ├── maze_solver.py
│   ├── robot_navigation.py
│   └── route_planning.py
├── Lab3/                              # Lab 3: AI Frameworks & Data Preparation
│   ├── README.md
│   ├── data_preprocessing.ipynb
│   ├── eda_visualization.ipynb
│   ├── feature_engineering.ipynb
│   └── outputs/                       # Exported visualization figures
└── Lab4/                              # Lab 4: Knowledge Representation & Reasoning
    ├── README.md
    ├── predicate_representation.py
    ├── semantic_network.py
    ├── frame_representation.py
    └── logical_rule_engine.py
```

---

## Running the Programs

Ensure **Python 3** is installed. Navigate to the repository root directory before executing any command.

### Running Search & Problem-Solving Experiments (Lab 1 & Lab 2)
Each script runs standalone with no external dependencies:

```bash
# Lab 1 examples
python3 Lab1/water_jug.py
python3 Lab1/vacuum_cleaner.py
python3 Lab1/missionaries_cannibals.py
python3 Lab1/state_space_representation.py

# Lab 2 examples
python3 Lab2/eight_puzzle_solver.py
python3 Lab2/graph_traversal.py
python3 Lab2/maze_solver.py
python3 Lab2/robot_navigation.py
python3 Lab2/route_planning.py
```

### Running Data Preparation & Visualization Notebooks (Lab 3)
Install required data science libraries and start Jupyter:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
jupyter notebook Lab3/
```
Open and execute [`data_preprocessing.ipynb`](Lab3/data_preprocessing.ipynb), [`eda_visualization.ipynb`](Lab3/eda_visualization.ipynb), or [`feature_engineering.ipynb`](Lab3/feature_engineering.ipynb).

### Running Knowledge Representation & Reasoning Systems (Lab 4)
Execute each experiment directly using standard Python:

```bash
# Experiment 1: Predicate Logic & Truth Inference
python3 Lab4/predicate_representation.py

# Experiment 2: Semantic Networks & Property Inheritance
python3 Lab4/semantic_network.py

# Experiment 3: Frame Structures & Slot Resolution
python3 Lab4/frame_representation.py

# Experiment 4: Forward Chaining Rule Engine & Conceptual Dependency
python3 Lab4/logical_rule_engine.py
```

---

## Documentation

Each laboratory directory contains a dedicated, comprehensive `README.md` file providing:
- Formal theoretical background and problem formulations
- Algorithm step-by-step procedures and complexity analysis
- Architectural and workflow diagrams
- Verified source code implementations and execution outputs
- Methodological observations, comparisons, and learning outcomes

---

## Progress

The repository reflects ongoing coursework in Artificial Intelligence:

- [x] **Lab 1**: AI Problem Solving, State Spaces, and Uninformed Search (Complete)
- [x] **Lab 2**: Search Algorithms, Graph Traversal, and Grid Navigation (Complete)
- [x] **Lab 3**: AI Programming Frameworks, Data Preprocessing, EDA, and Feature Engineering (Complete)
- [x] **Lab 4**: Knowledge Representation, Semantic Networks, Frames, and Rule Engines (Complete)
