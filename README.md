# Artificial Intelligence Laboratory

This repository contains practical implementations of problems, algorithms, frameworks, and knowledge representation models studied as part of my Artificial Intelligence laboratory coursework.

The experiments are implemented in Python and span across fundamental AI problem-solving paradigms: state-space representation, uninformed search algorithms, graph traversal and pathfinding, tabular data preparation and feature engineering pipelines, symbolic knowledge representation and automated logical reasoning, interpretable decision tree classifiers, foundational deep learning with PyTorch, as well as advanced topics in Explainable AI (Grad-CAM), Adversarial AI (FGSM), and unsupervised dimensionality reduction (PCA & SVD).

The primary objective is to gain hands-on, foundational experience in representing real-world problems mathematically and symbolically, designing search and reasoning algorithms, preparing data for intelligent decision-making systems, training neural architectures using gradient-based optimization, and evaluating model explainability and security guarantees.

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

### Lab 5 — Decision Tree Classifier

This lab covers white-box supervised classification using Scikit-Learn's `DecisionTreeClassifier` on the Campus Placement dataset. It demonstrates how to predict recruitment outcomes (`Placed` vs. `Not Placed`) while eliminating post-outcome target leakage, constructing leak-free preprocessing pipelines, evaluating multi-class metrics, visualizing tree architectures, and diagnosing overfitting.

| Experiment | Description | Technique / Deliverable |
|---|---|---|
| Placement Classification Pipeline | End-to-end classification workflow: EDA, leakage prevention, `ColumnTransformer` preprocessing, 80:20 stratified split, evaluation, tree visualization, and depth tuning. | [`Lab5/decision_tree_classifier.ipynb`](Lab5/decision_tree_classifier.ipynb) |
| Reusable Decision Tree Engine | Modular Python implementation providing functions for dataset loading, feature preparation, pipeline training, evaluation, and diagnostic plot exporting. | [`Lab5/decision_tree_model.py`](Lab5/decision_tree_model.py) |
| Diagnostic Visualizations | High-resolution exports of test-set confusion matrix, decision tree topology splits, and Gini feature importances. | [`Lab5/outputs/`](Lab5/outputs/) (`confusion_matrix.png`, `decision_tree.png`, `feature_importance.png`) |

**Concepts covered:**
- Feature matrix ($X$) and target vector ($y$) formulation
- Prevention of catastrophic target leakage (`salary` dropped due to post-placement dependency)
- Non-predictive identifier removal (`sl_no`)
- Leak-free preprocessing pipelines (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`)
- Stratified train-test splitting (80:20 ratio, `random_state=42`)
- Gini Impurity minimization and recursive binary partitioning (CART)
- Model evaluation: Accuracy ($69.77\%$), Precision ($75.76\%$), Recall ($83.33\%$), F1-Score ($79.37\%$), Confusion Matrix, and Classification Report
- Decision tree topology visualization (`plot_tree`) and continuous threshold analysis
- Gini feature importance (identifying secondary percentage `ssc_p` as dominant predictor at $44.78\%$)
- Variance inflation, overfitting mechanics at deep levels, and depth pre-pruning (`max_depth=2` achieving $79.07\%$ test accuracy)

[View Lab 5 – Source Code and Documentation](Lab5/README.md)

---

### Lab 6 — Deep Learning using PyTorch

This lab introduces fundamental deep learning concepts using **PyTorch** (`torch`), progressing from multidimensional tensor manipulation and automatic differentiation (`torch.autograd`) to parametric single-layer linear regression and non-linear classification with a Multi-Layer Perceptron (MLP).

| Experiment | Description | Technique / Deliverable |
|---|---|---|
| PyTorch Basics & Autograd | Multidimensional tensor creation, indexing, slicing, reshaping (`view`), matrix multiplication (`@`), dynamic computational graphs, and reverse-mode differentiation (`.backward()`). | [`Lab6/pytorch_basics.ipynb`](Lab6/pytorch_basics.ipynb) |
| Linear Regression using PyTorch | Parametric regression on synthetic data ($y=3.5x+12.0+\epsilon$): `torch.nn.Linear`, `nn.MSELoss`, `optim.SGD`, 150-epoch training loop, $R^2$ evaluation ($96.19\%$), and fit plotting. | [`Lab6/linear_regression.ipynb`](Lab6/linear_regression.ipynb) |
| Neural Network Fundamentals | Non-linear binary classification on synthetic moons data using a modular feedforward neural network (`Linear -> ReLU -> Linear`), `BCEWithLogitsLoss`, Adam optimization, and decision boundary visualization. | [`Lab6/neural_network_basics.py`](Lab6/neural_network_basics.py) |
| Deep Learning Visualizations | Exported loss curves and diagnostic plots: training loss descent, regression fit line, and non-linear classification decision boundary contour. | [`Lab6/outputs/`](Lab6/outputs/) (`training_loss.png`, `actual_vs_predicted.png`, `neural_network_results.png`) |

**Concepts covered:**
- Multidimensional PyTorch tensors, metadata (`shape`, `ndim`, `dtype`, `device`), and zero-copy NumPy interoperability
- Vectorized arithmetic, dot products, and matrix multiplication (`torch.matmul`, `@`)
- Dynamic computational graphs (Directed Acyclic Graphs) and Autograd gradient tracking (`requires_grad=True`, `grad_fn`)
- Analytical vs. automatic reverse-mode differentiation (`.backward()`, `x.grad`) and gradient suppression (`torch.no_grad()`)
- Subclassing `torch.nn.Module` for custom layer definitions and forward propagation
- Loss functions: Mean Squared Error (`nn.MSELoss`) and Binary Cross-Entropy (`nn.BCEWithLogitsLoss`)
- Optimizers: Stochastic Gradient Descent (`optim.SGD`) and Adaptive Moment Estimation (`optim.Adam`)
- Canonical 5-step PyTorch training lifecycle (forward pass $\to$ loss computation $\to$ `zero_grad` $\to$ `loss.backward` $\to$ `optimizer.step`)
- Non-linear representation learning via Rectified Linear Units (`ReLU`)
- Quantitative evaluation ($R^2$, RMSE, Accuracy: $95.00\%$, Precision, Recall, Confusion Matrix) and visual decision boundary mapping

[View Lab 6 – Source Code and Documentation](Lab6/README.md)

---

### Lab 7 — Explainable AI, Adversarial AI and Advanced Applications

This lab explores critical challenges in modern AI deployment: interpreting deep vision models using Explainable AI (**Grad-CAM**), evaluating model vulnerability under adversarial attacks (**FGSM**), and applying linear dimensionality reduction techniques (**PCA & SVD**).

| Experiment | Description | Technique / Deliverable |
|---|---|---|
| Grad-CAM Explainability | Visualizing CNN attention on CIFAR-10: forward/backward hook registration, Global Average Pooling of gradients, class activation heatmaps, and spatial feature localization. | [`Lab7/grad_cam_explainability.ipynb`](Lab7/grad_cam_explainability.ipynb) |
| FGSM Adversarial Attack | White-box adversarial generation on CIFAR-10 using the Fast Gradient Sign Method, evaluating robustness and accuracy collapse across perturbation budgets ($\epsilon \in [0, 0.30]$). | [`Lab7/fgsm_adversarial_attack.ipynb`](Lab7/fgsm_adversarial_attack.ipynb) |
| PCA and SVD Dimensionality Reduction | Unsupervised dimensionality reduction on the Iris dataset, scree plot analysis ($95.81\%$ retained variance), 2D class separation, and mathematical proof of PCA $\equiv$ SVD ($\Delta < 10^{-14}$). | [`Lab7/pca_svd_dimensionality_reduction.ipynb`](Lab7/pca_svd_dimensionality_reduction.ipynb) |
| Advanced AI Visualizations | Exported diagnostic figures: 3-panel Grad-CAM activation heatmap overlay, clean vs. adversarial comparison, and 2D PCA projection scatter plot. | [`Lab7/outputs/`](Lab7/outputs/) (`grad_cam_visualization.png`, `adversarial_comparison.png`, `pca_visualization.png`) |

**Concepts covered:**
- Explainable AI (XAI) and visual attribution for convolutional neural networks
- Dynamic PyTorch hook architecture (`register_forward_hook`, `register_full_backward_hook`)
- Gradient-Weighted Class Activation Mapping (Grad-CAM) mathematical formulation
- Adversarial vulnerability and the Linearity Hypothesis in deep models (Goodfellow et al.)
- Fast Gradient Sign Method (FGSM) and $\ell_\infty$-bounded input perturbation generation
- Empirical model robustness quantification as a function of perturbation magnitude ($\epsilon$)
- Feature standardization and variance maximization in linear projections
- Principal Component Analysis (spectral eigen-decomposition of sample covariance $\Sigma = V \Lambda V^T$)
- Singular Value Decomposition ($X = U \Sigma V^T$) and exact mathematical equivalence with PCA ($\lambda_i = \frac{s_i^2}{N-1}$)

[View Lab 7 – Source Code and Documentation](Lab7/README.md)

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

### Decision Tree Classification (CART Algorithm)
Non-parametric recursive partitioning that recursively minimizes Gini Impurity to construct interpretable, axis-parallel orthogonal decision boundaries.
- **Used in:** Student campus placement prediction (`Placed` vs. `Not Placed`) with depth pre-pruning to combat overfitting.

### Automatic Differentiation & Dynamic Graphs (PyTorch Autograd)
Reverse-mode automatic differentiation that dynamically constructs Directed Acyclic Graphs during forward propagation and evaluates exact parameter gradients via the chain rule.
- **Used in:** Computing gradients $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$ for parametric neural models, and input image gradients $\nabla_x \mathcal{L}$ for adversarial attacks.

### Parametric Deep Learning Optimization
Iterative optimization algorithms updating continuous parameter tensors along negative gradient vectors to minimize empirical loss surfaces.
- **Used in:** Linear regression using SGD with MSE loss, non-linear Multi-Layer Perceptron (MLP) classification using Adam, and lightweight CNN feature learning.

### Explainable AI (Grad-CAM)
Gradient-Weighted Class Activation Mapping that computes the gradient of target class logits with respect to final convolutional feature maps, applying Global Average Pooling to weight feature maps and rectifying with ReLU to produce discriminative spatial heatmaps.
- **Used in:** Explaining visual decision rationale on CIFAR-10 image classification.

### Adversarial Perturbation Generation (FGSM)
White-box adversarial attack synthesizing $\ell_\infty$-bounded perturbations along the gradient sign direction $x_{\text{adv}} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}), 0, 1)$ to test neural network robustness.
- **Used in:** Evaluating CNN resilience under adversarial noise across varying budgets ($\epsilon \in [0, 0.30]$).

### Dimensionality Reduction (PCA & SVD)
Projecting high-dimensional feature spaces onto maximal variance orthogonal axes via covariance eigen-decomposition and singular value matrix factorization ($X = U \Sigma V^T$).
- **Used in:** Compressing 4D Iris measurements into 2D space while preserving $95.81\%$ of total statistical variance.

---

## Technologies and Tools

| Technology / Library | Version / Role | Purpose |
|---|---|---|
| **Python 3** | `>= 3.10` | Core programming language for all algorithms and models. |
| **PyTorch (`torch`)** | `2.15.0+cpu` | Deep learning framework, tensor computation, autograd, CNNs, and hook mechanisms. |
| **Torchvision** | `0.29.2+cpu` | Computer vision datasets (CIFAR-10) and tensor image transformation pipelines. |
| **NumPy** | `2.5.3` | High-performance numerical computing, array operations, and SVD decomposition. |
| **Pandas** | `3.0.6` | Tabular data manipulation, cleaning, aggregation, and encoding. |
| **Matplotlib** | `3.11.2` | Plotting, axes formatting, heatmap blending, and figure visualization. |
| **Seaborn** | `0.13.2` | Statistical data visualization (distributions, heatmaps, boxplots). |
| **Scikit-learn** | `1.9.1` | Data preprocessing transformations, statistical feature selection, Decision Trees, Iris dataset, and PCA. |
| **Jupyter Notebook** | Interactive Notebooks | Interactive execution and visualization rendering for Labs 3, 5, 6, and 7. |
| **Git & GitHub** | Version Control | Source code tracking, documentation, and repository management. |

---

## Repository Structure

```text
Artificial_Intelligence/
├── README.md                          # Root laboratory repository documentation
├── docs/                              # Academic laboratory documentation portfolio
│   ├── diagrams/                      # System architecture & state-space diagrams
│   ├── algorithm-flowcharts/          # Procedural algorithm execution flowcharts
│   ├── complexity-analysis.pdf        # Algorithmic space-time complexity analysis
│   ├── final-report.pdf               # Comprehensive 13-page academic laboratory report
│   └── presentation.pdf               # 18-slide academic defense presentation
├── datasets/                          # Benchmark datasets for ML & deep learning
│   ├── Placement_Data_Full_Class.csv
│   ├── bank.csv
│   ├── cifar-10-python.tar.gz         # Official CIFAR-10 Python archive
│   └── cifar-10-python/               # Extracted CIFAR-10 batch files
│       └── cifar-10-batches-py/
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
├── Lab4/                              # Lab 4: Knowledge Representation & Reasoning
│   ├── README.md
│   ├── predicate_representation.py
│   ├── semantic_network.py
│   ├── frame_representation.py
│   └── logical_rule_engine.py
├── Lab5/                              # Lab 5: Decision Tree Classifier
│   ├── README.md
│   ├── decision_tree_classifier.ipynb
│   ├── decision_tree_model.py
│   └── outputs/                       # Exported confusion matrix, tree, feature importance
├── Lab6/                              # Lab 6: Deep Learning using PyTorch
│   ├── README.md
│   ├── pytorch_basics.ipynb
│   ├── linear_regression.ipynb
│   ├── neural_network_basics.py
│   └── outputs/                       # Exported loss curves, regression fit, decision boundary
└── Lab7/                              # Lab 7: Explainable AI, Adversarial AI & Advanced Applications
    ├── README.md
    ├── grad_cam_explainability.ipynb
    ├── fgsm_adversarial_attack.ipynb
    ├── pca_svd_dimensionality_reduction.ipynb
    └── outputs/                       # Exported Grad-CAM, adversarial comparison, PCA scatter
```

---

## Running the Programs

Ensure **Python 3** and required libraries are installed. Navigate to the repository root directory before executing any command:

```bash
cd /home/jeet/Desktop/Artificial_Intelligence
```

### Running Search & Problem-Solving Experiments (Lab 1 & Lab 2)
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

### Running Data Preparation Notebooks (Lab 3)
```bash
jupyter notebook Lab3/
```
Execute [`data_preprocessing.ipynb`](Lab3/data_preprocessing.ipynb), [`eda_visualization.ipynb`](Lab3/eda_visualization.ipynb), or [`feature_engineering.ipynb`](Lab3/feature_engineering.ipynb).

### Running Knowledge Representation & Reasoning Systems (Lab 4)
```bash
python3 Lab4/predicate_representation.py
python3 Lab4/semantic_network.py
python3 Lab4/frame_representation.py
python3 Lab4/logical_rule_engine.py
```

### Running Decision Tree Experiments (Lab 5)
```bash
python3 Lab5/decision_tree_model.py
jupyter notebook Lab5/decision_tree_classifier.ipynb
```

### Running PyTorch Deep Learning Experiments (Lab 6)
```bash
python3 Lab6/neural_network_basics.py
jupyter notebook Lab6/pytorch_basics.ipynb
jupyter notebook Lab6/linear_regression.ipynb
```

### Running Explainable AI, Adversarial AI & Advanced Applications (Lab 7)
```bash
jupyter notebook Lab7/grad_cam_explainability.ipynb
jupyter notebook Lab7/fgsm_adversarial_attack.ipynb
jupyter notebook Lab7/pca_svd_dimensionality_reduction.ipynb
```

---

## Documentation

### Academic Documentation Directory (`docs/`)

The complete, professional academic documentation for the laboratory coursework is housed in the [`docs/`](docs/) directory:

| Document / Deliverable | Description | Format |
|---|---|---|
| **[Final Laboratory Report](docs/final-report.pdf)** | Comprehensive 13-page academic report covering all seven units, problem formulations, algorithms, code components, empirical observations, and complexity analysis. | PDF |
| **[Academic Presentation](docs/presentation.pdf)** | Professional 18-slide academic defense presentation summarizing the laboratory's theoretical foundations, search strategies, ML/DL models, and key findings. | PDF |
| **[Complexity Analysis](docs/complexity-analysis.pdf)** | Detailed analysis of space-time bounds, branching factors, optimality, completeness, and informed vs. uninformed search trade-offs. | PDF |
| **[System Diagrams](docs/diagrams/)** | High-resolution technical diagrams depicting agent models, state spaces, KR graphs, PyTorch computational DAGs, Grad-CAM, and FGSM pipelines. | PNG / PUML / Draw.io |
| **[Algorithm Flowcharts](docs/algorithm-flowcharts/)** | High-resolution procedural flowcharts for BFS, DFS, CART, PyTorch training loops, rule engines, and dimensionality reduction. | PNG / PUML / Draw.io |

### Unit-Level Documentation

Each laboratory directory (`Lab1/` through `Lab7/`) contains a dedicated, comprehensive `README.md` file providing:
- Formal theoretical background and mathematical formulations
- Algorithm step-by-step procedures and complexity analysis
- Architectural, flowchart, and computational graph diagrams
- Verified source code implementations and execution outputs
- Methodological observations, performance comparisons, and learning outcomes

---

## Progress

The repository reflects ongoing coursework in Artificial Intelligence:

- [x] **Lab 1**: AI Problem Solving, State Spaces, and Uninformed Search (Complete)
- [x] **Lab 2**: Search Algorithms, Graph Traversal, and Grid Navigation (Complete)
- [x] **Lab 3**: AI Programming Frameworks, Data Preprocessing, EDA, and Feature Engineering (Complete)
- [x] **Lab 4**: Knowledge Representation, Semantic Networks, Frames, and Rule Engines (Complete)
- [x] **Lab 5**: Decision Tree Classification, Interpretability, and Overfitting Regularization (Complete)
- [x] **Lab 6**: Deep Learning using PyTorch, Autograd, Linear Regression, and Neural Networks (Complete)
- [x] **Lab 7**: Explainable AI (Grad-CAM), Adversarial Machine Learning (FGSM), and Dimensionality Reduction (PCA & SVD) (Complete)
