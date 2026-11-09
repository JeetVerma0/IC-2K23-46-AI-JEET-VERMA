"""
Lab 4: Knowledge Representation and Logical Reasoning
Experiment 2: Semantic Network Construction and Property Inheritance

Implements a semantic network with entities, relationship edges
(IS-A, HAS-A, PART-OF), property inheritance, and network queries.
"""

from collections import defaultdict


class SemanticNetwork:
    """Represents a knowledge graph of entities and directed relations."""

    def __init__(self):
        self.network = defaultdict(list)
        self.nodes = set()

    def add_relation(self, source, relation, target):
        """Add a directed relation between source and target entities."""
        self.nodes.add(source)
        self.nodes.add(target)
        self.network[source].append((relation, target))

    def get_relations(self, source, relation_type=None):
        """Query relations originating from a source node."""
        if source not in self.nodes:
            return []
        edges = self.network.get(source, [])
        if relation_type:
            return [target for rel, target in edges if rel.upper() == relation_type.upper()]
        return edges

    def query_property(self, entity, relation_type):
        """Query properties of an entity, traversing IS-A links for inheritance."""
        results = []
        visited = set()

        def traverse(current):
            if current in visited:
                return
            visited.add(current)
            for rel, target in self.network.get(current, []):
                if rel.upper() == relation_type.upper():
                    results.append(target)
                elif rel.upper() == "IS-A":
                    traverse(target)

        traverse(entity)
        return results

    def display(self):
        """Display the semantic network relationships."""
        print("=== SEMANTIC NETWORK RELATIONSHIPS ===")
        for source in sorted(self.nodes):
            edges = self.network.get(source, [])
            if edges:
                links = ", ".join(f"--({rel})--> {target}" for rel, target in edges)
                print(f"[{source}]: {links}")


def build_university_network():
    """Constructs a university knowledge network with students, professors, courses."""
    net = SemanticNetwork()
    # Taxonomy / IS-A Hierarchy
    net.add_relation("Student", "IS-A", "Person")
    net.add_relation("Professor", "IS-A", "Person")
    net.add_relation("Jeet", "IS-A", "Student")
    net.add_relation("Dr_Smith", "IS-A", "Professor")

    # HAS-A Attributes (inheritable properties)
    net.add_relation("Person", "HAS-A", "Brain")
    net.add_relation("Student", "HAS-A", "Student_ID")
    net.add_relation("Professor", "HAS-A", "Office")
    net.add_relation("Jeet", "HAS-A", "Laptop")

    # PART-OF Structural Relationships & Associations
    net.add_relation("CS_Department", "PART-OF", "Engineering_Faculty")
    net.add_relation("AI_Course", "PART-OF", "CS_Department")
    net.add_relation("Jeet", "ENROLLED_IN", "AI_Course")
    net.add_relation("Dr_Smith", "TEACHES", "AI_Course")
    return net


if __name__ == "__main__":
    net = build_university_network()
    net.display()

    print("\n--- Direct Relationship Queries ---")
    print(f"Jeet ENROLLED_IN: {net.get_relations('Jeet', 'ENROLLED_IN')}")
    print(f"Dr_Smith TEACHES: {net.get_relations('Dr_Smith', 'TEACHES')}")
    print(f"Unknown_Node query [Missing]: {net.get_relations('Unknown_Node', 'IS-A')}")

    print("\n--- Property Inheritance via IS-A ---")
    print(f"Properties of Jeet (Direct & Inherited HAS-A): {net.query_property('Jeet', 'HAS-A')}")
    print(f"Properties of Dr_Smith (Direct & Inherited HAS-A): {net.query_property('Dr_Smith', 'HAS-A')}")

    print("\n--- PART-OF Structural Queries ---")
    print(f"AI_Course PART-OF: {net.get_relations('AI_Course', 'PART-OF')}")
    print(f"CS_Department PART-OF: {net.get_relations('CS_Department', 'PART-OF')}")
