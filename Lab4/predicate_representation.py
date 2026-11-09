"""
Lab 4: Knowledge Representation and Logical Reasoning
Experiment 1: Predicate Representation and Logical Inference

Implements predicate logic, knowledge base storage, fact querying,
and basic logical inference using AND, OR, and NOT operations.
"""


class Predicate:
    """Represents a logical predicate with a name and arguments."""

    def __init__(self, name, *args):
        self.name = name
        self.args = tuple(args)

    def __repr__(self):
        return f"{self.name}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, self.args))


class KnowledgeBase:
    """Stores facts and infers conclusions using predicate logic."""

    def __init__(self):
        self.facts = set()

    def tell(self, predicate):
        """Add a factual predicate to the knowledge base."""
        self.facts.add(predicate)

    def ask(self, predicate):
        """Check if an exact predicate fact exists in the knowledge base."""
        return predicate in self.facts

    def query(self, name, *pattern_args):
        """Query facts with matching name; None acts as a wildcard argument."""
        results = []
        for f in self.facts:
            if f.name == name and len(f.args) == len(pattern_args):
                if all(p is None or p == a for p, a in zip(pattern_args, f.args)):
                    results.append(f)
        return results

    # Logical Inference Rules (AND, OR, NOT)
    def infer_course_eligibility(self, student, course, prereq):
        """AND Rule: Eligible if Enrolled AND has Passed prerequisite."""
        return self.ask(Predicate("Enrolled", student, course)) and self.ask(Predicate("Passed", student, prereq))

    def infer_exam_permission(self, student):
        """OR Rule: Permitted if GoodAttendance OR SpecialExemption."""
        return self.ask(Predicate("GoodAttendance", student)) or self.ask(Predicate("SpecialExemption", student))

    def infer_graduation_clearance(self, student):
        """NOT Rule: Cleared if CreditsCompleted AND NOT FeePending."""
        return self.ask(Predicate("CreditsCompleted", student)) and not self.ask(Predicate("FeePending", student))


if __name__ == "__main__":
    kb = KnowledgeBase()

    # Populate Knowledge Base with Predicate Facts
    facts = [
        Predicate("Student", "Jeet"), Predicate("Student", "Alex"),
        Predicate("Course", "AI"), Predicate("Enrolled", "Jeet", "AI"),
        Predicate("Enrolled", "Alex", "AI"), Predicate("Passed", "Jeet", "DataStructures"),
        Predicate("GoodAttendance", "Jeet"), Predicate("SpecialExemption", "Alex"),
        Predicate("CreditsCompleted", "Jeet"), Predicate("CreditsCompleted", "Alex"),
        Predicate("FeePending", "Alex")
    ]
    for fact in facts:
        kb.tell(fact)

    print("=== PREDICATE LOGIC KNOWLEDGE BASE ===")
    print("Stored Facts:", ", ".join(str(f) for f in sorted(kb.facts, key=lambda x: (x.name, x.args))))

    print("\n--- Fact & Wildcard Queries ---")
    print(f"Query Enrolled(Jeet, AI): {kb.ask(Predicate('Enrolled', 'Jeet', 'AI'))}")
    print(f"Query Enrolled(Jeet, Robotics) [Missing]: {kb.ask(Predicate('Enrolled', 'Jeet', 'Robotics'))}")
    ai_students = [m.args[0] for m in kb.query("Enrolled", None, "AI")]
    print(f"Wildcard Query Enrolled(?, AI): {ai_students}")

    print("\n--- Logical Inferences ---")
    print(f"[AND] Jeet eligible for AI: {kb.infer_course_eligibility('Jeet', 'AI', 'DataStructures')}")
    print(f"[AND] Alex eligible for AI: {kb.infer_course_eligibility('Alex', 'AI', 'DataStructures')}")
    print(f"[OR]  Jeet exam permission: {kb.infer_exam_permission('Jeet')}")
    print(f"[OR]  Alex exam permission: {kb.infer_exam_permission('Alex')}")
    print(f"[NOT] Jeet graduation clearance: {kb.infer_graduation_clearance('Jeet')}")
    print(f"[NOT] Alex graduation clearance: {kb.infer_graduation_clearance('Alex')}")
