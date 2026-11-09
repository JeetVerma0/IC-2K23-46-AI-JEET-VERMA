"""
Lab 4: Knowledge Representation and Logical Reasoning
Experiment 4: Rule-Based Knowledge Representation and Inference System

Implements a forward-chaining rule engine, a university knowledge base,
an educational medical knowledge representation, and conceptual dependency.
"""


class Rule:
    """Represents an IF-THEN production rule with conditions and conclusion."""

    def __init__(self, conditions, conclusion, description=""):
        self.conditions = set(conditions)
        self.conclusion = conclusion
        self.description = description

    def evaluate(self, facts):
        """Check if all conditions are satisfied by existing facts."""
        return self.conditions.issubset(facts)


class RuleEngine:
    """Production rule engine evaluating rules through forward chaining."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact):
        """Add an initial fact to the knowledge base."""
        self.facts.add(fact)

    def add_rule(self, rule):
        """Add an IF-THEN rule to the engine."""
        self.rules.append(rule)

    def forward_chain(self):
        """Iteratively evaluate rules and infer new facts until closure."""
        new_inferences = []
        cycle = 1
        while True:
            inferred = False
            for rule in self.rules:
                if rule.conclusion not in self.facts and rule.evaluate(self.facts):
                    self.facts.add(rule.conclusion)
                    new_inferences.append(rule.conclusion)
                    inferred = True
                    print(f"  [Cycle {cycle}] Triggered: {rule.description}")
                    print(f"             Derived: {rule.conclusion}")
            if not inferred:
                break
            cycle += 1
        return new_inferences


def demo_university_kb():
    """A & B. University Knowledge Base: prerequisites and eligibility."""
    print("=== UNIVERSITY KNOWLEDGE BASE & FORWARD CHAINING ===")
    engine = RuleEngine()
    for fact in [
        "Enrolled(Jeet, AI)", "Passed(Jeet, Data_Structures)",
        "Completed_Credits(Jeet, 80)", "Prerequisite(AI, Data_Structures)",
        "Good_Conduct(Jeet)"
    ]:
        engine.add_fact(fact)

    engine.add_rule(Rule(
        {"Enrolled(Jeet, AI)", "Passed(Jeet, Data_Structures)", "Prerequisite(AI, Data_Structures)"},
        "Eligible_For_Course(Jeet, AI)", "Rule 1: Course Prerequisite Rule"
    ))
    engine.add_rule(Rule(
        {"Eligible_For_Course(Jeet, AI)", "Good_Conduct(Jeet)"},
        "Permitted_For_Exam(Jeet, AI)", "Rule 2: Exam Eligibility Rule"
    ))
    engine.add_rule(Rule(
        {"Completed_Credits(Jeet, 80)", "Permitted_For_Exam(Jeet, AI)"},
        "Dean_List_Candidate(Jeet)", "Rule 3: Honors Candidacy Rule"
    ))

    print("Initial Facts:", sorted(engine.facts))
    print("\nRunning Forward Chaining:")
    derived = engine.forward_chain()
    print(f"Total New Facts Inferred: {len(derived)}")


def demo_medical_kb():
    """C. Medical Knowledge Representation (Educational Demonstration)."""
    print("\n=== MEDICAL KNOWLEDGE REPRESENTATION (EDUCATIONAL) ===")
    print("NOTE: Educational demonstration of rule chaining only, NOT a clinical tool.\n")
    engine = RuleEngine()
    for symptom in ["Symptom(Fever)", "Symptom(Cough)", "Symptom(Sore_Throat)", "Duration_Days(3)"]:
        engine.add_fact(symptom)

    engine.add_rule(Rule(
        {"Symptom(Fever)", "Symptom(Cough)", "Symptom(Sore_Throat)"},
        "Condition(Possible_Upper_Respiratory_Infection)", "Rule M1: Viral Pattern"
    ))
    engine.add_rule(Rule(
        {"Condition(Possible_Upper_Respiratory_Infection)", "Duration_Days(3)"},
        "Action(Recommend_Hydration_And_Clinical_Evaluation)", "Rule M2: Care Protocol"
    ))

    print("Observed Facts:", sorted(engine.facts))
    print("\nEvaluating Medical Inference Rules:")
    engine.forward_chain()


def demo_conceptual_dependency():
    """D. Conceptual Dependency (CD) Representation (Roger Schank)."""
    print("\n=== CONCEPTUAL DEPENDENCY (CD) REPRESENTATION ===")
    print("Sentence: 'Jeet gave an AI textbook to Alex.'")
    cd_frame = {
        "Primitive ACT": "ATRANS (Transfer of Possession)",
        "Actor": "Jeet",
        "Object": "AI Textbook",
        "From (Source)": "Jeet",
        "To (Recipient)": "Alex"
    }
    for role, entity in cd_frame.items():
        print(f"  {role:<32}: {entity}")
    print("\nExplanation:")
    print("  - ATRANS represents the primitive act of transferring possession/ownership.")
    print("  - 'Jeet' is the actor who initiates the transfer from himself to 'Alex'.")


if __name__ == "__main__":
    demo_university_kb()
    demo_medical_kb()
    demo_conceptual_dependency()
