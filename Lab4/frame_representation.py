"""
Lab 4: Knowledge Representation and Logical Reasoning
Experiment 3: Frame-Based Knowledge Representation

Implements frame data structures with slots, default values,
hierarchical inheritance, and query mechanisms.
"""


class Frame:
    """Represents a knowledge frame with slots, defaults, and inheritance."""

    def __init__(self, name, parent=None, slots=None):
        self.name = name
        self.parent = parent
        self.slots = slots if slots is not None else {}

    def set_slot(self, slot_name, value):
        """Set or update a slot value in this frame."""
        self.slots[slot_name] = value

    def get_slot(self, slot_name):
        """Retrieve slot value locally or inherit recursively from parent."""
        if slot_name in self.slots:
            return self.slots[slot_name]
        if self.parent:
            return self.parent.get_slot(slot_name)
        return None

    def get_all_slots(self):
        """Collect all slots including inherited ones."""
        all_slots = self.parent.get_all_slots() if self.parent else {}
        all_slots.update(self.slots)
        return all_slots

    def display(self):
        """Display the frame and resolved slot values."""
        parent_name = self.parent.name if self.parent else "None"
        print(f"Frame [{self.name}] (Parent: {parent_name}):")
        for k, v in self.get_all_slots().items():
            origin = "(Local)" if k in self.slots else "(Inherited)"
            print(f"  - {k}: {v} {origin}")


def setup_university_frames():
    """Initializes generic class frames and concrete instance frames."""
    # Base Person frame
    person = Frame("Person", slots={"species": "Homo Sapiens", "can_speak": True, "category": "Human"})

    # Subclasses inheriting from Person
    student = Frame("Student", parent=person, slots={"role": "Learner", "status": "Enrolled", "max_credits": 20})
    professor = Frame("Professor", parent=person, slots={"role": "Faculty", "can_grade": True, "designation": "Professor"})

    # Course frame
    course = Frame("Course", slots={"type": "Academic Course", "credits": 4, "grading": "Letter Grade"})

    # Concrete instances
    jeet = Frame("Jeet", parent=student, slots={"student_id": "CS101", "major": "AI", "status": "Active"})
    dr_smith = Frame("Dr_Smith", parent=professor, slots={"faculty_id": "FAC401", "office": "Room 302"})
    ai_course = Frame("AI_Course", parent=course, slots={"code": "CS401", "instructor": "Dr_Smith"})

    return jeet, dr_smith, ai_course


if __name__ == "__main__":
    jeet, dr_smith, ai_course = setup_university_frames()

    print("=== FRAME-BASED KNOWLEDGE REPRESENTATION ===")
    jeet.display()
    print()
    dr_smith.display()
    print()
    ai_course.display()

    print("\n--- Slot Queries & Inheritance ---")
    print(f"Jeet student_id (Local): {jeet.get_slot('student_id')}")
    print(f"Jeet status (Overridden local): {jeet.get_slot('status')}")
    print(f"Jeet max_credits (Inherited from Student): {jeet.get_slot('max_credits')}")
    print(f"Jeet species (Inherited from Person): {jeet.get_slot('species')}")

    print("\n--- Professor Frame Queries ---")
    print(f"Dr_Smith office (Local): {dr_smith.get_slot('office')}")
    print(f"Dr_Smith can_grade (Inherited from Professor): {dr_smith.get_slot('can_grade')}")
    print(f"Dr_Smith species (Inherited from Person): {dr_smith.get_slot('species')}")

    print("\n--- Missing Slot Query Handling ---")
    print(f"Jeet salary (Non-existent slot): {jeet.get_slot('salary')}")
