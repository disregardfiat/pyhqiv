"""
Pipeline for any useful thermo question without reference documents.

hqiv_answer_thermo(question) parses the question, builds the system from the
axiom only, and returns structured answer + plot code.
"""

from pyhqiv.thermo import hqiv_answer_thermo, TESTABLE_PREDICTIONS

questions = [
    "metallic hydrogen transition at 300 K",
    "Si melting at 10 GPa",
    "Argon critical point",
    "phase diagram H2 0-1000 GPa",
]

for q in questions:
    out = hqiv_answer_thermo(q)
    print("Question:", q)
    print("Answer:", out["answer"])
    print("Value:", out["value"], out["unit"])
    print("System:", out["system_used"])
    print()

print("--- Testable predictions (falsifiable) ---")
for p in TESTABLE_PREDICTIONS:
    print(f"  [{p['id']}] {p['statement']}")
    print(f"       Observable: {p['observable']}")
