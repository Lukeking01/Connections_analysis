
SCRABBLE_VALUES = {
    **dict.fromkeys("AEILNORSTU", 1),
    **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3),
    **dict.fromkeys("FHVWY", 4),
    "K": 5,
    **dict.fromkeys("JX", 8),
    **dict.fromkeys("QZ", 10),
}

DIFFICULTY_MAP = {
    "yellow": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
}

# ================================
# Define grouping keys
# ================================

group_keys = ["puzzle_date", "category"]
puzzle_keys = ["puzzle_date"]

