from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SEMANTIC = DATA / "semantic_source"
DERIVED = DATA / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


def save_table(table, filename):
    """Save a reproducible result without scattering CSVs around the project."""
    destination = DERIVED / filename
    table.to_csv(destination, index=False)
    return destination
