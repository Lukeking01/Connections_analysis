"""Load puzzle data and construct one observation per puzzle."""
import pandas as pd
from .paths import RAW, SEMANTIC, save_table

SPECIAL_PUZZLES = {"2024-04-01"}  # Emoji-only April Fools puzzle.


def load_data():
    """Read the three source CSVs; preserve literal strings such as 'NA'."""
    words = pd.read_csv(RAW / "connections2.csv", keep_default_na=False)
    words["difficulty"] = pd.to_numeric(words["difficulty"], errors="coerce")
    words["puzzle_date"] = pd.to_datetime(words["puzzle_date"])
    groups = pd.read_csv(SEMANTIC / "semantic_group_statistics.csv")
    pairs = pd.read_csv(SEMANTIC / "semantic_pair_statistics.csv")
    return words, groups, pairs


def quality_report(words):
    """Check the expected 16 words and four groups per puzzle."""
    result = words.groupby("puzzle_date").agg(
        word_rows=("word", "size"),
        unique_words=("word", "nunique"),
        groups=("category", "nunique"),
        difficulty=("difficulty", "first"),
    ).reset_index()
    result["valid_shape"] = (result.word_rows == 16) & (result.groups == 4)
    save_table(result, "quality_check.csv")
    return result


def puzzle_features(words, pairs):
    """Combine word length and within/across-group similarity into puzzle rows."""
    words = words.copy()
    words["word_length"] = words["word"].astype(str).str.len()
    basics = words.groupby("puzzle_date").agg(
        puzzle_number=("puzzle_number", "first"),
        difficulty=("difficulty", "first"),
        mean_word_length=("word_length", "mean"),
        sd_word_length=("word_length", "std"),
    ).reset_index()
    basics["puzzle_id"] = (basics["puzzle_date"].dt.strftime("%Y-%m-%d") + "_" +
                           basics["puzzle_number"].astype(str))
    summary = pairs.groupby(["puzzle_id", "pair_type"])["similarity"].mean().unstack()
    summary = summary.rename(columns={
        "Same group": "within_group_similarity",
        "Different groups": "cross_group_similarity",
    }).reset_index()
    puzzles = basics.merge(summary, on="puzzle_id", how="left")
    puzzles["similarity_gap"] = (puzzles["within_group_similarity"] -
                                  puzzles["cross_group_similarity"])
    save_table(puzzles, "puzzle_features.csv")
    return puzzles


def written_word_puzzles(puzzles):
    """Exclude the known emoji-only puzzle from text-based measurements."""
    return puzzles.loc[~puzzles["puzzle_date"].dt.strftime("%Y-%m-%d").isin(SPECIAL_PUZZLES)].copy()


def puzzle_examples(words):
    """Build dated puzzle records for solution-prediction experiments."""
    examples = []
    for date, group in words.groupby("puzzle_date", sort=True):
        group = group.sort_values(["color", "position"])
        terms = group["word"].astype(str).tolist()
        if len(terms) != 16 or len(set(terms)) != 16:
            continue
        if date.strftime("%Y-%m-%d") in SPECIAL_PUZZLES:
            continue
        examples.append({"date": date, "words": terms,
                         "colors": group["color"].tolist(),
                         "difficulty": group["difficulty"].iloc[0]})
    return examples
