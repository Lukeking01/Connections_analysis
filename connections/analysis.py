"""Course-level descriptive statistics, regression, PCA and K-means."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .paths import save_table

FEATURES = ["within_group_similarity", "cross_group_similarity",
            "similarity_gap", "mean_word_length", "sd_word_length"]
STRUCTURE_FEATURES = ["within_group_similarity", "cross_group_similarity",
                      "mean_word_length", "sd_word_length"]


def difficulty_trends(puzzles):
    rated = puzzles.dropna(subset=["difficulty"]).copy()
    rated["year"] = rated["puzzle_date"].dt.year
    table = rated.groupby("year")["difficulty"].agg(["count", "mean", "median"]).reset_index()
    save_table(table, "difficulty_by_year.csv")
    return table


def color_summary(groups):
    table = groups.groupby("color")["mean_similarity"].agg(["count", "mean", "std"]).reset_index()
    save_table(table, "color_similarity_summary.csv")
    return table


def difficulty_correlations(puzzles):
    rows = []
    for feature in FEATURES:
        # Pairwise deletion retains all available observations for each feature.
        complete = puzzles[[feature, "difficulty"]].dropna()
        rho, p_value = spearmanr(complete[feature], complete["difficulty"])
        rows.append({"feature": feature, "n": len(complete),
                     "spearman_correlation": rho, "p_value": p_value})
    table = pd.DataFrame(rows)
    save_table(table, "difficulty_correlations.csv")
    return table


def difficulty_regression(puzzles, encoder=None):
    """Fit the PyTorch difficulty regressor (requires sentence encoder)."""
    from .ml import fit_difficulty_mlp
    if encoder is None:
        from .clustering import load_encoder
        encoder = load_encoder()
    return fit_difficulty_mlp(puzzles, encoder)


def puzzle_structure(puzzles, n_clusters=4):
    """Cluster *puzzles*, not words. PCA is for visualizing those puzzles."""
    clean = puzzles.dropna(subset=STRUCTURE_FEATURES).copy()
    X = clean[STRUCTURE_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.loc[X.index].copy()
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled)
    clean["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    coordinates = pd.DataFrame({"puzzle_date": clean["puzzle_date"],
                                "PC1": points[:, 0], "PC2": points[:, 1],
                                "cluster": clean["cluster"]})
    summary = clean.groupby("cluster").agg(
        puzzles=("puzzle_date", "size"), mean_difficulty=("difficulty", "mean"),
        mean_within_similarity=("within_group_similarity", "mean"),
        mean_cross_similarity=("cross_group_similarity", "mean"),
    ).reset_index()
    save_table(coordinates, "pca_coordinates.csv")
    save_table(coordinates[["puzzle_date", "cluster"]], "puzzle_clusters.csv")
    save_table(summary, "cluster_summary.csv")
    return coordinates, summary, pca.explained_variance_ratio_


def vocabulary_tables(words):
    word_counts = words.groupby("word").agg(appearances=("word", "size"),
                                              puzzles_used=("puzzle_date", "nunique")).reset_index()
    categories = words.groupby("category")["puzzle_date"].nunique().rename("puzzles").reset_index()
    save_table(word_counts, "word_frequency.csv")
    save_table(categories, "category_frequency.csv")
    return word_counts, categories
