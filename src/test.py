import pandas as pd
import numpy as np

data = pd.read_csv(
    r"C:\Users\Administrator\data\connections.csv"
)

# Remove puzzles with missing words
data = data.dropna(subset=["word"])

# Only retain complete puzzles
complete = data.groupby("puzzle_date").filter(
    lambda x: len(x) == 16
)

# Create a list of puzzles
puzzles = []

for date, group in complete.groupby("puzzle_date"):
    puzzles.append({
        "date": date,
        "words": group["word"].tolist(),
        "colors": group["color"].tolist()
    })

print("Number of puzzles:", len(puzzles))
print(puzzles[0])

#%%

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Get all unique words
unique_words = complete["word"].unique().tolist()

# Generate embeddings
embeddings = embedding_model.encode(
    unique_words,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Dictionary mapping words to their vectors
word_embeddings = dict(zip(unique_words, embeddings))

#%%

import torch
import torch.nn as nn
from itertools import combinations
from sklearn.model_selection import train_test_split

def make_pairs(puzzle_list):
    X = []
    y = []

    for puzzle in puzzle_list:
        words = puzzle["words"]
        colors = puzzle["colors"]

        for i, j in combinations(range(16), 2):
            a = word_embeddings[words[i]]
            b = word_embeddings[words[j]]

            features = np.concatenate([
                np.abs(a - b),
                a * b
            ])

            X.append(features)
            y.append(int(colors[i] == colors[j]))

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )


# Split by puzzle, not individual words
train_puzzles, test_puzzles = train_test_split(
    puzzles, test_size=0.2, random_state=42
)

X_train, y_train = make_pairs(train_puzzles)
X_test, y_test = make_pairs(test_puzzles)

#%%

class ConnectionsNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


model = ConnectionsNN()

# Correct pairs are less common than incorrect pairs
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%

from torch.utils.data import DataLoader, TensorDataset

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True
)

for epoch in range(20):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = loss_function(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.3f}")
    
#%%

from functools import lru_cache

def solve_puzzle(words):
    model.eval()

    pair_scores = np.zeros((16, 16))

    with torch.no_grad():
        for i, j in combinations(range(16), 2):
            a = word_embeddings[words[i]]
            b = word_embeddings[words[j]]

            features = np.concatenate([
                np.abs(a - b),
                a * b
            ])

            x = torch.tensor(
                features, dtype=torch.float32
            ).unsqueeze(0)

            score = model(x).item()

            pair_scores[i, j] = score
            pair_scores[j, i] = score

    # Score every possible group of four
    group_scores = {}

    for group in combinations(range(16), 4):
        score = sum(
            pair_scores[i, j]
            for i, j in combinations(group, 2)
        )
        group_scores[group] = score

    # Find the best four non-overlapping groups
    @lru_cache(None)
    def best_partition(remaining):
        if not remaining:
            return 0, []

        first = remaining[0]
        best_score = float("-inf")
        best_groups = None

        for others in combinations(remaining[1:], 3):
            group = (first,) + others
            leftover = tuple(
                i for i in remaining if i not in group
            )

            rest_score, rest_groups = best_partition(leftover)
            score = group_scores[group] + rest_score

            if score > best_score:
                best_score = score
                best_groups = [group] + rest_groups

        return best_score, best_groups

    _, groups = best_partition(tuple(range(16)))

    return [[words[i] for i in group] for group in groups]

#%%

test_puzzle = test_puzzles[0]

predicted_groups = solve_puzzle(test_puzzle["words"])

for i, group in enumerate(predicted_groups):
    print(f"Group {i+1}: {group}")
    
#%%

def evaluate(puzzle_list):
    correct_groups = 0
    correct_puzzles = 0

    for puzzle in puzzle_list:
        predicted = solve_puzzle(puzzle["words"])

        actual = [
            set(
                word for word, color in zip(
                    puzzle["words"], puzzle["colors"]
                ) if color == c
            )
            for c in ["yellow", "green", "blue", "purple"]
        ]

        correct = sum(
            set(group) in actual
            for group in predicted
        )

        correct_groups += correct
        correct_puzzles += (correct == 4)

    print(
        "Group accuracy:",
        correct_groups / (4 * len(puzzle_list))
    )

    print(
        "Puzzle accuracy:",
        correct_puzzles / len(puzzle_list)
    )


evaluate(test_puzzles)
