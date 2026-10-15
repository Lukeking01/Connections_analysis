"""Train/evaluate the PyTorch solver with chronological splits."""
import argparse
import pandas as pd
from connections.data import puzzle_examples
from connections.clustering import load_encoder
from connections.training.train import train_solver, evaluate_solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--no-synonyms',action='store_true')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    words = pd.read_csv('data/raw/connections2.csv',keep_default_na=False,parse_dates=['puzzle_date'])
    puzzles = puzzle_examples(words)
    model, result = train_solver(puzzles,load_encoder(),epochs=args.epochs,
                                 use_synonyms=not args.no_synonyms,device=args.device)
    metrics = pd.DataFrame(evaluate_solver(result))
    print(metrics.to_string(index=False))
    from pathlib import Path
    output = Path('data/derived/torch_solver_evaluation.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(output,index=False)

if __name__ == '__main__':
    main()
