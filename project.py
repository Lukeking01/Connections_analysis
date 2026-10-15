"""Single command-line entry point. Run `python project.py --help`."""
import argparse
from connections.paths import RAW


def main():
    parser = argparse.ArgumentParser(description="Connections data and model workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    scrape = sub.add_parser("scrape", help="Fetch new puzzles into the canonical SQLite database")
    scrape.add_argument("--start", help="First date, YYYY-MM-DD; default: first puzzle")
    scrape.add_argument("--end", help="Last date, YYYY-MM-DD; default: yesterday")
    scrape.add_argument("--no-difficulty", action="store_true")
    build = sub.add_parser("build", help="Rebuild CSV tables from the SQLite database")
    build.add_argument("--semantic", action="store_true", help="Also recompute embedding-based tables")
    train = sub.add_parser("train", help="Train and evaluate PyTorch grouping variants")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--no-synonyms", action="store_true")
    sub.add_parser("difficulty", help="Train and evaluate PyTorch difficulty regressor")
    args = parser.parse_args()
    if args.command == "scrape":
        from connections.scraper import update_database
        from connections.database import read_words, save_csv
        result = update_database(start_date=args.start, end_date=args.end,
                                 include_difficulty=not args.no_difficulty)
        print(result)
        save_csv(read_words(RAW / "connections.db"), RAW / "connections2.csv")
    elif args.command == "build":
        from connections.database import read_words, save_csv, build_semantic_tables, MODEL_NAME
        from connections.paths import SEMANTIC
        words = read_words(RAW / "connections.db")
        save_csv(words, RAW / "connections2.csv")
        if args.semantic:
            groups, pairs = build_semantic_tables(words, MODEL_NAME)
            save_csv(groups, SEMANTIC / "semantic_group_statistics.csv")
            save_csv(pairs, SEMANTIC / "semantic_pair_statistics.csv")
    else:
        import pandas as pd
        from connections.data import puzzle_examples
        from connections.clustering import load_encoder
        from connections.ml import fit_pair_mlp, fit_difficulty_mlp
        words = pd.read_csv(RAW / "connections2.csv", keep_default_na=False,
                            parse_dates=["puzzle_date"])
        puzzles = puzzle_examples(words)
        encoder = load_encoder()
        if args.command == "train":
            _, metrics, _ = fit_pair_mlp(puzzles, encoder, epochs=args.epochs,
                                         use_synonyms=not args.no_synonyms)
        else:
            _, metrics = fit_difficulty_mlp(puzzles, encoder)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
