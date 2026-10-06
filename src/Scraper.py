import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, timedelta
from pathlib import Path

START_DATE = date(2023, 6, 12)
END_DATE = date(2026, 10, 6)

DB = Path("data/connections.db")
DATA = Path("data")
DATA.mkdir(exist_ok=True)


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    text = [x.strip() for x in soup.stripped_strings]

    start = text.index("Ready for the full solution?")
    text = text[start:]

    groups = []
    colors = ["Yellow", "Green", "Blue", "Purple"]

    for color in colors:
        i = text.index(color)
        category = text[i + 1]
        words = [w.strip() for w in text[i + 2].split("·")]

        if len(words) != 4:
            raise ValueError(f"{color}: {words}")

        groups.append((color.lower(), category, words))

    return groups


def scrape():
    conn = sqlite3.connect(DB)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS puzzles (
            puzzle_date TEXT PRIMARY KEY,
            puzzle_number INTEGER,
            url TEXT
        );

        CREATE TABLE IF NOT EXISTS groups (
            puzzle_date TEXT,
            puzzle_number INTEGER,
            color TEXT,
            category TEXT,
            PRIMARY KEY (puzzle_date, color)
        );

        CREATE TABLE IF NOT EXISTS words (
            puzzle_date TEXT,
            puzzle_number INTEGER,
            color TEXT,
            category TEXT,
            position INTEGER,
            word TEXT,
            PRIMARY KEY (puzzle_date, color, position)
        );
    """)

    puzzles = []
    groups = []
    words = []
    errors = []

    d = START_DATE
    dates = []

    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)

    print("=" * 60)
    print("CONNECTIONS SCRAPER")
    print("=" * 60)
    print(f"\nSource: https://www.fiveletterwords.io")
    print(f"Start:  {START_DATE}")
    print(f"End:    {END_DATE}")
    print(f"\nDates to process: {len(dates)}")
    print(f"\nDatabase: {DB}\n")

    for n, d in enumerate(dates, 1):
        url = f"https://www.fiveletterwords.io/connections/{d}"

        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20
            )
            r.raise_for_status()

            parsed = parse(r.text)

            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string if soup.title else ""
            puzzle_number = None

            if "#" in title:
                try:
                    puzzle_number = int(
                        title.split("#")[1].split()[0].replace(",", "")
                    )
                except:
                    pass

            puzzles.append({
                "puzzle_date": str(d),
                "puzzle_number": puzzle_number,
                "url": url
            })

            for color, category, ws in parsed:
                groups.append({
                    "puzzle_date": str(d),
                    "puzzle_number": puzzle_number,
                    "color": color,
                    "category": category
                })

                for pos, word in enumerate(ws, 1):
                    words.append({
                        "puzzle_date": str(d),
                        "puzzle_number": puzzle_number,
                        "color": color,
                        "category": category,
                        "position": pos,
                        "word": word
                    })

            print(f"[{n:4}/{len(dates)}] {d} ... OK")

        except Exception as e:
            errors.append({
                "puzzle_date": str(d),
                "error": str(e)
            })
            print(f"[{n:4}/{len(dates)}] {d} ... ERROR: {e}")

    for table in ["puzzles", "groups", "words"]:
        conn.execute(f"DELETE FROM {table}")

    for p in puzzles:
        conn.execute(
            "INSERT INTO puzzles VALUES (?, ?, ?)",
            (p["puzzle_date"], p["puzzle_number"], p["url"])
        )

    for g in groups:
        conn.execute(
            "INSERT INTO groups VALUES (?, ?, ?, ?)",
            tuple(g.values())
        )

    for w in words:
        conn.execute(
            "INSERT INTO words VALUES (?, ?, ?, ?, ?, ?)",
            tuple(w.values())
        )

    conn.commit()
    conn.close()

    pd.DataFrame(puzzles).to_csv(DATA / "puzzles.csv", index=False)
    pd.DataFrame(groups).to_csv(DATA / "groups.csv", index=False)
    pd.DataFrame(words).to_csv(DATA / "connections.csv", index=False)
    pd.DataFrame(errors).to_csv(DATA / "scrape_errors.csv", index=False)

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"\nSuccessful: {len(puzzles)}")
    print(f"Errors:     {len(errors)}")
    print(f"Database:   {DB}")


if __name__ == "__main__":
    scrape()