import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, timedelta
from pathlib import Path
import re
import time

START_DATE = date(2023, 6, 12)
END_DATE = date(2026, 10, 12)

DB = Path("data/connections.db")
DATA = Path("data")
DATA.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def puzzle_number_from_date(d: date) -> int:
    """Official numbering: #1 = 2023-06-12."""
    return (d - date(2023, 6, 12)).days + 1


def parse_groups(html: str):
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


def extract_puzzle_number_from_title(title: str) -> int | None:
    """Fallback parser for titles like '... (#367)' or '... (#1,218)'."""
    if not title or "#" not in title:
        return None
    m = re.search(r"#\s*([\d,]+)", title)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def get_companion_difficulty(puzzle_date: date, puzzle_number: int) -> float | None:
    """
    Try the modern Companion URL pattern.
    The date in the URL is usually the publication day of the Companion
    (often the day before the puzzle in Eastern time).
    We try puzzle_date and puzzle_date - 1 day.
    """
    candidates = [
        puzzle_date,
        puzzle_date - timedelta(days=1),
        puzzle_date + timedelta(days=1),  # rare edge cases
    ]

    for d in candidates:
        url = (
            f"https://www.nytimes.com/{d.year}/{d.month:02d}/{d.day:02d}/"
            f"crosswords/connections-companion-{puzzle_number}.html"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue

            # Look for the standard phrase
            m = re.search(
                r"[Tt]oday.?s difficulty is\s+(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5",
                r.text,
                re.IGNORECASE,
            )
            if m:
                return float(m.group(1))

            # Slightly looser fallback
            m = re.search(
                r"difficulty.*?(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5",
                r.text,
                re.IGNORECASE,
            )
            if m:
                return float(m.group(1))

        except Exception:
            continue

    return None


def scrape():
    conn = sqlite3.connect(DB)

    conn.executescript("""
        DROP TABLE IF EXISTS words;
        CREATE TABLE words (
            puzzle_date TEXT,
            puzzle_number INTEGER,
            difficulty REAL,
            color TEXT,
            category TEXT,
            position INTEGER,
            word TEXT,
            PRIMARY KEY (puzzle_number, color, position)
        );
    """)

    words = []
    errors = []

    dates = []
    d = START_DATE
    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)

    print("=" * 60)
    print("CONNECTIONS SCRAPER")
    print("=" * 60)
    print(f"Source:     https://www.fiveletterwords.io")
    print(f"Start:      {START_DATE}")
    print(f"End:        {END_DATE}")
    print(f"Dates:      {len(dates)}")
    print(f"Database:   {DB}\n")

    for n, d in enumerate(dates, 1):
        url = f"https://www.fiveletterwords.io/connections/{d}"
        puzzle_number = puzzle_number_from_date(d)  # primary, reliable source

        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()

            # Optional: cross-check number from title
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string if soup.title else ""
            title_number = extract_puzzle_number_from_title(title)
            if title_number is not None and title_number != puzzle_number:
                print(f"  ⚠ title number mismatch: date→{puzzle_number}, title→{title_number}")

            parsed = parse_groups(r.text)

            # Fetch official tester difficulty
            difficulty = get_companion_difficulty(d, puzzle_number)

            for color, category, ws in parsed:
                for pos, word in enumerate(ws, 1):
                    words.append({
                        "puzzle_date": str(d),
                        "puzzle_number": puzzle_number,
                        "difficulty": difficulty,
                        "color": color,
                        "category": category,
                        "position": pos,
                        "word": word,
                    })

            diff_str = f"{difficulty:.1f}" if difficulty is not None else "—"
            print(f"[{n:4}/{len(dates)}] {d}  #{puzzle_number:4}  diff={diff_str}  ... OK")

            # Be polite to both sites
            time.sleep(0.1)

        except Exception as e:
            errors.append({"puzzle_date": str(d), "error": str(e)})
            print(f"[{n:4}/{len(dates)}] {d}  ... ERROR: {e}")

    # Write to DB
    for table in ["words"]:
        conn.execute(f"DELETE FROM {table}")

    for w in words:
        conn.execute(
            "INSERT INTO words VALUES (?, ?, ?, ?, ?, ?, ?)",
            (w["puzzle_date"], w["puzzle_number"], w["difficulty"], w["color"],
             w["category"], w["position"], w["word"]),
        )

    conn.commit()
    conn.close()

    # CSVs
    pd.DataFrame(words).to_csv(DATA / "connections2.csv", index=False)
    pd.DataFrame(errors).to_csv(DATA / "scrape_errors.csv", index=False)

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Database:   {DB}")


if __name__ == "__main__":
    scrape()