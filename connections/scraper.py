"""Collect Connections solutions and optional difficulty ratings into SQLite.

This module makes network requests only when ``update_database`` is called.
It retains existing records and never drops the words table.
"""
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .paths import RAW

FIRST_PUZZLE = date(2023, 6, 12)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; educational research scraper)"}


def parse_groups(html):
    """Extract the four color/category/word groups from the source page."""
    text = [item.strip() for item in BeautifulSoup(html, "html.parser").stripped_strings]
    if "Ready for the full solution?" not in text:
        raise ValueError("Solution heading not found; the website layout may have changed")
    text = text[text.index("Ready for the full solution?"):]
    groups = []
    for color in ("Yellow", "Green", "Blue", "Purple"):
        if color not in text:
            raise ValueError(f"Missing {color} group")
        i = text.index(color)
        words = [word.strip() for word in text[i + 2].split("·")]
        if len(words) != 4:
            raise ValueError(f"Expected four {color} words, got {words}")
        groups.append((color.lower(), text[i + 1], words))
    return groups


def companion_difficulty(session, day, number):
    """Find a published Companion difficulty score, if one is accessible."""
    import re
    for offset in (0, -1, 1):
        candidate = day + timedelta(days=offset)
        url = (f"https://www.nytimes.com/{candidate:%Y/%m/%d}/"
               f"crosswords/connections-companion-{number}.html")
        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                continue
            match = re.search(r"today.?s difficulty is\s+(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5",
                              response.text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except requests.RequestException:
            continue
    return None


def update_database(start_date=None, end_date=None, database=None,
                    include_difficulty=True, delay=0.2):
    """Collect a date range, adding successful puzzles without erasing old ones.

    By default start at puzzle #1 and end yesterday. Failures are reported
    and logged, while successful existing puzzles are skipped.
    """
    import csv
    import time

    start = date.fromisoformat(start_date) if isinstance(start_date, str) else (start_date or FIRST_PUZZLE)
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else (end_date or date.today() - timedelta(days=1))
    if start < FIRST_PUZZLE or end < start or end > date.today():
        raise ValueError("Invalid date range")
    database = Path(database) if database else RAW / "connections.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    added = 0
    with sqlite3.connect(database) as connection, requests.Session() as session:
        session.headers.update(HEADERS)
        connection.execute("""CREATE TABLE IF NOT EXISTS words (
            puzzle_date TEXT, puzzle_number INTEGER, difficulty REAL,
            color TEXT, category TEXT, position INTEGER, word TEXT,
            PRIMARY KEY (puzzle_number, color, position))""")
        existing = {row[0] for row in connection.execute("SELECT DISTINCT puzzle_date FROM words")}
        day = start
        while day <= end:
            if day.isoformat() in existing:
                day += timedelta(days=1)
                continue
            number = (day - FIRST_PUZZLE).days + 1
            try:
                response = session.get(f"https://www.fiveletterwords.io/connections/{day}", timeout=20)
                response.raise_for_status()
                groups = parse_groups(response.text)
                difficulty = companion_difficulty(session, day, number) if include_difficulty else None
                rows = [(day.isoformat(), number, difficulty, color, category, position, word)
                        for color, category, words in groups
                        for position, word in enumerate(words, 1)]
                with connection:
                    connection.executemany("INSERT INTO words VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
                added += 1
                print(f"Added {day} (#{number})")
            except (requests.RequestException, ValueError, sqlite3.Error, IndexError) as error:
                errors.append((day.isoformat(), str(error)))
                print(f"Could not collect {day}: {error}")
            time.sleep(delay)
            day += timedelta(days=1)
    if errors:
        error_file = RAW / "scrape_errors.csv"
        with error_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["puzzle_date", "error"])
            writer.writerows(errors)
    print(f"Collection complete: {added} new puzzles, {len(errors)} errors")
    if errors and added == 0 and not existing:
        raise RuntimeError("No puzzles collected. Check connectivity or website layout.")
    return {"added": added, "errors": len(errors), "database": database}
