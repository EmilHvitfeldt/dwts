"""Scrape per-episode air dates for each Dancing with the Stars season.

Two sources are tried per season, because neither covers the whole show:

  "episodes" -- the season page's `Episodes` section, a `wikiepisodetable`
                whose release-date cell carries a machine-readable ISO date
                in parentheses. Present for seasons 1-6 and 17-34.
  "ratings"  -- the season page's `Ratings` section. Formats vary a lot
                (the date column is headed "Air date" or "Airdate", and the
                label column is "No.", "Show", or "Episode"), and some of
                these tables carry no date at all. Fills seasons 7-10 and
                13-16.

Seasons 11 and 12 have no dated table on their season pages at all; those
are covered downstream in build_air_dates.py by deriving dates from
elimination text in the already-scraped `status` column.

Output: data/raw/air-dates-season-NN.parquet, one row per episode.
"""

import re
import time
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup

from wiki import fetch_soup, table_records

DATA_DIR = Path("data") / "raw"
FIRST_SEASON = 1
LAST_SEASON = 34

# "September 16, 2025 ( 2025-09-16 )" -> prefer the parenthesised ISO form
ISO_DATE_RE = re.compile(r"\(\s*(\d{4}-\d{2}-\d{2})\s*\)")
LONG_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+(\d{1,2}),?\s+(\d{4})\b"
)
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

DATE_HEADER_RE = re.compile(r"(?i)\b(air\s*date|original\s+release\s+date|airdate)\b")


def parse_date(text: str) -> str | None:
    """Pull an ISO date string out of a cell, preferring explicit ISO markup."""
    m = ISO_DATE_RE.search(text)
    if m:
        return m.group(1)
    m = LONG_DATE_RE.search(text)
    if m:
        month, day, year = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def find_date_column(records: list[dict]) -> str | None:
    """Identify the header whose column holds dates.

    Matches on the header name first, then falls back to sniffing which
    column actually parses as a date -- some tables label it in ways the
    header regex will not anticipate.
    """
    if not records:
        return None
    headers = list(records[0].keys())

    for h in headers:
        if DATE_HEADER_RE.search(h or ""):
            return h

    best, best_hits = None, 0
    for h in headers:
        hits = sum(1 for r in records if parse_date(r.get(h) or ""))
        if hits > best_hits:
            best, best_hits = h, hits
    # require most rows to parse, so we don't latch onto a stray year
    if best is not None and best_hits >= max(2, len(records) // 2):
        return best
    return None


def parse_dated_table(table, season: int, source: str) -> list[dict]:
    """Flatten a table into {season, source, label, date} rows."""
    records = table_records(table)
    date_col = find_date_column(records)
    if date_col is None:
        return []

    out = []
    for idx, rec in enumerate(records):
        date = parse_date(rec.get(date_col) or "")
        if date is None:
            continue
        # everything except the date column is fair game for week matching
        label = " ".join(
            str(v) for k, v in rec.items() if k != date_col and v
        ).strip()
        out.append(
            {
                "season": season,
                "source": source,
                "episode_index": idx,
                "label": label,
                "air_date": date,
            }
        )
    return out


def parse_section(soup: BeautifulSoup, section_id: str, season: int, source: str):
    heading = soup.find(id=section_id)
    if heading is None:
        return []
    table = heading.find_next("table")
    if table is None:
        return []
    return parse_dated_table(table, season, source)


def scrape_season_dates(season: int) -> pl.DataFrame:
    soup = fetch_soup(season)
    rows = parse_section(soup, "Episodes", season, "episodes")
    rows += parse_section(soup, "Ratings", season, "ratings")
    schema = {
        "season": pl.Int64,
        "source": pl.String,
        "episode_index": pl.Int64,
        "label": pl.String,
        "air_date": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def scrape_all(first: int = FIRST_SEASON, last: int = LAST_SEASON, delay: float = 1.0):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in range(first, last + 1):
        df = scrape_season_dates(season)
        out = DATA_DIR / f"air-dates-season-{season:02d}.parquet"
        df.write_parquet(out)
        by_source = dict(
            df.group_by("source").agg(pl.len().alias("n")).iter_rows()
        ) if df.height else {}
        print(f"season {season:>2}: {df.height:>3} dated rows {by_source}")
        time.sleep(delay)


if __name__ == "__main__":
    scrape_all()
