"""Scrape Dancing with the Stars (US) weekly scoring data from Wikipedia.

Iterates over seasons 1-34, parsing each season's page and saving a tidy
weekly-scores table (joined with cast info) to data/season-{n}.parquet.
"""

import re
import time
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup

from wiki import expand_table_rows, fetch_soup, normalize_header

DATA_DIR = Path("data") / "raw"
FIRST_SEASON = 1
LAST_SEASON = 34


def parse_cast_table(soup: BeautifulSoup) -> pl.DataFrame:
    """Parse the 'Couples' table listing celebrity/pro pairs and elimination order."""
    heading = soup.find(id="Couples") or soup.find(id="Cast")
    table = heading.find_next("table") if heading else None
    if table is None:
        # fallback: first wikitable on the page
        table = soup.select_one("table.wikitable")

    grid = expand_table_rows(table)
    header, *body = grid
    header = [normalize_header(h) for h in header]

    records = []
    for row in body:
        rec = dict(zip(header, row))
        records.append(rec)

    df = pl.DataFrame(records)
    df = df.rename(
        {
            "Celebrity": "celebrity",
            "Notability": "notability",
            "Professional partner": "professional",
            "Status": "status",
        }
    )
    keep = [c for c in ("celebrity", "notability", "professional", "status") if c in df.columns]
    return df.select(keep)


SCORE_RE = re.compile(r"^\s*(\d+)\s*\(([^)]*)\)\s*$")


def parse_score_cell(cell: str) -> dict:
    """Parse a cell like '20 (7, 7, 6)' into total + individual judge scores."""
    cell = cell.strip()
    if not cell or cell.lower().startswith("no score"):
        return {"total_score": None, "judge_scores": None}
    m = SCORE_RE.match(cell)
    if not m:
        return {"total_score": None, "judge_scores": None}
    total = int(m.group(1))
    judges = [j.strip() for j in m.group(2).split(",")]
    judges = [int(j) for j in judges if j.strip().isdigit()]
    return {"total_score": total, "judge_scores": judges}


def parse_weekly_score_tables(soup: BeautifulSoup, season: int) -> pl.DataFrame:
    """Parse each 'Week N' section's table into a tidy long-format frame."""
    all_rows = []

    week_headers = soup.select("h3")
    for h in week_headers:
        heading_text = h.get_text(" ", strip=True)
        m = re.match(r"Week\s+(\d+)", heading_text)
        if not m:
            continue
        week_num = int(m.group(1))

        table = h.find_next("table", class_="wikitable")
        if table is None:
            continue

        grid = expand_table_rows(table)
        header, *body = grid
        header = [normalize_header(c) for c in header]

        for row in body:
            rec = dict(zip(header, row))
            couple_cell = rec.get("Couple", "")
            if not couple_cell:
                continue
            # group-dance rows list multiple couples in one cell, newline-joined
            couples = [c.strip() for c in couple_cell.split("\n") if c.strip()]
            score_info = parse_score_cell(rec.get("Scores", ""))
            is_group = len(couples) > 1
            for couple in couples:
                all_rows.append(
                    {
                        "season": season,
                        "week": week_num,
                        "week_label": heading_text,
                        "couple": couple,
                        "dance": rec.get("Dance"),
                        "music": rec.get("Music"),
                        "total_score": score_info["total_score"],
                        "judge_scores": score_info["judge_scores"],
                        "result": rec.get("Result"),
                        "is_group_dance": is_group,
                    }
                )

    return pl.DataFrame(all_rows)


def match_couple_to_cast(couple: str, cast_rows: list[dict]) -> dict | None:
    """Match an abbreviated 'Couple' string (e.g. 'Bill E. & Emma') to a cast row.

    Wikipedia abbreviates celebrities/professionals to first names, adding a
    last-initial when needed to disambiguate (e.g. 'Bill E.' vs 'Bill N.').
    We match by checking whether the abbreviation (periods stripped) is a
    prefix of the cast member's full name.
    """
    if " & " not in couple:
        return None
    ab_celeb, ab_pro = couple.split(" & ", 1)
    ab_celeb = ab_celeb.replace(".", "").strip()
    ab_pro = ab_pro.replace(".", "").strip()

    matches = [
        row
        for row in cast_rows
        if row["celebrity"].startswith(ab_celeb) and row["professional"].startswith(ab_pro)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def join_cast_into_weekly(cast: pl.DataFrame, weekly: pl.DataFrame) -> pl.DataFrame:
    """Attach celebrity/professional/notability/status columns to each weekly row."""
    cast_rows = cast.to_dicts()

    mapping = {}
    unmatched = []
    for couple in weekly.get_column("couple").unique().to_list():
        match = match_couple_to_cast(couple, cast_rows)
        if match is None:
            unmatched.append(couple)
        else:
            mapping[couple] = match

    if unmatched:
        print(f"  warning: {len(unmatched)} unmatched couple key(s): {unmatched}")

    def lookup(couple: str, field: str):
        return mapping.get(couple, {}).get(field)

    joined = weekly.with_columns(
        pl.col("couple")
        .map_elements(lambda c: lookup(c, "celebrity"), return_dtype=pl.String)
        .alias("celebrity"),
        pl.col("couple")
        .map_elements(lambda c: lookup(c, "notability"), return_dtype=pl.String)
        .alias("notability"),
        pl.col("couple")
        .map_elements(lambda c: lookup(c, "professional"), return_dtype=pl.String)
        .alias("professional"),
        pl.col("couple")
        .map_elements(lambda c: lookup(c, "status"), return_dtype=pl.String)
        .alias("status"),
    )
    return joined


def scrape_season(season: int) -> pl.DataFrame:
    soup = fetch_soup(season)
    cast = parse_cast_table(soup)
    weekly = parse_weekly_score_tables(soup, season)
    return join_cast_into_weekly(cast, weekly)


def save_season(season: int, out_dir: Path = DATA_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = scrape_season(season)
    out_path = out_dir / f"season-{season:02d}.parquet"
    df.write_parquet(out_path)
    return out_path


def scrape_all_seasons(
    first: int = FIRST_SEASON, last: int = LAST_SEASON, delay: float = 1.0
) -> None:
    failures = []
    for season in range(first, last + 1):
        print(f"season {season}...")
        try:
            out_path = save_season(season)
            print(f"  saved {out_path}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(season)
        time.sleep(delay)

    if failures:
        print(f"\nseasons that failed to scrape: {failures}")
    else:
        print("\nall seasons scraped successfully")


if __name__ == "__main__":
    scrape_all_seasons()
