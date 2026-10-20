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


SCORE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\(([^)]*)\)\s*$")

NO_SCORE = {"total_score": None, "judge_scores": None}


def parse_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_score_cell(cell: str) -> dict:
    """Parse a cell like '20 (7, 7, 6)' into total + individual judge scores."""
    cell = cell.strip()
    if not cell or cell.lower().startswith("no score"):
        return NO_SCORE
    m = SCORE_RE.match(cell)
    if not m:
        return NO_SCORE
    judges = [parse_number(j.strip()) for j in m.group(2).split(",")]
    return {
        "total_score": float(m.group(1)),
        "judge_scores": [j for j in judges if j is not None],
    }


# Season 15 labels its judge columns by surname instead of stating the order
JUDGE_SURNAMES = {
    "Inaba": "Carrie Ann Inaba",
    "Goodman": "Len Goodman",
    "Tonioli": "Bruno Tonioli",
    "Abdul": "Paula Abdul",
}

JudgeOrder = list[tuple[str, bool]]


def parse_judge_order(text: str) -> JudgeOrder:
    """Parse '... left to right: A, B (Night 1 only), C.' into (name, optional) pairs.

    `optional` marks a judge annotated in parentheses, who did not score every
    dance in the section.
    """
    names = text.split(":", 1)[1]
    names = re.sub(r"\s*\[\s*[^\]]{0,6}\s*\]", "", names)
    order = []
    for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", names):
        part = part.strip(" .")
        if part:
            order.append((re.sub(r"\s*\(.*?\)", "", part).strip(), "(" in part))
    return order


def is_judge_order(el) -> bool:
    if el.name != "p":
        return False
    text = el.get_text(" ", strip=True)
    return "left to right" in text and ":" in text


def find_season_judge_order(soup: BeautifulSoup) -> JudgeOrder | None:
    """The judge order stated above the first week, which applies unless overridden."""
    for el in soup.find_all(["h3", "p"]):
        if el.name == "h3" and re.match(r"Week\s+\d+", el.get_text(" ", strip=True)):
            return None
        if is_judge_order(el):
            return parse_judge_order(el.get_text(" ", strip=True))
    return None


def resolve_judges(order: JudgeOrder | None, n_scores: int) -> list[str] | None:
    """Names for `n_scores` scores, or None when the stated order does not fit."""
    if not order:
        return None
    names = [name for name, _ in order]
    if len(names) == n_scores:
        return names
    required = [name for name, optional in order if not optional]
    if len(required) == n_scores:
        return required
    return None


def week_tables(heading, season_order: JudgeOrder | None) -> list:
    """(table, judge order) for each wikitable in the section at a week heading.

    A week can restate the judge order (guest judges), which then applies to the
    tables that follow it; otherwise the season-level order applies.
    """
    tables = []
    order = None
    for el in heading.find_all_next(["h2", "h3", "p", "table"]):
        if el.name in ("h2", "h3"):
            break
        if is_judge_order(el):
            order = parse_judge_order(el.get_text(" ", strip=True))
        elif "wikitable" in (el.get("class") or []):
            tables.append((el, order or season_order))
    return tables


def find_column(header: list[str], name: str) -> int | None:
    return header.index(name) if name in header else None


def parse_week_table(table, order: JudgeOrder | None = None) -> list[dict]:
    """Parse one week table into rows of couple, dance, music, result and scores.

    Score columns vary: usually one 'Scores' column of cells like '20 (7, 7, 6)',
    but season 15 splits the judges into separate half-point columns under a
    two-row header, and two weeks (seasons 10 and 11) score technique and
    performance separately. A row is emitted per score, tagged with `score_type`
    in those weeks.

    `order` is the judge order stated for the table; it names the scores in
    `judges` (season 15 names them from its column headers instead).
    """
    grid = expand_table_rows(table)
    header = [normalize_header(c) for c in grid[0]]
    labels = [" ".join(h.lower().split()) for h in header]

    split_header = (
        len(grid) > 1 and grid[1][0] == grid[0][0] and labels.count("scores") > 1
    )
    body = grid[2:] if split_header else grid[1:]

    couple_col = find_column(header, "Couple")
    if couple_col is None:
        return []
    dance_col = next((i for i, h in enumerate(header) if h.startswith("Dance")), None)
    music_col = find_column(header, "Music")
    if music_col is None:
        music_col = next((i for i, h in enumerate(labels) if "music" in h), None)
    result_col = find_column(header, "Result")

    # (column, score_type); "alternate" means technical then performance rows
    score_cols = []
    judge_cols = []
    total_col = None
    judge_names = []
    if split_header:
        judge_cols = [i for i, h in enumerate(labels) if h == "scores"]
        judge_names = [JUDGE_SURNAMES.get(grid[1][i], grid[1][i]) for i in judge_cols]
        total_col = find_column(labels, "total")
    else:
        for i, h in enumerate(labels):
            if "technical" in h and "performance" in h:
                score_cols.append((i, "alternate"))
            elif "technical" in h:
                score_cols.append((i, "technical"))
            elif "performance" in h:
                score_cols.append((i, "performance"))
            elif h == "scores":
                score_cols.append((i, None))

    def cell(row, col):
        return row[col] if col is not None else None

    rows = []
    seen: dict[str, int] = {}
    for row in body:
        couple_cell = row[couple_col]
        if not couple_cell:
            continue
        # group-dance rows list multiple couples in one cell, newline-joined;
        # parenthesised lines like "( Season 4 winner )" are annotations
        couples = [
            c.strip()
            for c in couple_cell.split("\n")
            if c.strip() and not c.strip().startswith("(")
        ]
        if not couples:
            continue

        if split_header:
            pairs = [
                (name, parse_number(row[i])) for name, i in zip(judge_names, judge_cols)
            ]
            pairs = [(name, score) for name, score in pairs if score is not None]
            total = parse_number(row[total_col])
            # marathon bonus points are one spanned cell repeated across the
            # judge columns, so they do not sum to the total
            if not pairs or sum(score for _, score in pairs) != total:
                pairs = []
            scores = [
                (
                    None,
                    {
                        "total_score": total,
                        "judge_scores": [score for _, score in pairs] or None,
                        "judges": [name for name, _ in pairs] or None,
                    },
                )
            ]
        elif score_cols:
            scores = []
            for col, kind in score_cols:
                if kind == "alternate":
                    kind = ("technical", "performance")[seen.get(couple_cell, 0) % 2]
                    seen[couple_cell] = seen.get(couple_cell, 0) + 1
                info = parse_score_cell(row[col])
                judge_scores = info["judge_scores"]
                judges = resolve_judges(order, len(judge_scores)) if judge_scores else None
                scores.append((kind, {**info, "judges": judges}))
        else:
            scores = [(None, {**NO_SCORE, "judges": None})]

        for score_type, info in scores:
            for couple in couples:
                rows.append(
                    {
                        "couple": couple,
                        "dance": cell(row, dance_col),
                        "music": cell(row, music_col),
                        "total_score": info["total_score"],
                        "judge_scores": info["judge_scores"],
                        "judges": info["judges"],
                        "score_type": score_type,
                        "result": cell(row, result_col),
                        "is_group_dance": len(couples) > 1,
                    }
                )
    return rows


def parse_weekly_score_tables(soup: BeautifulSoup, season: int) -> pl.DataFrame:
    """Parse every table in each 'Week N' section into a tidy long-format frame."""
    all_rows = []
    season_order = find_season_judge_order(soup)

    for h in soup.select("h3"):
        heading_text = h.get_text(" ", strip=True)
        m = re.match(r"Week\s+(\d+)", heading_text)
        if not m:
            continue
        week_num = int(m.group(1))

        # unique within a week, so (season, week, performance_order) is a key
        performance_order = 0
        for table, order in week_tables(h, season_order):
            for rec in parse_week_table(table, order):
                performance_order += 1
                all_rows.append(
                    {
                        "season": season,
                        "week": week_num,
                        "week_label": heading_text,
                        "performance_order": performance_order,
                        **rec,
                    }
                )

    df = pl.DataFrame(all_rows).with_columns(
        pl.col("total_score").cast(pl.Float64),
        pl.col("judge_scores").cast(pl.List(pl.Float64)),
        pl.col("judges").cast(pl.List(pl.String)),
        pl.col("score_type").cast(pl.String),
    )

    unnamed = df.filter(
        pl.col("judge_scores").is_not_null() & pl.col("judges").is_null()
    )
    if unnamed.height:
        weeks = sorted(unnamed.get_column("week").unique().to_list())
        print(f"  warning: {unnamed.height} scored row(s) without judge names (weeks {weeks})")
    return df


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
