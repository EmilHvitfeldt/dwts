"""Scrape a date of birth for every Dancing with the Stars celebrity.

Celebrities are identified by the wikilink in the season page's Couples
table rather than by their name text. Name matching looks like it works
(about 91% of names resolve) but fails in the one way that matters: single
-name celebrities such as "Brandy", "Mario", "Romeo" and "Babyface" resolve
to an unrelated article and would silently attach a real but wrong person's
birthday. The link is unambiguous.

The celebrity cell's text is taken with the same normalisation the score
scraper uses, so it joins straight onto the `celebrity` column of
data/dwts_all_seasons.parquet -- including the "Name\\nSeason 4" strings the
all-stars cast table produces.

Birth dates come from Wikidata property P569, reached by resolving each
article title to its Wikidata item.

Output: data/celebrity_birthdates.parquet
"""

import time
import urllib.parse
from pathlib import Path

import polars as pl
import requests

from wiki import HEADERS, cell_text, expand_table_cells, fetch_soup, normalize_header

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "celebrity_birthdates.parquet"
FIRST_SEASON = 1
LAST_SEASON = 34

ENWIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
BATCH = 40

# Wikidata time precision codes; anything coarser than a day makes the age
# approximate, so it is recorded rather than silently rounded.
PRECISION = {11: "day", 10: "month", 9: "year"}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_json(url: str, params: dict, attempts: int = 5) -> dict:
    """GET with backoff. The Wikidata API returns 429 readily when batching."""
    for attempt in range(attempts):
        resp = SESSION.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                pass
        wait = 10 * (attempt + 1)
        print(f"  HTTP {resp.status_code}, retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"gave up on {url}")


def chunks(xs: list, n: int = BATCH):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


# --------------------------------------------------------------------------
# cast-table links
# --------------------------------------------------------------------------

def article_title(link) -> str | None:
    """Article title from a wikilink, or None if it is not one.

    Hrefs come back absolute ("https://en.wikipedia.org/wiki/Baron_Davis")
    rather than root-relative, so match on the /wiki/ segment instead of a
    prefix. Namespaced targets (File:, Help:) and red links are not people.
    """
    if link is None:
        return None
    href = link.get("href", "")
    marker = "/wiki/"
    if marker not in href:
        return None
    target = href.split(marker, 1)[1].split("#")[0].split("?")[0]
    title = urllib.parse.unquote(target).replace("_", " ").strip()
    if not title or ":" in title:
        return None
    return title


def celebrity_links(season: int) -> list[dict]:
    """Extract (celebrity text, article title) from a season's Couples table.

    The celebrity column is located by its header text rather than assumed to
    be the first cell, since some seasons' Couples/Cast tables lead with a
    different column (e.g. placement or an image).
    """
    soup = fetch_soup(season)
    heading = soup.find(id="Couples") or soup.find(id="Cast")
    table = heading.find_next("table") if heading else soup.select_one("table.wikitable")
    if table is None:
        return []

    grid = expand_table_cells(table)
    if not grid:
        return []
    header_row, *body_rows = grid
    headers = [normalize_header(cell_text(c)) if c is not None else "" for c in header_row]
    col_idx = headers.index("Celebrity") if "Celebrity" in headers else 0

    out = []
    seen_cells = set()
    for row in body_rows:
        if col_idx >= len(row) or row[col_idx] is None:
            continue
        cell = row[col_idx]
        if id(cell) in seen_cells:
            continue
        seen_cells.add(id(cell))
        title = article_title(cell.find("a", href=True))
        # read the text after finding the link: cell_text mutates <br> tags
        name = cell_text(cell)
        if not name:
            continue
        out.append({"season": season, "celebrity": name, "article_title": title})
    return out


def scrape_all_links(first: int = FIRST_SEASON, last: int = LAST_SEASON,
                     delay: float = 1.0) -> pl.DataFrame:
    rows = []
    for season in range(first, last + 1):
        found = celebrity_links(season)
        linked = sum(1 for r in found if r["article_title"])
        print(f"season {season:>2}: {linked}/{len(found)} celebrities linked")
        rows.extend(found)
        time.sleep(delay)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# wikidata lookup
# --------------------------------------------------------------------------

def titles_to_qids(titles: list[str]) -> dict[str, str]:
    """Resolve article titles to Wikidata item ids, following redirects."""
    qids: dict[str, str] = {}
    for batch in chunks(titles):
        data = get_json(
            ENWIKI_API,
            {
                "action": "query",
                "format": "json",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "titles": "|".join(batch),
                "redirects": 1,
            },
        )
        query = data.get("query", {})
        resolved = {}
        for norm in query.get("normalized", []):
            resolved[norm["from"]] = norm["to"]
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        by_title = {p["title"]: p for p in query.get("pages", {}).values()}
        for title in batch:
            final = resolved.get(title, title)
            final = redirects.get(final, final)
            page = by_title.get(final)
            if page and "missing" not in page:
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid:
                    qids[title] = qid
        time.sleep(1)
    return qids


def qids_to_birthdates(qids: list[str]) -> dict[str, tuple[str, str]]:
    """Fetch P569 (date of birth) for each item, as (iso_date, precision)."""
    out: dict[str, tuple[str, str]] = {}
    for batch in chunks(qids, 25):
        data = get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(batch),
                "props": "claims",
                "languages": "en",
            },
        )
        for qid, entity in data.get("entities", {}).items():
            claims = entity.get("claims", {}).get("P569") or []
            # prefer a claim with day precision over a vaguer one
            best = None
            for claim in claims:
                try:
                    value = claim["mainsnak"]["datavalue"]["value"]
                except (KeyError, TypeError):
                    continue
                precision = value.get("precision", 0)
                if best is None or precision > best[1]:
                    best = (value["time"], precision)
            if best is None:
                continue
            # "+1979-04-13T00:00:00Z" -> "1979-04-13"
            iso = best[0].lstrip("+")[:10]
            out[qid] = (iso, PRECISION.get(best[1], str(best[1])))
        time.sleep(1)
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("reading cast tables...")
    links = scrape_all_links()

    # one row per distinct celebrity string; the same person can appear in
    # several seasons (all-stars) under different cell text
    people = (
        links.drop_nulls("article_title")
        .group_by("celebrity")
        .agg(pl.col("article_title").first(), pl.col("season").min().alias("season"))
    )
    unlinked = links.filter(pl.col("article_title").is_null())
    if unlinked.height:
        print(f"\n{unlinked.height} celebrity cells had no wikilink:")
        print(unlinked.select("season", "celebrity"))

    titles = sorted(set(people.get_column("article_title").to_list()))
    print(f"\nresolving {len(titles)} article titles to Wikidata items...")
    qids = titles_to_qids(titles)
    print(f"  resolved {len(qids)}/{len(titles)}")

    print("fetching birth dates...")
    dob = qids_to_birthdates(sorted(set(qids.values())))
    print(f"  found P569 for {len(dob)}/{len(set(qids.values()))} items")

    out = (
        people.with_columns(
            pl.col("article_title")
            .replace_strict(qids, default=None, return_dtype=pl.String)
            .alias("wikidata_id")
        )
        .with_columns(
            pl.col("wikidata_id")
            .replace_strict(
                {q: v[0] for q, v in dob.items()}, default=None, return_dtype=pl.String
            )
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("birthdate"),
            pl.col("wikidata_id")
            .replace_strict(
                {q: v[1] for q, v in dob.items()}, default=None, return_dtype=pl.String
            )
            .alias("birthdate_precision"),
        )
        .select(
            "celebrity", "article_title", "wikidata_id", "birthdate",
            "birthdate_precision",
        )
        .sort("celebrity")
    )

    if out.height == 0:
        raise SystemExit("no celebrities resolved; refusing to write an empty table")

    out.write_parquet(OUT_PATH)
    found = out.get_column("birthdate").is_not_null().sum()
    print(f"\nwrote {out.height} celebrities to {OUT_PATH}")
    print(f"birth date found for {found}/{out.height} ({100 * found / out.height:.1f}%)")
    missing = out.filter(pl.col("birthdate").is_null())
    if missing.height:
        print("\nno birth date:")
        print(missing.select("celebrity", "article_title", "wikidata_id"))


if __name__ == "__main__":
    main()
