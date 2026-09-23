"""Shared helpers for scraping Dancing with the Stars data from Wikipedia.

"""

import re

import requests
from bs4 import BeautifulSoup

SEASON_URL = (
    "https://en.wikipedia.org/wiki/"
    "Dancing_with_the_Stars_(American_TV_series)_season_{season}"
)

HEADERS = {
    "User-Agent": (
        "dwts-scraper/0.1 (personal research project; "
        "contact: local-user@example.com)"
    )
}


def fetch_soup(season: int) -> BeautifulSoup:
    resp = requests.get(SEASON_URL.format(season=season), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


BR_SENTINEL = ""


def cell_text(cell) -> str:
    """Normalised text of a single table cell.

    get_text(strip=True) drops whitespace-only text nodes, so a plain "\\n"
    placeholder for <br> would vanish; use a non-whitespace sentinel instead
    and convert it back afterwards. Footnote markers like "[14]", "[ a ]"
    are stripped.
    """
    for br in cell.find_all("br"):
        br.replace_with(BR_SENTINEL)
    text = cell.get_text(" ", strip=True)
    text = re.sub(rf"\s*{BR_SENTINEL}\s*", "\n", text)
    return re.sub(r"\s*\[\s*[^\]]{0,6}\s*\]", "", text).strip()


def expand_table_rows(table) -> list[list[str]]:
    """Turn a wikitable into a grid of text cells, expanding rowspan/colspan.

    A rowspanned cell normally repeats its text on every row it covers.
    However, some Wikipedia "group dance" cells pack N newline-separated
    names into a single cell with rowspan=N (one distinct name per row,
    rather than the same text repeated) -- those are distributed
    sequentially instead of duplicated.
    """
    rows = table.find_all("tr")
    grid: list[list[str]] = []
    pending: dict[int, list[str]] = {}  # col -> queue of remaining row values

    for row in rows:
        cells = row.find_all(["th", "td"])
        out_row: list[str] = []
        col = 0

        def place_pending(col):
            while col in pending:
                queue = pending[col]
                out_row.append(queue.pop(0))
                if not queue:
                    del pending[col]
                col += 1
            return col

        col = place_pending(col)

        for cell in cells:
            col = place_pending(col)
            text = cell_text(cell)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            parts = text.split("\n")
            if rowspan > 1 and len(parts) == rowspan:
                # one distinct value per row this cell spans
                for i in range(colspan):
                    out_row.append(parts[0])
                    pending[col + i] = list(parts[1:])
            else:
                for i in range(colspan):
                    out_row.append(text)
                    if rowspan > 1:
                        pending[col + i] = [text] * (rowspan - 1)
            col += colspan
            col = place_pending(col)

        col = place_pending(col)
        grid.append(out_row)

    # pad ragged rows
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    return grid


def expand_table_cells(table) -> list[list]:
    """Turn a wikitable into a grid of the actual bs4 cell objects, expanding
    rowspan/colspan by repeating the spanning cell (not its text).

    Used where the caller needs to inspect a cell itself (e.g. for a link)
    rather than just its normalised text, so column position has to be
    resolved by header name instead of by raw cell index.
    """
    rows = table.find_all("tr")
    grid: list[list] = []
    pending: dict[int, tuple] = {}  # col -> (cell, rows remaining)

    for row in rows:
        out_row: list = []
        col = 0

        def place_pending(col):
            while col in pending:
                cell, remaining = pending[col]
                out_row.append(cell)
                remaining -= 1
                if remaining <= 0:
                    del pending[col]
                else:
                    pending[col] = (cell, remaining)
                col += 1
            return col

        col = place_pending(col)

        for cell in row.find_all(["th", "td"]):
            col = place_pending(col)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for i in range(colspan):
                out_row.append(cell)
                if rowspan > 1:
                    pending[col + i] = (cell, rowspan - 1)
            col += colspan
            col = place_pending(col)

        col = place_pending(col)
        grid.append(out_row)

    width = max(len(r) for r in grid)
    grid = [r + [None] * (width - len(r)) for r in grid]
    return grid


def normalize_header(text: str) -> str:
    """Strip trailing footnote markers (e.g. 'Celebrity [ 14 ]' -> 'Celebrity')."""
    return re.sub(r"\s*\[.*$", "", text).strip()


def table_records(table) -> list[dict]:
    """Flatten a wikitable into header-keyed dicts, one per body row."""
    grid = expand_table_rows(table)
    if not grid:
        return []
    header, *body = grid
    header = [normalize_header(h) for h in header]
    return [dict(zip(header, row)) for row in body]
