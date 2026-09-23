"""Build a (season, week) -> air_date table for Dancing with the Stars.

No single source covers the whole show, so dates are layered, most precise
first, and every row records which layer it came from in `date_source`:

  episodes     scraped from the season page's `Episodes` section
  ratings      scraped from the season page's `Ratings` section
  elimination  derived from the `status` text already in the score data
               ("Eliminated 2nd on September 30, 2013" dates that couple's
               final week). Covers seasons 11 and 12, which have no dated
               table on Wikipedia at all.
  interpolated linearly filled between known weeks of the same season

Because DWTS performance shows are broadcast live, the air date is also the
performance date.

Output: data/air_dates.parquet
"""

import re
from datetime import date, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SCORES_PATH = DATA_DIR / "dwts_all_seasons.parquet"
OUT_PATH = DATA_DIR / "air_dates.parquet"

SOURCE_PRIORITY = {"episodes": 0, "ratings": 1, "elimination": 2, "interpolated": 3}

# Episodes that are not the scored performance show. Only used to decide
# whether a purely positional (ordinal) mapping is trustworthy.
NON_PERFORMANCE_RE = re.compile(
    r"(?i)results|meet the cast|first look|special|encore|recap|clip show|"
    r"aftershow|after show|behind the scenes|most memorable moments"
)

ORDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October"
    "|November|December"
)


# --------------------------------------------------------------------------
# week-number extraction from an episode label
# --------------------------------------------------------------------------

def week_from_label(label: str) -> int | None:
    """Pull a week number out of an episode label, or None if it has none.

    Handles the several conventions Wikipedia uses across the show's run:
      "Performance Show: Week 3"  -> 3   (seasons 14-16)
      "Round Two" / "Round 2"     -> 2   (seasons 1-6)
      "Performance show 4"        -> 4   (season 13)
      "Episode 703"               -> 3   (seasons 7-9; last two digits)
    """
    m = re.search(r"(?i)\bweek\s+(\d+)", label)
    if m:
        return int(m.group(1))

    m = re.search(r"(?i)\bround\s+(\d+)", label)
    if m:
        return int(m.group(1))

    m = re.search(r"(?i)\bround\s+([a-z]+)", label)
    if m and m.group(1).lower() in ORDINAL_WORDS:
        return ORDINAL_WORDS[m.group(1).lower()]

    m = re.search(r"(?i)\b(?:performance|results)\s+show\s+(\d+)", label)
    if m:
        return int(m.group(1))

    # "Episode 701", "Episode 701A" -> season 7, week 1 (letter = results night)
    m = re.search(r"(?i)\bepisode\s+\d(\d\d)[a-z]?\b", label)
    if m:
        return int(m.group(1))

    return None


def theme_from_week_label(week_label: str) -> str | None:
    """'Week 8: Halloween Week' -> 'halloween week'."""
    if ":" not in week_label:
        return None
    return normalize_theme(week_label.split(":", 1)[1])


def normalize_theme(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(night|week|show|performance)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# layer 1 + 2: scraped episode/ratings tables
# --------------------------------------------------------------------------

def load_scraped() -> pl.DataFrame:
    files = sorted(RAW_DIR.glob("air-dates-season-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no air-dates-season-*.parquet in {RAW_DIR}; run scrape_air_dates.py first"
        )
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df.with_columns(pl.col("air_date").str.to_date("%Y-%m-%d"))


def drop_year_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Discard dates whose year disagrees with the rest of their season.

    Guards against editor typos: as of this writing season 34 episode 5 is
    dated "October 14, 2020" on Wikipedia, five years off, which would
    otherwise poison every age computed for that week.
    """
    modal_year = (
        df.group_by("season", "year")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .group_by("season")
        .first()
        .select("season", pl.col("year").alias("modal_year"))
    )
    joined = df.join(modal_year, on="season", how="left")
    dropped = joined.filter(pl.col("year") != pl.col("modal_year"))
    for row in dropped.iter_rows(named=True):
        print(
            f"  dropped season {row['season']} outlier date {row['air_date']} "
            f"(season is {row['modal_year']}): {row['label'][:60]}"
        )
    return joined.filter(pl.col("year") == pl.col("modal_year")).drop("modal_year")


def longest_nondecreasing(values: list) -> set[int]:
    """Indices of a longest non-decreasing subsequence of `values`.

    Used to decide which dates in an episode table to trust: the biggest
    self-consistent run wins and the stragglers are treated as typos.
    """
    n = len(values)
    if n == 0:
        return set()
    best_len = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] <= values[i] and best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best_len[i])
    out = set()
    while end != -1:
        out.add(end)
        end = prev[end]
    return out


def map_season_source(
    season: int,
    rows: list[dict],
    themes: dict,
    n_weeks: int,
    elim: dict[int, date],
) -> dict[int, date]:
    """Map one season's episodes (from one source) to week -> air_date.

    Wikipedia's episode tables are inconsistent enough that a naive mapping
    silently produces wrong dates, so each step below is guarded and the
    whole season's mapping is discarded if it fails a sanity check. It is
    better to fall through to the elimination layer than to emit a
    confidently wrong date.
    """
    # 1. Dates must not go backwards as episode number increases. Where they
    #    do, a cell is an editor typo (e.g. season 4 dates "Round 9 Results"
    #    as 2007-04-15 when the round aired 2007-05-14). Keep the largest
    #    consistent set rather than the first, so a single bad early row does
    #    not evict all the good rows after it.
    rows = list(rows)
    keep_idx = longest_nondecreasing([r["air_date"] for r in rows])
    kept = [r for i, r in enumerate(rows) if i in keep_idx]
    for i, r in enumerate(rows):
        if i not in keep_idx:
            print(
                f"  season {season}: dropped out-of-order date "
                f"{r['air_date']} for {r['label'][:50]!r}"
            )

    # 2. Extract a week number, preferring real performance shows over the
    #    results/special episodes that share a week.
    perf, other = [], []
    for r in kept:
        wk = week_from_label(r["label"])
        if wk is None:
            wk = match_theme(season, r["label"], themes)
        if wk is None:
            continue
        (other if NON_PERFORMANCE_RE.search(r["label"]) else perf).append((wk, r))

    wk2date: dict[int, date] = {}
    for wk, r in perf:
        wk2date.setdefault(wk, r["air_date"])
    for wk, r in other:
        wk2date.setdefault(wk, r["air_date"])

    # 3. Ordinal fallback: nothing matched by name, but dropping the obvious
    #    non-performance episodes leaves exactly one per week.
    if not wk2date:
        only_perf = [r for r in kept if not NON_PERFORMANCE_RE.search(r["label"])]
        if len(only_perf) == n_weeks:
            wk2date = {i + 1: r["air_date"] for i, r in enumerate(only_perf)}
            print(f"  season {season}: used ordinal mapping")

    wk2date = {w: d for w, d in wk2date.items() if 1 <= w <= n_weeks}
    if not wk2date:
        return {}

    # 4. The mapping must be internally monotonic.
    ordered = [wk2date[w] for w in sorted(wk2date)]
    if any(b <= a for a, b in zip(ordered, ordered[1:])):
        print(f"  season {season}: discarded mapping (weeks not in date order)")
        return {}

    # 5. Cross-check against the independent elimination dates. This catches
    #    seasons where the episode numbering does not correspond to the score
    #    table's weeks at all -- season 6 splits "Round 1" across three
    #    nights, so every later round is offset by a week.
    shared = sorted(set(wk2date) & set(elim))
    if len(shared) >= 3:
        diffs = sorted(abs((wk2date[w] - elim[w]).days) for w in shared)
        median = diffs[len(diffs) // 2]
        if median > 3:
            print(
                f"  season {season}: discarded mapping (disagrees with "
                f"elimination dates by {median}d median)"
            )
            return {}

    return wk2date


def map_scraped_to_weeks(
    scraped: pl.DataFrame, weeks: pl.DataFrame, elim: pl.DataFrame
) -> pl.DataFrame:
    """Attach a week number to each scraped episode row."""
    themes = {
        (r["season"], r["week"]): theme_from_week_label(r["week_label"])
        for r in weeks.iter_rows(named=True)
        if r["week_label"]
    }
    n_weeks = {
        r["season"]: r["n_weeks"]
        for r in weeks.group_by("season")
        .agg(pl.col("week").max().alias("n_weeks"))
        .iter_rows(named=True)
    }
    elim_map: dict[int, dict[int, date]] = {}
    for r in elim.iter_rows(named=True):
        elim_map.setdefault(r["season"], {})[r["week"]] = r["air_date"]

    out = []
    for (season,), grp in scraped.sort("season", "source", "episode_index").group_by(
        "season", maintain_order=True
    ):
        # prefer the richer source when a season has both
        for source in ("episodes", "ratings"):
            rows = grp.filter(pl.col("source") == source)
            if rows.height == 0:
                continue
            mapping = map_season_source(
                season,
                rows.iter_rows(named=True),
                themes,
                n_weeks.get(season, 0),
                elim_map.get(season, {}),
            )
            for wk, d in mapping.items():
                out.append(
                    {
                        "season": season,
                        "week": wk,
                        "air_date": d,
                        "date_source": source,
                    }
                )
            break  # this source is the best available; don't mix in the other

    return pl.DataFrame(
        out,
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "air_date": pl.Date,
            "date_source": pl.String,
        },
    )


def match_theme(season: int, label: str, themes: dict) -> int | None:
    """Match an episode title against the themed week labels of that season."""
    norm = normalize_theme(label)
    if not norm:
        return None
    hits = [
        wk
        for (s, wk), theme in themes.items()
        if s == season and theme and theme in norm
    ]
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------
# layer 3: derive dates from elimination text in the score data
# --------------------------------------------------------------------------

def dates_from_eliminations(scores: pl.DataFrame) -> pl.DataFrame:
    """A couple's last week is the week they were eliminated on `status`'s date."""
    per_couple = (
        scores.filter(
            pl.col("celebrity").is_not_null(), pl.col("status").is_not_null()
        )
        .group_by("season", "celebrity", "status")
        .agg(pl.col("week").max().alias("week"))
        .with_columns(
            pl.col("status")
            .str.extract(rf"((?:{MONTH_NAMES})\s+\d{{1,2}},\s+\d{{4}})")
            .str.to_date("%B %d, %Y", strict=False)
            .alias("air_date")
        )
        .drop_nulls("air_date")
    )
    return (
        per_couple.group_by("season", "week")
        .agg(pl.col("air_date").min())
        .with_columns(pl.lit("elimination").alias("date_source"))
    )


def calibrate_elimination(elim: pl.DataFrame, mapped: pl.DataFrame) -> pl.DataFrame:
    """Shift elimination dates onto the performance night.

    Through season 16 the results show aired the day *after* the performance,
    so a "Eliminated 2nd on ..." date is one day late; from season 17 the
    single combined show makes them identical. Rather than hard-coding that
    era boundary, measure the offset per season against the scraped dates
    (seasons where both layers exist agree on 0 or +1 with almost no noise),
    and for seasons with no scraped dates at all borrow the median offset of
    nearby seasons.
    """
    shared = mapped.join(elim, on=["season", "week"], suffix="_elim").with_columns(
        (pl.col("air_date_elim") - pl.col("air_date")).dt.total_days().alias("offset")
    )
    measured = {
        r["season"]: int(r["offset"])
        for r in shared.group_by("season")
        .agg(pl.col("offset").median().alias("offset"), pl.len().alias("n"))
        .filter(pl.col("n") >= 3)
        .iter_rows(named=True)
    }

    offsets = {}
    for season in elim.get_column("season").unique().to_list():
        if season in measured:
            offsets[season] = measured[season]
            continue
        nearby = [
            off for s, off in measured.items() if abs(s - season) <= 3
        ]
        offsets[season] = round(sorted(nearby)[len(nearby) // 2]) if nearby else 0
        print(
            f"  season {season}: no scraped dates to calibrate against, "
            f"borrowing offset {offsets[season]}d from neighbouring seasons"
        )

    return elim.with_columns(
        (
            pl.col("air_date")
            - pl.duration(
                days=pl.col("season").replace_strict(offsets, return_dtype=pl.Int64)
            )
        ).alias("air_date")
    )


# --------------------------------------------------------------------------
# layer 4: interpolation
# --------------------------------------------------------------------------

def interpolate_season(weeks: list[int], known: dict[int, date]) -> dict[int, date]:
    """Fill missing weeks from the known ones, assuming ~7-day spacing."""
    if not known:
        return {}
    anchors = sorted(known)
    filled = {}
    for wk in weeks:
        if wk in known:
            continue
        before = [a for a in anchors if a < wk]
        after = [a for a in anchors if a > wk]
        if before and after:
            lo, hi = before[-1], after[0]
            span = (known[hi] - known[lo]).days
            offset = round(span * (wk - lo) / (hi - lo))
            filled[wk] = known[lo] + timedelta(days=offset)
        elif before:
            lo = before[-1]
            filled[wk] = known[lo] + timedelta(days=7 * (wk - lo))
        else:
            hi = after[0]
            filled[wk] = known[hi] - timedelta(days=7 * (hi - wk))
    return filled


# --------------------------------------------------------------------------
# assembly + validation
# --------------------------------------------------------------------------

def combine(layers: pl.DataFrame, weeks: pl.DataFrame) -> pl.DataFrame:
    """Keep the highest-priority date per (season, week), then interpolate gaps."""
    best = (
        layers.with_columns(
            pl.col("date_source").replace_strict(SOURCE_PRIORITY, return_dtype=pl.Int64).alias("prio")
        )
        .sort("season", "week", "prio", "air_date")
        .group_by("season", "week", maintain_order=True)
        .first()
        .drop("prio")
    )

    known = {
        (r["season"], r["week"]): r["air_date"] for r in best.iter_rows(named=True)
    }
    all_weeks: dict[int, list[int]] = {}
    for r in weeks.iter_rows(named=True):
        all_weeks.setdefault(r["season"], []).append(r["week"])

    filled_rows = []
    for season, wks in all_weeks.items():
        season_known = {
            wk: d for (s, wk), d in known.items() if s == season
        }
        for wk, d in interpolate_season(sorted(wks), season_known).items():
            filled_rows.append(
                {
                    "season": season,
                    "week": wk,
                    "air_date": d,
                    "date_source": "interpolated",
                }
            )

    if filled_rows:
        best = pl.concat(
            [best, pl.DataFrame(filled_rows, schema=best.schema)], how="vertical"
        )
    return best.sort("season", "week")


def validate(air_dates: pl.DataFrame, weeks: pl.DataFrame) -> None:
    problems = []

    # 1. every (season, week) in the score data has a date
    missing = weeks.join(air_dates, on=["season", "week"], how="anti")
    if missing.height:
        problems.append(f"{missing.height} (season, week) slots have no date:\n{missing}")

    # 2. dates strictly increase with week inside a season
    non_mono = (
        air_dates.sort("season", "week")
        .with_columns(pl.col("air_date").diff().over("season").alias("delta"))
        .filter(pl.col("delta").is_not_null(), pl.col("delta").dt.total_days() <= 0)
    )
    if non_mono.height:
        problems.append(f"{non_mono.height} non-monotonic week->date steps:\n{non_mono}")

    # 3. one calendar year per season (no DWTS season straddles a new year)
    multiyear = (
        air_dates.with_columns(pl.col("air_date").dt.year().alias("y"))
        .group_by("season")
        .agg(pl.col("y").n_unique().alias("n_years"))
        .filter(pl.col("n_years") > 1)
    )
    if multiyear.height:
        problems.append(f"seasons spanning >1 calendar year:\n{multiyear}")

    if problems:
        raise SystemExit("VALIDATION FAILED\n\n" + "\n\n".join(problems))

    # advisory: unusual week-to-week spacing is worth eyeballing, not fatal
    odd = (
        air_dates.sort("season", "week")
        .with_columns(pl.col("air_date").diff().over("season").alias("delta"))
        .filter(
            pl.col("delta").is_not_null(),
            (pl.col("delta").dt.total_days() < 5) | (pl.col("delta").dt.total_days() > 14),
        )
    )
    if odd.height:
        print(f"\nnote: {odd.height} week gaps outside 5-14 days (not an error):")
        print(odd)


def main() -> None:
    scores = pl.read_parquet(SCORES_PATH)
    weeks = scores.select("season", "week", "week_label").unique(
        subset=["season", "week"]
    )

    print("loading scraped episode/ratings dates...")
    scraped = load_scraped().with_columns(pl.col("air_date").dt.year().alias("year"))
    scraped = drop_year_outliers(scraped).drop("year")

    print("mapping episodes to weeks...")
    elim = dates_from_eliminations(scores)
    mapped = map_scraped_to_weeks(scraped, weeks, elim)

    print("calibrating elimination dates onto the performance night...")
    elim = calibrate_elimination(elim, mapped)

    layers = pl.concat(
        [mapped, elim.select(mapped.columns)], how="vertical"
    )
    air_dates = combine(layers, weeks)

    validate(air_dates, weeks)

    air_dates.write_parquet(OUT_PATH)

    summary = (
        air_dates.group_by("date_source")
        .agg(pl.len().alias("n"))
        .with_columns((100 * pl.col("n") / air_dates.height).round(1).alias("pct"))
        .sort("n", descending=True)
    )
    print(f"\nwrote {air_dates.height} (season, week) dates to {OUT_PATH}")
    print(summary)


if __name__ == "__main__":
    main()
