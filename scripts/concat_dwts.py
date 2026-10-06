"""Combine all per-season DWTS parquet files into a single tidy dataset."""

from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
AIR_DATES_PATH = DATA_DIR / "air_dates.parquet"
BIRTHDATES_PATH = DATA_DIR / "celebrity_birthdates.parquet"
OUT_PATH = DATA_DIR / "dwts_all_seasons.parquet"


def concat_seasons(data_dir: Path = RAW_DIR) -> pl.DataFrame:
    season_files = sorted(
        data_dir.glob("season-*.parquet"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not season_files:
        raise FileNotFoundError(f"No season-*.parquet files found in {data_dir}")

    frames = [pl.read_parquet(f) for f in season_files]
    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined


def attach_air_dates(combined: pl.DataFrame) -> pl.DataFrame:
    """Add air_date / date_source from air_dates.parquet, if it has been built.

    Left join, so a missing date never silently drops score rows.
    """
    if not AIR_DATES_PATH.exists():
        print(f"note: {AIR_DATES_PATH} not found; run build_air_dates.py to add dates")
        return combined

    air_dates = pl.read_parquet(AIR_DATES_PATH)
    joined = combined.join(air_dates, on=["season", "week"], how="left")
    missing = joined.get_column("air_date").null_count()
    if missing:
        print(f"warning: {missing} rows have no air date")
    return joined


def attach_ages(combined: pl.DataFrame) -> pl.DataFrame:
    """Add birthdate and the celebrity's age at the performance.

    `age` is exact years (e.g. 46.4), so floor it for a whole-number age.
    Requires air_date, since the age is measured at the performance rather
    than at some fixed point in the season.
    """
    if not BIRTHDATES_PATH.exists():
        print(f"note: {BIRTHDATES_PATH} not found; run scrape_birthdates.py to add ages")
        return combined
    if "air_date" not in combined.columns:
        print("note: no air_date column, so ages cannot be computed")
        return combined

    birthdates = pl.read_parquet(BIRTHDATES_PATH).select(
        "celebrity", "birthdate", "birthdate_precision"
    )
    joined = combined.join(birthdates, on="celebrity", how="left").with_columns(
        (
            (pl.col("air_date") - pl.col("birthdate")).dt.total_days() / 365.2425
        ).alias("age")
    )

    known = joined.filter(pl.col("celebrity").is_not_null())
    missing = known.get_column("age").null_count()
    if missing:
        names = (
            known.filter(pl.col("age").is_null())
            .get_column("celebrity")
            .unique()
            .to_list()
        )
        print(f"note: {missing} rows have no age ({len(names)} celebrities: {names})")

    # Willow Shields was 14 in season 20, so the floor has to sit below that;
    # this is a guard against a mis-resolved person, not a realism check.
    implausible = joined.filter(
        pl.col("age").is_not_null(), (pl.col("age") < 10) | (pl.col("age") > 95)
    )
    if implausible.height:
        raise ValueError(
            f"VALIDATION FAILED: {implausible.height} rows have an implausible age\n"
            f"{implausible.select('season', 'celebrity', 'air_date', 'birthdate', 'age')}"
        )

    return joined


if __name__ == "__main__":
    try:
        combined = attach_ages(attach_air_dates(concat_seasons()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    combined.write_parquet(OUT_PATH)
    print(f"combined {combined.get_column('season').n_unique()} seasons")
    print(f"wrote {combined.height} rows to {OUT_PATH}")
    print(combined)
