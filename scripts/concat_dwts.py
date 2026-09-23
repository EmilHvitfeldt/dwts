"""Combine all per-season DWTS parquet files into a single tidy dataset."""

from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
AIR_DATES_PATH = DATA_DIR / "air_dates.parquet"
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


if __name__ == "__main__":
    combined = attach_air_dates(concat_seasons())
    combined.write_parquet(OUT_PATH)
    print(f"combined {combined.get_column('season').n_unique()} seasons")
    print(f"wrote {combined.height} rows to {OUT_PATH}")
    print(combined)
