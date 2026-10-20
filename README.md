# Dancing with the Stars (DWTS)

Repo that collects data from Wikipedia pages about each Dancing with the Stars season.

## Data

- `data/dwts_all_seasons.parquet`: one row per couple per scored (or unscored) performance, seasons 1 to 34.
- `data/judge_scores.parquet`: one row per individual judge score, for joining judges onto the performances.
- `data/raw/`: the per-season scrapes the combined file is built from.

### Judge scores

In `dwts_all_seasons.parquet`, `judge_scores` is a list of scores (half points in season 15) and `judges` is a list of the same length naming who gave each one, in the order Wikipedia lists them. Both are null for unscored dances (relays, marathons, dance-offs, group dances). `judges` is also null for the four season 20 week 9 team-up dances, where Wikipedia does not say which judges scored.

`score_type` is `technical` or `performance` in the two weeks that scored them separately (season 10 and 11, week 4) and null otherwise. Those weeks have one row per score.

### Keys

`(season, week, performance_order)` is unique. `performance_order` is the position of the row within the week's tables, in the order Wikipedia lists them. `(season, week, couple)` is not unique, since a couple can dance more than once in a week.

`judge_scores.parquet` has the same key columns plus `couple`, `judge` and `score`, so it joins back with `season`, `week` and `performance_order`.

## Rebuilding

```
uv run python scripts/scrape_dwts.py
uv run python scripts/concat_dwts.py
```
