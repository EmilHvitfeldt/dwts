"""Scrape a place of birth for every Dancing with the Stars celebrity.

Celebrities are identified by the wikilink in the season page's Couples
table, exactly as in scrape_birthdates.py, and for the same reason: name
matching quietly resolves single-name contestants like "Brandy", "Mario"
and "Romeo" to unrelated articles.

Places come from Wikidata property P19 (place of birth). The raw value is a
settlement, so the country and the state/province are found by walking the
P131 ("located in the administrative territorial entity") chain upwards:

  country  the nearest P17 on the birthplace itself or one of its ancestors
  state    the nearest ancestor that is a first-level subdivision of a
           country that actually has named states or provinces

`birth_state` is deliberately null where a state makes no sense: the UK,
Ireland, Cuba and so on are recorded with a country only. The same is true
for places whose administrative chain is too thin to identify a subdivision.

A handful of celebrities have no P19 on Wikidata at all; those are covered
by MANUAL_BIRTHPLACES below, read off the English Wikipedia article.

Output: data/celebrity_birthplaces.parquet
"""

import time
from pathlib import Path

import polars as pl

from scrape_birthdates import (
    WIKIDATA_API,
    chunks,
    get_json,
    scrape_all_links,
    titles_to_qids,
)

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "celebrity_birthplaces.parquet"

# Countries whose first-level subdivisions are worth recording as a "state".
# Anywhere else gets a null state rather than a county, region or city-state
# that would not be what someone means by the word.
STATE_LIKE_COUNTRIES = {
    "Q30",    # United States
    "Q16",    # Canada
    "Q408",   # Australia
    "Q96",    # Mexico
    "Q155",   # Brazil
    "Q668",   # India
    "Q183",   # Germany
}

# Territories and dependencies that are their own "country" for this purpose
# even though Wikidata hangs them off a sovereign state via P17.
TERRITORY_AS_COUNTRY = {
    "Q1183",  # Puerto Rico
}

# Places that count as a state without sitting directly under the country in
# the P131 chain. Washington, D.C. is a federal district rather than a state,
# but it is what belongs in the column for someone born there.
SUBDIVISION_ITEMS = {
    "Q61",  # Washington, D.C.
}

MAX_P131_DEPTH = 6

# Hand-entered places, keyed by article title, for the celebrities Wikidata
# has no usable P19 for (mostly recent reality-TV contestants whose items are
# stubs) and the one it has wrong. Each is sourced from the English Wikipedia
# article, and a null field means the article does not say. Applied over the
# Wikidata result, so a value appearing upstream later does not silently
# disagree with what is recorded here; unused entries are reported on each
# run so this list does not rot.
MANUAL_BIRTHPLACES = {
    # article title: (city, state, country, note)
    "Brooks Nader": ("Baton Rouge", "Louisiana", "United States", "born and raised"),
    "Charity Lawson": ("Columbus", "Georgia", "United States", "born and raised"),
    "Christine Chiu": (None, None, "Taiwan", "article redirects to Bling Empire#Cast"),
    "Cody Rigsby": (None, "California", "United States", "born in California, no city given"),
    "Harry Jowsey": ("Yeppoon", "Queensland", "Australia", "infobox"),
    "Jen Affleck": (None, None, "United States", "described as American, no birthplace given"),
    "Jenn Tran": ("Hillsdale", "New Jersey", "United States", "infobox"),
    "Matt James (television personality)": (
        None, "North Carolina", "United States", "raised in Raleigh; birth city not given",
    ),
    "Whitney Leavitt": ("American Fork", "Utah", "United States", "infobox"),
    # Wikidata gives Mudgee, New South Wales, which belongs to someone else;
    # the rodeo cowboy who danced in season 8 was born in Phoenix.
    "Ty Murray": ("Phoenix", "Arizona", "United States", "corrects a wrong Wikidata P19"),
}


def apply_manual(out: pl.DataFrame) -> pl.DataFrame:
    """Overlay MANUAL_BIRTHPLACES onto the scraped table."""
    unused = set(MANUAL_BIRTHPLACES) - set(out.get_column("article_title").to_list())
    if unused:
        print(f"\nnote: {len(unused)} manual entries matched no celebrity: {sorted(unused)}")

    fields = {"birth_city": 0, "birth_state": 1, "birth_country": 2}
    return out.with_columns(
        pl.coalesce(
            pl.col("article_title").replace_strict(
                {k: v[i] for k, v in MANUAL_BIRTHPLACES.items()},
                default=None,
                return_dtype=pl.String,
            ),
            # a manual row with a null field means "unknown", not "fall back",
            # so blank the scraped value for every celebrity in the table
            pl.when(pl.col("article_title").is_in(list(MANUAL_BIRTHPLACES)))
            .then(None)
            .otherwise(pl.col(name)),
        ).alias(name)
        for name, i in fields.items()
    )


def entity_claims(qids: list[str], batch: int = 25) -> dict[str, dict]:
    """Fetch claims for a set of items, keyed by item id."""
    out: dict[str, dict] = {}
    for chunk in chunks(sorted(set(qids)), batch):
        data = get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(chunk),
                "props": "claims",
                "languages": "en",
            },
        )
        for qid, entity in data.get("entities", {}).items():
            out[qid] = entity.get("claims", {})
        time.sleep(1)
    return out


def entity_labels(qids: list[str], batch: int = 50) -> dict[str, str]:
    """Fetch English labels for a set of items."""
    out: dict[str, str] = {}
    for chunk in chunks(sorted(set(qids)), batch):
        data = get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(chunk),
                "props": "labels",
                "languages": "en",
            },
        )
        for qid, entity in data.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                out[qid] = label
        time.sleep(1)
    return out


def claim_ids(claims: dict, prop: str) -> list[str]:
    """Item ids for one property, preferred-rank statements first.

    Several people have more than one P19 (a hospital and the town it is in,
    or a disputed birthplace); ordering by rank puts the one Wikidata editors
    settled on at the front.
    """
    out_pref, out_norm = [], []
    for claim in claims.get(prop, []):
        if claim.get("rank") == "deprecated":
            continue
        snak = claim.get("mainsnak", {})
        value = snak.get("datavalue", {}).get("value")
        if not isinstance(value, dict) or "id" not in value:
            continue
        (out_pref if claim.get("rank") == "preferred" else out_norm).append(value["id"])
    return out_pref + out_norm


def resolve_place_chain(place: str, claims: dict[str, dict]) -> list[str]:
    """The birthplace followed by its P131 ancestors, nearest first."""
    chain = [place]
    seen = {place}
    current = place
    for _ in range(MAX_P131_DEPTH):
        parents = claim_ids(claims.get(current, {}), "P131")
        parent = next((p for p in parents if p not in seen), None)
        if parent is None:
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return chain


def country_of(chain: list[str], claims: dict[str, dict]) -> str | None:
    """Nearest P17 along the chain, preferring a territory over its sovereign."""
    for qid in chain:
        if qid in TERRITORY_AS_COUNTRY:
            return qid
    for qid in chain:
        countries = claim_ids(claims.get(qid, {}), "P17")
        if countries:
            return countries[0]
    return None


def state_of(chain: list[str], country: str | None, claims: dict[str, dict]) -> str | None:
    """First-level subdivision along the chain, where one is meaningful.

    A first-level subdivision is identified structurally, as the chain entry
    that sits directly under the country: Philadelphia -> Philadelphia
    County -> Pennsylvania -> United States picks Pennsylvania. Matching on
    P31 instead would need a class per country and still miss cases, since
    Hidalgo and Ontario do not share a class with the U.S. states.

    The chain is walked outermost-first so the entry closest to the country
    wins when a place declares more than one parent.
    """
    if country not in STATE_LIKE_COUNTRIES:
        return None
    for qid in reversed(chain):
        if qid == country:
            continue
        if qid in SUBDIVISION_ITEMS:
            return qid
        if country in claim_ids(claims.get(qid, {}), "P131"):
            return qid
    return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("reading cast tables...")
    links = scrape_all_links()

    people = (
        links.drop_nulls("article_title")
        .group_by("celebrity")
        .agg(pl.col("article_title").first())
    )
    unlinked = links.filter(pl.col("article_title").is_null())
    if unlinked.height:
        print(f"\n{unlinked.height} celebrity cells had no wikilink:")
        print(unlinked.select("season", "celebrity"))

    titles = sorted(set(people.get_column("article_title").to_list()))
    print(f"\nresolving {len(titles)} article titles to Wikidata items...")
    qids = titles_to_qids(titles)
    print(f"  resolved {len(qids)}/{len(titles)}")

    print("fetching places of birth...")
    person_claims = entity_claims(sorted(set(qids.values())))
    birthplace = {
        qid: claim_ids(claims, "P19")[0]
        for qid, claims in person_claims.items()
        if claim_ids(claims, "P19")
    }
    print(f"  found P19 for {len(birthplace)}/{len(person_claims)} people")

    print("walking the administrative chain above each birthplace...")
    place_claims: dict[str, dict] = {}
    frontier = sorted(set(birthplace.values()))
    for _ in range(MAX_P131_DEPTH):
        todo = [q for q in frontier if q not in place_claims]
        if not todo:
            break
        place_claims.update(entity_claims(todo))
        frontier = [
            p
            for q in todo
            for p in claim_ids(place_claims.get(q, {}), "P131")
        ]

    resolved = {}
    for person_qid, place in birthplace.items():
        chain = resolve_place_chain(place, place_claims)
        country = country_of(chain, place_claims)
        resolved[person_qid] = (place, state_of(chain, country, place_claims), country)

    label_ids = {q for triple in resolved.values() for q in triple if q}
    print(f"labelling {len(label_ids)} places...")
    labels = entity_labels(sorted(label_ids))

    def field(index: int) -> dict[str, str]:
        return {
            person: labels[triple[index]]
            for person, triple in resolved.items()
            if triple[index] and triple[index] in labels
        }

    out = (
        people.with_columns(
            pl.col("article_title")
            .replace_strict(qids, default=None, return_dtype=pl.String)
            .alias("wikidata_id")
        )
        .with_columns(
            pl.col("wikidata_id")
            .replace_strict(field(0), default=None, return_dtype=pl.String)
            .alias("birth_city"),
            pl.col("wikidata_id")
            .replace_strict(field(1), default=None, return_dtype=pl.String)
            .alias("birth_state"),
            pl.col("wikidata_id")
            .replace_strict(field(2), default=None, return_dtype=pl.String)
            .alias("birth_country"),
        )
        .select(
            "celebrity", "article_title", "wikidata_id",
            "birth_city", "birth_state", "birth_country",
        )
        .sort("celebrity")
    )

    if out.height == 0:
        raise SystemExit("no celebrities resolved; refusing to write an empty table")

    out = apply_manual(out)
    out.write_parquet(OUT_PATH)
    found = out.get_column("birth_country").is_not_null().sum()
    print(f"\nwrote {out.height} celebrities to {OUT_PATH}")
    print(f"country found for {found}/{out.height} ({100 * found / out.height:.1f}%)")
    print(
        out.group_by("birth_country")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    missing = out.filter(pl.col("birth_country").is_null())
    if missing.height:
        print("\nno place of birth:")
        print(missing.select("celebrity", "article_title", "wikidata_id"))


if __name__ == "__main__":
    main()
