#!/usr/bin/env python3
"""
Convert food-security CSV data files to dataset-centric JSON-LD.

Reads fileA (validation), fileB (discovery), and fileC (joins) CSVs and
generates one JSON-LD file per curated dataset using schema.org vocabulary,
following Natasha Noy's dataset-centric schema design.

Usage:
    python convert.py [--data-dir ../../data] [--include-joins]
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


CONTEXT = {
    "@vocab": "https://schema.org/",
    "dud": "https://nationaldataplatform.org/data-usage-descriptor/",
}


def slugify(name: str) -> str:
    """Convert dataset name to a filename-safe slug."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80]


def normalize(name: str) -> str:
    """Normalize a dataset name for matching."""
    s = name.lower().strip()
    # Remove common parenthetical abbreviations for matching
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def load_registry(data_dir: Path) -> tuple[dict, dict, dict]:
    """Load curated dataset registry from seed + additional CSVs.

    Returns:
        registry: {dataset_id: {name, aliases, slug, reviews[]}}
        alias_index: {normalized_name: dataset_id}
        name_to_id: {exact_canonical_name: dataset_id} for fileA matching
    """
    registry = {}
    alias_index = {}
    name_to_id = {}

    for csv_file in ["usda_seed_datasets.csv", "additional_discovered_datasets.csv"]:
        path = data_dir / csv_file
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                did = row["dataset_id"].strip()
                name = row["dataset_name"].strip()
                raw_aliases = row.get("aliases", "")
                aliases = [a.strip() for a in raw_aliases.split(";") if a.strip()]

                registry[did] = {
                    "name": name,
                    "aliases": aliases,
                    "slug": slugify(name),
                    "group": row.get("group_name", ""),
                    "reviews": [],
                }

                # Exact name lookup (for fileA which uses different IDs)
                name_to_id[name] = did

                # Index canonical name + all aliases (normalized)
                for variant in [name] + aliases:
                    norm = normalize(variant)
                    if norm and norm not in alias_index:
                        alias_index[norm] = did

    # Auto-register datasets that appear in fileA but not in seed/additional CSVs
    # (e.g., NHANES, NSCH, EQI, HCUP NIS from cross-domain overlap)
    file_a = data_dir / "fileA_usda_publication_dataset_pairs.csv"
    if file_a.exists():
        with open(file_a, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("validated_dataset_name", "").strip()
                if not name:
                    continue
                if name not in name_to_id and normalize(name) not in alias_index:
                    # Use the MongoDB ObjectID from fileA as identifier
                    oid = row.get("validated_dataset_id", "").strip()
                    did = oid or slugify(name)
                    if did not in registry:
                        registry[did] = {
                            "name": name,
                            "aliases": [],
                            "slug": slugify(name),
                            "group": "cross-domain",
                            "reviews": [],
                        }
                        name_to_id[name] = did
                        alias_index[normalize(name)] = did

    return registry, alias_index, name_to_id


def match_name(name: str, alias_index: dict) -> str | None:
    """Try to match a free-text dataset name to a curated dataset ID."""
    norm = normalize(name)
    if norm in alias_index:
        return alias_index[norm]

    # Try matching without trailing qualifiers (e.g., "ACS 5-year estimates")
    for key, did in alias_index.items():
        if key in norm or norm in key:
            return did

    return None


def make_citation(row: dict) -> dict:
    """Build a schema.org ScholarlyArticle citation from a CSV row."""
    citation = {
        "@type": "ScholarlyArticle",
        "name": row.get("publication_title", ""),
    }
    doi = row.get("publication_doi", "").strip()
    if doi:
        citation["identifier"] = doi
    # Use Dimensions URL for traceability (no custom namespace needed)
    pub_id = row.get("publication_id", "").strip()
    if pub_id:
        citation["url"] = f"https://app.dimensions.ai/details/publication/{pub_id}"
    return citation


def make_rating(name: str, value) -> dict | None:
    """Build a schema.org Rating entry."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return {"@type": "Rating", "name": name, "ratingValue": v}


def process_file_a(data_dir: Path, registry: dict, name_to_id: dict,
                    alias_index: dict) -> dict:
    """Process fileA (validation reviews). Returns {dataset_id: [reviews]}.

    fileA uses MongoDB ObjectIDs which differ from the registry UUIDs,
    so we match by validated_dataset_name instead.
    """
    reviews = defaultdict(list)
    path = data_dir / "fileA_usda_publication_dataset_pairs.csv"
    matched = 0
    total = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            name = row.get("validated_dataset_name", "").strip()
            # Try exact name match first, then alias index
            did = name_to_id.get(name) or match_name(name, alias_index)
            if not did:
                continue
            matched += 1

            ratings = []
            r = make_rating("mention", row.get("confidence_score_mention"))
            if r:
                ratings.append(r)
            r = make_rating("use", row.get("confidence_score_use"))
            if r:
                ratings.append(r)

            review = {
                "@type": "Review",
                "reviewAspect": "validation",
                "reviewBody": row.get("validation_reasoning", ""),
                "citation": make_citation(row),
                "reviewRating": ratings,
                "dud:dataset_usage_status": "validated",
            }
            reviews[did].append(review)

    print(f"  fileA: {matched}/{total} rows matched to registry")
    return reviews


def process_file_b(data_dir: Path, registry: dict, alias_index: dict) -> dict:
    """Process fileB (discovery reviews). Returns {dataset_id: [reviews]}."""
    reviews = defaultdict(list)
    path = data_dir / "fileB_additional_datasets_publication_pairs.csv"
    matched = 0
    total = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            name = row.get("new_name", "").strip()
            if not name:
                continue

            did = match_name(name, alias_index)
            if not did:
                continue
            matched += 1

            ratings = []
            r = make_rating("mention", row.get("new_confidence_score_mention"))
            if r:
                ratings.append(r)
            r = make_rating("use", row.get("new_confidence_score_use"))
            if r:
                ratings.append(r)

            review = {
                "@type": "Review",
                "reviewAspect": "discovery",
                "reviewBody": row.get("new_description", ""),
                "citation": make_citation(row),
                "reviewRating": ratings,
                "dud:dataset_usage_status": "discovered",
            }
            source = row.get("new_source", "").strip()
            if source:
                review["sdPublisher"] = {
                    "@type": "Organization",
                    "name": source,
                }
            domain = row.get("new_domain", "").strip()
            if domain:
                review["about"] = domain

            reviews[did].append(review)

    print(f"  fileB: {matched}/{total} rows matched to registry")
    return reviews


def process_file_c(data_dir: Path, registry: dict, alias_index: dict) -> dict:
    """Process fileC (join reviews). Returns {dataset_id: [reviews]}."""
    reviews = defaultdict(list)
    path = data_dir / "fileC_dataset_joins.csv"
    matched = 0
    total = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            ds1 = row.get("join_dataset1", "").strip()
            ds2 = row.get("join_dataset2", "").strip()
            did1 = match_name(ds1, alias_index)
            did2 = match_name(ds2, alias_index)

            if not did1 and not did2:
                continue
            matched += 1

            citation = make_citation(row)

            join_rating = make_rating(
                "join_confidence", row.get("join_confidence_score")
            )

            # Add review to dataset 1 if matched
            if did1:
                review = {
                    "@type": "Review",
                    "reviewAspect": "join",
                    "reviewBody": row.get("join_context", ""),
                    "citation": citation,
                    "dud:joinDetail": {
                        "partnerDataset": ds2,
                        "joinType": row.get("join_join_type", ""),
                        "joinKeys": row.get("join_join_keys", ""),
                        "methodology": row.get("join_methodology", ""),
                    },
                }
                if join_rating:
                    review["reviewRating"] = [join_rating]
                reviews[did1].append(review)

            # Add symmetric review to dataset 2 if matched
            if did2:
                review = {
                    "@type": "Review",
                    "reviewAspect": "join",
                    "reviewBody": row.get("join_context", ""),
                    "citation": citation,
                    "dud:joinDetail": {
                        "partnerDataset": ds1,
                        "joinType": row.get("join_join_type", ""),
                        "joinKeys": row.get("join_join_keys", ""),
                        "methodology": row.get("join_methodology", ""),
                    },
                }
                if join_rating:
                    review["reviewRating"] = [join_rating]
                reviews[did2].append(review)

    print(f"  fileC: {matched}/{total} rows matched to registry (at least one side)")
    return reviews


def assemble_dataset(did: str, info: dict) -> dict:
    """Build the final JSON-LD document for one dataset."""
    doc = {
        "@context": CONTEXT,
        "@type": "Dataset",
        "name": info["name"],
        "identifier": did,
    }
    if info["aliases"]:
        doc["alternateName"] = info["aliases"]

    reviews = info["reviews"]
    if reviews:
        # Sort: validation first, then discovery, then join; within each by DOI
        aspect_order = {"validation": 0, "discovery": 1, "join": 2}
        reviews.sort(
            key=lambda r: (
                aspect_order.get(r.get("reviewAspect", ""), 9),
                r.get("citation", {}).get("identifier", ""),
            )
        )
        doc["review"] = reviews

    return doc


def build_catalog(registry: dict, out_dir: Path) -> dict:
    """Build a DataCatalog referencing all dataset files."""
    datasets = []
    for did, info in sorted(registry.items(), key=lambda x: x[1]["name"]):
        if info["reviews"]:
            datasets.append(
                {
                    "@type": "Dataset",
                    "name": info["name"],
                    "identifier": did,
                    "url": f"datasets/{info['slug']}.jsonld",
                }
            )
    return {
        "@context": CONTEXT,
        "@type": "DataCatalog",
        "name": "Food Security Data-Usage Descriptors",
        "description": (
            "Dataset-centric data-usage descriptors linking food security datasets "
            "to scientific publications, with LLM-validated confidence scores."
        ),
        "dataset": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert CSVs to JSON-LD")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data",
        help="Path to data/ directory",
    )
    parser.add_argument(
        "--include-joins",
        action="store_true",
        default=True,
        help="Include join reviews from fileC (default: true)",
    )
    parser.add_argument(
        "--no-joins",
        action="store_true",
        help="Exclude join reviews from fileC",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    include_joins = not args.no_joins

    print(f"Data directory: {data_dir}")
    out_dir = data_dir / "json-ld"
    ds_dir = out_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load registry
    print("\nLoading dataset registry...")
    registry, alias_index, name_to_id = load_registry(data_dir)
    print(f"  {len(registry)} curated datasets loaded")
    print(f"  {len(alias_index)} alias entries indexed")

    # 2. Process CSVs
    print("\nProcessing CSV files...")
    a_reviews = process_file_a(data_dir, registry, name_to_id, alias_index)
    b_reviews = process_file_b(data_dir, registry, alias_index)

    if include_joins:
        c_reviews = process_file_c(data_dir, registry, alias_index)
    else:
        c_reviews = {}
        print("  fileC: skipped (--no-joins)")

    # 3. Merge reviews into registry
    for did in registry:
        registry[did]["reviews"] = (
            a_reviews.get(did, [])
            + b_reviews.get(did, [])
            + c_reviews.get(did, [])
        )

    # 4. Write JSON-LD files
    print("\nWriting JSON-LD files...")
    written = 0
    empty = 0
    for did, info in registry.items():
        doc = assemble_dataset(did, info)
        path = ds_dir / f"{info['slug']}.jsonld"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        if info["reviews"]:
            written += 1
        else:
            empty += 1

    # 5. Write context file
    context_path = out_dir / "context.jsonld"
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump({"@context": CONTEXT}, f, indent=2)

    # 6. Write catalog
    catalog = build_catalog(registry, out_dir)
    catalog_path = out_dir / "catalog.jsonld"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # 7. Summary
    total_reviews = sum(len(info["reviews"]) for info in registry.values())
    val_count = sum(
        1
        for info in registry.values()
        for r in info["reviews"]
        if r.get("reviewAspect") == "validation"
    )
    disc_count = sum(
        1
        for info in registry.values()
        for r in info["reviews"]
        if r.get("reviewAspect") == "discovery"
    )
    join_count = sum(
        1
        for info in registry.values()
        for r in info["reviews"]
        if r.get("reviewAspect") == "join"
    )

    print(f"\n{'='*50}")
    print(f"  Datasets with reviews: {written}")
    print(f"  Datasets empty:        {empty}")
    print(f"  Total reviews:         {total_reviews}")
    print(f"    validation:          {val_count}")
    print(f"    discovery:           {disc_count}")
    print(f"    join:                {join_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
