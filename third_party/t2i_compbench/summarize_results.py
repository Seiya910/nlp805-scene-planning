#!/usr/bin/env python3
"""Summarize normalized T2I-CompBench spatial evaluation records.

The upstream evaluator produces benchmark-specific files. This script deliberately
accepts a small normalized JSONL bridge so that the project does not depend on a
particular upstream output filename or copy the benchmark implementation.
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("direct", "unchecked_plan", "checked_plan")


def read_records(path):
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")

        required = {"prompt_id", "condition", "seed", "spatial_score", "both_objects_detected"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {', '.join(sorted(missing))}")
        if record["condition"] not in CONDITIONS:
            raise ValueError(f"line {line_number}: unknown condition {record['condition']!r}")
        if (
            not isinstance(record["prompt_id"], (str, int))
            or isinstance(record["prompt_id"], bool)
            or not str(record["prompt_id"]).strip()
        ):
            raise ValueError(f"line {line_number}: prompt_id must be a nonempty string or integer")
        if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
            raise ValueError(f"line {line_number}: seed must be an integer")
        score = record["spatial_score"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
            or not math.isfinite(score)
        ):
            raise ValueError(f"line {line_number}: spatial_score must be between 0 and 1")
        if not isinstance(record["both_objects_detected"], bool):
            raise ValueError(f"line {line_number}: both_objects_detected must be boolean")
        key = (str(record["prompt_id"]), record["condition"], record["seed"])
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate prompt, condition, and seed")
        seen.add(key)
        records.append(record)

    if not records:
        raise ValueError("input contains no records")
    return records


def prompt_means(records):
    values = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (str(record["prompt_id"]), record["condition"])
        values[key[0]][key[1]].append(float(record["spatial_score"]))

    means = {}
    for prompt_id, by_condition in values.items():
        means[prompt_id] = {
            condition: statistics.mean(scores)
            for condition, scores in by_condition.items()
        }
    return means


def check_paired_seeds(records):
    seeds = defaultdict(lambda: defaultdict(set))
    for record in records:
        seeds[str(record["prompt_id"])][record["condition"]].add(record["seed"])

    for prompt_id, by_condition in seeds.items():
        direct = by_condition.get("direct")
        if direct is None:
            continue
        for condition in CONDITIONS[1:]:
            if condition in by_condition and by_condition[condition] != direct:
                raise ValueError(
                    f"prompt {prompt_id}: {condition} and direct must use the same seeds"
                )


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(differences, rounds, seed):
    if not differences:
        return None
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choice(differences) for _ in differences)
        for _ in range(rounds)
    ]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def summarize(records, rounds=2000, seed=42):
    check_paired_seeds(records)
    means = prompt_means(records)
    summary = {}
    for condition in CONDITIONS:
        prompt_scores = [row[condition] for row in means.values() if condition in row]
        image_rows = [row for row in records if row["condition"] == condition]
        summary[condition] = {
            "prompt_count": len(prompt_scores),
            "image_count": len(image_rows),
            "mean_spatial_score": statistics.mean(prompt_scores) if prompt_scores else None,
            "object_detection_coverage": (
                statistics.mean(row["both_objects_detected"] for row in image_rows)
                if image_rows else None
            ),
        }

    paired = {
        condition: [
            means[prompt_id][condition] - means[prompt_id]["direct"]
            for prompt_id in means
            if "direct" in means[prompt_id] and condition in means[prompt_id]
        ]
        for condition in CONDITIONS
        if condition != "direct"
    }
    differences = {
        condition: {
            "prompt_count": len(values),
            "mean_difference_vs_direct": statistics.mean(values) if values else None,
            "bootstrap_95_ci": bootstrap_ci(values, rounds, seed),
        }
        for condition, values in paired.items()
    }
    return {"conditions": summary, "paired_differences_vs_direct": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="normalized JSONL result file")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_rounds < 1:
        parser.error("--bootstrap-rounds must be positive")

    result = summarize(read_records(args.input), args.bootstrap_rounds, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
