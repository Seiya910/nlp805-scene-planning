"""Summarize measured latency and pair ScenePlan with direct generation.

Pairs must share the model checkpoint, hardware, settings, prompt, seed, and
repeat index. Warmup and model-loading time are excluded from latency_s.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_FIELDS = (
    "model", "model_id", "revision", "device", "dtype", "hardware",
    "torch_version", "diffusers_version", "steps", "guidance_scale", "width", "height",
)
SAMPLE_FIELDS = ("prompt_id", "seed", "repeat")


def number(value, name, line, *, allow_none=False):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"line {line}: {name} must be a finite nonnegative number")


def read_records(path):
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected an object")
        required = set(CONFIG_FIELDS + SAMPLE_FIELDS + ("condition", "latency_s", "stage_seconds", "agent_calls"))
        missing = required - row.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {', '.join(sorted(missing))}")
        if row["condition"] not in ("direct", "sceneplan"):
            raise ValueError(f"line {line_number}: unknown condition")
        for field in ("model", "model_id", "device", "dtype", "hardware", "prompt_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"line {line_number}: {field} must be nonempty text")
        for field in ("seed", "repeat", "steps", "width", "height"):
            if type(row[field]) is not int or row[field] < (1 if field in ("steps", "width", "height") else 0):
                raise ValueError(f"line {line_number}: invalid {field}")
        for field in ("latency_s", "guidance_scale"):
            number(row[field], field, line_number)
        if "model_load_s" in row:
            number(row["model_load_s"], "model_load_s", line_number)
        for field in ("gpu_baseline_allocated_mib", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib"):
            number(row.get(field), field, line_number, allow_none=True)
        for field in ("stage_seconds", "agent_calls"):
            if not isinstance(row[field], dict):
                raise ValueError(f"line {line_number}: {field} must be an object")
            for name, value in row[field].items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"line {line_number}: invalid {field} name")
                if field == "agent_calls":
                    if type(value) is not int or value < 0:
                        raise ValueError(f"line {line_number}: invalid agent call count")
                else:
                    number(value, f"stage_seconds.{name}", line_number)
        key = tuple(row[field] for field in CONFIG_FIELDS + SAMPLE_FIELDS) + (row["condition"],)
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate model/condition/prompt/seed/repeat")
        seen.add(key)
        records.append(row)
    if not records:
        raise ValueError("no latency records found")
    return records


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def condition_summary(rows):
    stages = sorted({name for row in rows for name in row["stage_seconds"]})
    agents = sorted({name for row in rows for name in row["agent_calls"]})
    load_times = sorted({row["model_load_s"] for row in rows if row.get("model_load_s") is not None})
    peak_memory = [row["gpu_peak_allocated_mib"] for row in rows
                   if row.get("gpu_peak_allocated_mib") is not None]
    extra_memory = [max(0, row["gpu_peak_allocated_mib"] - row["gpu_baseline_allocated_mib"])
                    for row in rows if row.get("gpu_peak_allocated_mib") is not None
                    and row.get("gpu_baseline_allocated_mib") is not None]
    return {
        "latency_s": stats([row["latency_s"] for row in rows]),
        "model_load_s_separate": stats(load_times),
        "stage_seconds": {
            name: stats([row["stage_seconds"][name] for row in rows if name in row["stage_seconds"]])
            for name in stages
        },
        "agent_calls": {name: stats([row["agent_calls"].get(name, 0) for row in rows]) for name in agents},
        "gpu_peak_allocated_mib": stats(peak_memory),
        "gpu_extra_allocated_mib": stats(extra_memory),
    }


def paired_summary(rows):
    by_sample = defaultdict(dict)
    for row in rows:
        sample = tuple(row[field] for field in SAMPLE_FIELDS)
        by_sample[sample][row["condition"]] = row
    pairs = [(conditions["direct"], conditions["sceneplan"])
             for conditions in by_sample.values()
             if "direct" in conditions and "sceneplan" in conditions]
    differences = [scene["latency_s"] - direct["latency_s"] for direct, scene in pairs]
    ratios = [scene["latency_s"] / direct["latency_s"] for direct, scene in pairs
              if direct["latency_s"] > 0]
    memory_differences = [scene["gpu_peak_allocated_mib"] - direct["gpu_peak_allocated_mib"]
                          for direct, scene in pairs
                          if direct.get("gpu_peak_allocated_mib") is not None
                          and scene.get("gpu_peak_allocated_mib") is not None]
    return {
        "matched_runs": len(pairs),
        "unmatched_runs": sum(len(conditions) for conditions in by_sample.values()
                              if len(conditions) == 1),
        "sceneplan_minus_direct_s": stats(differences),
        "sceneplan_divided_by_direct": stats(ratios),
        "gpu_peak_delta_mib": stats(memory_differences),
    }


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[field] for field in CONFIG_FIELDS)].append(row)
    groups = []
    for config_key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        by_condition = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        groups.append({
            "config": dict(zip(CONFIG_FIELDS, config_key)),
            "conditions": {name: condition_summary(group) for name, group in sorted(by_condition.items())},
            "paired_sceneplan_vs_direct": paired_summary(rows),
        })
    return {
        "timing_definition": "warm runs; model loading and warmup excluded; stage times inclusive",
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="also save summary as JSON")
    args = parser.parse_args()
    try:
        result = summarize(read_records(args.input))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
