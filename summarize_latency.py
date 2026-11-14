"""汇总真实 latency JSONL；只对相同设置、prompt 和 seed 做配对。"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * 0.95
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[lo] + (values[hi] - values[lo]) * (pos - lo)}


def read_records(paths):
    rows, seen = [], set()
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("comparison_id", "run_id", "model", "condition", "prompt_id", "prompt_sha256",
                        "seed", "repeat", "status", "latency_s", "stage_seconds", "agent_calls",
                        "model_load_s", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "checkpoint",
                        "checkpoint_identity", "revision", "generation", "scheduler", "environment", "timing")
            if not isinstance(row, dict) or any(k not in row for k in required):
                raise ValueError(f"{path}:{line_no}: not a current latency record")
            if row["condition"] not in ("direct", "sceneplan") or row["status"] not in ("ok", "failed"):
                raise ValueError(f"{path}:{line_no}: invalid condition/status")
            key = tuple(row[k] for k in ("comparison_id", "condition", "prompt_id", "prompt_sha256", "seed", "repeat"))
            if key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate run; use repeat indices for repeated measurements")
            seen.add(key)
            if not isinstance(row["stage_seconds"], dict) or not isinstance(row["agent_calls"], dict):
                raise ValueError(f"{path}:{line_no}: stages and agent calls must be objects")
            numbers = list(row["stage_seconds"].values()) + [row["model_load_s"], row["gpu_peak_allocated_mib"], row["gpu_peak_reserved_mib"]]
            if row["status"] == "ok":
                numbers.append(row["latency_s"])
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in numbers):
                raise ValueError(f"{path}:{line_no}: invalid timing value")
            if any(type(x) is not int or x < 0 for x in row["agent_calls"].values()):
                raise ValueError(f"{path}:{line_no}: invalid agent count")
            rows.append(row)
    if not rows:
        raise ValueError("no records")
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["comparison_id"]].append(row)
    output = []
    for group_id, group in grouped.items():
        conditions, paired = {}, defaultdict(dict)
        for condition in ("direct", "sceneplan"):
            all_rows = [r for r in group if r["condition"] == condition]
            if not all_rows:
                continue
            good = [r for r in all_rows if r["status"] == "ok"]
            # load time 每个 session 只取一次，不按图片数量重复统计。
            loads = {r["run_id"]: r["model_load_s"] for r in all_rows}
            stages = sorted({name for r in good for name in r["stage_seconds"]})
            agents = sorted({name for r in good for name in r["agent_calls"]})
            conditions[condition] = {
                "successes": len(good), "failures": len(all_rows) - len(good),
                "latency_s": stats([r["latency_s"] for r in good]),
                "model_load_s": stats(list(loads.values())),
                "stage_seconds": {name: stats([r["stage_seconds"][name] for r in good if name in r["stage_seconds"]]) for name in stages},
                "agent_calls": {name: stats([r["agent_calls"].get(name, 0) for r in good]) for name in agents},
                "total_agent_calls": stats([sum(r["agent_calls"].values()) for r in good]),
                "gpu_peak_allocated_mib": stats([r["gpu_peak_allocated_mib"] for r in good]),
                "gpu_peak_reserved_mib": stats([r["gpu_peak_reserved_mib"] for r in good]),
            }
            for r in good:
                key = tuple(r[k] for k in ("prompt_id", "prompt_sha256", "seed", "repeat"))
                paired[key][condition] = r
        pairs = [(r["direct"]["latency_s"], r["sceneplan"]["latency_s"])
                 for r in paired.values() if "direct" in r and "sceneplan" in r]
        first = group[0]
        output.append({
            "comparison_id": group_id,
            "settings": {k: first[k] for k in ("model", "checkpoint", "checkpoint_identity", "revision", "generation", "scheduler", "environment", "timing")},
            "conditions": conditions,
            "paired": {
                "n": len(pairs),
                "unmatched_successes": sum(len(r) for r in paired.values() if len(r) == 1),
                "sceneplan_minus_direct_s": stats([b - a for a, b in pairs]),
                "sceneplan_divided_by_direct": stats([b / a for a, b in pairs if a > 0]),
                "ratio_of_paired_means": (sum(b for _, b in pairs) / sum(a for a, _ in pairs)) if pairs and sum(a for a, _ in pairs) > 0 else None,
            },
        })
    return {"note": "seconds; warmup/load excluded from latency; stages inclusive (do not sum nested or parallel stages)", "groups": output}


def write_csv(summary, path):
    fields = ["comparison_id", "model", "condition", "successes", "failures", "mean_s", "median_s", "p95_s",
              "mcp_mean_s", "hcd_mean_s", "agent_calls_mean", "peak_allocated_mib_mean",
              "peak_reserved_mib_mean", "paired_n", "paired_overhead_s", "paired_ratio"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in summary["groups"]:
            for condition, result in group["conditions"].items():
                latency = result["latency_s"] or {}
                writer.writerow({
                    "comparison_id": group["comparison_id"],
                    "model": group["settings"]["model"], "condition": condition,
                    "successes": result["successes"], "failures": result["failures"],
                    "mean_s": latency.get("mean"), "median_s": latency.get("median"), "p95_s": latency.get("p95"),
                    "mcp_mean_s": (result["stage_seconds"].get("mcp") or {}).get("mean"),
                    "hcd_mean_s": (result["stage_seconds"].get("hcd") or {}).get("mean"),
                    "agent_calls_mean": (result["total_agent_calls"] or {}).get("mean"),
                    "peak_allocated_mib_mean": (result["gpu_peak_allocated_mib"] or {}).get("mean"),
                    "peak_reserved_mib_mean": (result["gpu_peak_reserved_mib"] or {}).get("mean"),
                    "paired_n": group["paired"]["n"],
                    "paired_overhead_s": (group["paired"]["sceneplan_minus_direct_s"] or {}).get("mean"),
                    "paired_ratio": group["paired"]["ratio_of_paired_means"],
                })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="+", help="one or more JSONL files")
    p.add_argument("--output", help="save full JSON summary")
    p.add_argument("--csv", help="save a small result table")
    args = p.parse_args()
    try:
        summary = summarize(read_records(args.input))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    encoded = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    if args.csv:
        write_csv(summary, args.csv)
    print(encoded)


if __name__ == "__main__":
    main()
