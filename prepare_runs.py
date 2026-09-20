"""Prepare a paired experiment manifest; this script does not run any models."""

import argparse
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = ("direct", "unchecked_plan", "checked_plan")


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    generation, planner = config.get("generation"), config.get("planner")
    if not isinstance(generation, dict) or not isinstance(planner, dict):
        raise ValueError("Configuration requires generation and planner objects.")
    for name, section in (("generation", generation), ("planner", planner)):
        if not isinstance(section.get("model_id"), str) or not section["model_id"].strip():
            raise ValueError(f"{name}.model_id must be a nonempty string.")
    seeds = generation.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("generation.seeds must be a nonempty list.")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Diffusion seeds must be nonnegative integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Diffusion seeds must be unique.")
    for key in ("num_inference_steps", "width", "height"):
        _positive_integer(generation.get(key), f"generation.{key}")
    if generation["width"] % 8 or generation["height"] % 8:
        raise ValueError("Image dimensions must be divisible by 8.")
    guidance = generation.get("guidance_scale")
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("generation.guidance_scale must be finite and nonnegative.")
    _positive_integer(planner.get("checking_passes"), "planner.checking_passes")


def prepare_runs(prompts_path, config_path, output_path, overwrite=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config(config)
    prompts, ids, texts = [], set(), set()
    for line_number, line in enumerate(Path(prompts_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("prompt_id", "prompt")
        ):
            raise ValueError(f"Invalid prompt row on line {line_number}.")
        normalized = " ".join(row["prompt"].split())
        if row["prompt_id"] in ids or normalized in texts:
            raise ValueError(f"Duplicate prompt ID or text on line {line_number}.")
        ids.add(row["prompt_id"])
        texts.add(normalized)
        prompts.append(row)
    if not prompts:
        raise ValueError("The prompt file is empty.")
    generation, planner = config["generation"], config["planner"]
    settings = {key: generation[key] for key in
                ("num_inference_steps", "guidance_scale", "width", "height")}
    runs = []
    for prompt in prompts:
        for condition in CONDITIONS:
            for seed in generation["seeds"]:
                run = {
                    "prompt_id": prompt["prompt_id"],
                    "original_prompt": prompt["prompt"],
                    "condition": condition,
                    "seed": seed,
                    "generator_model_id": generation["model_id"],
                    "generation_settings": settings,
                    "planner_model_id": None if condition == "direct" else planner["model_id"],
                    "checking_passes": planner["checking_passes"] if condition == "checked_plan" else 0,
                    "status": "planned",
                }
                identity = json.dumps(run, sort_keys=True, ensure_ascii=False)
                run["run_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                runs.append(run)
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("Run ID collision; use longer IDs before proceeding.")
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
        for run in runs:
            stream.write(json.dumps(run, ensure_ascii=False) + "\n")
    return len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        count = prepare_runs(args.prompts, args.config, args.output, args.overwrite)
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    print(f"Saved {count} planned runs. No models were run or images generated.")


if __name__ == "__main__":
    main()
