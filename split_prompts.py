"""Create fixed, disjoint development and evaluation prompt subsets."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def create_splits(input_path, output_dir, dev_size=10, eval_size=50, seed=42,
                  overwrite=False):
    if any(type(value) is not int for value in (dev_size, eval_size, seed)):
        raise ValueError("Sizes and seed must be integers.")
    if dev_size < 0 or eval_size < 1:
        raise ValueError("dev_size must be nonnegative and eval_size positive.")
    source = Path(input_path).read_bytes()
    # Collapse whitespace so repeated prompts do not enter both subsets.
    prompts = list(dict.fromkeys(
        " ".join(line.split()) for line in source.decode("utf-8-sig").splitlines()
        if line.strip()
    ))
    if len(prompts) < dev_size + eval_size:
        raise ValueError(
            f"Need {dev_size + eval_size} unique prompts; found {len(prompts)}."
        )
    sampled = random.Random(seed).sample(prompts, dev_size + eval_size)
    rows = [{"prompt_id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
             "prompt": text} for text in sampled]
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Prompt ID collision; use longer IDs before proceeding.")
    subsets = {"dev": rows[:dev_size], "eval": rows[dev_size:]}
    metadata = {
        "seed": seed,
        "source_file": Path(input_path).name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "unique_source_prompts": len(prompts),
        "normalization": "Collapse whitespace; preserve case and punctuation.",
        "dev_size": dev_size,
        "eval_size": eval_size,
        "dev_ids": [row["prompt_id"] for row in subsets["dev"]],
        "eval_ids": [row["prompt_id"] for row in subsets["eval"]],
    }
    destination = Path(output_dir)
    files = {
        f"{name}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in entries)
        for name, entries in subsets.items()
    }
    files["metadata.json"] = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    if not overwrite:
        existing = [str(destination / name) for name in files
                    if (destination / name).exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        with (destination / name).open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(contents)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 text, one prompt per line")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dev-size", type=int, default=10)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = create_splits(args.input, args.output_dir, args.dev_size,
                                 args.eval_size, args.seed, args.overwrite)
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    print(f"Saved {metadata['dev_size']} development and {metadata['eval_size']} evaluation prompts.")


if __name__ == "__main__":
    main()
