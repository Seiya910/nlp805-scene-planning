import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split prompts into development and evaluation sets.")
    parser.add_argument("input", type=Path, help="text file with one prompt per line")
    parser.add_argument("--dev-count", type=int, default=10)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dev_count < 0 or args.eval_count < 1:
        parser.error("--dev-count must be nonnegative and --eval-count must be positive")

    lines = args.input.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    prompts = list(dict.fromkeys(prompts))
    total = args.dev_count + args.eval_count

    if len(prompts) < total:
        parser.error(f"At least {total} different prompts are needed")

    random.Random(args.seed).shuffle(prompts)

    Path("data").mkdir(exist_ok=True)
    Path("data/dev.json").write_text(
        json.dumps(prompts[:args.dev_count], indent=2), encoding="utf-8"
    )
    Path("data/eval.json").write_text(
        json.dumps(prompts[args.dev_count:total], indent=2), encoding="utf-8"
    )

    print(f"Saved {args.dev_count} development prompts and {args.eval_count} evaluation prompts.")


if __name__ == "__main__":
    main()
