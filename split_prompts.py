import json
import random
import sys
from pathlib import Path


if len(sys.argv) != 2:
    print("Usage: python3 split_prompts.py prompts.txt")
    raise SystemExit(1)

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
prompts = [line.strip() for line in lines if line.strip()]
prompts = list(dict.fromkeys(prompts))

if len(prompts) < 60:
    print("At least 60 different prompts are needed.")
    raise SystemExit(1)

random.seed(42)
random.shuffle(prompts)

Path("data").mkdir(exist_ok=True)
Path("data/dev.json").write_text(json.dumps(prompts[:10], indent=2), encoding="utf-8")
Path("data/eval.json").write_text(json.dumps(prompts[10:60], indent=2), encoding="utf-8")

print("Saved 10 development prompts and 50 evaluation prompts.")
