# Scene planning for text-to-image generation

NLP805 course project by Jinjie Wei.

This project studies whether a simple scene plan can help a text-to-image model follow spatial relations. We compare direct prompting, planning without checking, and planning with one checking step.

- [Proposal](https://www.overleaf.com/5674814873rnyjzpnvchnf#3ca2de)
- [Slides](proposal_JinjieWei.pptx)

## Status

So far, the repository only contains utilities for splitting prompts, checking the scene-plan format, and preparing runs with fixed seeds. Image generation and evaluation are not implemented yet, so there are no results here.

## Run

```bash
python3 scene_plan.py example_plan.json --render
python3 -m unittest discover -v
```

Experiment settings are in `config.json`. The pilot uses spatial prompts from [T2I-CompBench](https://papers.neurips.cc/paper_files/paper/2023/hash/f8ad010cdd9143dbb0e9308c093aff24-Abstract-Datasets_and_Benchmarks.html).

```bash
python3 split_prompts.py --input data/spatial_prompts.txt --output-dir data/splits --dev-size 10 --eval-size 50 --seed 42
python3 prepare_runs.py --prompts data/splits/eval.jsonl --config config.json --output runs/pilot.jsonl
```

These commands prepare metadata only. They do not generate images.
