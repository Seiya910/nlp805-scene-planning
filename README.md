# Scene Planning for Text-to-Image Generation

NLP805 course project by Jinjie Wei.

The project asks whether a scene plan and a consistency-checking step help a text-to-image model follow spatial relations. The pilot compares **direct prompting**, **unchecked planning**, and **checked planning**, using the same image generator and diffusion seeds.

- [Overleaf proposal](https://www.overleaf.com/project/6aa3a4cd48d9c33c0f86d534)
- [Presentation slides](Scene_generation_Jinjie_Wei_real_failure.pptx)

## Current status

The proposal and initial project utilities are available. The original presentation is preserved; its broader scene-generation ideas motivate the smaller spatial-relation pilot described here.

Implemented utilities cover prompt splitting, JSON scene-plan validation, deterministic prompt rendering, and experiment-manifest preparation. **Model inference, the LLM checking pass, image generation, and benchmark evaluation are not implemented yet. No experimental results are reported.**

## Files

| File | Purpose |
| --- | --- |
| `split_prompts.py` | Create reproducible, disjoint development and evaluation sets. |
| `scene_plan.py` | Check object IDs, normalized boxes, and simple spatial relations; render a plan as text. |
| `prepare_runs.py` | Prepare the three-condition experiment manifest with shared seeds. |
| `config.json` | Proposed model IDs, split sizes, and generation settings. |
| `example_plan.json` | Handwritten format example, not a model output or benchmark sample. |
| `test_data_pipeline.py`, `test_scene_plan.py` | Tests for the utilities. |

## Quick start

The starter utilities use **Python 3.9 or newer and the standard library only**. They do not download model weights or require a GPU.

Validate and render the example:

```bash
python3 scene_plan.py example_plan.json --render
```

Run the tests:

```bash
python3 -m unittest discover -v
```

For the actual pilot, obtain spatial prompts from the [official T2I-CompBench dataset directory](https://github.com/Karine-Huang/T2I-CompBench/tree/main/examples/dataset). Prepare `data/spatial_prompts.txt` with one eligible prompt per line, restricted to two objects and the proposed 2D relations: left of, right of, above, and below. Record the source revision and the eligibility decisions before sampling. The splitting utility checks text uniqueness; it does not determine semantic eligibility.

```bash
python3 split_prompts.py --input data/spatial_prompts.txt --output-dir data/splits --dev-size 10 --eval-size 50 --seed 42
python3 prepare_runs.py --prompts data/splits/eval.jsonl --config config.json --output runs/pilot.jsonl
```

At least 60 unique eligible prompts are needed for these split sizes. The output contains 10 development prompts and 50 held-out evaluation prompts. With four seeds and three conditions, `runs/pilot.jsonl` contains **600 planned runs**, not generated images. Existing output files are protected unless `--overwrite` is passed. Data and generated run files are ignored by Git.

The split command's arguments control sampling; the `split` and top-level `seed` entries in `config.json` document the corresponding planned settings. The manifest command reads the model and generation settings from that configuration.

## Scene-plan format

Each object has a unique ID, a description, and a box `[x_min, y_min, x_max, y_max]` normalized to `[0, 1]`. The image origin is at the top left, with y increasing downward. Relations reference object IDs and use `left_of`, `right_of`, `above`, or `below`.

The local checker compares box centers and reports contradictions. It is a simple development check, **not the proposed LLM checking pass or the official image evaluator**. It does not repair a plan. Rendered text describes the plan but does not enforce bounding boxes in SDXL.

## Planned experiment

- Use [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) to propose a plan and perform one checking pass. Reuse the same initial plan for the unchecked and checked conditions.
- Use the frozen [SDXL 1.0 base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for all conditions. The current configuration values are proposed defaults, pending a smoke test.
- Keep the original prompt as the evaluation target. Refine instructions on development prompts, then freeze them before held-out evaluation.
- Report spatial-relation scores, object-detection coverage, paired comparisons, and selected failures using the [official benchmark evaluator](https://github.com/Karine-Huang/T2I-CompBench). The four-seed subset is a course pilot, not a reproduction of full-benchmark results.
- Add Transformers, PyTorch, Diffusers, NumPy, and pandas when implementing inference and evaluation. Record tested package versions, model revisions, and the evaluator commit at that stage.

## Reading

1. Huang et al. (NeurIPS 2023). [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://papers.neurips.cc/paper_files/paper/2023/hash/f8ad010cdd9143dbb0e9308c093aff24-Abstract-Datasets_and_Benchmarks.html).
2. Feng et al. (NeurIPS 2023). [LayoutGPT: Compositional Visual Planning and Generation with Large Language Models](https://papers.nips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html).
