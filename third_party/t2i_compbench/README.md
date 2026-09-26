# T2I-CompBench

This folder records the T2I-CompBench integration used to evaluate the scene-planning
experiments in this project.

Upstream repository: [Karine-Huang/T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)

Checked upstream revision:

```text
4aa404212eb5d06e5adbcd9cee696c750d0d25a5
```

Use the upstream `UniDet_eval/2D_spatial_eval.py` evaluator. This repository keeps only
the small result-summary script below, not the benchmark's model files.

The script accepts one normalized JSON object per line:

```json
{"prompt_id":"p01","condition":"direct","seed":0,"spatial_score":0.75,"both_objects_detected":true}
```

The expected conditions are `direct`, `unchecked_plan`, and `checked_plan`.

Run:

```bash
python3 third_party/t2i_compbench/summarize_results.py results.jsonl
```
