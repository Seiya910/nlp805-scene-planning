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

## ScenePlan ablation

`layout_repair.py` takes a single plan JSON or one cached MCP plan per JSONL row
with a `plan` field. It reports joint infeasibility and invalid layouts separately.
Satisfaction and displacement are calculated on valid plans with directional relations.

In the HCD integration, call `prepare_conditions(cached_plan)` from
`sceneplan_fusion.py` once per prompt. It returns the conditions and joint repair
status. Each condition contains the layout to pass to HCD and a `blend` function.
Call `blend(z_full, z_smooth, k)` at HCD's final blend before advancing to the next
timestep, with `k` counted from 1 to 20. Use the same SDXL-base checkpoint,
scheduler, guidance scale of 7.0, and ten seed IDs in all four conditions.
The project requires the original HCD implementation to generate the latents.

After running the upstream spatial evaluator, use the same JSONL schema above with
conditions `mccd`, `joint_repair`, `scheduled_fusion`, and `sceneplan`.
Keep scores for missing detections as returned by the evaluator.

```bash
python3 third_party/t2i_compbench/summarize_results.py results.jsonl --experiment sceneplan
```
