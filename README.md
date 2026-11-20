# Scene Planning for Text-to-Image Generation

NLP805 project by Jinjie Wei.

The idea is to check whether a simple scene plan can help a text-to-image model follow spatial relations such as left, right, above, and below.

The initial proposal planned to compare direct prompting, unchecked planning, and checked planning on a small subset of T2I-CompBench.

## Files

- `presentations/proposal_JinjieWei.pptx`: First project presentation
- `example_plan.json`: a small scene-plan example
- `scene_plan.py`: basic checks for objects, boxes, and relations
- `split_prompts.py`: split prompts into development and evaluation sets

## Links

- [Overleaf proposal (editable)](https://www.overleaf.com/5674814873rnyjzpnvchnf#3ca2de)
- [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)

## Usage

```bash
python3 scene_plan.py example_plan.json
python3 split_prompts.py prompts.txt
```

## ScenePlan extensions to MCCD

ScenePlan builds on MCCD's multi-agent scene parser and hierarchical compositional diffusion. [MCCD (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Li_MCCD_Multi-Agent_Collaboration-based_Compositional_Diffusion_for_Complex_Text-to-Image_Generation_CVPR_2025_paper.html)

- **Joint layout repair.** Parsed boxes can still violate directional constraints. ScenePlan adjusts their centers together to satisfy feasible constraints with minimal movement.
- **Spatially scheduled fusion.** MCCD's final blend uses a fixed weight. ScenePlan strengthens regional guidance near related objects early in denoising to improve placement.

Layout checks compare the original, sequentially repaired, and jointly repaired boxes for relation satisfaction and movement. A matched SDXL-base ablation on T2I-CompBench Spatial tests each change alone and both together. The reported score rises from 0.2369 for MCCD to 0.2730 with both changes. [Evaluation setup](third_party/t2i_compbench/README.md#sceneplan-ablation)
