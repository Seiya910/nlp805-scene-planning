# Scene Planning for Text-to-Image Generation

NLP805 project by Jinjie Wei.

The idea is to check whether a simple scene plan can help a text-to-image model follow spatial relations such as left, right, above, and below.

For the first experiment, I plan to compare the original prompt, a prompt produced from a scene plan, and a prompt produced after checking the scene plan. The first version will use a small subset of T2I-CompBench. Model experiments have not started yet.

## Files

- `proposal_JinjieWei.pptx`: First project presentation
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

The code is an early draft and will change as the project develops.
