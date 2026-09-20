"""Tests use synthetic prompts, not downloaded benchmark data."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from prepare_runs import CONDITIONS, prepare_runs
from split_prompts import create_splits


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "prompts.txt"
        self.source.write_text("\n".join(f"Object {i} is left of a plate." for i in range(65))
                               + "\n  Object 0  is left of a plate.\n\n", encoding="utf-8")
        self.config = {
            "seed": 42,
            "split": {"dev_size": 10, "eval_size": 50},
            "generation": {"model_id": "test-image-model", "seeds": [0, 1, 2, 3],
                           "num_inference_steps": 30, "guidance_scale": 7.5,
                           "width": 1024, "height": 1024},
            "planner": {"model_id": "test-planner", "checking_passes": 1},
        }
        self.config_path = self.root / "config.json"

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_splits_are_disjoint_reproducible_and_do_not_change_global_rng(self):
        state = random.getstate()
        first = create_splits(self.source, self.root / "a")
        second = create_splits(self.source, self.root / "b")
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, second)
        self.assertEqual(first["unique_source_prompts"], 65)
        self.assertEqual(len(first["dev_ids"]), 10)
        self.assertEqual(len(first["eval_ids"]), 50)
        self.assertFalse(set(first["dev_ids"]) & set(first["eval_ids"]))
        for filename in ("dev.jsonl", "eval.jsonl", "metadata.json"):
            self.assertEqual((self.root / "a" / filename).read_bytes(),
                             (self.root / "b" / filename).read_bytes())
        with self.assertRaises(FileExistsError):
            create_splits(self.source, self.root / "a")

    def test_insufficient_unique_prompts_rejected_before_writing(self):
        self.source.write_text("a cup left of a plate\na cup left of a plate\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique prompts"):
            create_splits(self.source, self.root / "splits")
        self.assertFalse((self.root / "splits").exists())

    def test_600_paired_runs_have_unique_ids_and_shared_seeds(self):
        create_splits(self.source, self.root / "splits")
        evaluation = self.root / "splits" / "eval.jsonl"
        self.write_config()
        output = self.root / "runs.jsonl"
        self.assertEqual(prepare_runs(evaluation, self.config_path, output), 600)
        runs = read_jsonl(output)
        self.assertEqual(len({run["run_id"] for run in runs}), 600)
        for prompt in read_jsonl(evaluation):
            for condition in CONDITIONS:
                group = [run for run in runs if run["prompt_id"] == prompt["prompt_id"]
                         and run["condition"] == condition]
                self.assertEqual({run["seed"] for run in group}, {0, 1, 2, 3})
                self.assertTrue(all(run["original_prompt"] == prompt["prompt"] for run in group))
                self.assertTrue(all(run["status"] == "planned" for run in group))
        self.assertTrue(all(run["planner_model_id"] is None for run in runs if run["condition"] == "direct"))
        self.assertTrue(all("image_path" not in run and "model_prompt" not in run for run in runs))
        with self.assertRaises(FileExistsError):
            prepare_runs(evaluation, self.config_path, output)
        before = output.read_bytes()
        prepare_runs(evaluation, self.config_path, output, overwrite=True)
        self.assertEqual(before, output.read_bytes())

    def test_duplicate_ids_seeds_and_invalid_rows_are_rejected(self):
        prompts = self.root / "eval.jsonl"
        prompts.write_text('{"prompt_id":"a","prompt":"cup left of plate"}\n', encoding="utf-8")
        self.config["generation"]["seeds"] = [0, 0]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_runs(prompts, self.config_path, self.root / "runs.jsonl")
        self.config["generation"]["seeds"] = [0, 1]
        self.write_config()
        prompts.write_text('{"prompt_id":"a","prompt":"cup left of plate"}\n'
                           '{"prompt_id":"a","prompt":"cup above plate"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_runs(prompts, self.config_path, self.root / "runs.jsonl")
        prompts.write_text('{"prompt_id":"a","prompt":false}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid prompt"):
            prepare_runs(prompts, self.config_path, self.root / "runs.jsonl")
        self.assertFalse((self.root / "runs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
