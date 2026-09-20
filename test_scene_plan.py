"""Small regression checks for scene-plan schema and coordinate sanity checks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scene_plan import render_plan, validate_plan


ROOT = Path(__file__).resolve().parent


class ScenePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "example_plan.json").read_text(encoding="utf-8"))

    def test_example_and_render(self):
        self.assertEqual(validate_plan(self.plan), [])
        before = copy.deepcopy(self.plan)
        rendered = render_plan(self.plan)
        self.assertIn("a red cup is to the left of a blue plate.", rendered)
        self.assertIn("(0.1, 0.35, 0.35, 0.65)", rendered)
        self.assertEqual(rendered, render_plan(self.plan))
        self.assertEqual(self.plan, before)

    def test_reversed_relation_can_be_rendered_unchecked(self):
        self.plan["relations"][0]["relation"] = "right_of"
        self.assertTrue(any("contradicts" in error for error in validate_plan(self.plan)))
        self.assertEqual(validate_plan(self.plan, check_relations=False), [])
        self.assertIn("to the right of", render_plan(self.plan))

    def test_image_y_axis_and_strict_direction(self):
        self.plan["objects"][0]["bbox"] = [0.1, 0.1, 0.3, 0.3]
        self.plan["objects"][1]["bbox"] = [0.1, 0.7, 0.3, 0.9]
        self.plan["relations"][0]["relation"] = "above"
        self.assertEqual(validate_plan(self.plan), [])
        self.plan["relations"][0]["relation"] = "below"
        self.assertTrue(validate_plan(self.plan))
        self.plan["relations"][0]["relation"] = "left_of"
        self.assertTrue(validate_plan(self.plan))  # Equal x centers are not left of each other.

    def test_invalid_boxes(self):
        for box in ([0.5, 0.1, 0.2, 0.8], [0, 0, 1, 0], [-0.1, 0, 1, 1],
                    [0, 0, 2, 1], [False, 0, 1, 1], [float("nan"), 0, 1, 1],
                    [0, 0, float("inf"), 1], [0, 0, 10 ** 500, 1], [0, 1], "box"):
            with self.subTest(box=box):
                self.plan["objects"][0]["bbox"] = box
                self.assertTrue(validate_plan(self.plan))
                with self.assertRaises(ValueError):
                    render_plan(self.plan)

    def test_duplicate_ids_and_unknown_references(self):
        self.plan["objects"][1]["id"] = "cup"
        errors = validate_plan(self.plan)
        self.assertTrue(any("Duplicate" in error for error in errors))
        self.assertTrue(any("known object" in error for error in errors))

    def test_malformed_json_shapes_do_not_raise(self):
        cases = [None, [], True, "text", 1, {}, {"objects": None, "relations": {}},
                 {"objects": [None, []], "relations": [None, []]},
                 {"objects": [{"id": []}, {"id": {}}],
                  "relations": [{"subject": [], "object": {}, "relation": []}]}]
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(validate_plan(value))

    def test_nonempty_fields_self_reference_and_unknown_relation(self):
        self.plan["prompt"] = " "
        self.plan["objects"][0]["description"] = ""
        self.plan["relations"][0]["object"] = "cup"
        self.plan["relations"][0]["relation"] = "inside"
        errors = validate_plan(self.plan)
        for expected in ("prompt", "description", "itself", "left_of"):
            self.assertTrue(any(expected in error for error in errors))

    def test_cli_success_and_failure(self):
        result = subprocess.run([sys.executable, str(ROOT / "scene_plan.py"),
                                 str(ROOT / "example_plan.json"), "--render"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Plan valid.", result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "malformed.json"
            malformed.write_text('{"objects": ', encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scene_plan.py"), str(malformed)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Cannot read plan", result.stderr)


if __name__ == "__main__":
    unittest.main()
