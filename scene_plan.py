import json
import math
import sys


def check_plan(plan):
    if not isinstance(plan, dict):
        return ["Plan must be a JSON object."]

    errors = []
    objects = plan.get("objects", [])
    relations = plan.get("relations", [])

    if not isinstance(objects, list):
        errors.append("Objects must be a list.")
        objects = []
    if not isinstance(relations, list):
        errors.append("Relations must be a list.")
        relations = []

    if len(objects) < 2:
        errors.append("The plan needs at least two objects.")

    ids = set()
    centers = {}
    for index, obj in enumerate(objects, 1):
        if not isinstance(obj, dict):
            errors.append(f"Object {index} must be a JSON object.")
            continue
        name = obj.get("id")
        box = obj.get("bbox", [])

        if not isinstance(name, str) or not name.strip():
            errors.append(f"Object {index} needs an ID.")
            continue
        if name in ids:
            errors.append("Object IDs must be unique: " + name)
            continue
        ids.add(name)

        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not all(
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and 0 <= x <= 1
                and math.isfinite(x)
                for x in box
            )
        ):
            errors.append("Invalid box for " + name)
            continue

        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            errors.append("Box must be between 0 and 1 for " + name)
            continue

        centers[name] = ((x0 + x1) / 2, (y0 + y1) / 2)

    for index, relation in enumerate(relations, 1):
        if not isinstance(relation, dict):
            errors.append(f"Relation {index} must be a JSON object.")
            continue
        first = relation.get("subject")
        second = relation.get("object")
        kind = relation.get("relation")

        if first not in centers or second not in centers:
            errors.append("Relation uses an unknown object.")
            continue

        x1, y1 = centers[first]
        x2, y2 = centers[second]
        correct = {
            "left_of": x1 < x2,
            "right_of": x1 > x2,
            "above": y1 < y2,
            "below": y1 > y2,
        }.get(kind)

        if correct is None:
            errors.append("Unknown relation: " + str(kind))
        elif not correct:
            errors.append("The boxes do not match the relation: " + kind)

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scene_plan.py plan.json")
        raise SystemExit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        plan = json.load(f)

    problems = check_plan(plan)
    if problems:
        for problem in problems:
            print("-", problem)
        raise SystemExit(1)

    print("Plan looks valid.")
