import json
import sys


def check_plan(plan):
    errors = []
    objects = plan.get("objects", [])
    relations = plan.get("relations", [])

    if len(objects) < 2:
        errors.append("The plan needs at least two objects.")

    ids = set()
    centers = {}
    for obj in objects:
        name = obj.get("id")
        box = obj.get("bbox", [])

        if not name or name in ids:
            errors.append("Object IDs must be unique.")
            continue
        ids.add(name)

        if len(box) != 4 or not all(isinstance(x, (int, float)) for x in box):
            errors.append("Invalid box for " + name)
            continue

        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            errors.append("Box must be between 0 and 1 for " + name)
            continue

        centers[name] = ((x0 + x1) / 2, (y0 + y1) / 2)

    for relation in relations:
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
