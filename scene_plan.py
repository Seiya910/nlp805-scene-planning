"""Validate and render a small scene plan using only the Python standard library.

Coordinates are normalized to [0, 1], with (0, 0) at the image's top left.
Directional checks compare box centers. This is a rule-based sanity check,
not the proposed LLM checking pass or the official image benchmark evaluator.
The checker reports issues without changing the plan.
"""

import argparse
import json
import math
import sys


RELATION_WORDS = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
}


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _valid_coordinate(value):
    # bool is an int subclass, but is not a meaningful coordinate.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
        and 0 <= value <= 1
    )


def validate_plan(plan, check_relations=True):
    """Return readable errors for malformed schema or contradictory box centers.

Set check_relations=False to check only the schema. Directional consistency
then remains unchecked, which is useful when serializing the unchecked-plan
condition. References and allowed relation names are still validated.
"""
    if not isinstance(plan, dict):
        return ["The plan must be a JSON object."]

    errors = []
    if not _nonempty_string(plan.get("prompt")):
        errors.append("prompt must be a nonempty string.")

    objects = plan.get("objects")
    if not isinstance(objects, list):
        errors.append("objects must be a list.")
        objects = []
    if len(objects) < 2:
        errors.append("objects must contain at least two objects.")

    object_ids = set()
    centers = {}
    for index, obj in enumerate(objects):
        label = "objects[{}]".format(index)
        if not isinstance(obj, dict):
            errors.append("{} must be an object.".format(label))
            continue
        object_id = obj.get("id")
        id_valid = _nonempty_string(object_id)
        duplicate = id_valid and object_id in object_ids
        if not id_valid:
            errors.append("{}.id must be a nonempty string.".format(label))
        elif duplicate:
            errors.append("Duplicate object id: {!r}.".format(object_id))
        else:
            object_ids.add(object_id)
        if not _nonempty_string(obj.get("description")):
            errors.append("{}.description must be a nonempty string.".format(label))

        box = obj.get("bbox")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(_valid_coordinate(value) for value in box)
        ):
            errors.append("{}.bbox must have four finite numbers in [0, 1].".format(label))
            continue
        x0, y0, x1, y1 = box
        if not (x0 < x1 and y0 < y1):
            errors.append("{}.bbox must satisfy x0 < x1 and y0 < y1.".format(label))
        elif id_valid and not duplicate:
            centers[object_id] = ((x0 + x1) / 2, (y0 + y1) / 2)

    relations = plan.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be a list.")
        relations = []
    for index, relation in enumerate(relations):
        label = "relations[{}]".format(index)
        if not isinstance(relation, dict):
            errors.append("{} must be an object.".format(label))
            continue
        subject = relation.get("subject")
        target = relation.get("object")
        name = relation.get("relation")
        references_valid = True
        for key, value in (("subject", subject), ("object", target)):
            if not isinstance(value, str) or value not in object_ids:
                errors.append("{}.{} must reference a known object id.".format(label, key))
                references_valid = False
        if references_valid and subject == target:
            errors.append("{} cannot relate an object to itself.".format(label))
            references_valid = False
        if not isinstance(name, str) or name not in RELATION_WORDS:
            errors.append("{}.relation must be left_of, right_of, above, or below.".format(label))
            continue
        if not (check_relations and references_valid and subject in centers and target in centers):
            continue
        sx, sy = centers[subject]
        tx, ty = centers[target]
        consistent = {
            "left_of": sx < tx,
            "right_of": sx > tx,
            "above": sy < ty,
            "below": sy > ty,
        }[name]
        if not consistent:
            errors.append("{} contradicts the box centers: {} {} {}.".format(label, subject, name, target))
    return errors


def render_plan(plan):
    """Serialize a schema-valid plan deterministically, without fixing its boxes.

An unchecked plan may contain conflicting relations and coordinates. Rendering
preserves them, rather than silently introducing a checking or revision step.
Textual coordinates describe a plan; they do not enforce image-generator boxes.
"""
    errors = validate_plan(plan, check_relations=False)
    if errors:
        raise ValueError("Invalid scene plan: " + " ".join(errors))
    descriptions = {obj["id"]: obj["description"].strip() for obj in plan["objects"]}
    parts = [plan["prompt"].strip(), "Scene layout (normalized coordinates, origin at top left):"]
    for obj in plan["objects"]:
        box = ", ".join(format(value, ".6g") for value in obj["bbox"])
        parts.append("{}: bounding box (x0, y0, x1, y1) = ({}).".format(descriptions[obj["id"]], box))
    for relation in plan["relations"]:
        parts.append("{} is {} {}.".format(
            descriptions[relation["subject"]],
            RELATION_WORDS[relation["relation"]],
            descriptions[relation["object"]],
        ))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="Path to a scene-plan JSON file")
    parser.add_argument("--render", action="store_true", help="Also print the text representation")
    args = parser.parse_args()
    try:
        with open(args.plan, encoding="utf-8") as source:
            plan = json.load(source)
    except (OSError, UnicodeError, ValueError) as error:
        print("Cannot read plan: {}".format(error), file=sys.stderr)
        return 1
    errors = validate_plan(plan)
    if errors:
        for error in errors:
            print("- " + error, file=sys.stderr)
        return 1
    print("Plan valid.")
    if args.render:
        print(render_plan(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
