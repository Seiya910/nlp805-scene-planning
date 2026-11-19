import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

MARGIN = 0.10
TOL = 1e-7
DIRECTIONS = {"left_of": (0, False), "right_of": (0, True),
              "above": (1, False), "below": (1, True)}


# Layout input：统一成左上角原点的归一化中心、半宽高和方向约束。
def read_layout(plan, *, skip_unknown_objects=False):
    objects = plan["objects"]
    indices = {obj["id"]: i for i, obj in enumerate(objects)}
    if len(indices) != len(objects):
        raise ValueError("duplicate object ID")

    boxes = np.asarray([obj.get("bbox") for obj in objects], dtype=float)
    if (not objects or boxes.shape != (len(objects), 4)
            or not np.isfinite(boxes).all()
            or np.any(boxes < 0) or np.any(boxes > 1)
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("expected valid normalized xyxy boxes")

    pairs = []
    for relation in plan.get("relations", []):
        a = indices.get(relation["subject"])
        b = indices.get(relation["object"])
        if a is None or b is None:
            if skip_unknown_objects:
                continue
            raise ValueError("relation references an object without a box")
        kind = relation["relation"]
        if kind in DIRECTIONS:
            axis, reverse = DIRECTIONS[kind]
            first, second = (b, a) if reverse else (a, b)
            pairs.append((2 * first + axis, 2 * second + axis))

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_sizes = (boxes[:, 2:] - boxes[:, :2]) / 2
    return centers.ravel(), half_sizes.ravel(), pairs


def satisfies(centers, pairs):
    return all(centers[b] - centers[a] >= MARGIN - TOL for a, b in pairs)


# Repair：sequential 按顺序处理每一对，joint 一次约束全部中心。
def sequential_centers(start, lo, hi, pairs):
    centers = start.copy()
    for a, b in pairs:
        if a == b or centers[b] - centers[a] >= MARGIN - TOL:
            continue
        if lo[a] + MARGIN > hi[b]:
            continue
        first = (centers[a] + centers[b] - MARGIN) / 2
        centers[a] = np.clip(first, lo[a], hi[b] - MARGIN)
        centers[b] = centers[a] + MARGIN
    return centers


def joint_centers(start, lo, hi, pairs):
    matrix = np.zeros((len(pairs), start.size))
    for row, (a, b) in enumerate(pairs):
        matrix[row, a] -= 1
        matrix[row, b] += 1
    margin = np.full(len(pairs), MARGIN)
    feasible = linprog(
        np.zeros_like(start), A_ub=-matrix, b_ub=-margin,
        bounds=list(zip(lo, hi)), method="highs",
    )
    if feasible.status == 2:
        return start, "infeasible"
    if not feasible.success:
        return start, "solver_error"

    result = minimize(
        lambda centers: np.sum((centers - start) ** 2),
        feasible.x,
        jac=lambda centers: 2 * (centers - start),
        bounds=Bounds(lo, hi),
        constraints=[LinearConstraint(matrix, margin, np.inf)],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 300},
    )
    if not result.success or not satisfies(result.x, pairs):
        return start, "solver_error"
    return result.x, "repaired"


def repair_layout(plan, method="joint"):
    if method not in ("joint", "sequential"):
        raise ValueError(f"unknown repair method: {method}")
    try:
        start, half_sizes, pairs = read_layout(plan)
    except (KeyError, TypeError, ValueError):
        return plan, "invalid_layout"
    if not pairs:
        return plan, "no_directional_relations"
    if satisfies(start, pairs):
        return plan, "already_satisfied"

    lo, hi = half_sizes, 1 - half_sizes
    if method == "sequential":
        centers, status = sequential_centers(start, lo, hi, pairs), "one_pass"
    else:
        centers, status = joint_centers(start, lo, hi, pairs)
        if status != "repaired":
            return plan, status

    repaired = deepcopy(plan)
    centers = centers.reshape(-1, 2)
    half_sizes = half_sizes.reshape(-1, 2)
    boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)
    for obj, bbox in zip(repaired["objects"], boxes):
        obj["bbox"] = np.clip(bbox, 0, 1).tolist()
    return repaired, status


# Layout evaluation：按场景统计约束满足率，按 box 汇总欧氏位移。
def evaluate_layouts(plans):
    methods = ("original", "sequential", "joint")
    totals = {
        method: {"satisfied": 0, "distance": 0.0, "boxes": 0}
        for method in methods
    }
    counts = dict(scenes=0, directional_scenes=0, invalid_layouts=0,
                  no_directional_relations=0, joint_infeasible=0, joint_solver_errors=0)
    for plan in plans:
        counts["scenes"] += 1
        try:
            start, _, pairs = read_layout(plan)
        except (KeyError, TypeError, ValueError):
            counts["invalid_layouts"] += 1
            continue
        if not pairs:
            counts["no_directional_relations"] += 1
            continue
        counts["directional_scenes"] += 1

        sequential, _ = repair_layout(plan, "sequential")
        joint, status = repair_layout(plan)
        counts["joint_infeasible"] += status == "infeasible"
        counts["joint_solver_errors"] += status == "solver_error"
        for method, result in zip(methods, (plan, sequential, joint)):
            centers, _, _ = read_layout(result)
            shifts = np.linalg.norm((centers - start).reshape(-1, 2), axis=1)
            totals[method]["satisfied"] += satisfies(centers, pairs)
            totals[method]["distance"] += float(shifts.sum())
            totals[method]["boxes"] += shifts.size

    eligible = counts["directional_scenes"]
    return {
        **counts,
        "methods": {
            method: {
                "satisfied_scenes": values["satisfied"],
                "scene_satisfaction_rate": values["satisfied"] / eligible if eligible else None,
                "mean_box_center_displacement": (
                    values["distance"] / values["boxes"] if values["boxes"] else None
                ),
            }
            for method, values in totals.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", type=Path, help="one plan JSON or JSONL with a plan field")
    args = parser.parse_args()
    content = args.plans.read_text(encoding="utf-8")
    if args.plans.suffix == ".jsonl":
        plans = [json.loads(line)["plan"] for line in content.splitlines() if line.strip()]
    else:
        plans = [json.loads(content)]
    if not plans:
        parser.error("no plans in input")
    print(json.dumps(evaluate_layouts(plans), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
