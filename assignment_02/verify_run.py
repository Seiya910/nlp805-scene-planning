"""Replay the official-data run and verify scores, article splits, and input hashes."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from framing import (Paragraph, check_official, encode, metrics, predict, read_labels,
                     read_table, save_json, split_articles, write_predictions)


def verify(run, data):
    results = json.loads((run / "results.json").read_text())
    manifest = json.loads((run / "split.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    info = json.loads((data / "dataset_info.json").read_text())
    assert info == results["dataset_info"], "Prepared data differ from this run."
    labels = read_labels(data / "labels.txt")
    records = read_table(data / "annotations.tsv", labels, data / "articles")
    splits = split_articles(records, manifest["seed"])
    groups = {}
    for name, indices in splits.items():
        groups[name] = {records[i].article for i in indices}
        assert sorted(groups[name]) == manifest[name]["group_ids"]
        assert len(indices) == manifest[name]["units"]
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"]), "Article leakage."
    source_paths = [data / "annotations.tsv", data / "labels.txt", data / "dataset_info.json",
                    data / "scorer-subtask-1.py"]
    source_paths += [data / "articles" / name for name in sorted(set.union(*groups.values()))]
    for path in source_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["files"][path.name], path
    test_rows = [records[i] for i in splits["test"]]
    # Gold labels are private and are omitted from the deliverable ZIP. Rebuild
    # this local scoring file from the authorized source and the saved split.
    if not (run / "test_gold.tsv").exists():
        write_predictions(run / "test_gold.tsv", test_rows, encode(test_rows, labels), labels)
    gold = read_table(run / "test_gold.tsv", labels)
    assert [r.key for r in gold] == [r.key for r in test_rows]
    np.testing.assert_array_equal(encode(gold, labels), encode(test_rows, labels))
    scorer = data / "scorer-subtask-1.py"
    checked = {}
    for kind, model in results["models"].items():
        replay = run / f"{kind}_replayed.tsv"
        predict(SimpleNamespace(model=run / f"{kind}.joblib", dataset_jsonl=None,
                                paragraphs=run / "test_boundaries.tsv", articles=data / "articles", out=replay))
        assert replay.read_bytes() == (run / f"{kind}_predictions.tsv").read_bytes()
        guesses = read_table(replay, labels)
        score = metrics(encode(gold, labels), encode(guesses, labels), labels)
        assert score == model["test"], "Saved per-label metrics differ."
        checked[kind] = check_official(scorer, run / "test_gold.tsv", replay, data / "labels.txt", score)
    selected = results["selected_by_validation"]
    predict(SimpleNamespace(model=run / "selected.joblib", dataset_jsonl=None,
                            paragraphs=run / "test_boundaries.tsv", articles=data / "articles",
                            out=run / "selected_predictions.tsv"))
    assert (run / "selected_predictions.tsv").read_bytes() == (run / f"{selected}_predictions.tsv").read_bytes()

    # A real scorer check for the two details that are easy to implement wrongly:
    # empty rows, absent classes, and matching paragraph keys when row order differs.
    fixtures = data / "scorer-checks"
    fixtures.mkdir(exist_ok=True)
    for case in ["empty", "mixed"]:
        truth = np.zeros((1 if case == "empty" else 6, 14), dtype=np.int8)
        guessed = truth.copy()
        if case == "mixed":
            truth[1, 0] = 1
            guessed[2, 0] = 1
            truth[3, [0, 12]] = 1
            guessed[3, [0, 1]] = 1
            truth[4, 1] = guessed[4, 1] = 1
            truth[5, :] = guessed[5, :] = 1
        fixture_rows = [Paragraph(f"fixture_{i}.txt", 0, 1, "x", ()) for i in range(len(truth))]
        gold_path, pred_path = fixtures / f"{case}_gold.tsv", fixtures / f"{case}_pred.tsv"
        write_predictions(gold_path, fixture_rows, truth, labels)
        write_predictions(pred_path, fixture_rows[::-1], guessed[::-1], labels)
        check_official(scorer, gold_path, pred_path, data / "labels.txt", metrics(truth, guessed, labels))
    lookup = {r.key: r.text for r in test_rows}
    for line in (run / "errors.jsonl").read_text().splitlines():
        error = json.loads(line)
        if "text_excerpt" in error:
            key = error["article"], error["start"], error["end"]
            assert error["text_excerpt"] == " ".join(lookup[key].split()[:12])
    verification = {
        "data_kind": "designated-training", "official_scorer_verified": True,
        "source_articles": len(set.union(*groups.values())), "paragraphs": len(records),
        "split_articles": {k: len(v) for k, v in groups.items()},
        "split_paragraphs": {k: len(v) for k, v in splits.items()},
        "article_overlap": 0, "split_ids_reproduced": True, "source_hashes_verified": len(source_paths),
        "all_model_predictions_reproduced": True, "all_per_label_metrics_reproduced": True,
        "selected_model_file_reproduced": True, "selected_by_validation": selected,
        "official_scores": checked, "scorer_fixture_files_checked": 2,
        "scorer_fixture_paragraph_cases": 7, "error_excerpts_match_original_offsets": True,
    }
    save_json(run / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    args = p.parse_args()
    verify(args.run, args.data_dir)
