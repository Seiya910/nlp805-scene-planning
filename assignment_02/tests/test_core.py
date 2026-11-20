"""Check the failure-prone parts of the experiment, rather than model quality."""

import tempfile
import csv
import json
import unittest
from pathlib import Path

import numpy as np

from framing import (Paragraph, encode, fit_heads, make_features, metrics,
                     probabilities, read_jsonl, read_table, split_articles, write_predictions)
from prepare_hf import prepare
from prepare_official import canonical_labels


LABELS = ["Frame_" + str(i) for i in range(14)]


class CoreTests(unittest.TestCase):
    def test_official_no_framing_marker_becomes_empty(self):
        self.assertEqual(canonical_labels(["No framing"]), ([], True))
        self.assertEqual(canonical_labels([""]), ([], False))
        self.assertEqual(canonical_labels(["Economic", ""]), (["Economic"], False))

    def test_no_framing_cannot_coexist_with_a_frame(self):
        with self.assertRaisesRegex(ValueError, "must not coexist"):
            canonical_labels(["No framing", "Economic"])

    def test_public_preparation_merges_text_and_excludes_other_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            for filename, rows in [
                ("combined_train.csv", [(1, "Example text.", 0), (15, "Outside taxonomy.", 14)]),
                ("combined_validation.csv", [(2, "  EXAMPLE   text. ", 1)]),
            ]:
                with (source / filename).open("w", newline="") as h:
                    w = csv.writer(h)
                    w.writerow(["code_frames", "text", "label"])
                    w.writerows(rows)
            destination = root / "prepared.jsonl"
            prepare(source, destination)
            rows = [json.loads(x) for x in destination.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["labels"], ["Economic", "Capacity_and_resources"])
            self.assertNotIn("None", rows[0]["labels"])

    def test_standard_metric_does_not_add_none(self):
        y = np.zeros((1, 14), dtype=np.int8)
        standard = metrics(y, y, LABELS, include_none=False)
        self.assertNotIn("None", standard["per_label"])
        self.assertEqual(standard["micro_f1"], 0.0)
        self.assertEqual(metrics(y, y, LABELS)["micro_f1"], 1.0)

    def test_jsonl_prediction_rejects_labelled_records(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "inputs.jsonl"
            p.write_text(json.dumps({"id": "a.txt", "text": "A news snippet", "labels": ["Frame_0"]}) + "\n")
            with self.assertRaisesRegex(ValueError, "omit the gold labels"):
                read_jsonl(p, LABELS, gold=False)

    def test_cyrillic_offsets_preserve_crlf(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "article.txt").write_bytes("Заглавие\r\n\r\nНов абзац".encode("utf-8"))
            start = len("Заглавие\r\n\r\n")
            (root / "gold.tsv").write_text(f"article.txt\t{start}\t{start + 9}\tFrame_0\n")
            rows = read_table(root / "gold.tsv", LABELS, root)
            self.assertEqual(rows[0].text, "Нов абзац")

    def test_prediction_input_rejects_gold_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "boundaries.tsv"
            p.write_text("article.txt\t0\t3\tFrame_0\n")
            with self.assertRaisesRegex(ValueError, "without gold labels"):
                read_table(p, LABELS, gold=False)

    def test_duplicate_and_unknown_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "gold.tsv"
            for text in ["article.txt\t0\t3\tUNKNOWN\n",
                         "article.txt\t0\t3\tFrame_0\narticle.txt\t0\t3\tFrame_0\n"]:
                p.write_text(text)
                with self.assertRaises(ValueError):
                    read_table(p, LABELS)

    def test_article_splits_have_no_overlap_and_replay(self):
        rows = [Paragraph(f"article_{i}.txt", j, j + 1, "текст", ())
                for i in range(10) for j in range(3)]
        first, second = split_articles(rows, 42), split_articles(rows, 42)
        groups = [{rows[i].article for i in ids} for ids in first.values()]
        self.assertEqual(set.union(*groups), {r.article for r in rows})
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_empty_sets_receive_none_credit(self):
        empty = np.zeros((2, 14), dtype=int)
        result = metrics(empty, empty, LABELS)
        self.assertEqual(result["micro_f1"], 1.0)
        self.assertEqual(result["per_label"]["None"]["support"], 2)
        # The official scorer assigns F1=1 to classes absent from both inputs.
        self.assertEqual(result["macro_f1"], 1.0)
        guessed = empty.copy()
        guessed[0, 0] = 1
        self.assertEqual(metrics(empty, guessed, LABELS)["micro_f1"], 0.5)

    def test_empty_prediction_tsv_roundtrip(self):
        rows = [Paragraph("article.txt", 0, 3, "abc", ())]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "pred.tsv"
            write_predictions(p, rows, np.zeros((1, 14)), LABELS)
            self.assertEqual(p.read_bytes(), b"article.txt\t0\t3\n")
            np.testing.assert_array_equal(encode(read_table(p, LABELS), LABELS), np.zeros((1, 14)))

    def test_constant_heads_and_training_only_vocabulary(self):
        features = make_features("word")
        x = features.fit_transform(["здраве болница", "болница лекар", "икономика пари"])
        features.transform(["невижданадума"])
        self.assertNotIn("невижданадума", features.vocabulary_)
        y = np.zeros((3, 14), dtype=np.int8)
        y[:, 0] = 1
        y[0, 1] = 1
        probs = probabilities(fit_heads(x, y, 1.0, 42), x)
        np.testing.assert_array_equal(probs[:, 0], np.ones(3))
        np.testing.assert_array_equal(probs[:, 2:], np.zeros((3, 12)))
        self.assertTrue(((probs[:, 1] > 0) & (probs[:, 1] < 1)).all())


if __name__ == "__main__":
    unittest.main()
