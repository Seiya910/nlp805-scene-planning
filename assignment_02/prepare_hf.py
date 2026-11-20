"""Download a pinned public Hugging Face corpus and prepare an honest open-data task.

The release contains annotated sentences/phrases, not full articles or paragraph
IDs. We keep that distinction explicit instead of inventing article boundaries.
"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "jmLuis/MediaFrameCorpus-PhilippineFrameCorpus-Combined"
REVISION = "c25ba4a28d8b7c69dfd95341b1d3bf8f8f99cc50"
FRAME_LABELS = [
    "Economic", "Capacity_and_resources", "Morality", "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence", "Policy_prescription_and_evaluation",
    "Crime_and_punishment", "Security_and_defense", "Health_and_safety", "Quality_of_life",
    "Cultural_identity", "Public_opinion", "Political", "External_regulation_and_reputation",
]


def normalized_text(text):
    """Match superficial text variants without looking at any labels."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def prepare(source, destination):
    groups = {}
    raw_rows = other_rows = empty_rows = 0
    files = {}
    for name in ["combined_train.csv", "combined_validation.csv"]:
        path = source / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_rows += 1
                code = int(row["code_frames"])
                if code not in range(1, 16) or int(row["label"]) != code - 1:
                    raise ValueError(f"Unexpected code/zero-based label pair in {name}: {row}")
                key = normalized_text(row["text"])
                if not key:
                    empty_rows += 1
                    continue
                group = groups.setdefault(key, {"text": " ".join(row["text"].split()),
                                                "codes": set(), "source_rows": 0})
                group["source_rows"] += 1
                # Code 15 means Other, not None. It is outside the selected taxonomy.
                # Other-only texts are excluded; Other is never relabeled as no framing.
                if code == 15:
                    other_rows += 1
                else:
                    group["codes"].add(code)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    multi = retained = other_only = 0
    with destination.open("w", encoding="utf-8") as handle:
        for key, group in sorted(groups.items()):
            if not group["codes"]:
                other_only += 1
                continue
            labels = [FRAME_LABELS[x - 1] for x in sorted(group["codes"])]
            # Identical normalized text receives the union of its published labels.
            # This is a declared derived task, not a claim about original paragraph gold.
            record = {"id": "MFC_PFC_" + hashlib.sha256(key.encode()).hexdigest()[:24] + ".txt",
                      "text": group["text"], "labels": labels, "source_rows": group["source_rows"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            retained += 1
            multi += len(labels) > 1
            counts.update(labels)
    info = {
        "repository": REPOSITORY, "revision": REVISION,
        "source_url": "https://huggingface.co/datasets/" + REPOSITORY,
        "data_kind": "public-annotated", "unit": "normalized_news_snippet",
        "label_source": "Dataset card reports agreement by at least two annotators; not independently reannotated.",
        "source_article_ids_available": False,
        "split_policy": "Combine the published row splits, merge exact normalized text, then create fresh disjoint text-unit splits.",
        "label_policy": "Union published codes 1-14 for each normalized text; exclude code-15-only text units.",
        "metric_policy": "Standard multilabel micro/macro F1 over 14 frames; zero_division=0; no extra None class.",
        "raw_rows": raw_rows, "empty_rows": empty_rows, "other_annotation_rows": other_rows,
        "normalized_text_groups": len(groups), "other_only_groups_excluded": other_only,
        "retained_units": retained, "multilabel_units": multi, "label_counts": dict(counts),
        "input_sha256": files, "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "course_status": "Alternative open-data experiment; not designated SemEval-2027 training/evaluation data.",
        "license": "Not specified in the dataset card. Raw and prepared texts are excluded from the project ZIP.",
    }
    destination.with_name("dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    destination.with_name("labels.txt").write_text("\n".join(FRAME_LABELS) + "\n")
    print(json.dumps({k: info[k] for k in ["raw_rows", "retained_units", "multilabel_units", "other_only_groups_excluded"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/hf_mfc_pfc")
    parser.add_argument("--out", default="data/hf_mfc_pfc/prepared.jsonl")
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"repository": REPOSITORY, "revision": REVISION, "files": {}}
    old = json.loads((root / "download.json").read_text()) if (root / "download.json").exists() else {}
    for name in ["combined_train.csv", "combined_validation.csv", "README.md"]:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
        path = root / name
        cached = old.get("files", {}).get(name, {})
        expected = cached.get("sha256") if old.get("revision") == REVISION else None
        valid_cache = path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if not valid_cache:
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
        metadata["files"][name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "bytes": path.stat().st_size}
    (root / "download.json").write_text(json.dumps(metadata, indent=2) + "\n")
    prepare(root, Path(args.out))


if __name__ == "__main__":
    main()
