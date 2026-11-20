"""Prepare the English training release downloaded from the official Files tab.

Keep private data outside the submission archive. Article bytes and supplied
offsets are preserved; only the six explicit 'No framing' markers are normalized.
"""

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from framing import read_table, save_json


DATA_SHA256 = "717c4bdbea4aba14e25bfc2527e51286b031420e7e25b634dc4bbd604e9cf8dd"
KIT_SHA256 = "0116006d5e92f6900e1d4fdc348968952cd6edf77208d8daf6ac709966837590"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(archive, destination):
    """Reject archive paths that could escape the requested data directory."""
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {archive}")
        for member in bundle.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            bundle.extract(member, destination)


def canonical_labels(values):
    """The release uses both an empty field and 'No framing' for no frame."""
    values = [x for x in values if x]
    if "No framing" in values:
        if values != ["No framing"]:
            raise ValueError("No framing must not coexist with a real frame.")
        return [], True
    return values, False


def prepare(data_zip, starter_zip, out):
    if sha256(data_zip) != DATA_SHA256 or sha256(starter_zip) != KIT_SHA256:
        raise ValueError("These files differ from the pinned official release; check the package versions.")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty preparation directory.")
    out.mkdir(parents=True, exist_ok=True)
    extract(data_zip, out / "source")
    extract(starter_zip, out / "starter")
    root = out / "source/data"
    batches = [
        root / "initial-training-data/subtask1/SEMEVAL23_FRAMING_INITIAL_TRAIN-B/EN",
        root / "training-second-batch-oct/EN",
    ]
    label_source = batches[0].parent / "labels.txt"
    labels = label_source.read_text(encoding="utf-8-sig").splitlines()
    if len(labels) != 14 or len(set(labels)) != 14:
        raise ValueError("Expected the 14 machine-readable labels in the release.")
    (out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    articles = out / "articles"
    articles.mkdir()
    rows, seen, originals, audit = [], set(), {}, []
    counts, document_hashes = Counter(), defaultdict(list)
    batch_counts = []
    for batch in batches:
        gold_path = batch / "subtask-1-annotations.txt"
        with gold_path.open(encoding="utf-8-sig", newline="") as h:
            source_rows = list(csv.reader(h, delimiter="\t"))
        reference_path = batch / "subtask-1-annotations-with-paragraphs.txt"
        with reference_path.open(encoding="utf-8-sig", newline="") as h:
            references = {(r[0], int(r[1]), int(r[2])): r[-1]
                          for r in csv.reader(h, delimiter="\t") if len(r) >= 4}
        names = sorted({r[0] for r in source_rows})
        for name in names:
            # Copy bytes, not decoded text: newline conversion changes offsets.
            src = batch / "raw-documents" / name
            if name in originals:
                raise ValueError(f"Duplicate article filename across releases: {name}")
            originals[name] = sha256(src)
            shutil.copyfile(src, articles / name)
            normalized = " ".join(src.read_bytes().decode("utf-8").split()).casefold()
            document_hashes[hashlib.sha256(normalized.encode()).hexdigest()].append(name)
        for line, row in enumerate(source_rows, 1):
            name, start, end = row[:3]
            key = name, int(start), int(end)
            if key in seen:
                raise ValueError(f"Duplicate paragraph key: {key}")
            seen.add(key)
            assigned, normalized = canonical_labels(row[3:])
            if set(assigned) - set(labels):
                raise ValueError(f"Unknown labels in {gold_path}:{line}")
            if normalized:
                audit.append({"kind": "no_framing_to_empty", "article": name,
                              "start": int(start), "end": int(end)})
            text = (articles / name).read_bytes().decode("utf-8")
            if not 0 <= int(start) < int(end) <= len(text):
                raise ValueError(f"Invalid supplied offsets: {key}")
            if key not in references:
                audit.append({"kind": "missing_auxiliary_reference", "article": name,
                              "start": int(start), "end": int(end)})
            elif " ".join(references[key].split()) != " ".join(text[int(start):int(end)].split()):
                audit.append({"kind": "auxiliary_text_mismatch", "article": name,
                              "start": int(start), "end": int(end)})
            rows.append([name, start, end] + assigned)
            counts.update(assigned)
        batch_counts.append({"member": str(gold_path.relative_to(out / "source")),
                             "articles": len(names), "paragraphs": len(source_rows),
                             "sha256": sha256(gold_path)})
    duplicates = [names for names in document_hashes.values() if len(names) > 1]
    if duplicates:
        raise ValueError(f"Duplicate complete articles need shared split groups: {duplicates}")
    # The official scorer accepts a three-column row for no framing. A trailing
    # empty fourth field is interpreted as an invalid label by this scorer version.
    with (out / "annotations.tsv").open("w", encoding="utf-8", newline="") as h:
        csv.writer(h, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE).writerows(rows)
    records = read_table(out / "annotations.tsv", labels, articles)
    scorer_source = out / "starter/starter-kit/subtask1/scorer-subtask-1.py"
    shutil.copyfile(scorer_source, out / "scorer-subtask-1.py")
    info = {
        "data_kind": "designated-training", "unit": "article_paragraph", "language": "English",
        "competition_url": "https://artist.math.unipd.it/competitions/4/",
        "package_name": "full-data_2026_10_14", "download_filename": data_zip.name,
        "download_date": "2026-11-09", "source_articles": len(originals),
        "raw_rows": len(rows), "retained_units": len(records),
        "multilabel_units": sum(len(r.labels) > 1 for r in records),
        "empty_gold_units": sum(not r.labels for r in records),
        "label_counts": dict(counts), "batches": batch_counts,
        "audit_counts": dict(Counter(x["kind"] for x in audit)),
        "article_sha256": originals, "duplicate_article_groups": duplicates,
        "data_zip_sha256": sha256(data_zip), "starter_zip_sha256": sha256(starter_zip),
        "official_scorer_sha256": sha256(out / "scorer-subtask-1.py"),
        "offset_policy": "Slice the original UTF-8 text without newline conversion; preserve supplied IDs and offsets.",
        "annotation_policy": "Use aggregate gold TSV; map sole No framing markers to empty sets. Do not alter offsets or drop auxiliary mismatches.",
        "metric_policy": "Official scorer: 14 frames plus None, zero_division=1.",
        "redistribution": "Private research data. Exclude raw texts, gold annotations and official package files from the deliverable ZIP.",
    }
    save_json(out / "dataset_info.json", info)
    save_json(out / "data_audit.json", audit)
    print(json.dumps({k: info[k] for k in ["source_articles", "retained_units", "multilabel_units",
                                          "empty_gold_units", "audit_counts"]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-zip", required=True, type=Path)
    p.add_argument("--starter-zip", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    prepare(args.data_zip, args.starter_zip, args.out)


if __name__ == "__main__":
    main()
