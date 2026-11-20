"""CPU-friendly English news framing with explicit dataset and metric provenance.

The native TSV interface preserves supplied article/paragraph identifiers.
The public experiment groups unique snippets because no article IDs are released.
All tuning uses validation labels only; the test set does not select the model.
"""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion


# 读段落和标签 / Load paragraphs and labels
@dataclass
class Paragraph:
    article: str
    start: int
    end: int
    text: str
    labels: tuple

    @property
    def key(self):
        return self.article, self.start, self.end


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_labels(path):
    labels = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if len(labels) != 14 or len(set(labels)) != 14 or any(not x or x.strip() != x for x in labels):
        raise ValueError("Supply exactly 14 unique framing labels, one per line.")
    if "None" in labels:
        raise ValueError("None is an evaluation label, not a predicted framing category.")
    return labels


def read_table(path, labels, articles=None, gold=True):
    """Read aggregated TSV; keep Unicode character offsets and raw newlines."""
    records, seen, cache = [], set(), {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_no, fields in enumerate(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_no}: expected filename, start and end.")
            name, start, end = fields[:3]
            if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".txt"):
                raise ValueError(f"Unsafe or invalid article filename: {name!r}")
            if not gold and any(fields[3:]):
                raise ValueError("Prediction input must contain boundaries only, without gold labels.")
            assigned = tuple(x for x in fields[3:] if x)
            if len(set(assigned)) != len(assigned) or set(assigned) - set(labels):
                raise ValueError(f"{path}:{line_no}: duplicate or unknown machine labels: {assigned}")
            start, end = int(start), int(end)
            if not 0 <= start < end:
                raise ValueError(f"{path}:{line_no}: invalid offsets.")
            text = ""
            if articles is not None:
                if name not in cache:
                    # read_text() would change CRLF into LF and corrupt the supplied offsets.
                    cache[name] = (Path(articles) / name).read_bytes().decode("utf-8")
                source = cache[name]
                if end > len(source):
                    raise ValueError(f"{name}: end offset {end} exceeds {len(source)} characters.")
                text = source[start:end]
                if not text.strip():
                    raise ValueError(f"{name}:{start}-{end}: paragraph contains no text.")
            record = Paragraph(name, start, end, text, assigned)
            if record.key in seen:
                raise ValueError(f"Duplicate paragraph: {record.key}")
            seen.add(record.key)
            records.append(record)
    if not records:
        raise ValueError("The TSV file contains no paragraphs.")
    return records


def encode(records, labels):
    positions = {name: i for i, name in enumerate(labels)}
    y = np.zeros((len(records), len(labels)), dtype=np.int8)
    for row, record in enumerate(records):
        for name in record.labels:
            y[row, positions[name]] = 1
    return y


def read_jsonl(path, labels, gold=True):
    """Read prepared news snippets; these IDs are not original article IDs."""
    records, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name, text = row["id"], row["text"]
        if not gold and "labels" in row:
            raise ValueError("Prediction JSONL must omit the gold labels field.")
        assigned = tuple(row["labels"]) if gold else ()
        if not isinstance(text, str) or not text.strip() or name in seen:
            raise ValueError("Empty text or duplicate input ID in JSONL.")
        if len(assigned) != len(set(assigned)) or set(assigned) - set(labels):
            raise ValueError("Duplicate or unknown labels in JSONL.")
        seen.add(name)
        records.append(Paragraph(name, 0, len(text), text, assigned))
    if not records:
        raise ValueError("No JSONL records were found.")
    return records


# 计算 F1 / Compute F1 scores
def metrics(truth, prediction, labels, include_none=True):
    """Score fixed-label decisions, optionally adding None for the native task.

    The downloaded official scorer uses 14 frames plus None and zero_division=1.
    The earlier public-data experiment uses 14 frames and zero_division=0.
    """
    def with_none(y):
        return np.column_stack([y, ~y.astype(bool).any(axis=1)]).astype(np.int8)

    a, b = (with_none(truth), with_none(prediction)) if include_none else (truth, prediction)
    metric_labels = labels + ["None"] if include_none else labels
    undefined = 1 if include_none else 0
    precision, recall, f1, support = precision_recall_fscore_support(a, b, zero_division=undefined)
    return {
        "micro_f1": float(f1_score(a, b, average="micro", zero_division=undefined)),
        "macro_f1": float(f1_score(a, b, average="macro", zero_division=undefined)),
        "per_label": {label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for label, p, r, f, s in zip(metric_labels, precision, recall, f1, support)},
    }


# 按文章切分数据 / Split the data by article
def split_articles(records, seed):
    """Split grouping IDs 60/20/20; public-data IDs denote unique text snippets."""
    groups = np.array([r.article for r in records])
    if len(set(groups)) < 5:
        raise ValueError("At least five articles are needed for three disjoint article splits.")
    indices = np.arange(len(records))
    test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    pool, test = next(test_split.split(indices, groups=groups))
    val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_local, val_local = next(val_split.split(pool, groups=groups[pool]))
    return {"train": pool[train_local], "validation": pool[val_local], "test": test}


# TF-IDF 特征和分类器 / TF-IDF features and classifiers
def make_features(kind):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000,
                          sublinear_tf=True, lowercase=True, dtype=np.float64)
    if kind == "word":
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                          max_features=40000, sublinear_tf=True, dtype=np.float64)
    return FeatureUnion([("word", word), ("char", char)])


def fit_heads(x, y, c, seed, class_weight=None):
    heads = []
    for column in y.T:
        # 只有一种标签值时直接用常数 / Use a constant if only one class appears.
        if np.unique(column).size == 1:
            heads.append(float(column[0]))
            continue
        head = LogisticRegression(C=c, solver="liblinear", max_iter=2000,
                                  class_weight=class_weight, random_state=seed)
        head.fit(x, column)
        heads.append(head)
    return heads


def probabilities(heads, x):
    columns = []
    for head in heads:
        if isinstance(head, float):
            column = np.full(x.shape[0], head)
        else:
            column = head.predict_proba(x)[:, 1]
        columns.append(column)
    return np.column_stack(columns)


# 写预测文件并核对官方评分 / Write predictions and check official scores
def write_predictions(path, records, prediction, labels):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        for record, row in zip(records, prediction):
            chosen = [label for label, active in zip(labels, row) if active]
            # This scorer expects exactly three fields for an empty prediction.
            # A trailing tab becomes an invalid empty-string label in its parser.
            writer.writerow([record.article, record.start, record.end] + chosen)


def check_official(scorer, gold, predictions, label_file, expected):
    """Run the unmodified official CLI and check both scores before reporting them."""
    result = subprocess.run([sys.executable, str(scorer), "-g", str(gold), "-p", str(predictions),
                             "-f", str(label_file), "-o"], check=True, capture_output=True, text=True)
    fields = result.stdout.strip().split("\t")
    if len(fields) != 2:
        raise ValueError(f"Official scorer did not accept the files: {result.stdout} {result.stderr}")
    scores = dict(zip(["micro_f1", "macro_f1"], map(float, fields)))
    for key, value in scores.items():
        if abs(value - expected[key]) > 1e-12:
            raise ValueError(f"Official {key} differs: {value} versus {expected[key]}")
    return scores


# 训练三个模型，在验证集上选模型 / Train three models and select on validation
def train(args):
    labels = read_labels(args.labels)
    dataset_info = {}
    if args.dataset_jsonl:
        records = read_jsonl(args.dataset_jsonl, labels)
        info_path = Path(args.dataset_jsonl).with_name("dataset_info.json")
    else:
        if not args.annotations or not args.articles:
            raise ValueError("Supply --dataset-jsonl or both --articles and --annotations.")
        records = read_table(args.annotations, labels, args.articles)
        info_path = Path(args.annotations).with_name("dataset_info.json")
    if info_path.exists():
        dataset_info = json.loads(info_path.read_text())
    if args.data_kind == "public-annotated" and dataset_info.get("data_kind") != "public-annotated":
        raise ValueError("The public experiment needs the prepared dataset and its source manifest.")
    if dataset_info and dataset_info.get("data_kind") != args.data_kind:
        raise ValueError("The requested data kind contradicts the downloaded dataset manifest.")
    if args.data_kind == "designated-training" and (not args.official_scorer or
                                                    dataset_info.get("data_kind") != "designated-training"):
        raise ValueError("Official training requires its prepared source manifest and --official-scorer.")
    include_none = args.data_kind != "public-annotated"
    y = encode(records, labels)
    splits = split_articles(records, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty output directory so earlier experiments are not overwritten.")
    manifest = {name: {"group_ids": sorted({records[i].article for i in ids}), "units": len(ids)}
                for name, ids in splits.items()}
    save_json(out / "split.json", {"seed": args.seed, "unit": dataset_info.get("unit", "article_paragraph"), **manifest})
    train_ids = splits["train"]
    val_ids = splits["validation"]
    test_ids = splits["test"]
    texts = [r.text for r in records]
    test_records = [records[i] for i in test_ids]
    write_predictions(out / "test_gold.tsv", test_records, y[test_ids], labels)
    with (out / "test_boundaries.tsv").open("w", encoding="utf-8") as handle:
        for r in test_records:
            handle.write(f"{r.article}\t{r.start}\t{r.end}\n")
    if args.dataset_jsonl:
        with (out / "test_inputs.jsonl").open("w", encoding="utf-8") as handle:
            for r in test_records:
                handle.write(json.dumps({"id": r.article, "text": r.text}, ensure_ascii=False) + "\n")
    results, all_errors = {}, []
    model_kinds = ["word", "word_char", "word_char_balanced"]
    thresholds = np.round(np.arange(0.10, 0.61, 0.05), 2)
    for kind in model_kinds:
        print(f"Fitting {kind} on {len(train_ids)} training units...", flush=True)
        started = time.perf_counter()
        vectorizer = make_features(kind)
        # Vocabulary and IDF are learned from training paragraphs only.
        x_train = vectorizer.fit_transform([texts[i] for i in train_ids])
        x_val = vectorizer.transform([texts[i] for i in val_ids])
        best = None
        search = []
        class_weight = "balanced" if kind.endswith("_balanced") else None
        for c in [0.5, 2.0]:
            print(f"  C={c}", flush=True)
            heads = fit_heads(x_train, y[train_ids], c, args.seed, class_weight)
            probs = probabilities(heads, x_val)
            for threshold in thresholds:
                score = metrics(y[val_ids], probs >= threshold, labels, include_none)["micro_f1"]
                search.append({"c": c, "threshold": float(threshold), "validation_micro_f1": score})
                # Ties favor a threshold near 0.5 and then stronger regularization.
                rank = (score, -abs(float(threshold) - 0.5), -c)
                if best is None or rank > best[0]:
                    best = rank, heads, c, float(threshold)
        _, heads, c, threshold = best
        x_test = vectorizer.transform([texts[i] for i in test_ids])
        pred = probabilities(heads, x_test) >= threshold
        validation = metrics(y[val_ids], probabilities(heads, x_val) >= threshold, labels, include_none)
        test = metrics(y[test_ids], pred, labels, include_none)
        bundle = {"vectorizer": vectorizer, "heads": heads, "labels": labels,
                  "threshold": threshold, "kind": kind, "data_kind": args.data_kind,
                  "include_none_metric": include_none}
        joblib.dump(bundle, out / f"{kind}.joblib", compress=3)
        write_predictions(out / f"{kind}_predictions.tsv", test_records, pred, labels)
        results[kind] = {"c": c, "class_weight": class_weight, "threshold": threshold, "features": x_train.shape[1],
                         "constant_labels": [label for label, h in zip(labels, heads) if isinstance(h, float)],
                         "validation": validation, "test": test, "search": search,
                         "seconds": round(time.perf_counter() - started, 3)}
        if args.official_scorer:
            results[kind]["official_test"] = check_official(
                args.official_scorer, out / "test_gold.tsv", out / f"{kind}_predictions.tsv", args.labels, test)
        for r, actual, guessed in zip(test_records, y[test_ids], pred):
            if not np.array_equal(actual, guessed):
                all_errors.append({"model": kind, "article": r.article, "start": r.start, "end": r.end,
                                   "gold": [label for label, active in zip(labels, actual) if active],
                                   "predicted": [label for label, active in zip(labels, guessed) if active],
                                   "missed": [label for label, a, b in zip(labels, actual, guessed) if a and not b],
                                   "extra": [label for label, a, b in zip(labels, actual, guessed) if b and not a]})
    # Choose the final model by validation F1.
    selected = max(results, key=lambda k: results[k]["validation"]["micro_f1"])
    shutil.copyfile(out / f"{selected}.joblib", out / "selected.joblib")
    if args.dataset_jsonl:
        source_paths = [Path(args.dataset_jsonl), Path(args.labels)]
    else:
        article_names = sorted({r.article for r in records})
        source_paths = [Path(args.annotations), Path(args.labels)]
        source_paths.extend(Path(args.articles) / name for name in article_names)
    if info_path.exists():
        source_paths.append(info_path)
    if args.official_scorer:
        source_paths.append(Path(args.official_scorer))
    provenance = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                  "numpy": np.__version__, "seed": args.seed, "data_kind": args.data_kind,
                  "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  "dataset_info": dataset_info}
    save_json(out / "provenance.json", provenance)
    save_json(out / "results.json", {"language": "English", "subtask": "1", "data_kind": args.data_kind,
                                    "metric_status": "official_scorer_verified" if args.official_scorer else "local_evaluation",
                                    "macro_policy": "fixed_14_frames_zero_division_0" if not include_none else "official_14_frames_plus_None_zero_division_1",
                                    "dataset_info": dataset_info,
                                    "selected_by_validation": selected, "models": results})
    # Include only two short excerpts for the report. Full source texts remain
    # in the locally downloaded dataset and are not redistributed in the ZIP.
    lookup = {r.key: r.text for r in test_records}
    for error in [e for e in all_errors if e["model"] == selected][:2]:
        key = error["article"], error["start"], error["end"]
        error["text_excerpt"] = " ".join(lookup[key].split()[:12])
    with (out / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in all_errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    counts = {name: {"groups": len(value["group_ids"]), "units": value["units"]} for name, value in manifest.items()}
    print(json.dumps({"data_kind": args.data_kind, "splits": counts, "selected": selected,
                      "test_scores": {k: {m: v["test"][m] for m in ["micro_f1", "macro_f1"]}
                                           for k, v in results.items()}}, indent=2))


# 预测和评分 / Prediction and scoring
def predict(args):
    # Load only model files produced by this project; joblib is not a safe format
    # for opening arbitrary files obtained from untrusted sources.
    model = joblib.load(args.model)
    if args.dataset_jsonl:
        records = read_jsonl(args.dataset_jsonl, model["labels"], gold=False)
    else:
        if not args.paragraphs or not args.articles:
            raise ValueError("Supply --dataset-jsonl or both --articles and --paragraphs.")
        records = read_table(args.paragraphs, model["labels"], args.articles, gold=False)
    x = model["vectorizer"].transform([r.text for r in records])
    pred = probabilities(model["heads"], x) >= model["threshold"]
    write_predictions(args.out, records, pred, model["labels"])
    print(f"Wrote {len(records)} text-unit predictions to {args.out}")


def score(args):
    labels = read_labels(args.labels)
    gold = read_table(args.gold, labels)
    pred = read_table(args.pred, labels)
    by_key = {r.key: r for r in pred}
    if set(by_key) != {r.key for r in gold}:
        raise ValueError("Gold and prediction paragraph IDs/offsets must match exactly.")
    result = metrics(encode(gold, labels), encode([by_key[r.key] for r in gold], labels), labels,
                     include_none=not args.standard_multilabel)
    status = "standard_multilabel_local_evaluation" if args.standard_multilabel else "local_implementation_of_official_metric"
    if args.official_scorer:
        if args.standard_multilabel:
            raise ValueError("The official scorer uses None; do not combine it with --standard-multilabel.")
        check_official(args.official_scorer, args.gold, args.pred, args.labels, result)
        status = "official_scorer_verified"
    print(json.dumps({"status": status, **result}, indent=2))


# 生成 LaTeX 表格和分析 / Write LaTeX tables and analysis
def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def report(args):
    """Write measured report fragments from a completed run; never substitute trial scores."""
    run, target = Path(args.run), Path(args.report_dir)
    results = json.loads((run / "results.json").read_text())
    if results["data_kind"] not in ["public-annotated", "designated-training"]:
        raise ValueError("A completed public-data or designated-training run is required.")
    official = results["data_kind"] == "designated-training"
    if official and results["metric_status"] != "official_scorer_verified":
        raise ValueError("Verify the official scorer before writing the formal report.")
    split = json.loads((run / "split.json").read_text())
    info = results["dataset_info"]
    target.mkdir(parents=True, exist_ok=True)
    macros = {
        "RunSeed": split["seed"], "RawRows": info["raw_rows"], "TotalUnits": info["retained_units"],
        "MultiLabelUnits": info["multilabel_units"],
        "ExcludedOtherUnits": info.get("other_only_groups_excluded", 0),
    }
    for name, prefix in [("train", "Train"), ("validation", "Val"), ("test", "Test")]:
        macros[prefix + "Units"] = split[name]["units"]
        macros[prefix + "Articles"] = len(split[name]["group_ids"])
    names = {"word": "word baseline", "word_char": "word and character model",
             "word_char_balanced": "model with class balancing"}
    selected = results["selected_by_validation"]
    macros["SelectedModel"] = names[selected]
    chosen = results["models"][selected]
    chosen_test = chosen.get("official_test", chosen["test"])
    macros.update({"SelectedMicro": f"{chosen_test['micro_f1']:.3f}",
                   "SelectedMacro": f"{chosen_test['macro_f1']:.3f}",
                   "ValidationSeed": split["seed"] + 1})
    if official:
        macros.update({"SourceArticles": info["source_articles"], "EmptyGoldUnits": info["empty_gold_units"],
                       "NormalizedNoFraming": info["audit_counts"]["no_framing_to_empty"],
                       "AuxMismatch": info["audit_counts"]["auxiliary_text_mismatch"],
                       "AuxMissing": info["audit_counts"]["missing_auxiliary_reference"]})
    def macro_value(key, value):
        return f"{value:,}" if isinstance(value, int) and key not in ["RunSeed", "ValidationSeed"] else str(value)
    text = ["\\newcommand{\\" + key + "}{" + macro_value(key, value) + "}" for key, value in macros.items()]
    (target / "run_info.tex").write_text("\n".join(text) + "\n")
    # Keep model names on one line; a paragraph column would stretch their words.
    # Group the two test metrics and let the table fill the text column evenly.
    table_names = {"word": "Word baseline", "word_char": "Word and character",
                   "word_char_balanced": "With class balancing"}
    score_columns = [("validation", "micro_f1"), ("test", "micro_f1"), ("test", "macro_f1")]
    best_scores = [max(results["models"][kind][split_name][metric] for kind in table_names)
                   for split_name, metric in score_columns]
    rows = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}", r"\toprule",
            r" & Validation & \multicolumn{2}{c}{Test}\\",
            r"\cmidrule(lr){2-2}\cmidrule(l){3-4}",
            r"Model & micro F1 & micro F1 & macro F1\\", r"\midrule"]
    for index, (kind, title) in enumerate(table_names.items()):
        m = results["models"][kind]
        cells = [title]
        for (split_name, metric), best in zip(score_columns, best_scores):
            value = m[split_name][metric]
            number = f"{value:.3f}"
            cells.append(r"\textbf{" + number + "}" if value == best else number)
        rows.append(" & ".join(cells) + r" \\")
        if index < len(table_names) - 1:
            rows.append(r"\addlinespace[3pt]")
    rows.extend([r"\bottomrule", r"\end{tabular*}"])
    (target / "results.tex").write_text("\n".join(rows) + "\n")
    analysis = [f"The {names[selected]} has the highest validation micro F1 and is selected as the final model."]
    for previous, current, change in [("word", "word_char", "Adding character features"),
                                      ("word_char", "word_char_balanced", "Adding class balancing")]:
        before, after = results["models"][previous]["test"], results["models"][current]["test"]
        delta_micro = after["micro_f1"] - before["micro_f1"]
        delta_macro = after["macro_f1"] - before["macro_f1"]
        micro_verb = "increases" if delta_micro >= 0 else "reduces"
        macro_verb = "increases" if delta_macro >= 0 else "reduces"
        macro_phrase = "macro F1" if micro_verb == macro_verb else f"{macro_verb} macro F1"
        analysis.append(f"{change} {micro_verb} test micro F1 by {abs(delta_micro):.3f} and "
                        f"{macro_phrase} by {abs(delta_macro):.3f}.")
    (target / "analysis.tex").write_text("\n".join(analysis) + "\n")
    errors = [json.loads(line) for line in (run / "errors.jsonl").read_text().splitlines()]
    errors = [e for e in errors if e["model"] == selected][:2]
    short_names = dict(zip(results["models"][selected]["test"]["per_label"], [
        "Economic", "Capacity", "Morality", "Fairness", "Legality", "Policy", "Crime",
        "Security", "Health", "Quality of life", "Culture", "Public opinion", "Political", "External regulation"]))
    def display(values):
        return ", ".join(short_names.get(x, x.replace("_", " ")) for x in values) or "empty"
    examples = []
    for i, e in enumerate(errors, 1):
        excerpt = tex_escape(e.get("text_excerpt", "").strip(' "'))
        location = f"\\texttt{{{tex_escape(e['article'])}}}, {e['start']}--{e['end']}. " if official else ""
        examples.append(f"\\paragraph{{Example {i}.}} {location}\\emph{{``{excerpt}''}}. "
                        f"Gold: {tex_escape(display(e['gold']))}; prediction: {tex_escape(display(e['predicted']))}.")
    (target / "errors.tex").write_text(("\n".join(examples) or
                                       "All test predictions match their gold label sets.") + "\n")
    print(f"Updated measured report fragments in {target}")


# 命令行入口 / Command-line entry point
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Compare three systems on disjoint grouping-ID splits.")
    for name in ["labels", "out"]:
        p.add_argument("--" + name, required=True)
    for name in ["articles", "annotations", "dataset-jsonl"]:
        p.add_argument("--" + name)
    p.add_argument("--data-kind", choices=["trial", "designated-training", "public-annotated"], required=True)
    p.add_argument("--official-scorer")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)
    p = commands.add_parser("predict", help="Predict from an unlabeled paragraph-boundary TSV.")
    for name in ["model", "out"]:
        p.add_argument("--" + name, required=True)
    for name in ["articles", "paragraphs", "dataset-jsonl"]:
        p.add_argument("--" + name)
    p.set_defaults(func=predict)
    p = commands.add_parser("score", help="Compute explicitly unofficial local proxy metrics.")
    for name in ["gold", "pred", "labels"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--standard-multilabel", action="store_true")
    p.add_argument("--official-scorer")
    p.set_defaults(func=score)
    p = commands.add_parser("report", help="Generate LaTeX tables from one completed run.")
    p.add_argument("--run", required=True)
    p.add_argument("--report-dir", default="report")
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
