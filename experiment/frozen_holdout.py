"""Freeze v1 pipelines, then evaluate a later local campaign without fitting.

Joblib is pickle-based: load only bundles produced locally by this project.
Hashes detect accidental changes; they do not make untrusted pickles safe.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
from typing import Any, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import confusion_matrix

from experiment.ml_baseline import (
    LABELS, MODEL_NAMES, IncidentDataset, _classification_metrics,
    build_model, load_incident_dataset, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCES = (
    "experiment/ml_baseline.py",
    "experiment/frozen_holdout.py",
    "experiment/data_engineering/build_local_incidents.py",
    "experiment/data_engineering/incident_schema.py",
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
    }


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Dataset timestamps must include a timezone")
    return result.astimezone(UTC)


def _provenance(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_ids, systems, starts, ends = set(), set(), [], []
    hashes: set[str] = set()
    for record in records:
        if (record.get("schema_version") != 1 or record.get("source") != "local"
                or record.get("task") != "local_fault_classification"):
            raise ValueError("Only schema-v1 local fault-classification records are allowed")
        provenance = record.get("provenance", {})
        run_id, system = provenance.get("run_id"), record.get("system_name")
        if not run_id or not system:
            raise ValueError("Campaign run_id and system_name are required")
        run_ids.add(str(run_id))
        systems.add(str(system))
        start, end = _timestamp(record["window_start"]), _timestamp(record["window_end"])
        if start >= end:
            raise ValueError("Invalid incident time window")
        starts.append(start)
        ends.append(end)
        for key in ("docker_logs_sha256", "probes_sha256"):
            if not provenance.get(key):
                raise ValueError(f"Missing provenance hash: {key}")
            hashes.add(str(provenance[key]))
    return {
        "run_ids": sorted(run_ids), "systems": sorted(systems),
        "earliest_window_start": min(starts).isoformat(),
        "latest_window_end": max(ends).isoformat(),
        "raw_telemetry_hashes": sorted(hashes),
    }


def _validate_metric_schema(dataset: IncidentDataset, expected: list[str]) -> None:
    if any(sorted(row["metric_features"]) != expected for row in dataset.rows):
        raise ValueError("Metric feature schema differs from the frozen training schema")


def freeze_models(input_path: Path, output_dir: Path, *, seed: int = 20260825) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Frozen bundle already exists; refusing to overwrite: {output_dir}")
    dataset = load_incident_dataset(input_path)
    provenance = _provenance(input_path)
    _validate_metric_schema(dataset, dataset.metric_feature_names)
    models = {name: build_model(name, seed=seed).fit(dataset.rows, dataset.labels) for name in MODEL_NAMES}
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "models.joblib"
    joblib.dump(models, artifact, compress=3)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Frozen full-v1 fit for subsequent holdout-only prediction",
        "random_seed": seed,
        "model_names": list(MODEL_NAMES),
        "model_sha256": sha256_file(artifact),
        "versions": runtime_versions(),
        "source_sha256": {name: sha256_file(ROOT / name) for name in FROZEN_SOURCES},
        "training": {
            "path": str(input_path), "sha256": sha256_file(input_path),
            "records": len(dataset.rows), "class_counts": dataset.class_counts,
            "incident_ids": dataset.incident_ids, "split_groups": dataset.groups,
            "metric_feature_names": dataset.metric_feature_names,
            "provenance": provenance, "leakage_controls": dataset.leakage_audit,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_names") != list(MODEL_NAMES):
        raise ValueError("Unsupported frozen bundle")
    if runtime_versions() != manifest["versions"]:
        raise ValueError("Runtime versions differ from the frozen environment")
    if set(manifest["source_sha256"]) != set(FROZEN_SOURCES):
        raise ValueError("Frozen source list is incomplete")
    for name, expected_hash in manifest["source_sha256"].items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"Frozen preprocessing/evaluation source changed: {name}")
    if sha256_file(bundle_dir / "models.joblib") != manifest["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    return manifest


def evaluate_holdout(bundle_dir: Path, input_path: Path) -> dict[str, Any]:
    manifest = verify_bundle(bundle_dir)
    dataset = load_incident_dataset(input_path)
    provenance = _provenance(input_path)
    training = manifest["training"]
    if sha256_file(input_path) == training["sha256"]:
        raise ValueError("Holdout cannot be the training dataset")
    for name, test_values, train_values in (
        ("incident IDs", dataset.incident_ids, training["incident_ids"]),
        ("split groups", dataset.groups, training["split_groups"]),
        ("campaign run IDs", provenance["run_ids"], training["provenance"]["run_ids"]),
        ("raw telemetry", provenance["raw_telemetry_hashes"], training["provenance"]["raw_telemetry_hashes"]),
    ):
        if set(test_values).intersection(train_values):
            raise ValueError(f"Training/holdout overlap in {name}")
    if provenance["systems"] != training["provenance"]["systems"]:
        raise ValueError("This protocol evaluates the same local system only")
    lower_bound = max(_timestamp(manifest["created_at"]), _timestamp(training["provenance"]["latest_window_end"]))
    if _timestamp(provenance["earliest_window_start"]) <= lower_bound:
        raise ValueError("Holdout telemetry must start after the model was frozen")
    _validate_metric_schema(dataset, training["metric_feature_names"])

    # Never call fit/fit_transform here, including on text or metric preprocessing.
    models = joblib.load(bundle_dir / "models.joblib")
    results = {}
    for name in MODEL_NAMES:
        model = models[name]
        predicted = [str(value) for value in model.predict(dataset.rows)]
        probabilities = model.predict_proba(dataset.rows)
        classes = list(model.named_steps["classifier"].classes_)
        matrix = confusion_matrix(dataset.labels, predicted, labels=LABELS)
        results[name] = {
            "metrics": _classification_metrics(dataset.labels, predicted),
            "confusion_matrix": {
                actual: {label: int(matrix[i, j]) for j, label in enumerate(LABELS)}
                for i, actual in enumerate(LABELS)
            },
            "predictions": [
                {"incident_id": incident_id, "actual": actual, "predicted": guess,
                 "correct": actual == guess,
                 "confidence": round(float(probabilities[i, classes.index(guess)]), 6)}
                for i, (incident_id, actual, guess) in enumerate(zip(dataset.incident_ids, dataset.labels, predicted))
            ],
        }
    return {
        "schema_version": 1, "experiment_id": "local-frozen-holdout-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "bundle": {"path": str(bundle_dir), "manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
                   "model_sha256": manifest["model_sha256"], "frozen_at": manifest["created_at"]},
        "training": {key: training[key] for key in ("path", "sha256", "records", "class_counts")},
        "test": {"path": str(input_path), "sha256": sha256_file(input_path),
                 "records": len(dataset.rows), "class_counts": dataset.class_counts,
                 "metric_feature_count": len(dataset.metric_feature_names), "provenance": provenance},
        "evaluation_design": {
            "method": "separate_later_campaign_frozen_prediction",
            "test_fitting_or_tuning": False,
            "training_test_disjoint": True,
            "captured_after_freeze": True,
            "metric_schema_unchanged": True,
            "source_and_runtime_verified": True,
        },
        "leakage_controls": dataset.leakage_audit,
        "models": results,
        "comparison": {"logs_plus_metrics_minus_logs_only_macro_f1": round(
            results["logs_plus_metrics"]["metrics"]["macro_f1"] - results["logs_only"]["metrics"]["macro_f1"], 6)},
        "limitations": [
            "Same system and known fault classes; not cross-system or unknown-fault generalisation.",
            "Workload and duration change together, so their individual effects cannot be isolated.",
            "Complete incident windows include recovery evidence; this is retrospective classification, not online early diagnosis.",
            "Strong operational fault signatures remain; a small test set cannot establish broad reliability.",
            "Classifier probabilities are not calibrated confidence guarantees.",
        ],
    }


def render_holdout_figure(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((path.parent / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Database", "HTTP 500", "Service stopped"]
    for axis, name, title in zip(axes, MODEL_NAMES, ("Logs only", "Logs + metrics")):
        model = result["models"][name]
        matrix = np.array([[model["confusion_matrix"][a][b] for b in LABELS] for a in LABELS])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        axis.set_xticks(range(3), labels, rotation=20, ha="right")
        axis.set_yticks(range(3), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"{title} | Macro F1 = {model['metrics']['macro_f1']:.3f}")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.suptitle(f"Frozen v1 models: {result['test']['records']}-incident independent local holdout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--input", type=Path, default=Path("data/processed/incidents/local-campaign-v1.jsonl"))
    freeze.add_argument("--output-dir", type=Path, default=Path("data/models/local-v1-frozen"))
    freeze.add_argument("--seed", type=int, default=20260825)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, default=Path("data/models/local-v1-frozen"))
    evaluate.add_argument("--input", type=Path, default=Path("data/processed/incidents/local-campaign-v2.jsonl"))
    evaluate.add_argument("--output", type=Path, default=Path("data/results/local-frozen-holdout-v2.json"))
    evaluate.add_argument("--figure", type=Path, default=Path("data/results/figures/local-frozen-holdout-v2.png"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_models(args.input, args.output_dir, seed=args.seed)
        print(json.dumps({"bundle": str(args.output_dir), "frozen_at": result["created_at"],
                          "training_records": result["training"]["records"], "model_sha256": result["model_sha256"]}, indent=2))
    else:
        if args.output.exists() or args.figure.exists():
            raise FileExistsError("Holdout outputs already exist; refusing to overwrite")
        result = evaluate_holdout(args.bundle, args.input)
        render_holdout_figure(result, args.figure)
        result["artifacts"] = {"figure": str(args.figure), "figure_sha256": sha256_file(args.figure)}
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "metrics": {name: result["models"][name]["metrics"] for name in MODEL_NAMES}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
