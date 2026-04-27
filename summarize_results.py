import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


FIELDS = [
    "model",
    "image_size",
    "optimizer",
    "lr",
    "batch_size",
    "weight_decay",
    "scheduler",
    "normalization",
    "augmentation",
    "best_epoch",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "confusion_matrix_path",
    "test_confusion_matrix_path",
    "run_dir",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def summarize_run(run_dir: Path) -> Optional[dict[str, Any]]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics" / "best_validation_metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None

    config = read_json(config_path)
    metrics = read_json(metrics_path)
    test_metrics_path = run_dir / "metrics" / "final_test_metrics.json"
    test_metrics = read_json(test_metrics_path) if test_metrics_path.exists() else {}
    args = config.get("args", {})
    model_config = config.get("model_config", {})
    normalization = config.get("normalization", {})
    confusion_path = run_dir / "figures" / "best_validation_confusion_matrix.png"
    test_confusion_path = run_dir / "figures" / "final_test_confusion_matrix.png"

    return {
        "model": config.get("model"),
        "image_size": "x".join(str(value) for value in model_config.get("image_size", [])),
        "optimizer": args.get("optimizer"),
        "lr": args.get("lr"),
        "batch_size": args.get("batch_size"),
        "weight_decay": args.get("weight_decay"),
        "scheduler": args.get("scheduler"),
        "normalization": normalization.get("mode", args.get("normalization", "imagenet")),
        "augmentation": config.get("augmentation", args.get("augmentation", "none")),
        "best_epoch": metrics.get("best_epoch", metrics.get("epoch")),
        "val_accuracy": metrics.get("accuracy"),
        "val_precision": metrics.get("precision"),
        "val_recall": metrics.get("recall"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "confusion_matrix_path": str(confusion_path) if confusion_path.exists() else "",
        "test_confusion_matrix_path": str(test_confusion_path) if test_confusion_path.exists() else "",
        "run_dir": str(run_dir),
    }


def format_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(FIELDS) + " |"
    separator = "| " + " | ".join(["---"] * len(FIELDS)) + " |"
    lines = [header, separator]
    for row in rows:
        values = []
        for field in FIELDS:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation results from runs/* directories.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional output file. Defaults to stdout.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(path for path in runs_dir.glob("*") if path.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: float(row["val_accuracy"] or 0.0), reverse=True)

    if args.format == "markdown":
        output = format_markdown(rows) + "\n"
    else:
        from io import StringIO

        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        output = handle.getvalue()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
