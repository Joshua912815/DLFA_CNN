import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from main import (
    DATA_DIR,
    MODEL_CONFIGS,
    SqueezeExcitation,
    build_model,
    evaluate,
    get_device,
    initialize_lazy_layers,
    list_image_samples,
    make_dataloader,
    plot_confusion_matrix,
    resolve_normalization,
    split_train_val,
    write_json,
)


class LegacyConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class LegacyResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.conv1 = LegacyConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SqueezeExcitation(out_channels)
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.activation(x + residual)


class LegacyResidualSEPriceCNN(nn.Module):
    """BatchNorm version used by the first res_se_cnn attempt."""

    def __init__(self, channels: tuple[int, ...], num_classes: int = 2) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.features = nn.Sequential(
            LegacyConvBNAct(3, c0, kernel_size=(5, 3), padding=(2, 1)),
            LegacyResidualSEBlock(c0, c1, stride=(2, 2)),
            LegacyResidualSEBlock(c1, c1),
            LegacyResidualSEBlock(c1, c2, stride=(2, 2)),
            LegacyResidualSEBlock(c2, c2),
            LegacyResidualSEBlock(c2, c3, stride=(2, 2)),
            LegacyResidualSEBlock(c3, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(c3, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def normalization_from_config(
    config_payload: dict[str, Any],
    train_split: list[tuple[Path, int, str]],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalization = config_payload.get("normalization", {})
    if "mean" in normalization and "std" in normalization:
        return tuple(normalization["mean"]), tuple(normalization["std"])

    args = config_payload.get("args", {})
    mode = args.get("normalization", "imagenet")
    sample_cap = int(args.get("normalization_samples", 4096))
    seed = int(args.get("seed", 42))
    return resolve_normalization(mode, train_split, image_size, sample_cap, seed)


def build_model_for_checkpoint(model_name: str, checkpoint_state_dict: dict[str, torch.Tensor]) -> nn.Module:
    if model_name == "res_se_cnn" and any(key.endswith("running_mean") for key in checkpoint_state_dict):
        return LegacyResidualSEPriceCNN(MODEL_CONFIGS[model_name].channels)
    return build_model(model_name)


def evaluate_run(
    run_dir: Path,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        return None

    config_payload = read_json(config_path)
    model_name = config_payload["model"]
    model_config = MODEL_CONFIGS[model_name]
    args = config_payload.get("args", {})

    train_samples = list_image_samples(data_dir / "train")
    train_split, _ = split_train_val(
        train_samples,
        float(args.get("val_ratio", 0.2)),
        args.get("val_split", "time"),
        int(args.get("seed", 42)),
    )
    mean, std = normalization_from_config(config_payload, train_split, model_config.image_size)

    test_samples = list_image_samples(data_dir / "test")
    test_loader = make_dataloader(
        test_samples,
        model_config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        mean=mean,
        std=std,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_for_checkpoint(model_name, checkpoint["model_state_dict"]).to(device)
    initialize_lazy_layers(model, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.get("label_smoothing", 0.0)))
    metrics = evaluate(model, test_loader, criterion, device)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["model"] = model_name
    metrics["run_dir"] = str(run_dir)

    metrics_path = run_dir / "metrics" / "final_test_metrics.json"
    figure_path = run_dir / "figures" / "final_test_confusion_matrix.png"
    write_json(metrics_path, metrics)
    plot_confusion_matrix(metrics["confusion_matrix"], figure_path, "Final Test Confusion Matrix")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved best checkpoints on the 2013 test set.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--run-name", default=None, help="Evaluate one run directory by name.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)
    device = get_device()
    if args.run_name:
        run_dirs = [runs_dir / args.run_name]
    else:
        run_dirs = sorted(path for path in runs_dir.glob("*") if path.is_dir() and not path.name.startswith("debug_"))

    print(f"device: {device}")
    evaluated = 0
    for run_dir in run_dirs:
        metrics = evaluate_run(run_dir, data_dir, args.batch_size, args.num_workers, device)
        if metrics is None:
            continue
        evaluated += 1
        print(
            f"{run_dir.name}: test_acc={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"confusion={metrics['confusion_matrix']}"
        )
    print(f"evaluated runs: {evaluated}")


if __name__ == "__main__":
    main()
