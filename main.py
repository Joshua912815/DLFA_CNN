import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("project1/image")
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    image_size: tuple[int, int]  # (height, width)
    architecture: str
    channels: tuple[int, ...]
    paper_fc_features: Optional[int] = None
    description: str = ""


MODEL_CONFIGS = {
    "cnn5": ModelConfig(
        "cnn5",
        image_size=(32, 15),
        architecture="paper",
        channels=(64, 128),
        paper_fc_features=15360,
        description="5-day CNN baseline from Figure 3.",
    ),
    "cnn20": ModelConfig(
        "cnn20",
        image_size=(64, 60),
        architecture="paper",
        channels=(64, 128, 256),
        paper_fc_features=46080,
        description="20-day CNN baseline from Figure 3.",
    ),
    "cnn60": ModelConfig(
        "cnn60",
        image_size=(96, 180),
        architecture="paper",
        channels=(64, 128, 256, 512),
        paper_fc_features=184320,
        description="60-day CNN baseline from Figure 3.",
    ),
    "res_se_cnn": ModelConfig(
        "res_se_cnn",
        image_size=(96, 180),
        architecture="residual_se",
        channels=(32, 64, 128, 256),
        description="Stronger custom residual CNN with BatchNorm, SE attention, dropout, and global pooling.",
    ),
}

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_precision",
    "val_recall",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_sample_date(path: Path) -> str:
    match = re.search(r"-(\d{8})$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse date from image filename: {path}")
    return match.group(1)


def list_image_samples(root: Path) -> list[tuple[Path, int, str]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[tuple[Path, int, str]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = int(class_dir.name)
        for image_path in sorted(class_dir.glob("*.png")):
            samples.append((image_path, label, parse_sample_date(image_path)))
    if not samples:
        raise RuntimeError(f"No PNG images found under {root}")
    return samples


def split_train_val(
    samples: list[tuple[Path, int, str]],
    val_ratio: float,
    split: str,
    seed: int,
) -> tuple[list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    if split == "time":
        ordered = sorted(samples, key=lambda item: (item[2], str(item[0])))
        val_count = max(1, int(round(len(ordered) * val_ratio)))
        return ordered[:-val_count], ordered[-val_count:]

    if split == "stratified":
        rng = random.Random(seed)
        train_samples: list[tuple[Path, int, str]] = []
        val_samples: list[tuple[Path, int, str]] = []
        labels = sorted({label for _, label, _ in samples})
        for label in labels:
            class_samples = [sample for sample in samples if sample[1] == label]
            rng.shuffle(class_samples)
            val_count = max(1, int(round(len(class_samples) * val_ratio)))
            val_samples.extend(class_samples[:val_count])
            train_samples.extend(class_samples[val_count:])
        return train_samples, val_samples

    raise ValueError("--val-split must be 'time' or 'stratified'")


def class_counts(samples: list[tuple[Path, int, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, label, _ in samples:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def date_range(samples: list[tuple[Path, int, str]]) -> dict[str, object]:
    if not samples:
        return {"start": None, "end": None}
    dates = sorted(sample[2] for sample in samples)
    return {"start": dates[0], "end": dates[-1]}


def limit_samples(samples: list[tuple[Path, int, str]], limit: Optional[int]) -> list[tuple[Path, int, str]]:
    if limit is None or limit <= 0 or limit >= len(samples):
        return samples

    labels = sorted({sample[1] for sample in samples})
    if not labels:
        return samples

    per_label = max(1, limit // len(labels))
    remainder = limit - per_label * len(labels)
    selected: list[tuple[Path, int, str]] = []
    for label in labels:
        label_samples = [sample for sample in samples if sample[1] == label]
        take = per_label + (1 if remainder > 0 else 0)
        selected.extend(label_samples[:take])
        remainder -= 1
    return sorted(selected[:limit], key=lambda item: (item[2], str(item[0])))


class PriceTrendImageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int, str]],
        image_size: tuple[int, int],
        mean: tuple[float, float, float] = IMAGE_NET_MEAN,
        std: tuple[float, float, float] = IMAGE_NET_STD,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label, _ = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size[1], self.image_size[0]), _bilinear_resample())
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(label, dtype=torch.long)


def _bilinear_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


class PaperCNN(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int = 2) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in config.channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(5, 3),
                        padding=(2, 1),
                    ),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ConvBNAct(nn.Sequential):
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


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
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


class ResidualSEPriceCNN(nn.Module):
    """Custom model for full 96x180 price-trend images."""

    def __init__(self, config: ModelConfig, num_classes: int = 2) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2, c3 = config.channels
        self.features = nn.Sequential(
            ConvBNAct(3, c0, kernel_size=(5, 3), padding=(2, 1)),
            ResidualSEBlock(c0, c1, stride=(2, 2)),
            ResidualSEBlock(c1, c1),
            ResidualSEBlock(c1, c2, stride=(2, 2)),
            ResidualSEBlock(c2, c2),
            ResidualSEBlock(c2, c3, stride=(2, 2)),
            ResidualSEBlock(c3, c3),
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


def build_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name}; choose from {sorted(MODEL_CONFIGS)}")
    config = MODEL_CONFIGS[model_name]
    if config.architecture == "paper":
        return PaperCNN(config)
    if config.architecture == "residual_se":
        return ResidualSEPriceCNN(config)
    raise ValueError(f"Unknown architecture for {model_name}: {config.architecture}")


def model_names_from_arg(model_arg: str) -> list[str]:
    if model_arg == "paper":
        return [name for name, config in MODEL_CONFIGS.items() if config.architecture == "paper"]
    return list(MODEL_CONFIGS) if model_arg == "all" else [model_arg]


def model_config_payload(config: ModelConfig, actual_features: int) -> dict[str, object]:
    return {
        "name": config.name,
        "architecture": config.architecture,
        "image_size": list(config.image_size),
        "channels": list(config.channels),
        "feature_map_elements_before_classifier": actual_features,
        "paper_fc_label": config.paper_fc_features,
        "description": config.description,
    }


def make_dataloader(
    samples: list[tuple[Path, int, str]],
    config: ModelConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PriceTrendImageDataset(samples=samples, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(2, 2, dtype=torch.long)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        for actual, predicted in zip(labels.cpu(), predictions.cpu()):
            confusion[actual.long(), predicted.long()] += 1

    tn = confusion[0, 0].item()
    fp = confusion[0, 1].item()
    fn = confusion[1, 0].item()
    tp = confusion[1, 1].item()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion.tolist(),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_training_curves(log_path: Path, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, float]] = []
    with log_path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})

    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in rows], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix: list[list[int]], output_path: Path, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib-cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_run_dir(args: argparse.Namespace, model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.run_name:
        name = sanitize_name(f"{args.run_name}_{model_name}")
    else:
        name = sanitize_name(
            f"{timestamp}_{model_name}_lr{args.lr}_bs{args.batch_size}_split{args.val_split}_vr{args.val_ratio}"
        )
    base_dir = Path(args.output_dir) / name
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = Path(args.output_dir) / f"{name}_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def initialize_lazy_layers(model: nn.Module, config: ModelConfig, device: torch.device) -> int:
    model.eval()
    dummy = torch.zeros(1, 3, config.image_size[0], config.image_size[1], device=device)
    with torch.no_grad():
        features = model.features(dummy)
        _ = model(dummy)
    return int(features.numel())


def smoke_check(args: argparse.Namespace, device: torch.device) -> None:
    train_samples = list_image_samples(Path(args.data_dir) / "train")
    test_samples = list_image_samples(Path(args.data_dir) / "test")
    train_split, val_split = split_train_val(train_samples, args.val_ratio, args.val_split, args.seed)
    train_split = limit_samples(train_split, args.limit_train_samples)
    val_split = limit_samples(val_split, args.limit_val_samples)
    test_samples = limit_samples(test_samples, args.limit_test_samples)

    print(f"device: {device}")
    print(f"train samples after split: {len(train_split)}")
    print(f"validation samples: {len(val_split)}")
    print(f"test samples: {len(test_samples)}")

    model_names = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for model_name in model_names:
        config = MODEL_CONFIGS[model_name]
        model = build_model(model_name).to(device)
        actual_features = initialize_lazy_layers(model, config, device)
        batch = next(
            iter(
                make_dataloader(
                    train_split,
                    config,
                    batch_size=min(args.batch_size, 4),
                    shuffle=False,
                    num_workers=0,
                    device=device,
                )
            )
        )
        images, labels = batch
        logits = model(images.to(device))
        print(
            f"{model_name}: input={tuple(images.shape)} logits={tuple(logits.shape)} "
            f"feature_map_elements={actual_features} paper_fc_label={config.paper_fc_features} "
            f"labels={labels.tolist()}"
        )


def train(args: argparse.Namespace, device: torch.device, model_name: str) -> None:
    config = MODEL_CONFIGS[model_name]
    train_samples = list_image_samples(Path(args.data_dir) / "train")
    test_samples = list_image_samples(Path(args.data_dir) / "test")
    train_split, val_split = split_train_val(train_samples, args.val_ratio, args.val_split, args.seed)
    train_split = limit_samples(train_split, args.limit_train_samples)
    val_split = limit_samples(val_split, args.limit_val_samples)
    test_samples = limit_samples(test_samples, args.limit_test_samples)

    train_loader = make_dataloader(train_split, config, args.batch_size, True, args.num_workers, device)
    val_loader = make_dataloader(val_split, config, args.batch_size, False, args.num_workers, device)
    test_loader = make_dataloader(test_samples, config, args.batch_size, False, args.num_workers, device)

    model = build_model(model_name).to(device)
    actual_features = initialize_lazy_layers(model, config, device)
    run_dir = make_run_dir(args, model_name)
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"model: {model_name}")
    print(f"image_size: {config.image_size}")
    print(f"feature_map_elements: {actual_features}; paper_fc_label: {config.paper_fc_features}")
    print(f"train/val/test: {len(train_split)}/{len(val_split)}/{len(test_samples)}")
    print(f"run_dir: {run_dir}")

    write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "model": model_name,
            "device": str(device),
            "model_config": model_config_payload(config, actual_features),
        },
    )
    write_json(
        run_dir / "split_summary.json",
        {
            "split_method": args.val_split,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "train": {
                "size": len(train_split),
                "class_counts": class_counts(train_split),
                "date_range": date_range(train_split),
            },
            "validation": {
                "size": len(val_split),
                "class_counts": class_counts(val_split),
                "date_range": date_range(val_split),
            },
            "test": {
                "size": len(test_samples),
                "class_counts": class_counts(test_samples),
                "date_range": date_range(test_samples),
                "note": "2013 out-of-sample set; do not use for model selection.",
            },
        },
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = metrics_dir / "metrics.csv"
    checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    best_val_accuracy = -1.0
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
            writer.writerow(row)
            handle.flush()

            print(
                f"epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f}"
            )

            if float(val_metrics["accuracy"]) > best_val_accuracy:
                best_val_accuracy = float(val_metrics["accuracy"])
                torch.save(
                    {
                        "model": model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "config": model_config_payload(config, actual_features),
                    },
                    checkpoint_path,
                )
                write_json(metrics_dir / "best_validation_metrics.json", {"epoch": epoch, **val_metrics})

            torch.save(
                {
                    "model": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": model_config_payload(config, actual_features),
                },
                latest_checkpoint_path,
            )

    plot_training_curves(log_path, figures_dir / "training_curves.png")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation_metrics = evaluate(model, val_loader, criterion, device)
    write_json(
        metrics_dir / "best_validation_metrics.json",
        {"best_epoch": checkpoint["epoch"], **best_validation_metrics},
    )
    plot_confusion_matrix(
        best_validation_metrics["confusion_matrix"],
        figures_dir / "best_validation_confusion_matrix.png",
        "Best Validation Confusion Matrix",
    )

    print(f"best validation accuracy: {best_val_accuracy:.4f}")
    print(f"best checkpoint: {checkpoint_path}")
    print(f"training log: {log_path}")
    print(f"training curves: {figures_dir / 'training_curves.png'}")

    if args.eval_test:
        test_metrics = evaluate(model, test_loader, criterion, device)
        write_json(metrics_dir / "final_test_metrics.json", test_metrics)
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            figures_dir / "final_test_confusion_matrix.png",
            "Final Test Confusion Matrix",
        )
        print(
            "final test: "
            f"loss={test_metrics['loss']:.4f} accuracy={test_metrics['accuracy']:.4f} "
            f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
            f"confusion_matrix={test_metrics['confusion_matrix']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN baselines for DL Finance Project 1.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path containing image/train and image/test folders.")
    parser.add_argument("--model", choices=["paper", "all", *MODEL_CONFIGS.keys()], default="cnn60")
    parser.add_argument("--epochs", type=int, default=0, help="Use 0 for a dataset/model smoke check only.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-split", choices=["time", "stratified"], default="time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None, help="Optional run name; model name is appended automatically.")
    parser.add_argument("--limit-train-samples", type=int, default=None, help="Debug only: cap training samples.")
    parser.add_argument("--limit-val-samples", type=int, default=None, help="Debug only: cap validation samples.")
    parser.add_argument("--limit-test-samples", type=int, default=None, help="Debug only: cap test samples.")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate 2013 test set once after training finishes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.epochs == 0:
        smoke_check(args, device)
        return

    for model_name in model_names_from_arg(args.model):
        train(args, device, model_name)


if __name__ == "__main__":
    main()
