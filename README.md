# DLFA CNN Project 1

This repo implements the three CNN baseline models from `DL_Finance_Project_1.pdf` for binary stock return prediction from price trend images.

## Environment

The code selects device in this order:

```text
cuda -> mps -> cpu
```

So server training will use CUDA automatically when available.

## Data Paths

The script uses the actual project data path:

```text
project1/image/train
project1/image/test
```

Images are loaded from class folders `0` and `1`. PNG images are converted from `RGBA` to `RGB` before normalization.

The 2013 test folder is reserved for final out-of-sample evaluation only. Do not use test accuracy for model selection or hyperparameter tuning.

## Baseline Models

`main.py` provides the three CNNs from Figure 3:

```text
cnn5   5-day image baseline,  input 32x15,  conv channels 64 -> 128
cnn20  20-day image baseline, input 64x60,  conv channels 64 -> 128 -> 256
cnn60  60-day image baseline, input 96x180, conv channels 64 -> 128 -> 256 -> 512
```

Each block is:

```text
5x3 Conv2d -> LeakyReLU -> 2x1 MaxPool
```

The final classifier uses logits for `CrossEntropyLoss`; no explicit softmax is needed during training.

## Smoke Test

Run this before training to verify dataset paths and model forward shapes:

```bash

python main.py --model all --epochs 0 --batch-size 4 --num-workers 0
```

## Training

Train one baseline:

```bash

python main.py \
  --model cnn60 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --val-split time \
  --val-ratio 0.2 \
  --run-name cnn60_baseline
```

Train all three baselines sequentially:

```bash

python main.py \
  --model all \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --val-split time \
  --val-ratio 0.2 \
  --run-name paper_baselines
```

Debug the full output pipeline on a tiny balanced subset:

```bash

python main.py \
  --model cnn5 \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --limit-train-samples 32 \
  --limit-val-samples 16 \
  --run-name debug_output
```

Only after the final model and hyperparameters are fixed, run test evaluation once:

```bash

python main.py \
  --model cnn60 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --val-split time \
  --val-ratio 0.2 \
  --run-name cnn60_final \
  --eval-test
```

## Output Management

Each training run writes to:

```text
runs/<run-name>_<model>/
```

The run folder contains:

```text
config.json
split_summary.json
checkpoints/best.pt
checkpoints/latest.pt
metrics/metrics.csv
metrics/best_validation_metrics.json
metrics/final_test_metrics.json          # only when --eval-test is used
figures/training_curves.png
figures/best_validation_confusion_matrix.png
figures/final_test_confusion_matrix.png  # only when --eval-test is used
```

`runs/` is ignored by git because it may contain large model checkpoints and generated figures.

## Collaboration Rules

- Use validation metrics for tuning learning rate, batch size, epoch count, and model choice.
- Keep 2013 test metrics out of tuning discussions until the final model is fixed.
- When sharing results, include the full run folder name and `config.json` settings.
- Commit source code and docs, not generated checkpoints or plots.
