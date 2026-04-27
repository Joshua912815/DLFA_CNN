# DL Finance Project 1 Report Draft

## 1. Task and Data

- Task: use CNN models on price-trend images to predict whether the future 5-day stock return is positive or negative.
- Data split:
  - Train source: `project1/image/train`, covering 2010-2012.
  - Validation: last 20% of the train period by image date, used for model selection and hyperparameter tuning.
  - Test source: `project1/image/test`, 2013 out-of-sample period.
- Labels: binary classes `0` and `1`.
- Preprocessing:
  - PNG images are converted from RGBA to RGB.
  - Default normalization is ImageNet mean/std.
  - Advanced attempts also test dataset-estimated normalization and light pixel augmentation.
- Test majority-class baseline:
  - Test class counts: class 0 = 20,814, class 1 = 22,067.
  - Majority-class accuracy = 51.46%.

## 2. Models and Improvement Attempts

- Paper baselines from Figure 3:
  - `cnn5`: 5-day image, 32x15 input, channels 64 -> 128.
  - `cnn20`: 20-day image, 64x60 input, channels 64 -> 128 -> 256.
  - `cnn60`: 60-day image, 96x180 input, channels 64 -> 128 -> 256 -> 512.
- Custom CNN:
  - `res_se_cnn`: residual blocks, GroupNorm, SE channel attention, Dropout, global average pooling.
  - Legacy first attempt used BatchNorm and was unstable; retained as a reported failed/less stable attempt.
- Advanced CNN / ResNet attempts:
  - `resnet18_scratch`: standard torchvision ResNet18 from scratch.
  - `chart_resnet18_gn`: price-chart ResNet18 with 5x3 stride-1 stem, no initial max-pool, GroupNorm.
  - `chart_resnet18_se`: `chart_resnet18_gn` plus SE attention.
- Hyperparameter and preprocessing attempts:
  - AdamW, lower LR `3e-4`, cosine scheduler, weight decay `1e-4`.
  - Label smoothing `0.03`, gradient clipping `1.0`, early stopping.
  - Dataset normalization.
  - Light augmentation: brightness/contrast jitter and random erasing.
  - No-label-smoothing ablation.

## 3. Results

All model choices were based on validation performance. Test metrics below are out-of-sample 2013 evaluations of saved best-validation checkpoints.

| Attempt | Run | Val Acc | Test Acc | Test Prec | Test Rec | Test Confusion [[TN,FP],[FN,TP]] |
|---|---|---:|---:|---:|---:|---|
| chart_resnet18_se | `advanced_resnet_stable_chart_resnet18_se` | 66.31% | 58.19% | 59.25% | 60.09% | `[[11694, 9120], [8807, 13260]]` |
| res_se_cnn | `server_res_se_cnn_stable_20ep_res_se_cnn` | 65.17% | 57.72% | 59.65% | 55.10% | `[[12589, 8225], [9907, 12160]]` |
| chart_resnet18_gn + light aug | `advanced_chart_resnet18_gn_lightaug_chart_resnet18_gn` | 64.89% | 58.16% | 58.60% | 63.69% | `[[10886, 9928], [8013, 14054]]` |
| chart_resnet18_gn + dataset norm | `advanced_chart_resnet18_gn_datasetnorm_chart_resnet18_gn` | 64.33% | 58.23% | 59.72% | 57.87% | `[[12200, 8614], [9296, 12771]]` |
| cnn20 | `server_paper_stable_20ep_cnn20` | 64.04% | 56.90% | 58.72% | 54.67% | `[[12335, 8479], [10004, 12063]]` |
| cnn60 | `server_paper_stable_20ep_cnn60` | 63.81% | 57.27% | 58.94% | 55.90% | `[[12222, 8592], [9731, 12336]]` |
| chart_resnet18_gn | `advanced_resnet_stable_chart_resnet18_gn` | 63.09% | 59.08% | 59.75% | 62.77% | `[[11482, 9332], [8216, 13851]]` |
| resnet18_scratch | `advanced_resnet_stable_resnet18_scratch` | 62.48% | 54.01% | 61.16% | 29.10% | `[[16737, 4077], [15646, 6421]]` |
| res_se_cnn legacy BatchNorm | `server_res_se_cnn_10ep_res_se_cnn` | 62.37% | 57.88% | 58.20% | 64.40% | `[[10607, 10207], [7855, 14212]]` |
| chart_resnet18_gn, no smoothing | `advanced_chart_resnet18_gn_no_smoothing_chart_resnet18_gn` | 62.26% | 59.03% | 59.88% | 61.78% | `[[11679, 9135], [8433, 13634]]` |
| cnn5 | `server_paper_stable_20ep_cnn5` | 61.90% | 52.02% | 53.54% | 51.20% | `[[11009, 9805], [10768, 11299]]` |

Primary final model by validation selection:

- `chart_resnet18_se`
- Validation accuracy: 66.31%.
- Out-of-sample test accuracy: 58.19%.
- Test precision: 59.25%.
- Test recall: 60.09%.

Best observed test accuracy among all reported attempts:

- `chart_resnet18_gn`: 59.08%.
- This is reported as diagnostic analysis, not as the model-selection rule.

## 4. Analysis of Model Structures and Hyperparameters

- Paper baselines:
  - `cnn5` is weakest; short-window images likely lose too much trend context.
  - `cnn20` and `cnn60` are stronger and similar; deeper/full-window image helps, but not dramatically.
  - `cnn60` does not dominate `cnn20`, suggesting more spatial context can also increase overfitting/noise.
- Standard ResNet:
  - `resnet18_scratch` is weak on validation and test.
  - It has high precision but very low recall on test, meaning it predicts positive labels too conservatively.
  - Likely cause: standard ResNet stem and max-pool downsample sparse price-chart lines too aggressively.
- Chart-adapted ResNet:
  - `chart_resnet18_gn` improves over standard ResNet by preserving early spatial detail.
  - `chart_resnet18_se` gives the best validation score, showing channel attention helps learn useful price-chart features.
  - However, `chart_resnet18_se` has lower test accuracy than `chart_resnet18_gn`, indicating some validation overfitting.
- Custom `res_se_cnn`:
  - Stable GroupNorm version improved validation over paper baselines.
  - Test accuracy is close to the advanced models but not best.
  - The earlier BatchNorm version was unstable; GroupNorm is better for this time-split setting.
- Hyperparameters:
  - AdamW + cosine scheduler + lower LR stabilizes training compared with earlier runs.
  - Label smoothing improves validation for `chart_resnet18_gn` from 62.26% to 63.09%.
  - No-smoothing has slightly higher test accuracy but worse validation, so it should not be selected using test results.
  - Light augmentation starts poorly but reaches 64.89% validation; it may regularize later training but can also damage thin chart lines.
  - Dataset normalization improves `chart_resnet18_gn` validation from 63.09% to 64.33%, but not enough to beat SE attention.
- Validation-test gap:
  - Validation accuracies are around 62-66%, but test accuracies are around 52-59%.
  - This suggests regime shift from 2010-2012 to 2013 and weak/noisy financial prediction signal.
  - Models with highest validation are not always highest test, so test must not be used for model selection.

## 5. Pros and Cons of CNN Image-Based Stock Prediction

- Pros:
  - CNNs can learn local visual patterns such as trend shape, volatility clusters, and moving-average interactions.
  - Image representation can encode multiple technical indicators without manually designing tabular features.
  - Convolutions share parameters spatially, which helps learn repeated chart patterns across stocks and dates.
  - Advanced CNN blocks such as residual connections and SE attention can improve feature learning.
- Cons:
  - Stock return predictability is weak; labels are noisy and close to random.
  - Image conversion may discard exact numeric price/volume information.
  - CNNs can overfit visual artifacts rather than robust economic signals.
  - Standard computer-vision architectures may be inappropriate because chart images are sparse and time-directional.
  - Data augmentation must be conservative; flips and rotations are invalid for time-series charts.
  - Validation performance may not transfer to a future market regime.

## 6. Conclusion

- Best validation-selected model: `chart_resnet18_se`.
- Final out-of-sample test accuracy for this model: 58.19%.
- Best observed out-of-sample test accuracy among all attempts: 59.08% from `chart_resnet18_gn`.
- The main successful improvements were:
  - Replacing standard ResNet stem with chart-specific stem.
  - Adding SE attention.
  - Using AdamW/cosine/label smoothing/gradient clipping.
  - Trying dataset normalization and conservative augmentation.
- The main unsuccessful or weaker attempts were:
  - Standard ResNet18 from scratch.
  - Very short 5-day image baseline.
  - Removing label smoothing.
  - Early BatchNorm custom CNN due unstable validation behavior.
