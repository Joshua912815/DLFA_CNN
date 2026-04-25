# DL Finance Project 1 TODO List

## 1. 搞清数据与运行环境

- 确认数据目录：当前实际路径是 `project1/image/train` 和 `project1/image/test`，不是 starter code 里的 `./data/image/...`。
- 统计数据规模：
  - train/0: 50,098
  - train/1: 49,828
  - test/0: 20,814
  - test/1: 22,067
- 处理图像通道问题：图片是 `RGBA`，而 starter code 的 Normalize 只给了 3 通道参数；需要统一转成 `RGB` 或改成 4 通道处理。
- 检查运行环境：优先使用服务器/本地 `mcts` conda 环境；代码按 `cuda -> mps -> cpu` 自动选择设备。
- 维护 `requirements.txt`，便于团队在新环境复现实验依赖。

## 2. 实现可运行 Baseline

- 修正数据加载代码：
  - 使用正确数据路径。
  - 用 `ImageFolder` 加载 `train/test`。
  - 从 `project1/image/train` 中固定切出 validation set，用于调参和选择最佳模型。
  - 优先按时间序列切分 validation set：从文件名日期中解析样本日期，将 2010-2012 训练样本的最后一段时间作为 validation；如实现成本过高，再使用固定随机种子的 8:2 stratified split。
  - 保持 `project1/image/test` 作为 2013 年 out-of-sample test set，不参与训练、调参或 early stopping。
  - 加入 `transforms.Resize` 如有需要，但默认应保持原图 `180x96`。
  - 加入 `transforms.Lambda(lambda img: img.convert("RGB"))` 或自定义 loader。
- 实现至少一个 CNN baseline：
  - Conv + LeakyReLU + MaxPool 堆叠。
  - Flatten + Fully Connected 输出 2 类。
  - Loss 使用 `CrossEntropyLoss`。
  - Optimizer 可先用 `Adam`。
- 完成训练、测试、保存结果：
  - 输出 train loss / train accuracy。
  - 输出 validation loss / validation accuracy。
  - 根据 validation accuracy 或 validation loss 保存最佳模型权重和实验日志。
  - 每次实验保存 `config.json`、`split_summary.json`、`metrics.csv`、最佳/最新 checkpoint、训练曲线和混淆矩阵到独立 run 目录。
  - 在所有模型与超参数确定后，最后只对最佳模型运行一次 2013 test set 评估。

## 3. 复现或参考论文 CNN 结构

- 参考 PDF Figure 3 的 CNN：
  - 5x3 convolution。
  - channel 数可按 64、128、256、512 递增。
  - LeakyReLU。
  - 2x1 MaxPool。
  - 最后 FC + Softmax / logits。
- 因本项目图像是 64-day price trend image，对应 Figure 3 里较深的 60-day image 网络最相关。
- 不强行照抄 FC 输入维度，实际实现中用 dummy forward 或 adaptive pooling 自动推断尺寸，避免 shape 错误。

## 4. 做实验对比

- 至少跑通这些实验：
  - PDF Figure 3 的 `cnn5`、`cnn20`、`cnn60` 三个 baseline。
  - 自定义增强模型 `res_se_cnn`：使用 96x180 全图输入，包含 BatchNorm、残差连接、SE channel attention、Dropout 和 global average pooling。
  - 调整 learning rate。
  - 调整 batch size。
  - 调整 weight decay 和 epoch 数。
- 建议记录每组实验：
  - 模型结构。
  - optimizer。
  - learning rate。
  - batch size。
  - epoch 数。
  - train accuracy。
  - validation accuracy。
  - validation precision / recall / confusion matrix。
- 禁止用 2013 年 test accuracy 反复挑选模型或超参数，避免 look-ahead bias 和数据泄露。
- 最终模型确定后，只记录一次 test accuracy，并在报告中明确说明 test set 没有参与调参。
- 原论文给出的参考精度大约在 50% 到 54% 区间；报告中不要只追求很高 accuracy，要解释金融预测任务本身信号弱。

## 5. 写报告 PDF

- 报告至少 2 页。
- 必须包含：
  - 任务说明：用价格趋势图预测未来 5 日收益正负。
  - 数据说明：2010-2012 训练，2013 测试，二分类标签。
  - 模型结构与超参数。
  - out-of-sample test accuracy。
  - train / validation / test 的划分方式，并说明 validation 用于调参、2013 test 只用于最终评估。
  - 至少一张结果表。
  - confusion matrix / precision / recall 等辅助分析。
  - 不同模型结构、超参数对结果的影响。
  - CNN 图像方法预测股票收益的优缺点。
  - 所有尝试，包括失败或效果不好的尝试。
  - 如借鉴开源代码，明确说明来源。

## 6. 最终提交物

- 全部代码。
- 可复现实验的运行说明。
- 训练/测试结果记录。
- validation 调参记录和最终 test 结果记录。
- 至少 2 页 PDF 报告。
- 截止时间：2026-05-06 22:00。

## Assumptions

- 以 `project1/image/train` 作为训练集，以 `project1/image/test` 作为最终 out-of-sample 测试集。
- 必须从 `project1/image/train` 中切出 validation set 进行调参；默认采用按时间切分，备选方案是固定随机种子的 8:2 stratified split。
- `project1/image/test` 只在最终模型确定后评估一次，不能用于模型选择或超参数搜索。
- 最终报告的核心指标是 test classification accuracy，其他指标用于支持分析。
