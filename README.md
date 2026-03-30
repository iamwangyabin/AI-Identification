# AI-Identification

多分类 AI 图像来源识别训练代码。当前实现使用冻结的 DINOv3 ConvNeXt Tiny 作为通用特征骨干，从中高层提取多尺度特征，并通过轻量局部分支与全局分支分别建模伪影线索和生成风格信息，最终完成 37 类分类。

## 1. 数据说明

仓库当前使用以下文件作为数据清单与类别映射：

- `train.csv`
- `test.csv`
- `class_map.json`

CSV 每行字段如下：

- `path`: 相对 `data_root` 的相对路径
- `label`: 类别 id

当前清单采用精简格式，只保留训练必需字段；如果需要类别名，请通过 `class_map.json` 中的 `id_to_class` 反查。

这里的任务不是简单真假二分类，而是多分类来源识别。`All_Real` 表示真实图像类，其余类别主要是不同生成模型或伪造来源，`traditional` 表示传统篡改类。

当前 `class_map.json` 已经根据现有训练集和测试集清单做过修正，与实际样本分布一致。

## 2. 模型结构

主模型定义在 `src/model.py`，核心类为 `ConvNeXtForgeryClassifier`。

### Backbone

- 使用 `timm` 的 `convnext_tiny.dinov3_lvd1689m`
- `features_only=True`
- 默认提取 `out_indices=(1, 2, 3)` 三个中高层特征
- backbone 全冻结，不参与反向传播

默认多尺度特征尺寸为：

- stage 2: `[B, 192, 28, 28]`
- stage 3: `[B, 384, 14, 14]`
- stage 4: `[B, 768, 7, 7]`

### 多尺度特征融合

每层特征先经过：

- `1x1 Conv`
- `BatchNorm2d`
- `GELU`

统一对齐到 `align_dim=192`。

### 局部分支

局部分支 `LocalArtifactBranch` 负责建模局部伪影线索：

- 先用 `feature - avg_pool(feature)` 提取高频残差
- 再过轻量 depthwise + pointwise 卷积模块
- 最后做全局平均池化与最大池化并拼接
- 输出 `local_dim=128`

### 全局分支

全局分支 `GlobalStyleBranch` 负责建模生成风格信息：

- 先做全局平均池化
- 经 MLP 生成通道门控
- 回乘到特征图后再池化
- 输出 `global_dim=128`

### 分类头

每一层都会保留一个基础池化特征：

- pooled feature: `192`
- local feature: `128`
- global feature: `128`

每层共 `448` 维，3 层拼接后得到 `1344` 维融合表示，最终进入：

- `LayerNorm`
- `Linear(1344 -> 512)`
- `GELU`
- `Dropout`
- `Linear(512 -> num_classes)`

## 3. 训练与评测增强

增强定义在 `src/augment.py`，当前实现的是“随机后处理扰动池”，不是每次把所有扰动都叠加。

### 训练阶段

训练时先按 `--postprocess-prob` 判断是否触发增强，触发后从扰动池中随机抽取 `1` 到 `--postprocess-max-ops` 个操作。

### 测试阶段

测试时也支持同类随机后处理，参数独立：

- `--eval-postprocess-prob`
- `--eval-postprocess-max-ops`

这更贴近真实传播场景，但会让评测结果带随机波动。如果要固定 benchmark，可将 `--eval-postprocess-prob 0.0`。

### 当前后处理扰动池

- JPEG 压缩
- WebP 压缩
- Gaussian blur
- Median blur
- 随机裁剪再缩放
- 随机降采样再恢复
- 多插值重采样恢复
- 锐化
- 亮度扰动
- 对比度扰动
- 饱和度扰动
- Gamma 调整
- Gaussian noise

## 4. 代码结构

```text
.
├── README.md
├── class_map.json
├── test.csv
├── train.csv
├── train.py
└── src
    ├── __init__.py
    ├── augment.py
    ├── data.py
    └── model.py
```

模块职责如下：

- `src/data.py`: CSV/JSONL 数据集读取与路径拼接
- `src/model.py`: 冻结多尺度 ConvNeXt 伪造识别模型
- `src/augment.py`: 随机后处理扰动
- `train.py`: 训练、验证、checkpoint、CLI

## 5. 运行方式

建议环境：

- Python 3.13+
- PyTorch
- torchvision
- timm 1.0.26+
- Pillow
- numpy
- swanlab

### 查看参数

```bash
python3 train.py --help
```

### 基本训练

```bash
python3 train.py \
  --data-root /your/dataset/root \
  --batch-size 64 \
  --epochs 20 \
  --amp
```

### 使用 SwanLab 记录

先安装：

```bash
python3 -m pip install swanlab
```

再启动训练：

```bash
python3 train.py \
  --data-root /your/dataset/root \
  --use-swanlab \
  --swanlab-project AI-Identification \
  --swanlab-experiment-name convnext-tiny-dinov3-baseline
```

当前会记录：

- 训练过程中的 batch loss、top1、top5、学习率
- 每个 epoch 的 train/val loss、top1、top5
- best top1
- 模型总参数量与可训练参数量

推荐直接通过 `--data-root` 提供数据集根目录，清单里的 `path` 会自动拼接到该目录下。

### 关闭测试增强

```bash
python3 train.py \
  --data-root /your/dataset/root \
  --eval-postprocess-prob 0.0
```

### 只做验证

```bash
python3 train.py \
  --data-root /your/dataset/root \
  --resume outputs/best.pt \
  --eval-only
```

## 6. 当前实现特点

- backbone 冻结，训练成本低
- 使用多尺度特征，补足单层高层语义对细粒度伪影刻画不足的问题
- 局部与全局双分支分别关注伪影和风格
- 训练与测试都支持随机后处理扰动，更接近真实传播链路

## 7. 注意事项

- 当前测试增强是随机的，因此多次评测结果可能略有波动
- 如果你需要严格可复现的评测，请固定随机种子并关闭测试增强
- 如果你需要“标准测试”和“鲁棒测试”两套结果，建议分别运行一次
