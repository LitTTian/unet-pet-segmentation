# Oxford Pet 二分类语义分割（TransUNet/UNet）

基于 PyTorch 实现的牛津宠物（Oxford-IIIT Pet Dataset）二分类语义分割项目，支持 UNet、TransUNet 等模型，针对宠物/背景分割场景优化，兼顾精度、速度与可重复性。

## 项目简介

本项目专注于 **宠物二分类语义分割**（将图像中的宠物与背景分离），基于经典分割架构 UNet 和 TransUNet（CNN+Transformer 混合架构），适配 Oxford-IIIT Pet 数据集的特点：
- 数据集包含 37 种宠物，共 7349 张图像，每张图像均有对应的像素级掩码（trimap）；
- 任务目标：输出二值掩码（宠物=1，背景=0），核心评价指标为 IoU（交并比）；

### 核心特性
- 多模型对比实验：UNet、TransUNet（可扩展 UNet++/Attention UNet）；
- 优化训练策略：损失（BCE）、余弦退火调度、早停、梯度裁剪；
- 高可重复性：固定全流程随机种子、数据加载确定性优化；
- 灵活配置：命令行参数控制模型、训练参数、路径等；
- 可视化支持：WandB 实时监控训练过程（loss、IoU、Dice、学习率）；

## 环境依赖

- 使用`conda`可以快速创建隔离环境：
```bash
conda create -n oxford-pet-segmentation python=3.10
pip install -r requirements.txt
```

## 数据集准备

### 数据集下载
从 [Oxford-IIIT Pet Dataset 官网](https://www.robots.ox.ac.uk/~vgg/data/pets/) 下载以下文件：
1. 图像文件：`images.tar.gz`（解压后为宠物图像）；
2. 掩码文件：`annotations.tar.gz`（解压后包含 `trimaps/` 文件夹，为像素级掩码）；
3. 划分文件：`annotations.tar.gz` 中包含 `trainval.txt` 和 `test.txt`（训练/验证/测试集划分）。

### 数据集目录结构
```
datasets/
└── Oxford-IIIT_Pet_Dataset/
    ├── images/                # 所有宠物图像（.jpg）
    ├── annotations/
    │   ├── trimaps/           # 像素级掩码（.png）
    │   ├── trainval.txt       # 训练+验证集划分
    │   └── test.txt           # 测试集划分
    └── ...
```

### 掩码预处理
数据集原始掩码为三值（1=宠物，2=背景，3=边界），项目中已内置处理逻辑：自动转换为二值掩码（1=宠物，0=背景），无需手动处理。

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/LitTTian/unet-pet-segmentation
cd unet-pet-segmentation
```

### 2. 配置参数（可选）
通过命令行参数配置训练细节，支持的核心参数如下（完整参数见 `--help`）：
```bash
# 核心参数说明
--model: 模型类型（unet/transunet/transunetp，默认 transunet）
--batch-size: 批次大小（默认 24）
--lr: 初始学习率（默认 1e-4）
--epochs: 训练轮数（默认 200）
--img-size: 输入图像尺寸（默认 (256, 256)）
--loss-function: 损失函数（默认 BCEWithLogitsLoss）
--seed: 随机种子（默认 42）
--num-workers: 数据加载进程数（默认 1）
--image-dir: 图像文件夹路径（默认数据集路径）
--mask-dir: 掩码文件夹路径（默认数据集路径）
```

### 3. 启动训练
```bash
python train.py --model unet --batch-size 24 --lr 1e-4
```

### 4. 查看训练结果
- 日志文件：保存在 `logs/train.log` 目录，记录训练过程中的关键指标；
- 模型权重：保存在 `checkpoints/{model}_{RUN_ID}/` 目录，包含：
  - 最佳模型（`best_model.pth`，基于 Val IoU 选择）；
  - 保存所有模型（启用 `--save-all-checkpoints`并设置`--checkpoint-save-freq 1`）；
- WandB 面板：实时查看 loss 曲线、IoU 变化、学习率调度等。

## 模型架构

### 1. TransUNet（核心推荐）
- 编码器：4 层下采样（DoubleConv + MaxPool），通道数 3 -> 32 -> 64 -> 12 -> 256 -> 512；
- 瓶颈层：PatchEmbedding + Transformer 层（可配置头数/层数），捕捉全局特征；
- 解码器：4 层上采样（Upsample + 拼接 + SingleConv），通道数 512 -> 256 -> 128 -> 64 -> 16；
- 输出层：1×1 卷积映射到 1 通道（二分类）。



<div style="background-color:white; margin:auto; text-align:center;">
    <image src="./assets/transunet_arch.png"/>
    <span style="fontcolor:gray; font-size:small;">图：TransUNet 模型架构示意图（来自论文<a href="https://arxiv.org/abs/2102.04306">TransUNet</a>）</span>
</div>


### 2. UNet（基线模型）
- 经典编码-解码架构，跳跃连接保留细节特征；
- 通道数与论文中一致，无 Transformer 瓶颈层，训练速度更快。

<div style="background-color:white; margin:auto; text-align:center;">
    <image src="./assets/unet_arch.png"/>
    <span style="fontcolor:gray; font-size:small;">图：UNet 模型架构示意图（来自论文<a href="https://arxiv.org/abs/1505.04597">U-Net</a>）</span>
</div>

### 3. TransUNetP
- 在TransUNet基础上，讲输出的通道数从16改为32，测试其影响。（实验结果显示影响不大）

## 训练策略优化

### 1. 损失函数
- `nn.BCEWithLogitsLoss`：基础二分类交叉熵损失，数值稳定；

### 2. 学习率调度
- 预热阶段（前 5 轮）：线性提升学习率至初始值（默认值: 1e-4）；
- 主训练阶段：余弦退火调度，自适应降低学习率。

### 3. 正则化
- Dropout：仅在编码器/Transformer 层添加（概率 0.1），避免过拟合；
- 权重衰减（L2 正则）：默认 1e-5，抑制参数冗余；
- 早停：基于 Val IoU，连续 20 轮无提升则停止训练。

## 结果展示

- 模型`UNet`，`TransUNet`和`TransUNetP`在`batch size=24`，`learning rate=1e-4`下结果对比：

<div style="background-color:white; margin:auto; text-align:center;">
    <image src="./assets/model_comparison.png"/>
    <span style="fontcolor:gray; font-size:small;">图. 训练损失、验证IoU、验证Dice</span>
</div>

- 几个模型基本都在`30-60`个epoch内收敛，之后过拟合严重，训练过程使用了早停策略；


- 模型`TransUNet`在`batch size`为`16`和`24`下结果展示：

<div style="background-color:white; margin:auto; text-align:center;">
    <image src="./assets/transunet_comparison.png"/>
    <span style="fontcolor:gray; font-size:small;">图. 训练损失、验证IoU、验证Dice</span>
</div>

- `batch size=16`时，损失和IoU曲线略好一点，因为语义分割不适合使用过大的`batch size`，但训练时间更长。

- 实际上，我保存了每个epoch的模型权重，所以我可以绘制所有epoch上的测试集IoU和Dice分数曲线，如下所示：
<div style="background-color:white; margin:auto; text-align:center;">
    <image src="./assets/model_test_comparison.png"/>
    <span style="fontcolor:gray; font-size:small;">图. TransUNet在测试集上的IoU和Dice分数随epoch变化曲线</span>
</div>

- 同时，我也发现把`BCEWithLogitsLoss`作为损失函数时，损失最小的点并不一定是测试集上IoU和Dice分数最高的点。

| Model Name | Best Val IoU | Val Dice | Test IoU | Test Dice | #Epoch | duration(/e) |
|------------|--------------|----------|----------|-----------| --------| --------------- |
| UNet       |    80.824%    |  88.770%  |  79.631%  |  87.794%   | <b>40</b> | <b>35.2</b> |
| TransUnet  |    82.150%    |  89.613% | 80.801% | 88.562% | 59 | 40.0 |
| TransUnetP |   81.537%    |  89.140% | 80.475% | 88.325% | 71 | 46.0 |
| TransUnet16 | <b>82.855%</b> | <b>90.032%</b> | <b>81.617%</b> | <b>89.136%</b> | 52 | 47.6 |


## 项目结构

```
oxford-pet-segmentation/
├── checkpoints/          # 模型权重保存目录
├── logs/                 # 训练日志目录
├── model/                # 模型定义
│   ├── unet.py           # UNet 模型
│   ├── transunet.py      # TransUNet 模型
│   └── ...
├── utils/                # 工具函数
│   ├── dataset.py        # 数据集加载（OxfordPetDataset）
│   └── ...
├── trainer.py            # 训练/验证核心逻辑
├── train.py              # 主训练入口
├── examples.ipynb        # 示例 Jupyter Notebook
├── requirements.txt      # 环境依赖
└── README.md             # 项目说明
```

## 示例
- 在`examples.ipynb`中提供了使用训练后的模型输出预测图片，在测试集上计算IoU和Dice分数的示例代码。

- 如下图所示，展示了部分测试集上的预测结果：
<div style="background-color:white; margin:auto; text-align:center; flex-direction:column;">
    <image src="./assets/prediction_1.png"/>
    <image src="./assets/prediction_2.png"/>
    <image src="./assets/prediction_3.png"/>
    <p style="fontcolor:gray; font-size:small;">图. 测试集预测结果示例（左：原图；中：真实掩码；右：模型预测掩码）</p>
</div>

## 扩展与自定义

### 1. 添加新模型
在 `model/` 目录下新增模型文件（如 `unetpp.py`），并在 `train.py` 的 `模型初始化` 部分添加`model`的添加新的模型映射。

### 2. 自定义数据增强
- 在 `utils/dataset.py` 的 `OxfordPetDataset` 类中：
    - `_apply_sync_transforms`方法主要用于训练集同步处理`image`和`mask`（主要针对几何变换）
    - `_get_transforms`方法用于定义所有图像的通用预处理（如归一化、尺寸调整等）；

### 3. 损失函数
在 `trainer.py` 中的 `损失函数和优化器` 部分添加新的 `criterion`，可以优先尝试结合BCE与DiceLoss的组合损失。

## 引用
本项目的 U-Net 模型实现参考了开源仓库 `Pytorch-UNet`：
- 仓库地址：https://github.com/milesial/Pytorch-UNet
- 作者：Milesial
- 许可证：GNU General Public License v3.0 (GPL-3.0)
- 核心参考：U-Net 基础模块（DoubleConv、Down、Up 等）的实现逻辑、数据加载与训练流程设计。


```
@misc{ronneberger2015unetconvolutionalnetworksbiomedical,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1505.04597}, 
}

@misc{chen2021transunettransformersmakestrong,
      title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation}, 
      author={Jieneng Chen and Yongyi Lu and Qihang Yu and Xiangde Luo and Ehsan Adeli and Yan Wang and Le Lu and Alan L. Yuille and Yuyin Zhou},
      year={2021},
      eprint={2102.04306},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2102.04306}, 
}
```

## 致谢
- 数据集来源：Oxford-IIIT Pet Dataset（University of Oxford）；
- 模型参考：TransUNet 官方实现、PyTorch 官方 UNet 示例；
- 工具支持：WandB 可视化平台、PyTorch 深度学习框架。