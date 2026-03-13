# A Dual-Stream UNet With Parallel Channel–Spatial Interaction and Aggregation for Change Detection（DCSI_UNet）

[😊 Paper](https://ieeexplore.ieee.org/document/11299285)

This repository contains the official implementation of **DCSI_UNet**, a novel architecture for change detection in remote sensing images. The model leverages dual cross-modal similarity interactions to effectively capture multi-scale features and improve change detection accuracy.

## 📄 Paper

**DCSI_UNet: Dual Cross-modal Similarity Interactive U-Net for Change Detection**  
Authors: Chaoyv  
Published in: IEEE Transactions on Geoscience and Remote Sensing  
DOI: [10.1109/TGRS.2025.3643508](https://ieeexplore.ieee.org/document/11299285)  

**Abstract:**  
Change detection (CD) aims to analyze changes in objects between bitemporal images, playing a crucial role in geographical observations. However, hidden difference features and insufficient interfeature interaction both lead to suboptimal performance of methods. To address these challenges, this article proposes a dual-stream UNet with parallel channel–spatial interaction and aggregation (DCSI-UNet). First, the encoding part utilizes a multilayer dual-stream convolutional neural network (CNN) to extract multilayer features of bitemporal images and then processes them through a channel group interaction module (CGIM) and a spatial Gaussian attention module (SGAM) to generate channel and spatial interaction feature maps, respectively. Specifically, the CGIM splits the bitemporal feature maps into multiple channelwise subspaces and applies interactive attention to enhance cross-channel feature correlation. The SGAM calculates the mixed mean and joint variance at the spatial level of bitemporal feature maps, which establish a Gaussian attention kernel (GAK) capturing hidden difference features. Second, the decoding part is designed in two phases: skip connections and interaction-feature aggregation. The former decodes the interactive feature maps from CGIM and SGAM separately. The latter fuses multilayer channel and spatial interactive features from the encoding part, which suppresses localization loss and exploits richer semantic consistency of changed regions. Finally, comprehensive experiments on three public datasets demonstrate that the proposed method outperforms existing methods in both quantitative and qualitative evaluations. The code is available at https://github.com/ZChaoyv/DCSI-UNet. Usage: Please refer to the README.md file for instructions on environment setup, dataset preparation, training, and inference. If you find this work useful, please cite our paper.

## 🚀 Features

- **State-of-the-art Performance**: Achieves superior results on LEVIR-CD, WHU-CD, and CDD datasets
- **Multi-scale Feature Learning**: Dual cross-modal similarity interactions capture both local and global changes
- **Flexible Backbone Support**: Compatible with ResNet and UNet backbones
- **Comprehensive Evaluation**: Includes training, validation, and testing pipelines

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+ (tested with 2.10.0)
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with 8GB+ VRAM (recommended)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ZChaoyv/DCSI-UNet.git
   cd DCSI_UNet
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   conda create -n dcsi_unet python=3.10
   conda activate dcsi_unet
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Datasets

This project supports three benchmark change detection datasets:

### LEVIR-CD
- **Download**: [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR)

### WHU-CD
- **Download**: [WHU-CD Dataset](https://aistudio.baidu.com/datasetdetail/251669)

### CDD
- **Download**: [CDD Dataset](https://drive.google.com/file/d/1GX656JqqOyBi\_Ef0w65kDGVto-nHrNs9)

### Data Preparation

1. Download and extract the datasets to your local directory
2. Update the dataset paths in `data_config.py`:
   ```python
   if data_name == 'LEVIR-CD-256-list':
       self.root_dir = '/path/to/your/LEVIR-CD-256-list'
   elif data_name == 'WHU-CD-256-list':
       self.root_dir = '/path/to/your/WHU-CD-256-list'
   elif data_name == 'CDD-CD-256-list':
       self.root_dir = '/path/to/your/CDD-CD-256-list'
   ```

3. Ensure each dataset has the following structure:
   ```
   dataset/
   ├── A/          # Pre-event images
   ├── B/          # Post-event images
   ├── label/      # Ground truth change masks
   └── list/       # Train/val/test split lists
       ├── train.txt
       ├── val.txt
       └── test.txt
   ```

## 🎯 Training

### Basic Training

Train DCSI_UNet on LEVIR-CD dataset:

```bash
python3 /your path/main_cd.py
```
```bash
script/train.sh
```

### Advanced Training Options

- **Adjust batch size for memory:**
  ```bash
  --batch_size 8  # Reduce if CUDA out of memory
  ```

- **Multi-GPU training:**
  ```bash
  --gpu_ids 0,1,2,3  # Use multiple GPUs
  ```

- **Custom learning rate:**
  ```bash
  --lr 5e-3  # Lower learning rate for fine-tuning
  ```

### Training Output

The training script will:
- Save checkpoints in `checkpoints/{project_name}/`
- Log training progress with tqdm progress bars
- Generate validation visualizations in `vis/{project_name}/`

## 🔍 Evaluation

### Evaluate Pretrained Model

```bash
python3 /your path/eval_cd.py
```
```bash
script/eval.sh
```

### Evaluation Metrics

The evaluation computes:
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **Kappa Coefficient**: Agreement measure
- **Precision/Recall**: Individual accuracy metrics

Results are saved in `vis/{project_name}/` with:
- Change detection maps
- Confusion matrices
- Performance metrics

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 8`
- Use gradient accumulation
- Enable PyTorch memory optimization:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```

### Checkpoint Loading Issues
- Ensure PyTorch version compatibility
- Check `weights_only=False` in `torch.load()` calls
- Verify checkpoint file integrity

### Dataset Path Errors
- Update paths in `data_config.py`
- Ensure correct dataset structure
- Check file permissions

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhang2025dual,
  title={A Dual-Stream UNet With Parallel Channel--Spatial Interaction and Aggregation for Change Detection},
  author={Zhang, Chaoyu and Yu, Qiuze and Shang, Yanli and Liu, Fanghong and Zhang, Haowen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={1--15},
  year={2025},
  publisher={IEEE}
}
```

---

# 中文版本 / Chinese Version

## A Dual-Stream UNet With Parallel Channel–Spatial Interaction and Aggregation for Change Detection（DCSI_UNet）

[😊 论文](https://ieeexplore.ieee.org/document/11299285)

该仓库包含 **DCSI_UNet** 的官方实现，这是一种用于遥感图像变化检测的新型架构。该模型利用双交叉模态相似性交互来有效捕获多尺度特征并提高变化检测准确性。

## 📄 论文

**DCSI_UNet: 双交叉模态相似性交互 U-Net 用于变化检测**  
作者：Chaoyv  
发表期刊：IEEE 地球科学与遥感汇刊  
DOI: [10.1109/TGRS.2025.3643508](https://ieeexplore.ieee.org/document/11299285)  

**摘要：**  
变化检测（CD）旨在分析双时相图像中目标的变化，在地理观测中发挥着至关重要的作用。然而，隐藏差异特征和特征间交互不足都会导致现有方法的性能欠佳。为了应对这些挑战，本文提出了一种具有并行通道-空间交互和聚合的双流统一网络（DCSI-UNet）。首先，编码部分利用多层双流卷积神经网络（CNN）提取双时相图像的多层特征，然后分别通过通道组交互模块（CGIM）和空间高斯注意力模块（SGAM）处理这些特征，生成通道交互特征图和空间交互特征图。具体而言，CGIM将双时相特征图分割成多个通道子空间，并应用交互式注意力机制来增强跨通道特征的相关性。SGAM计算双时相特征图空间层面的混合均值和联合方差，从而构建一个能够捕获隐藏差异特征的高斯注意力核（GAK）。其次，解码部分设计为两阶段：跳跃连接和交互特征聚合。前者分别解码来自 CGIM 和 SGAM 的交互特征图。后者融合来自编码部分的多层通道和空间交互特征，从而抑制定位损失并利用变化区域更丰富的语义一致性。最后，在三个公开数据集上的综合实验表明，所提出的方法在定量和定性评估中均优于现有方法。代码可在 https://github.com/ZChaoyv/DCSI-UNet 获取。使用方法：请参阅 README.md 文件，了解环境设置、数据集准备、训练和推理说明。如果您觉得这项工作有用，请引用我们的论文。

## 🚀 功能特点

- **最先进的性能**：在 LEVIR-CD、WHU-CD 和 CDD 数据集上取得优异结果
- **多尺度特征学习**：双交叉模态相似性交互捕获局部和全局变化
- **灵活的主干网络支持**：兼容 ResNet 和 UNet 主干网络
- **全面评估**：包括训练、验证和测试流程

## 📋 要求

- Python 3.8+
- PyTorch 1.10+（测试版本 2.10.0）
- CUDA 11.0+（用于 GPU 加速）
- 8GB+ RAM（推荐 16GB）
- NVIDIA GPU 8GB+ VRAM（推荐）

## 🛠️ 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/ZChaoyv/DCSI-UNet.git
   cd DCSI_UNet
   ```

2. **创建虚拟环境（推荐）：**
   ```bash
   conda create -n dcsi_unet python=3.10
   conda activate dcsi_unet
   ```

3. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 数据集

本项目支持三个基准变化检测数据集：

### LEVIR-CD
- **下载**：[LEVIR-CD 数据集](https://justchenhao.github.io/LEVIR)

### WHU-CD
- **下载**：[WHU-CD 数据集](https://aistudio.baidu.com/datasetdetail/251669)

### CDD
- **下载**：[CDD 数据集](https://drive.google.com/file/d/1GX656JqqOyBi\_Ef0w65kDGVto-nHrNs9)

### 数据准备

1. 下载并解压数据集到本地目录
2. 在 `data_config.py` 中更新数据集路径：
   ```python
   if data_name == 'LEVIR-CD-256-list':
       self.root_dir = '/path/to/your/LEVIR-CD-256-list'
   elif data_name == 'WHU-CD-256-list':
       self.root_dir = '/path/to/your/WHU-CD-256-list'
   elif data_name == 'CDD-CD-256-list':
       self.root_dir = '/path/to/your/CDD-CD-256-list'
   ```

3. 确保每个数据集具有以下结构：
   ```
   dataset/
   ├── A/          # 事件前图像
   ├── B/          # 事件后图像
   ├── label/      # 真实变化掩码
   └── list/       # 训练/验证/测试分割列表
       ├── train.txt
       ├── val.txt
       └── test.txt
   ```

## 🎯 训练

### 基本训练

在 LEVIR-CD 数据集上训练 DCSI_UNet：

```bash
python3 /your path/main_cd.py
```
```bash
script/train.sh
```

### 高级训练选项

- **调整批大小以适应内存：**
  ```bash
  --batch_size 8  # 如果 CUDA 内存不足则减少
  ```

- **多 GPU 训练：**
  ```bash
  --gpu_ids 0,1,2,3  # 使用多个 GPU
  ```

- **自定义学习率：**
  ```bash
  --lr 5e-3  # 微调时的较低学习率
  ```

### 训练输出

训练脚本将：
- 在 `checkpoints/{project_name}/` 中保存检查点
- 使用 tqdm 进度条记录训练进度
- 在 `vis/{project_name}/` 中生成验证可视化

## 🔍 评估

### 评估预训练模型

```bash
python3 /your path/eval_cd.py
```
```bash
script/eval.sh
```

### 评估指标

评估计算：
- **F1 分数**：精确率和召回率的调和平均值
- **IoU（交并比）**：预测与真实标签的重叠
- **Kappa 系数**：一致性度量
- **精确率/召回率**：单个准确性指标

结果保存在 `vis/{project_name}/` 中，包括：
- 变化检测图
- 混淆矩阵
- 性能指标

## 🔧 故障排除

### CUDA 内存不足
- 减少批大小：`--batch_size 8`
- 使用梯度累积
- 启用 PyTorch 内存优化：
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  ```

### 检查点加载问题
- 确保 PyTorch 版本兼容性
- 检查 `torch.load()` 调用中的 `weights_only=False`
- 验证检查点文件完整性

### 数据集路径错误
- 在 `data_config.py` 中更新路径
- 确保正确的数据集结构
- 检查文件权限

## 🤝 贡献

我们欢迎贡献！请：
1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 提交拉取请求

## 📄 引用

如果您在研究中使用此代码，请引用我们的论文：

```bibtex
@article{zhang2025dual,
  title={A Dual-Stream UNet With Parallel Channel--Spatial Interaction and Aggregation for Change Detection},
  author={Zhang, Chaoyu and Yu, Qiuze and Shang, Yanli and Liu, Fanghong and Zhang, Haowen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={1--15},
  year={2025},
  publisher={IEEE}
}
```

*Last updated: 2026-03-13*