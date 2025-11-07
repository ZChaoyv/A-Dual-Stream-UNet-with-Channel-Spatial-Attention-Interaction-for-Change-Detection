# (DCSI-UNet) A Dual-Stream UNet with Parallel Channel-Spatial Interaction and Aggregation for Change Detection
A Dual-Stream UNet with Parallel Channel-Spatial Interaction and Aggregation for Change Detection
This repository contains the official implementation of the paper "A Dual-Stream UNet with Parallel Channel-Spatial Interaction and Aggregation for Change Detection". The proposed method enhances change detection performance by leveraging a dual-stream UNet architecture combined with interactive channel-spatial attention mechanisms, effectively capturing both global context and local detail changes between bi-temporal images.
Key Features
Implementation of the dual-stream UNet backbone for parallel feature extraction from bi-temporal data
Channel-spatial attention interaction modules to emphasize discriminative change-related features
End-to-end training pipeline with support for common change detection datasets (e.g., LEVIR-CD, WHU-CD)
Preprocessing scripts and evaluation metrics (mIoU, F1-score, etc.) for quantitative analysis
Usage
Refer to the README.md for environment setup, dataset preparation, training, and inference instructions.
If you find this work useful, please cite our paper.

## 1. 🌱 Environment setup
The experiments are consistently conducted on the workstation with AMD EPYC 7262 8-Core CPU and GPU of NVIDIA RTX 3090 with 24G of video memory, Python 3.9.20, PyTorch 2.4.1, CUDA 11.8, cuDNN 9.1.0. All library dependencies are detailed in requirements.txt.

## 2. :truck: Dataset <a name="dataset"></a>
- [x] [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR/)
- A benchmark dataset for building change detection, consisting of ultra-high-resolution (0.5 m) bitemporal remote sensing images from Google Earth (covering multiple cities in Texas, USA). Download link: https://justchenhao.github.io/LEVIR.
    
- [x] [WHU-CD Dataset](https://aistudio.baidu.com/datasetdetail/251669)
- Constructed from remote sensing images before and after the Christchurch earthquake (New Zealand), focusing on post-disaster building reconstruction changes. Download link: https://aistudio.baidu.com/datasetdetail/251669.

- [x] [CDD Dataset](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)
- Focuses on diverse changes (e.g., buildings, vehicles, seasonal vegetation variations) using Google Earth imagery. Download link: https://drive.google.com/file/d/1GX656JqqOyBi\_Ef0w65kDGVto-nHrNs9.


