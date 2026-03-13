import random
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


################################################################################
#                                                                              
#              🌈 DATA AUGMENTATION & TRANSFORMATION TOOLS                      
#                  🌈 数据增强与图像变换工具中心                                 
#                                                                              
#   Description: This script provides synchronized augmentation for image       
#   pairs and labels, including geometric shifts and color normalization.       
#   代码说明：该脚本为图像对和标签提供同步增强，包括几何变换与色彩归一化。              
#                                                                              
################################################################################


# ==============================================================================
# [Global Utils] Tensor conversion and normalization
# [全局工具] 张量转换与归一化
# ==============================================================================
def to_tensor_and_norm(imgs, labels):
    """Standardize images to [-1, 1] and labels to tensors / 图像标准化与标签转张量"""
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0) for img in labels]
    # Normalize with mean and std 0.5 / 使用均值和标准差 0.5 进行归一化
    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for img in imgs]
    return imgs, labels


# ==============================================================================
# [CDDataAugmentation] Main class for synchronized CD data augmentation
# [CDDataAugmentation] 变化检测数据同步增强主类
# ==============================================================================


class CDDataAugmentation:
    def __init__(self, img_size, with_random_hflip=False, with_random_vflip=False,
                 with_random_rot=False, with_random_crop=False, 
                 with_scale_random_crop=False, with_random_blur=False):
        self.img_size = img_size
        self.img_size_dynamic = (self.img_size is None)
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

    def transform(self, imgs, labels, to_tensor=True):
        """Apply a series of transforms to a list of images and labels / 对图像列表和标签执行变换"""
        
        # 1. Initialization: Convert to PIL / 初始化：转换为 PIL 格式
        imgs = [TF.to_pil_image(img) for img in imgs]
        labels = [TF.to_pil_image(img) for img in labels]

        # 2. Resizing / 尺寸调整
        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3) for img in imgs]
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0) for img in labels]

        # 3. Geometric Transforms (Flipping & Rotation) / 几何变换（翻转与旋转）
        # Synchronized horizontal flip / 同步水平翻转
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        # Discrete rotation (90, 180, 270) / 离散旋转
        if self.with_random_rot and random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        # 4. Cropping & Scaling / 裁剪与缩放
        if self.with_random_crop:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size).get_params(imgs[0], scale=(0.8, 1.0), ratio=(1, 1))
            imgs = [TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size), interpolation=Image.BICUBIC) for img in imgs]
            labels = [TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size), interpolation=Image.NEAREST) for img in labels]

        # 5. Gaussian Blur / 高斯模糊
        if self.with_random_blur and random.random() > 0.5:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius)) for img in imgs]

        # 6. Final conversion to Tensor / 最终转换为张量
        if to_tensor:
            imgs, labels = to_tensor_and_norm(imgs, labels)

        return imgs, labels


# ==============================================================================
# [Low-level Utils] Pixel-level cropping and resizing helpers
# [底层工具] 像素级裁剪与调整辅助函数
# ==============================================================================
def pil_crop(image, box, cropsize, default_value):
    """Handles edge cases for image cropping / 处理图像裁剪的边界情况"""
    img = np.array(image)
    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    # Fill actual content / 填充实际内容
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return Image.fromarray(cont)

def get_random_crop_box(imgsize, cropsize):
    """Calculate random coordinates for cropping / 计算随机裁剪坐标"""
    h, w = imgsize
    # Logic for image/container spacing / 图像与容器间隙逻辑
    # ... [Implementation detail / 实现细节]
    return # (Returns 8-tuple of coordinates / 返回8元坐标组)