import os
from PIL import Image
import numpy as np
from torch.utils import data
from datasets.data_utils import CDDataAugmentation


################################################################################
#                                                                              
#              🖼️ DATASET LOADING & AUGMENTATION PIPELINE                      
#                  🖼️ 数据集加载与数据增强                                
#                                                                              
#   Description: This script defines the Dataset classes for Change Detection,  
#   handling paired image loading (A & B) and their corresponding labels.       
#   代码说明：该脚本定义了变化检测的数据集类，处理成对图像（A 和 B）及其对应标签的加载。   
#                                                                              
################################################################################


# --- Dataset Structure Configuration / 数据集结构配置 ---
IMG_FOLDER_NAME = "A"           # Pre-event images / 事前图像
IMG_POST_FOLDER_NAME = 'B'      # Post-event images / 事后图像
LIST_FOLDER_NAME = 'list'       # Train/Val split lists / 训练验证名单
ANNOT_FOLDER_NAME = "label"     # Ground truth masks / 变化标签

IGNORE = 255
label_suffix = '.png' 

# ==============================================================================
# [Helper Functions] Path management and name loading
# [辅助函数] 路径管理与文件名加载
# ==============================================================================
def load_img_name_list(dataset_path):
    """Read image names from txt file / 从 txt 文件读取图像文件名"""
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    # Match label suffix with images / 将标签后缀与图像匹配
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


# ==============================================================================
# [ImageDataset] Base class for image-pair loading
# [ImageDataset] 用于图像对加载的基础类
# ==============================================================================


class ImageDataset(data.Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split 
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        
        # Initialize Augmentation / 初始化数据增强
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,   # Horizontal flip / 水平翻转
                with_random_vflip=True,   # Vertical flip / 垂直翻转
                with_scale_random_crop=True, # Random crop / 随机裁剪
                with_random_blur=True,    # Random blur / 随机模糊
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __getitem__(self, index):
        """Fetch a pair of images / 获取一对图像"""
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        # Apply transformation / 执行增强变换
        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        return self.A_size


# ==============================================================================
# [CDDataset] Change Detection Dataset with Labels
# [CDDataset] 带标签的变化检测数据集类
# ==============================================================================
class CDDataset(ImageDataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None, to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train, to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        """Fetch image pairs and the change label / 获取图像对与变化标签"""
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)
        L_path = get_label_path(self.root_dir, name)

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        label = np.array(Image.open(L_path), dtype=np.uint8)

        # Thresholding: Convert to binary mask / 阈值处理：转换为二值掩码
        label[label > 127] = 255
        label[label <= 127] = 0

        # Normalize label to [0, 1] / 将标签归一化至 [0, 1]
        if self.label_transform == 'norm':
            label = label // 255

        # Synchronized transformation for Image A, B and Label
        # 对图像 A、B 及标签执行同步的数据增强
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        
        return {'name': name, 'A': img, 'B': img_B, 'L': label}