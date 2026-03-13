import numpy as np


################################################################################
#                                                                              
#              📊 EVALUATION METRICS & PERFORMANCE TOOLS                       
#                  📊 评估指标与性能衡量工具中心                                 
#                                                                              
#   Description: This script defines metrics for Change Detection, including    
#   Confusion Matrix, F1-score, IoU, and Kappa coefficient calculation.         
#   代码说明：该脚本定义了变化检测的评价指标，包括混淆矩阵、F1、IoU 及 Kappa 系数计算。  
#                                                                              
################################################################################


# ==============================================================================
# [AverageMeter] Base class to compute and store running averages
# [AverageMeter] 基础类，用于计算并存储运行过程中的平均值
# ==============================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        """Update the metrics / 更新指标状态"""
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        # Calculate scores from the accumulated confusion matrix
        # 从累计的混淆矩阵中计算评分
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


# ==============================================================================
# [ConfuseMatrixMeter] Specialized meter for Confusion Matrix tracking
# [ConfuseMatrixMeter] 专门用于追踪混淆矩阵的计量类
# ==============================================================================
class ConfuseMatrixMeter(AverageMeter):
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """
        Get current confusion matrix and update the F1 score
        获取当前混淆矩阵并更新 F1 得分
        """
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        """Final score report / 最终评分报告"""
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    """Calculate Harmonic Mean / 计算调和平均值"""
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


# ==============================================================================
# [Core Math] Confusion Matrix to Metrics (F1, IoU, Kappa, etc.)
# [核心计算] 混淆矩阵转各项指标（F1, IoU, Kappa 等）
# ==============================================================================


def cm2F1(confusion_matrix):
    """Convert confusion matrix to Mean F1 score / 混淆矩阵转平均 F1"""
    hist = confusion_matrix
    tp = np.diag(hist) # True Positives / 真阳性
    sum_a1 = hist.sum(axis=1) # Ground Truth counts / 真值统计
    sum_a0 = hist.sum(axis=0) # Prediction counts / 预测值统计

    # Accuracy / 准确率
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    # Recall / 召回率
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # Precision / 精确率
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score formula: 2 * (P * R) / (P + R)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return np.nanmean(F1)


def cm2score(confusion_matrix):
    """Comprehensive score calculation / 综合得分计算"""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    
    mean_recall = np.nanmean(recall)
    mean_precision = np.nanmean(precision)
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    # Kappa Coefficient calculation / Kappa 系数计算
    pe = (hist[0].sum() * hist[:,0].sum() + hist[1].sum() * hist[:,1].sum()) / hist.sum() ** 2 + np.finfo(np.float32).eps
    kc = (acc - pe) / (1 - pe + np.finfo(np.float32).eps)

    # Intersection over Union (IoU) / 交并比
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    # Pack scores into a dictionary / 将评分打包入字典
    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1, 'mrecall':mean_recall, 'mprecision':mean_precision}
    score_dict.update({'kc': kc})
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """
    Generate Confusion Matrix for a set of predictions
    计算一组预测结果的混淆矩阵
    """
    def __fast_hist(label_gt, label_pred):
        # Use bincount for efficient histogram calculation
        # 使用 bincount 进行高效的直方图统计
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
        
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix