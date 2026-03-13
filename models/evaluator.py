import os
import numpy as np
import matplotlib.pyplot as plt
import utils

from tqdm import tqdm  
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from numpy.core.fromnumeric import choose
from utils import de_norm


################################################################################
#                                                                              
#              🔍 MODEL EVALUATION & VISUALIZATION SYSTEM                      
#                  🔍 模型评估与结果可视化系统                                   
#                        👤 AUTHOR: Caoyv                                      
#                      📄 PAPER: DCSI_UNet                                     
#           🔗 https://ieeexplore.ieee.org/document/11299285                   
#                                                                              
#   Description: This class handles the evaluation of the DCSI_UNet model,      
#   generating visual change maps and calculating quantitative metrics.        
#   代码说明：该类负责 DCSI_UNet 模型的评估，生成变化检测可视化图并计算定量指标。      
#                                                                              
################################################################################


# ==============================================================================
# [CDEvaluator] Core Class for Change Detection Model Evaluation
# [CDEvaluator] 变化检测模型评估核心类
# ==============================================================================
class CDEvaluator():

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        
        # STEP 1: Define the network and move it to the specific device
        # 第一步：定义网络结构并将其移动至指定设备
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        self.net_G.to(self.device)
        
        print(f"Using device: {self.device}")

        # STEP 2: Initialize logger and environment settings
        # 第二步：初始化日志记录器与环境设置
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.batch_size = args.batch_size
        self.steps_per_epoch = len(dataloader)

        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # Ensure directories exist for weights and visualization
        # 确保权重和可视化结果的目录存在
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # STEP 3: Setup the metric meter (Confusion Matrix)
        # 第三步：设置指标计量器（混淆矩阵）
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        """ Load the model weights / 加载模型权重 """
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if os.path.exists(ckpt_path):
            self.logger.write(f'Loading checkpoint from {ckpt_path}...\n')
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # Use strict=False to bypass unexpected keys like total_ops
            # 使用 strict=False 以跳过多余的键值（如 total_ops）
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=False)
            
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_epoch_id = checkpoint.get('best_epoch_id', 0)
            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.best_val_acc, self.best_epoch_id))
        else:
            raise FileNotFoundError('No such checkpoint %s' % ckpt_path)

    def _visualize_pred(self):
        """ Convert model output to a viewable 0-255 image / 将模型输出转换为 0-255 的可视图像 """
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        return pred * 255

    def _update_metric(self):
        """ Update the running accuracy metrics / 更新实时运行的准确率指标 """
        target = self.batch['L'].to(self.device).detach()
        G_pred = torch.argmax(self.G_pred.detach(), dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self, pbar):
        """ Process results and save visual outputs for each batch / 处理结果并保存每个批次的可视化输出 """
        running_acc = self._update_metric()
        
        # Update progress bar info / 更新进度条信息
        if np.mod(self.batch_id, 10) == 0:
            pbar.set_postfix(mF1=f"{running_acc:.5f}")

        # Save visualization images / 保存可视化图片
        if np.mod(self.batch_id, 1) == 0:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_gt = utils.make_numpy_grid(self.batch['L'])

            # Concatenate Input A, Input B, Prediction, and GT / 拼接输入 A, 输入 B, 预测图和标签
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(self.vis_dir, 'eval_' + str(self.batch_id + 1) + '.jpg')
            plt.imsave(file_name, vis)

    def _forward_pass(self, batch):
        """ Run the model inference / 执行模型推理 """
        self.batch = batch

        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        # Multi-scale outputs for Deep Supervision / 用于深监督的多尺度输出
        self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
        # Final combined prediction / 最终合并后的预测
        self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3

    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        """ Execute the full evaluation pipeline / 执行完整的评估流程 """
        self._load_checkpoint(checkpoint_name)

        self.logger.write('Begin evaluation...\n')
        self.running_metric.clear()
        self.is_training = False
        self.net_G.eval()

        # Initialize progress bar / 初始化进度条
        pbar = tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader), desc="Evaluating")
        
        for self.batch_id, batch in pbar:
            with torch.no_grad(): # Disable gradient calculation / 禁用梯度计算
                self._forward_pass(batch)
            self._collect_running_batch_states(pbar)
            
        # Finalize and save scores / 汇总并保存评分
        self._collect_epoch_states()
        print("\nEvaluation Complete.")

    def _collect_epoch_states(self):
        """ Calculate final scores for the entire dataset / 计算整个数据集的最终评分 """
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = scores_dict['mf1']

        # Separate average and per-class metrics / 分离平均指标和类别指标
        avg_keys = ['acc', 'miou', 'mf1', 'mrecall', 'mprecision', 'kc']
        class_keys = ['iou_0', 'iou_1', 'F1_0', 'F1_1', 'precision_0', 'precision_1', 'recall_0', 'recall_1']
        
        message_avg = ''
        for k in avg_keys:
            if k in scores_dict:
                message_avg += '%s: %.5f ' % (k, scores_dict[k])
        
        message_class = ''
        for k in class_keys:
            if k in scores_dict:
                message_class += '%s: %.5f ' % (k, scores_dict[k])
        
        self.logger.write('%s\n' % message_avg)
        self.logger.write('%s\n' % message_class)