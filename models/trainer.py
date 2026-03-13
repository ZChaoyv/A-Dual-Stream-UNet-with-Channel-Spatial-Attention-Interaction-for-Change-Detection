import numpy as np
import os
import time
import utils
import torch
import torch.optim as optim

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
from misc.logger_tool import Logger, Timer
from thop import profile
from tqdm import tqdm


################################################################################
#                                                                              
#              🚀 MODEL TRAINING & PIPELINE MANAGEMENT                         
#                  🚀 模型训练与流水线管理中心                                   
#                        👤 AUTHOR: Caoyv                                      
#                      📄 PAPER: DCSI_UNet                                     
#           🔗 https://ieeexplore.ieee.org/document/11299285                   
#                                                                              
#   Description: This class centralizes all dataset paths and preprocessing    
#   settings. It maps dataset aliases to their respective physical locations.   
#   代码说明：该类统一管理所有数据集路径与预处理设置，将数据集别名映射至物理存储路径。       
#                                                                              
################################################################################


class CDTrainer():

    def __init__(self, args, dataloaders):
        self.dataloaders = dataloaders
        self.n_class = args.n_class

        # STEP 1: Define the network (Generator)
        # 第一步：定义网络结构（生成器）
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        # STEP 2: Profile model complexity (FLOPs/Params)
        # 第二步：分析模型复杂度（计算量与参数量）
        x1 = torch.rand(1, 3, 256, 256).cuda()
        x2 = torch.rand(1, 3, 256, 256).cuda()
        input_data = (x1, x2)
        flops, params = profile(self.net_G, inputs=input_data)
        flops_in_giga = flops / 1e9
        print(f"Model FLOPs: {flops_in_giga:.4f} GFLOPs")
        print(f"Model Parameters: {params / 1e6:.2f} M")

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        # STEP 3: Setup Optimizer and LR Scheduler
        # 第三步：设置优化器与学习率调度器
        self.lr = args.lr
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        # STEP 4: Initialize logging, timers, and metrics
        # 第四步：初始化日志记录、计时器与评估指标
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.timer = Timer()
        self.batch_size = args.batch_size

        # STEP 5: Define training state variables
        # 第五步：定义训练状态变量
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.train_loader = None
        self.val_loader = None
        self.vis_dir = args.vis_dir

        # STEP 6: Define Loss Function
        # 第六步：定义损失函数
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        else:
            raise NotImplemented(args.loss)

        # STEP 7: Load or initialize accuracy history
        # 第七步：加载或初始化准确率历史记录
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    # --------------------------------------------------------------------------
    # Checkpoint & LR Management / 权重与学习率管理
    # --------------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        """ Load model weights and training progress / 加载模型权重与训练进度 """
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device, weights_only=False)
            
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        else:
            print('training from scratch...')

    def _save_checkpoint(self, ckpt_name):
        """ Save weights and states to disk / 将权重与状态保存至磁盘 """
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    # --------------------------------------------------------------------------
    # Forward & Backward Pass / 前向传播与反向传播
    # --------------------------------------------------------------------------

    def _forward_pass(self, batch):
        """ Network forward propagation / 网络前向传播 """
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
        self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3

    def _backward_G(self):
        """ Loss calculation and backpropagation / 损失计算与反向传播 """
        gt = self.batch['L'].to(self.device).long()
        # Summing multi-stage losses for deep supervision
        # 累加多阶段损失以进行深监督训练
        self.G_loss = self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + self._pxl_loss(self.G_pred3, gt)
        self.G_loss.backward()

    # --------------------------------------------------------------------------
    # Metrics & Visualization / 指标计算与可视化
    # --------------------------------------------------------------------------

    def _update_metric(self):
        """ Calculate accuracy and update confusion matrix / 计算准确率并更新混淆矩阵 """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def de_norm(self, tensor_data):
        return tensor_data * 0.5 + 0.5

    def _collect_running_batch_states(self):
        """ Log real-time training status per batch / 记录每个批次的实时训练状态 """
        running_acc = self._update_metric()
        m = len(self.dataloaders['train']) if self.is_training else len(self.dataloaders['val'])
        imps, est = self._timer_update()

        if np.mod(self.batch_id, 100) == 1:
            message = '训练模式: %s | Epoch: %d/%d | Batch: %d/%d | 速度: %.2f img/s | 预计剩余: %.2fh | 损失: %.5f | mF1: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est, self.G_loss.item(), running_acc)
            
            if (self.is_training and self.train_loader is not None) or (not self.is_training and self.val_loader is not None):
                tqdm.write(message.rstrip('\n'))
            else:
                self.logger.write(message)

        # Update tqdm progress bar description / 更新 tqdm 进度条描述
        if self.is_training and self.train_loader is not None:
            self.train_loader.set_postfix({'Loss': f'{self.G_loss.item():.5f}', 'mF1': f'{running_acc:.5f}', 'Speed': f'{imps * self.batch_size:.2f}'})
        elif not self.is_training and self.val_loader is not None:
            self.val_loader.set_postfix({'mF1': f'{running_acc:.5f}'})

        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(self.de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(self.de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(self.vis_dir, 'istrain_' + str(self.is_training) + '_' + str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')

    def _collect_epoch_states(self):
        """ Summarize metrics at the end of an epoch / 在 Epoch 结束时汇总指标 """
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n\n')

    def _update_checkpoints(self):
        """ Save last and best checkpoints / 更新并保存最新与最佳模型 """
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n\n'
                          % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n\n')

    def _update_training_acc_curve(self):
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()

    # --------------------------------------------------------------------------
    # Main Training Logic / 主训练逻辑
    # --------------------------------------------------------------------------

    def train_models(self):
        """ Execute full training and validation pipeline / 执行完整的训练与验证流程 """
        self._load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            # PHASE: Training / 阶段：训练
            self._clear_cache()
            self.is_training = True
            starttime = time.time()
            self.net_G.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            
            self.train_loader = tqdm(self.dataloaders['train'], desc=f'Epoch {self.epoch_id+1}/{self.max_num_epochs}', unit='batch')
            for self.batch_id, batch in enumerate(self.train_loader, 0):
                self._forward_pass(batch)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            self.logger.write('epoch time: %0.3f \n' % (time.time() - starttime))

            # PHASE: Evaluation / 阶段：验证评估
            self.logger.write('Begin evaluation val...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            self.val_loader = tqdm(self.dataloaders['val'], desc=f'Val Epoch {self.epoch_id+1}/{self.max_num_epochs}', unit='batch')
            for self.batch_id, batch in enumerate(self.val_loader, 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            
            self._collect_epoch_states()
            self._update_val_acc_curve()
            self._update_checkpoints()