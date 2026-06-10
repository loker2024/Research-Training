"""训练框架"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def resolve_device(device='auto'):
    """Resolve a requested training device.

    ``auto`` prefers CUDA for Windows training, then MPS for Mac smoke tests,
    and finally CPU.
    """
    if device is None:
        device = 'auto'
    requested = str(device).lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if requested == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class Trainer:
    """时序预测模型训练器"""

    def __init__(self, model, device='auto', lr=1e-3, weight_decay=1e-5, seed=216):
        """
        Args:
            model: 模型实例
            device: 设备 ('cuda' 或 'cpu')
            lr: 学习率
            weight_decay: 权重衰减
            seed: 随机种子，用于保证可复现性
        """
        self.seed = seed
        self._set_seed(seed)

        self.model = model
        self.device = resolve_device(device)
        self.model.to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_r2 = []
        self.val_r2 = []
        self.best_val_loss = float('inf')
        self.best_val_r2 = 0.0
        self.best_model_state = None

    @staticmethod
    def _set_seed(seed):
        """设置随机种子，保证可复现性"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def compute_r2(self, pred, target):
        """计算 R² 决定系数（准确率的一种表示）"""
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()

    def train_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_r2 = 0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            pred_y = self.model(batch_x)
            loss = self.criterion(pred_y, batch_y)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 计算 R²
            r2 = self.compute_r2(pred_y, batch_y)

            total_loss += loss.item()
            total_r2 += r2
            n_batches += 1

        return total_loss / n_batches, total_r2 / n_batches

    @torch.no_grad()
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_r2 = 0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred_y = self.model(batch_x)
            loss = self.criterion(pred_y, batch_y)
            r2 = self.compute_r2(pred_y, batch_y)

            total_loss += loss.item()
            total_r2 += r2
            n_batches += 1

        return total_loss / n_batches, total_r2 / n_batches

    def train(self, train_loader, val_loader, epochs=100, patience=10,
              save_dir='checkpoints', model_name='model', log_dir=None):
        """完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            patience: 早停耐心值
            save_dir: 模型保存目录
            model_name: 模型名称
            log_dir: TensorBoard 日志目录（None 则不记录）

        Returns:
            训练历史
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}.pt')

        # TensorBoard
        writer = None
        if log_dir is not None:
            tb_path = os.path.join(log_dir, model_name)
            writer = SummaryWriter(log_dir=tb_path)
            print(f'TensorBoard 日志: {tb_path}')

        print(f'开始训练: {model_name}')
        print(f'设备: {self.device}')
        print(f'训练样本: {len(train_loader.dataset)}')
        print(f'验证样本: {len(val_loader.dataset)}')
        print(f'批次大小: {train_loader.batch_size}')
        print('=' * 80)

        early_stop_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_r2 = self.train_epoch(train_loader)

            # 验证
            val_loss, val_r2 = self.evaluate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_r2.append(train_r2)
            self.val_r2.append(val_r2)

            # TensorBoard 记录
            if writer is not None:
                writer.add_scalars('Loss', {
                    'train': train_loss, 'val': val_loss
                }, epoch)
                writer.add_scalars('R2', {
                    'train': train_r2, 'val': val_r2
                }, epoch)
                writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_r2 = val_r2
                self.best_model_state = self.model.state_dict().copy()
                torch.save(self.best_model_state, save_path)
                early_stop_counter = 0
                best_str = ' ✓'
            else:
                early_stop_counter += 1
                best_str = ''

            # 打印进度（包含准确率 R²）
            print(f'Epoch {epoch:3d}/{epochs} | '
                  f'Train Loss: {train_loss:.6f} R²: {train_r2:.4f} | '
                  f'Val Loss: {val_loss:.6f} R²: {val_r2:.4f}{best_str} | '
                  f'Time: {epoch_time:.1f}s')

            # 早停检查
            if early_stop_counter >= patience:
                print(f'\n早停触发: {patience} 轮验证损失未改善')
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f'\n加载最佳模型: val_loss={self.best_val_loss:.6f}, val_R²={self.best_val_r2:.4f}')

        print('=' * 80)

        if writer is not None:
            writer.close()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2
        }

    @torch.no_grad()
    def predict(self, dataloader):
        """预测并返回结果

        Args:
            dataloader: 数据加载器

        Returns:
            predictions: 预测值 (N, horizon, features)
            targets: 真实值 (N, horizon, features)
        """
        self.model.eval()
        predictions = []
        targets = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred_y = self.model(batch_x)

            predictions.append(pred_y.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        return predictions, targets

    @torch.no_grad()
    def compute_metrics(self, predictions, targets, target_idx=None):
        """计算评估指标

        Args:
            predictions: 预测值 (N, horizon, features)
            targets: 真实值 (N, horizon, features)
            target_idx: 目标列索引（计算单变量指标）

        Returns:
            metrics: 指标字典
        """
        # MSE
        mse = np.mean((predictions - targets) ** 2)

        # MAE
        mae = np.mean(np.abs(predictions - targets))

        # MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        # R² 决定系数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

        # 如果指定了目标列，计算单变量指标
        if target_idx is not None:
            pred_target = predictions[:, :, target_idx]
            tgt_target = targets[:, :, target_idx]

            metrics['MSE_target'] = np.mean((pred_target - tgt_target) ** 2)
            metrics['MAE_target'] = np.mean(np.abs(pred_target - tgt_target))
            metrics['MAPE_target'] = np.mean(
                np.abs((tgt_target - pred_target) / (tgt_target + 1e-8))
            ) * 100

            ss_res_target = np.sum((tgt_target - pred_target) ** 2)
            ss_tot_target = np.sum((tgt_target - np.mean(tgt_target)) ** 2)
            metrics['R2_target'] = 1 - ss_res_target / (ss_tot_target + 1e-8)

        return metrics

    def save_history(self, save_path):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2
        }
        np.save(save_path, history)
        print(f'训练历史已保存至: {save_path}')
