"""时序数据集加载器"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """时序数据集

    从预处理好的 npz 文件中加载数据
    """

    def __init__(self, data_dir, dataset_name, horizon, split='train'):
        """
        Args:
            data_dir: 数据目录 (e.g., 'data/processed')
            dataset_name: 数据集名称 (e.g., 'ETTh1')
            horizon: 预测步长 (24, 48, 96, 168, 336)
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.horizon = horizon
        self.split = split

        # 加载数据
        data_path = os.path.join(
            data_dir, dataset_name, f'h{horizon}', f'{split}.npz'
        )
        data = np.load(data_path)
        self.X = torch.FloatTensor(data['X'])  # (N, lookback, features)
        self.Y = torch.FloatTensor(data['Y'])  # (N, horizon, features)

        # 加载元数据
        meta_path = os.path.join(data_dir, dataset_name, 'meta.json')
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        print(f'加载 {dataset_name} {split} 数据: X={self.X.shape}, Y={self.Y.shape}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    @property
    def input_size(self):
        """输入特征数"""
        return self.X.shape[2]

    @property
    def lookback(self):
        """回看窗口"""
        return self.X.shape[1]

    @property
    def target_idx(self):
        """目标列索引"""
        return self.meta['target_idx']


def create_dataloaders(data_dir, dataset_name, horizon, batch_size=32, num_workers=0):
    """创建训练/验证/测试数据加载器

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        horizon: 预测步长
        batch_size: 批次大小
        num_workers: 工作进程数

    Returns:
        train_loader, val_loader, test_loader, input_size
    """
    train_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, 'train')
    val_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, 'val')
    test_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, 'test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    input_size = train_dataset.input_size

    return train_loader, val_loader, test_loader, input_size
