import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Random Forest
class NeuralRandomForest(nn.Module):
    def __init__(self, input_dim, num_trees=100, tree_depth=5, num_classes=2):
        super(NeuralRandomForest, self).__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.num_classes = num_classes

        # 每个"树"是一个小的神经网络
        self.trees = nn.ModuleList([
            self._build_tree(input_dim, tree_depth, num_classes)
            for _ in range(num_trees)
        ])

        # 随机特征选择（模拟随机森林的特征采样）
        self.feature_masks = [
            torch.randperm(input_dim)[:max(1, input_dim // 3)]  # 选择约1/3的特征
            for _ in range(num_trees)
        ]

    def _build_tree(self, input_dim, depth, num_classes):
        layers = []
        current_dim = input_dim

        # 构建树的深度结构
        for i in range(depth):
            layers.extend([
                nn.Linear(current_dim, current_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim *= 2

        # 输出层
        layers.append(nn.Linear(current_dim, num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []

        for i, tree in enumerate(self.trees):
            # 应用特征mask
            masked_x = x[:, self.feature_masks[i]]

            # 通过单个树
            tree_output = tree(masked_x)
            outputs.append(F.softmax(tree_output, dim=1))

        # 集成所有树的输出（平均）
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

# SVM
class NeuralSVM(nn.Module):
    def __init__(self, input_dim, num_classes=2, margin=1.0):
        super(NeuralSVM, self).__init__()
        self.margin = margin

        # SVM可以看作是一个线性层 + Hinge Loss
        self.linear = nn.Linear(input_dim, num_classes)

        # 添加非线性能力
        self.non_linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 可以选择使用线性或非线性版本
        return self.non_linear(x)  # 或者 return self.linear(x)

class SVMLoss(nn.Module):
    def __init__(self, margin=1.0, regularization=0.01):
        super(SVMLoss, self).__init__()
        self.margin = margin
        self.regularization = regularization

    def forward(self, outputs, targets):
        # 将目标转换为one-hot编码
        batch_size = outputs.size(0)
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # 计算Hinge Loss
        correct_scores = outputs.gather(1, targets.unsqueeze(1)).squeeze()
        margins = outputs - correct_scores.unsqueeze(1) + self.margin
        margins = margins * (1 - targets_one_hot)  # 忽略正确类别的margin
        margins = F.relu(margins)

        hinge_loss = margins.sum(dim=1).mean()

        return hinge_loss
# 使用示例

# GBDT
class NeuralGradientBoosting(nn.Module):
    def __init__(self, input_dim, num_weak_learners=50, hidden_dim=64, num_classes=2, learning_rate=0.1):
        super(NeuralGradientBoosting, self).__init__()
        self.num_weak_learners = num_weak_learners
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # 弱学习器集合（小型神经网络）
        self.weak_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_weak_learners)
        ])

        # 每个弱学习器的权重
        self.learner_weights = nn.Parameter(torch.ones(num_weak_learners) / num_weak_learners)

        # 初始预测
        self.initial_prediction = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, num_learners=None):
        if num_learners is None:
            num_learners = self.num_weak_learners

        # 初始预测
        batch_size = x.size(0)
        predictions = self.initial_prediction.unsqueeze(0).expand(batch_size, -1)

        # 逐步添加弱学习器的预测
        for i in range(min(num_learners, self.num_weak_learners)):
            weak_pred = self.weak_learners[i](x)
            weighted_pred = weak_pred * self.learner_weights[i]
            predictions = predictions + self.learning_rate * weighted_pred

        return F.softmax(predictions, dim=1)

class GradientBoostingTrainer:
    def __init__(self, model, criterion, lr=0.01):
        self.model = model
        self.criterion = criterion
        self.lr = lr

    def train_sequential(self, train_loader, epochs_per_learner=10):
        self.model.train()
        # 初始预测训练
        print("Training initial prediction...")
        initial_optimizer = torch.optim.Adam([self.model.initial_prediction], lr=self.lr)
        self._train_single_component(train_loader, initial_optimizer, epochs_per_learner)

        # 顺序训练每个弱学习器
        for i, learner in enumerate(self.model.weak_learners):
            print(f"Training weak learner {i + 1}/{self.model.num_weak_learners}")

            # 冻结之前的学习器
            for prev_learner in self.model.weak_learners[:i]:
                for param in prev_learner.parameters():
                    param.requires_grad = False

            # 训练当前学习器
            optimizer = torch.optim.Adam(learner.parameters(), lr=self.lr)
            self._train_single_component(train_loader, optimizer, epochs_per_learner, learner_idx=i)

    def _train_single_component(self, train_loader, optimizer, epochs, learner_idx=None):
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()

                if learner_idx is not None:
                    # 使用指定数量的学习器进行预测
                    outputs = self.model(batch_x, num_learners=learner_idx + 1)
                else:
                    outputs = self.model(batch_x, num_learners=0)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0:
                print(f'  Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')

# 数据准备
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]# 示例使用
