import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
from sklearn.model_selection import train_test_split
from pre_train_Multi_View_E_model import *

def load_your_data(path, random_seed = 42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    matrix = pd.read_csv(path, index_col=0).values
    # 提取1和0的索引
    ones_indices = np.argwhere(matrix == 1)
    zeros_indices = np.argwhere(matrix == 0)

    # 转换为[行,列,值]格式
    ones_data = [[idx[0], idx[1], 1] for idx in ones_indices]
    zeros_data = [[idx[0], idx[1], 0] for idx in zeros_indices]

    # 平衡采样
    sampled_zeros = random.sample(zeros_data, len(ones_data))
    all_data = ones_data + sampled_zeros
    random.shuffle(all_data)
    # 转换为numpy数组
    data_array = np.array(all_data)
    return data_array

class MyDataset(Dataset):
    def __init__(self, RNA_s_f, drug_s_f, RNA_index, drug_index, label):
        super(MyDataset, self).__init__()
        self.RNA_s_f = RNA_s_f
        self.drug_s_f = drug_s_f
        self.drug_index = drug_index
        self.RNA_index = RNA_index
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.RNA_s_f[self.RNA_index[index]], self.drug_s_f[self.drug_index[index]], self.label[index]

class MyDataset1(Dataset):
    def __init__(self, RNA_s_f, drug_s_f, RNA_index, drug_index):
        super(MyDataset1, self).__init__()
        self.RNA_s_f = RNA_s_f
        self.drug_s_f = drug_s_f
        self.drug_index = drug_index
        self.RNA_index = RNA_index
    def __len__(self):
        return len(self.drug_index)
    def __getitem__(self, index):
        return self.RNA_s_f[self.RNA_index[index]], self.drug_s_f[self.drug_index[index]], self.RNA_index[index], self.drug_index[index]

class MyDataset2(Dataset):
    def __init__(self, RNA_index, drug_index, label):
        super(MyDataset2, self).__init__()
        self.drug_index = drug_index
        self.RNA_index = RNA_index
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return  self.RNA_index[index], self.drug_index[index], self.label[index]

def getDataload_train_ME(RNA_s_f, drug_s_f, RNA_index, drug_index, label, batch_size):
    dataset = MyDataset(RNA_s_f, drug_s_f, RNA_index, drug_index, label)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0)
    return dataset

def getDataload_add(RNA_s_f, drug_s_f, RNA_index, drug_index, batch_size):
    dataset = MyDataset1(RNA_s_f, drug_s_f, RNA_index, drug_index)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0)
    return dataset

def getDataload(RNA_index, drug_index, label, batch_size):
    dataset = MyDataset2(RNA_index, drug_index, label)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0)
    return dataset

def convert_to_bipartite_adjacency(matrix, normalize=True):
    """
    将 n×m 矩阵转换为 (n+m)×(n+m) 的二分图邻接矩阵

    Args:
        matrix: numpy数组或torch张量，形状为 [n, m]
        normalize: 是否对矩阵进行归一化

    Returns:
        adj_matrix: 形状为 [n+m, n+m] 的对称邻接矩阵
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    n, m = matrix.shape

    # 创建全零的 (n+m)×(n+m) 矩阵
    adj_matrix = np.zeros((n + m, n + m))

    # 填充右上块 (n×m)
    adj_matrix[:n, n:] = matrix

    # 填充左下块 (m×n) - 转置
    adj_matrix[n:, :n] = matrix.T

    # 如果需要归一化
    if normalize:
        # 对矩阵进行归一化，确保值在0-1之间
        if np.max(adj_matrix) > 0:
            adj_matrix = adj_matrix / np.max(adj_matrix)

    return adj_matrix

def adjacency_matrix_to_edge_index(adj_matrix, threshold=0.1, use_weights=True):
    """
    将关联矩阵转换为PyG需要的edge_index格式
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)

    n_nodes = adj_matrix.shape[0]

    # 创建掩码
    mask = (adj_matrix > threshold)

    # 移除自环
    mask = mask & ~torch.eye(n_nodes, dtype=torch.bool)

    # 获取非零元素的索引和值
    edge_index = mask.nonzero(as_tuple=False).t()

    if use_weights:
        edge_weight = adj_matrix[mask]
        return edge_index, edge_weight
    else:
        return edge_index, None

def convert_to_bipartite_adjacency_with_diagonal(matrix, normalize=True, self_connection_weight=1.0):
    """
    将 n×m 矩阵转换为 (n+m)×(n+m) 的二分图邻接矩阵，并添加自连接

    Args:
        matrix: numpy数组或torch张量，形状为 [n, m]
        normalize: 是否对矩阵进行归一化
        self_connection_weight: 自连接的权重

    Returns:
        adj_matrix: 形状为 [n+m, n+m] 的对称邻接矩阵
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    n, m = matrix.shape

    # 创建全零的 (n+m)×(n+m) 矩阵
    adj_matrix = np.zeros((n + m, n + m))

    # 填充右上块 (n×m)
    adj_matrix[:n, n:] = matrix

    # 填充左下块 (m×n) - 转置
    adj_matrix[n:, :n] = matrix.T

    # 添加自连接（对角线）
    np.fill_diagonal(adj_matrix, self_connection_weight)

    # 如果需要归一化
    if normalize:
        # 对矩阵进行归一化，确保值在0-1之间
        if np.max(adj_matrix) > 0:
            adj_matrix = adj_matrix / np.max(adj_matrix)

    return adj_matrix

def get_Multi_view_mol_data(model, device):

    model.eval()
    # 初始化图构建器
    graph_builder = MultiViewGraphBuilder()

    # 加载数据
    print("Loading drug data...")
    dataset = HeteroMolecularDataset('./data/DRUG.xlsx', graph_builder, max_samples=500)  # 限制样本数量以加速训练
    print(f"Loaded {len(dataset)} valid molecules")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=400, shuffle=False, collate_fn=hetero_collate_fn  # 小批量以处理异构数据
    )

    with torch.no_grad():
        for batch in dataloader:
            # 包装异构数据
            # molecule_batch = SimpleHeteroWrapper(batch['molecule']).to(device)
            # element_batch = SimpleHeteroWrapper(batch['element']).to(device)
            # drug_batch = SimpleHeteroWrapper(batch['drug']).to(device)

            # 训练单个样本（因为异构数据批处理复杂）
            batch_loss = 0
            z_mol = []
            z_elem = []
            z_drug = []
            for i in range(len(batch['molecule'])):
                mol_data = SimpleHeteroWrapper([batch['molecule'][i]]).to(device)
                elem_data = SimpleHeteroWrapper([batch['element'][i]]).to(device)
                drug_data = SimpleHeteroWrapper([batch['drug'][i]]).to(device)

                # 前向传播
                z_m, z_e, z_d = model(mol_data, elem_data, drug_data, mode='pretrain')
                z_mol.append(z_m)
                z_elem.append(z_e)
                z_drug.append(z_d)

                # 计算对比损失
            z_mol = torch.cat(z_mol, dim=0)
            z_elem = torch.cat(z_elem, dim=0)
            z_drug = torch.cat(z_drug, dim=0)
    return  z_mol, z_elem, z_drug

def drug_cold_split(indexs, labels, num_drugs, test_size=0.2, random_state=42):

    all_drugs = np.arange(num_drugs)

    train_drugs, test_drugs = train_test_split(
        all_drugs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_mask = np.isin(indexs[:,1], train_drugs)
    test_mask = np.isin(indexs[:,1], test_drugs)

    train_index = indexs[train_mask]
    train_label = labels[train_mask]

    test_index = indexs[test_mask]
    test_label = labels[test_mask]

    return train_index, train_label, test_index, test_label, train_drugs, test_drugs

def rna_cold_split(indexs, labels, num_rnas, test_size=0.2, random_state=42):

    all_rnas = np.arange(num_rnas)

    train_rnas, test_rnas = train_test_split(
        all_rnas,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_mask = np.isin(indexs[:,0], train_rnas)
    test_mask = np.isin(indexs[:,0], test_rnas)

    train_index = indexs[train_mask]
    train_label = labels[train_mask]

    test_index = indexs[test_mask]
    test_label = labels[test_mask]

    return train_index, train_label, test_index, test_label, train_rnas, test_rnas

def both_cold_split(indexs, labels, num_rnas, num_drugs, test_size=0.2, random_state=42):

    all_rnas = np.arange(num_rnas)
    all_drugs = np.arange(num_drugs)

    train_rnas, test_rnas = train_test_split(
        all_rnas,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_drugs, test_drugs = train_test_split(
        all_drugs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_mask = (
        np.isin(indexs[:,0], train_rnas) &
        np.isin(indexs[:,1], train_drugs)
    )

    test_mask = (
        np.isin(indexs[:,0], test_rnas) &
        np.isin(indexs[:,1], test_drugs)
    )

    train_index = indexs[train_mask]
    train_label = labels[train_mask]

    test_index = indexs[test_mask]
    test_label = labels[test_mask]

    return train_index, train_label, test_index, test_label