import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.Multi_View_Enhancement import *
class HeteroMolecularDataset(Dataset):
    """异构分子数据集"""

    def __init__(self, excel_file, graph_builder, max_samples=1000):
        self.df = pd.read_excel(excel_file)
        self.graph_builder = graph_builder

        # 限制样本数量以加速训练
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(max_samples, random_state=42)

        # 预处理数据
        self.process_data()

    def process_data(self):
        """处理数据"""
        self.valid_indices = []
        self.graphs = {'molecule': [], 'element': [], 'drug': []}

        print("Processing molecular data...")
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(self.df)} molecules")

            smiles = row['isosmiles']

            try:
                # 构建三个视图
                mol_graph = self.graph_builder.build_molecule_view(smiles)
                elem_graph = self.graph_builder.build_element_view(smiles)
                drug_graph = self.graph_builder.build_drug_view(smiles)

                if mol_graph is not None and elem_graph is not None and drug_graph is not None:
                    self.valid_indices.append(idx)
                    self.graphs['molecule'].append(mol_graph)
                    self.graphs['element'].append(elem_graph)
                    self.graphs['drug'].append(drug_graph)
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                continue

        print(f"Successfully processed {len(self.valid_indices)} molecules")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return {
            'molecule': self.graphs['molecule'][idx],
            'element': self.graphs['element'][idx],
            'drug': self.graphs['drug'][idx],
            'cid': self.df.iloc[actual_idx]['PUBCHEM_CID']
        }


def hetero_collate_fn(batch):
    """异构数据批处理函数"""
    # 对于异构数据，我们需要分别处理每个图
    # 这里简化处理：返回原始异构数据列表
    return {
        'molecule': [item['molecule'] for item in batch],
        'element': [item['element'] for item in batch],
        'drug': [item['drug'] for item in batch],
        'cids': [item['cid'] for item in batch]
    }


class SimpleHeteroWrapper:
    """简化异构数据包装器 - 用于批处理"""

    def __init__(self, hetero_data_list):
        self.hetero_data_list = hetero_data_list

        # 添加缺失的属性
        if hetero_data_list:
            # 使用第一个数据样本的属性
            self.node_types = hetero_data_list[0].node_types
            self.edge_types = hetero_data_list[0].edge_types
        else:
            self.node_types = []
            self.edge_types = []

    def to(self, device):
        """将数据移动到设备"""
        for data in self.hetero_data_list:
            for node_type in data.node_types:
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    data[node_type].x = data[node_type].x.to(device)

            for edge_type in data.edge_types:
                edge_store = data[edge_type]
                if hasattr(edge_store, 'edge_index') and edge_store.edge_index is not None:
                    edge_store.edge_index = edge_store.edge_index.to(device)
                if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                    edge_store.edge_attr = edge_store.edge_attr.to(device)
        # 确保包装器本身知道设备信息
        self.device = device

        return self

    # 添加这些方法以支持属性访问
    def __getitem__(self, key):
        # 如果key是节点类型，返回第一个数据样本的对应节点
        if isinstance(key, str) and key in self.node_types:
            return self.hetero_data_list[0][key]
        # 如果key是边类型元组，返回第一个数据样本的对应边
        elif isinstance(key, tuple) and len(key) == 3 and key in self.edge_types:
            return self.hetero_data_list[0][key]
        else:
            raise KeyError(f"Key {key} not found")

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.node_types
        elif isinstance(key, tuple) and len(key) == 3:
            return key in self.edge_types
        return False


# 示例使用
if __name__ == "__main__":
    # 检查设备
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化图构建器
    graph_builder = MultiViewGraphBuilder()

    # 加载数据
    print("Loading data...")
    dataset = HeteroMolecularDataset('./data/DRUG.xlsx', graph_builder, max_samples=500)  # 限制样本数量以加速训练
    print(f"Loaded {len(dataset)} valid molecules")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=100, shuffle=True, collate_fn=hetero_collate_fn  # 小批量以处理异构数据
    )

    # 修正节点维度定义
    node_dims = {
        'atom': 8,  # 修正为实际的原子特征维度
        'fragment': 4,  # 片段特征维度
        'element': 3,  # 元素特征维度
        'functional_group': 2,  # 官能团特征维度
        'dnode': 128  # 药物节点特征维度
    }

    # 边维度保持不变
    edge_dims = {
        ('atom', 'bond', 'atom'): 4,
        ('fragment', 'reaction', 'fragment'): 1,
        ('atom', 'join', 'fragment'): 1,
        ('atom', 'AE', 'element'): 1,
        ('fragment', 'FrFu', 'functional_group'): 1,
        ('element', 'EE', 'element'): 1,
        ('functional_group', 'FuFu', 'functional_group'): 1,
        ('element', 'EFu', 'functional_group'): 1,
        ('atom', 'AD', 'dnode'): 1,
        ('fragment', 'FrD', 'dnode'): 1
    }

    # 初始化模型
    print("Initializing model...")
    model = Multi_view_Heterogeneous_Encoder(
        node_dims=node_dims,
        edge_dims=edge_dims,
        hidden_dim=128,  # 减小隐藏维度以 加速训练
        projection_dim=64
    ).to(device)

    # 损失函数和训练器
    contrastive_loss = ContrastiveLoss(temperature=0.1)
    trainer = Trainer(model, contrastive_loss, device)

    # 训练对比学习
    print("Starting contrastive pre-training...")
    min_loss = 10000
    for epoch in range(50):  # 减少训练轮数以加速
        epoch_loss = 0
        batch_count = 0
        print('-----------------------------epoch:{}---------------------------------------'.format(epoch + 1))
        for batch in tqdm((dataloader), desc='Iteration'):
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
            loss_mol_elem = contrastive_loss(z_mol, z_elem)
            loss_mol_drug = contrastive_loss(z_mol, z_drug)
            loss_elem_drug = contrastive_loss(z_elem, z_drug)

            loss = loss_mol_elem + loss_mol_drug + loss_elem_drug
            batch_loss += loss.item()

            # 反向传播
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            epoch_loss += batch_loss / len(batch['molecule'])
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"Processed {batch_count} batches")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}, Average Contrastive Loss: {avg_loss:.4f}")
        if(min_loss > avg_loss):
            min_loss = avg_loss
            torch.save(model, './pretrain_model/Multi_view_E_model.pth')


    print("Pre-training completed!")
