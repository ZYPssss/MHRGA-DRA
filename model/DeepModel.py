import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np

class DeepModel(nn.Module):
    def __init__(self, num_features, input_dim, hidden_channels, num_heads, num_layers=2, dropout=0.6, use_edge_weights=True):
        super(DeepModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights
        self.num_nodes = num_features
        self.hidden_channels = hidden_channels
        self.node_embeddings = nn.Embedding(num_features, input_dim)

        # 第一层GAT - 多头注意力
        self.conv1 = GATConv(
            input_dim,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1 if use_edge_weights else None  # 添加edge_dim支持边权重
        )

        # 中间层（如果有多个层）
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=1 if use_edge_weights else None
                )
            )

        # 最后一层 - 单头注意力用于输出
        if num_layers > 1:
            self.conv_out = GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                edge_dim=1 if use_edge_weights else None
            )
        else:
            self.conv_out = self.conv1

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_channels + 64 * 3, 128),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.hidden_channels)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, 64),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, edge_index, RNAs_index, drugs_index, args, z_mol, z_elem, z_drug, edge_weight=None):
        # 如果使用边权重，将其传递给GAT层
        node_ids = torch.arange(self.num_nodes, device=self.node_embeddings.weight.device)
        x = self.node_embeddings(node_ids)
        if self.use_edge_weights and edge_weight is not None:
            # 第一层
            x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 中间层
            for conv in self.convs:
                x = F.elu(conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
                x = F.dropout(x, p=self.dropout, training=self.training)

            # 输出层
            if self.num_layers > 1:
                x = self.conv_out(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
        else:
            # 不使用边权重
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

            for conv in self.convs:
                x = F.elu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.num_layers > 1:
                x = self.conv_out(x, edge_index)
        All_RNAs_embedding = x[:args.num_RNAs]
        All_Drugs_embedding = x[args.num_RNAs: args.num_RNAs + args.num_drugs]
        RNAs_embedding = All_RNAs_embedding[RNAs_index]
        Drugs_embedding = All_Drugs_embedding[drugs_index]
        z_m = z_mol[drugs_index]
        z_e = z_elem[drugs_index]
        z_d = z_drug[drugs_index]
        Drugs_embedding = torch.cat((z_m, z_e, z_d, Drugs_embedding), dim=1)
        Drugs_embedding = self.fusion(Drugs_embedding)
        drug_RNA_embedding = torch.cat((RNAs_embedding, Drugs_embedding), dim=1)
        predictions = self.predictor(drug_RNA_embedding)
        predictions = torch.sigmoid(predictions)
        return predictions



class DeepModel_Visual(nn.Module):
    def __init__(self, num_features, input_dim, hidden_channels, num_heads, num_layers=2, dropout=0.6, use_edge_weights=True):
        super(DeepModel_Visual, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights
        self.num_nodes = num_features
        self.hidden_channels = hidden_channels
        self.node_embeddings = nn.Embedding(num_features, input_dim)

        # 第一层GAT - 多头注意力
        self.conv1 = GATConv(
            input_dim,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1 if use_edge_weights else None  # 添加edge_dim支持边权重
        )

        # 中间层（如果有多个层）
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=1 if use_edge_weights else None
                )
            )

        # 最后一层 - 单头注意力用于输出
        if num_layers > 1:
            self.conv_out = GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                edge_dim=1 if use_edge_weights else None
            )
        else:
            self.conv_out = self.conv1

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_channels + 64 *3, 128),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.hidden_channels)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, 64),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )

    def forward(self, edge_index, RNAs_index, drugs_index, args, z_mol, z_elem, z_drug, edge_weight=None):
        # 如果使用边权重，将其传递给GAT层
        node_ids = torch.arange(self.num_nodes, device=self.node_embeddings.weight.device)
        x = self.node_embeddings(node_ids)
        if self.use_edge_weights and edge_weight is not None:
            # 第一层
            x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 中间层
            for conv in self.convs:
                x = F.elu(conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
                x = F.dropout(x, p=self.dropout, training=self.training)

            # 输出层
            if self.num_layers > 1:
                x = self.conv_out(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
        else:
            # 不使用边权重
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

            for conv in self.convs:
                x = F.elu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.num_layers > 1:
                x = self.conv_out(x, edge_index)
        All_RNAs_embedding = x[:args.num_RNAs]
        All_Drugs_embedding = x[args.num_RNAs: args.num_RNAs + args.num_drugs]
        RNAs_embedding = All_RNAs_embedding[RNAs_index]
        Drugs_embedding = All_Drugs_embedding[drugs_index]
        z_m = z_mol[drugs_index]
        z_e = z_elem[drugs_index]
        z_d = z_drug[drugs_index]
        Drugs_embedding = torch.cat((z_m, z_e, z_d, Drugs_embedding), dim=1)
        Drugs_embedding = self.fusion(Drugs_embedding)
        drug_RNA_embedding = torch.cat((RNAs_embedding, Drugs_embedding), dim=1)
        predictions = self.predictor(drug_RNA_embedding)
        predictions = torch.sigmoid(predictions)
        return predictions, drug_RNA_embedding








