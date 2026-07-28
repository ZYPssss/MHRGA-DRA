import random
import argparse
from torch.optim import Adam
from dataloader import load_your_data, getDataload, convert_to_bipartite_adjacency, adjacency_matrix_to_edge_index, get_Multi_view_mol_data
from train_test import train_test_cold
from model.DeepModel import *
from dataloader import drug_cold_split, rna_cold_split, both_cold_split
def set_seed(seed=42):
    # 基础设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

random_seed = 42
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="RNA-drug training script.")
parser.add_argument("--data_path", default='./data/')
parser.add_argument("--batch", type=int, default=20000)
parser.add_argument("--m_input_dim", type=int, default=6656)
parser.add_argument("--input_dim", type=int, default=64)
parser.add_argument("--channels_dim", type=int, default=64)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--rf_epochs", type=int, default=1000)
parser.add_argument("--svm_epochs", type=int, default=1000)
parser.add_argument("--gb_epochs", type=int, default=1000)
parser.add_argument("--num_RNAs", type=int, default=6430)
parser.add_argument("--num_drugs", type=int, default=316)
parser.add_argument("--rf_lr", type=float, default=0.001)
parser.add_argument("--svm_lr", type=float, default=0.001)
parser.add_argument("--gb_lr", type=float, default=0.001)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--cold_type", type=str, default='rna_cold')




args = parser.parse_args()
argsdict = vars(args)


def main_drug_cold():

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)
    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值


    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # Multi_view_E_model = torch.load('./model/trained_model/Multi_view_E_model.pth', map_location=device)
    # z_mol, z_elem, z_drug = get_Multi_view_mol_data(Multi_view_E_model, device)
    z_mol = torch.load('./pretrain_data/z_mol.pth', map_location=device)
    z_elem = torch.load('./pretrain_data/z_elem.pth', map_location=device)
    z_drug = torch.load('./pretrain_data/z_drug.pth', map_location=device)

    index_train, label_train, index_test, label_test, _, _ = drug_cold_split(indexs, labels, m, test_size=0.2, random_state=random_seed)


    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    test_dataloader = getDataload(index_test[:, 0], index_test[:, 1], label_test, args.batch)

    add_adj = np.load('./pretrain_data/{}/add_adj.npy'.format(args.cold_type))

    # 1. 转换为二分图邻接矩阵
    bipartite_adj = convert_to_bipartite_adjacency(add_adj, normalize=False)

    # 2. 转换为边索引和边权重
    edge_index, edge_weight = adjacency_matrix_to_edge_index(bipartite_adj, 0.4)

    # 4. 初始化支持边权重的GAT模型
    model = DeepModel(
        num_features=n + m,
        input_dim=args.input_dim,
        hidden_channels=args.channels_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.6,
        use_edge_weights=True
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_test_cold(model, train_dataloader, test_dataloader, edge_index, edge_weight, args, criterion, optimizer, 200,
               device, z_mol, z_elem, z_drug, args.cold_type)

def main_rna_cold():

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)
    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值


    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # Multi_view_E_model = torch.load('./model/trained_model/Multi_view_E_model.pth', map_location=device)
    # z_mol, z_elem, z_drug = get_Multi_view_mol_data(Multi_view_E_model, device)
    z_mol = torch.load('./pretrain_data/z_mol.pth', map_location=device)
    z_elem = torch.load('./pretrain_data/z_elem.pth', map_location=device)
    z_drug = torch.load('./pretrain_data/z_drug.pth', map_location=device)

    index_train, label_train, index_test, label_test, _, _ = rna_cold_split(indexs, labels, n, test_size=0.2, random_state=random_seed)


    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    test_dataloader = getDataload(index_test[:, 0], index_test[:, 1], label_test, args.batch)

    add_adj = np.load('./pretrain_data/{}/add_adj.npy'.format(args.cold_type))

    # 1. 转换为二分图邻接矩阵
    bipartite_adj = convert_to_bipartite_adjacency(add_adj, normalize=False)

    # 2. 转换为边索引和边权重
    edge_index, edge_weight = adjacency_matrix_to_edge_index(bipartite_adj, 0.4)

    # 4. 初始化支持边权重的GAT模型
    model = DeepModel(
        num_features=n + m,
        input_dim=args.input_dim,
        hidden_channels=args.channels_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.6,
        use_edge_weights=True
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_test_cold(model, train_dataloader, test_dataloader, edge_index, edge_weight, args, criterion, optimizer, 200,
               device, z_mol, z_elem, z_drug, args.cold_type)


def main_both_cold():

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)
    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值


    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # Multi_view_E_model = torch.load('./model/trained_model/Multi_view_E_model.pth', map_location=device)
    # z_mol, z_elem, z_drug = get_Multi_view_mol_data(Multi_view_E_model, device)
    z_mol = torch.load('./pretrain_data/z_mol.pth', map_location=device)
    z_elem = torch.load('./pretrain_data/z_elem.pth', map_location=device)
    z_drug = torch.load('./pretrain_data/z_drug.pth', map_location=device)

    index_train, label_train, index_test, label_test = both_cold_split(indexs, labels, n, m, test_size=0.1, random_state=random_seed)


    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    test_dataloader = getDataload(index_test[:, 0], index_test[:, 1], label_test, args.batch)

    add_adj = np.load('./pretrain_data/{}/add_adj.npy'.format(args.cold_type))

    # 1. 转换为二分图邻接矩阵
    bipartite_adj = convert_to_bipartite_adjacency(add_adj, normalize=False)

    # 2. 转换为边索引和边权重
    edge_index, edge_weight = adjacency_matrix_to_edge_index(bipartite_adj, 0.4)

    # 4. 初始化支持边权重的GAT模型
    model = DeepModel(
        num_features=n + m,
        input_dim=args.input_dim,
        hidden_channels=args.channels_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.6,
        use_edge_weights=True
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_test_cold(model, train_dataloader, test_dataloader, edge_index, edge_weight, args, criterion, optimizer, 200,
               device, z_mol, z_elem, z_drug, args.cold_type)



if __name__ == '__main__':
    # args.cold_type = 'drug_cold'
    # main_drug_cold()
    # args.cold_type = 'rna_cold'
    # main_rna_cold()
    args.cold_type = 'both_cold'
    main_both_cold()