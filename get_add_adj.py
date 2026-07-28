import os

import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.model_selection import StratifiedKFold
from dataloader import load_your_data, getDataload_add, getDataload, drug_cold_split, rna_cold_split, both_cold_split
from train_test import Add_edge_weigth,train_model
import joblib
from model.DeepModel import *

from model.Machine_Learning_Enhancement import NeuralSVM, NeuralRandomForest, NeuralGradientBoosting

random_seed = 42
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="cell-drug training script.")
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




args = parser.parse_args()
argsdict = vars(args)

def get_warm_add_adj():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    # 使用分层5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = random_seed)

    results = []
    flod = 1
    n = args.num_RNAs
    m = args.num_drugs
    for train_idx, test_idx in skf.split(indexs, labels):
        # 获取训练和测试数据
        print('-----------------------flod{}--------------------'.format(flod))
        index_train, index_test = indexs[train_idx], indexs[test_idx]
        label_train, label_test = labels[train_idx], labels[test_idx]
        train_set = set(tuple(idx) for idx in index_train)
        adj = np.zeros((n, m))
        # 生成所有其他索引
        other_indices = []
        for i in range(n):
            for j in range(m):
                if (i, j) not in train_set:
                    other_indices.append([i, j])
        index_add = np.array(other_indices)

        for i in range(len(index_train)):
            adj[index_train[i][0]][index_train[i][1]] = label_train[i]

        train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
        # test_dataloader = getDataload(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test, args.batch)
        add_dataloader = getDataload_add(RNA_s_f, drug_s_f, index_add[:, 0], index_add[:, 1], args.batch)

        #rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)

        rf_model = joblib.load('./pretrain_model/warm/random_forest_model{}.pkl'.format(flod))
        svm_model = joblib.load('./pretrain_model/warm/svm_model{}.pkl'.format(flod))
        gb_model = joblib.load('./pretrain_model/warm/gradient_boosting_model{}.pkl'.format(flod))
        add_adj = Add_edge_weigth(adj, add_dataloader, args, rf_model, svm_model, gb_model, device)

        folder_path = './pretrain_data/warm'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + '/add_adj{}.npy'.format(flod), add_adj)

        flod += 1

def get_drug_cold_add_adj():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    # 使用分层5折交叉验证

    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # 获取训练和测试数据
    index_train, label_train, index_test, label_test, train_drugs, test_drugs = drug_cold_split(indexs, labels, len(drug_s_f), test_size=0.2, random_state=random_seed)
    train_set = set(tuple(idx) for idx in index_train)
    adj = np.zeros((n, m))
    # 生成所有其他索引
    other_indices = []
    for i in range(n):
        for j in range(m):
            if (i, j) not in train_set:
                other_indices.append([i, j])
    index_add = np.array(other_indices)

    for i in range(len(index_train)):
        adj[index_train[i][0]][index_train[i][1]] = label_train[i]

    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    # test_dataloader = getDataload(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test, args.batch)
    add_dataloader = getDataload_add(RNA_s_f, drug_s_f, index_add[:, 0], index_add[:, 1], args.batch)

    # rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)

    rf_model = joblib.load('./pretrain_model/drug_cold/random_forest_model.pkl')
    svm_model = joblib.load('./pretrain_model/drug_cold/svm_model.pkl')
    gb_model = joblib.load('./pretrain_model/drug_cold/gradient_boosting_model.pkl')
    add_adj = Add_edge_weigth(adj, add_dataloader, args, rf_model, svm_model, gb_model, device)

    folder_path = './pretrain_data/drug_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path + '/add_adj.npy', add_adj)

def get_rna_cold_add_adj():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    # 使用分层5折交叉验证

    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # 获取训练和测试数据
    index_train, label_train, index_test, label_test, train_rnas, test_rnas = rna_cold_split(indexs, labels, len(RNA_s_f), test_size=0.2, random_state=random_seed)
    train_set = set(tuple(idx) for idx in index_train)
    adj = np.zeros((n, m))
    # 生成所有其他索引
    other_indices = []
    for i in range(n):
        for j in range(m):
            if (i, j) not in train_set:
                other_indices.append([i, j])
    index_add = np.array(other_indices)

    for i in range(len(index_train)):
        adj[index_train[i][0]][index_train[i][1]] = label_train[i]

    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    # test_dataloader = getDataload(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test, args.batch)
    add_dataloader = getDataload_add(RNA_s_f, drug_s_f, index_add[:, 0], index_add[:, 1], args.batch)

    # rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)

    rf_model = joblib.load('./pretrain_model/rna_cold/random_forest_model.pkl')
    svm_model = joblib.load('./pretrain_model/rna_cold/svm_model.pkl')
    gb_model = joblib.load('./pretrain_model/rna_cold/gradient_boosting_model.pkl')
    add_adj = Add_edge_weigth(adj, add_dataloader, args, rf_model, svm_model, gb_model, device)

    folder_path = './pretrain_data/rna_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path + '/add_adj.npy', add_adj)

def get_both_cold_add_adj():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    # 使用分层5折交叉验证

    results = []
    n = args.num_RNAs
    m = args.num_drugs
    # 获取训练和测试数据
    index_train, label_train, index_test, label_test = both_cold_split(indexs, labels, len(RNA_s_f), len(drug_s_f), test_size=0.1, random_state=random_seed)
    train_set = set(tuple(idx) for idx in index_train)
    adj = np.zeros((n, m))
    # 生成所有其他索引
    other_indices = []
    for i in range(n):
        for j in range(m):
            if (i, j) not in train_set:
                other_indices.append([i, j])
    index_add = np.array(other_indices)

    for i in range(len(index_train)):
        adj[index_train[i][0]][index_train[i][1]] = label_train[i]

    train_dataloader = getDataload(index_train[:, 0], index_train[:, 1], label_train, args.batch)
    # test_dataloader = getDataload(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test, args.batch)
    add_dataloader = getDataload_add(RNA_s_f, drug_s_f, index_add[:, 0], index_add[:, 1], args.batch)

    # rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)

    rf_model = joblib.load('./pretrain_model/both_cold/random_forest_model.pkl')
    svm_model = joblib.load('./pretrain_model/both_cold/svm_model.pkl')
    gb_model = joblib.load('./pretrain_model/both_cold/gradient_boosting_model.pkl')
    add_adj = Add_edge_weigth(adj, add_dataloader, args, rf_model, svm_model, gb_model, device)

    folder_path = './pretrain_data/both_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path + '/add_adj.npy', add_adj)


if __name__ == '__main__':
    # get_warm_add_adj()
    # get_drug_cold_add_adj()
    # get_rna_cold_add_adj()
    get_both_cold_add_adj()




