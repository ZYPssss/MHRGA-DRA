import os

import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.model_selection import StratifiedKFold
from dataloader import load_your_data, getDataload_train_ME
from train_test import Machine_learning_Enhancement_train, Machine_learning_Enhancement_test
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import joblib
#from thundersvm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from dataloader import drug_cold_split, rna_cold_split, both_cold_split
from model.Machine_Learning_Enhancement import NeuralSVM, NeuralRandomForest, NeuralGradientBoosting

random_seed = 42
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description="cell-drug training script.")
parser.add_argument("--data_path", default='./data/')
parser.add_argument("--batch", type=int, default=20000)
parser.add_argument("--m_input_dim", type=int, default=6656)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--rf_epochs", type=int, default=1000)
parser.add_argument("--svm_epochs", type=int, default=1000)
parser.add_argument("--gb_epochs", type=int, default=1000)
parser.add_argument("--rf_lr", type=float, default=0.001)
parser.add_argument("--svm_lr", type=float, default=0.001)
parser.add_argument("--gb_lr", type=float, default=0.001)



args = parser.parse_args()
argsdict = vars(args)

def pre_train_EML():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed = random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    # 使用分层5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = random_seed)

    results = []
    flod = 1
    for train_idx, test_idx in skf.split(indexs, labels):
        # 获取训练和测试数据
        print('-----------------------flod{}--------------------'.format(flod))
        index_train, index_test = indexs[train_idx], indexs[test_idx]
        label_train, label_test = labels[train_idx], labels[test_idx]
        train_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_train[:, 0], index_train[:, 1], label_train, args.batch)
        test_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test, args.batch)


        # 初始化机器学习模型
        # rf_model = NeuralRandomForest(args.m_input_dim, num_trees=50, num_classes=args.num_classes)
        # svm_model = NeuralSVM(args.m_input_dim, num_classes=args.num_classes)
        # gb_model = NeuralGradientBoosting(args.m_input_dim, num_weak_learners=30, num_classes=args.num_classes)

        rf_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
        gb_model = GradientBoostingClassifier()
        svm_model = svm.SVC(kernel='rbf', gamma=20, probability=True, max_iter=1000)

        rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)
        folder_path = './pretrain_model/warm'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        joblib.dump(rf_model, folder_path + '/random_forest_model{}.pkl'.format(flod))
        joblib.dump(svm_model, folder_path + '/svm_model{}.pkl'.format(flod))
        joblib.dump(gb_model, folder_path + '/gradient_boosting_model{}.pkl'.format(flod))
        #Machine_learning_Enhancement_test(test_dataloader, args, rf_model, svm_model, gb_model, device)
        flod += 1


def pre_train_EML_drug_cold():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed=random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    results = []
    # 获取训练和测试数据
    print('-----------------------train_drug_cold--------------------')
    index_train, label_train, index_test, label_test, train_drugs, test_drugs = drug_cold_split(indexs, labels, len(drug_s_f), test_size=0.2, random_state=random_seed)
    train_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_train[:, 0], index_train[:, 1], label_train,
                                            args.batch)
    test_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test,
                                           args.batch)

    rf_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    gb_model = GradientBoostingClassifier()
    svm_model = svm.SVC(kernel='rbf', gamma=20, probability=True, max_iter=1000)

    rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model,
                                                                       gb_model, device)

    folder_path = './pretrain_model/drug_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    joblib.dump(rf_model, folder_path + '/random_forest_model.pkl')
    joblib.dump(svm_model, folder_path + '/svm_model.pkl')
    joblib.dump(gb_model, folder_path + '/gradient_boosting_model.pkl')


    Machine_learning_Enhancement_test(test_dataloader, args, rf_model, svm_model, gb_model, device)

def pre_train_EML_rna_cold():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed=random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    results = []
    # 获取训练和测试数据
    print('-----------------------train_rna_cold--------------------')
    index_train, label_train, index_test, label_test, train_rnas, test_rnas = rna_cold_split(indexs, labels, len(RNA_s_f), test_size=0.2, random_state=random_seed)
    train_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_train[:, 0], index_train[:, 1], label_train,
                                            args.batch)
    test_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test,
                                           args.batch)

    rf_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    gb_model = GradientBoostingClassifier()
    svm_model = svm.SVC(kernel='rbf', gamma=20, probability=True, max_iter=1000)

    rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model,
                                                                       gb_model, device)

    folder_path = './pretrain_model/rna_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    joblib.dump(rf_model, folder_path + '/random_forest_model.pkl')
    joblib.dump(svm_model, folder_path + '/svm_model.pkl')
    joblib.dump(gb_model, folder_path + '/gradient_boosting_model.pkl')

    Machine_learning_Enhancement_test(test_dataloader, args, rf_model, svm_model, gb_model, device)

def pre_train_EML_both_cold():
    drug_s_f = np.loadtxt(argsdict['data_path'] + 'DRUG_similarity.csv', delimiter=',', dtype=float)
    RNA_s_f = np.loadtxt(argsdict['data_path'] + 'RNA_similarity.csv', delimiter=',', dtype=float)

    data_array = load_your_data(argsdict['data_path'] + 'DRUG_RNA_MATRIX.csv', random_seed=random_seed)

    indexs = data_array[:, :2]  # 行和列索引
    labels = data_array[:, 2]  # 标签值

    results = []
    # 获取训练和测试数据
    print('-----------------------train_both_cold--------------------')
    index_train, label_train, index_test, label_test = both_cold_split(indexs, labels, len(RNA_s_f), len(drug_s_f), test_size=0.1, random_state=random_seed)
    train_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_train[:, 0], index_train[:, 1], label_train,
                                            args.batch)
    test_dataloader = getDataload_train_ME(RNA_s_f, drug_s_f, index_test[:, 0], index_test[:, 1], label_test,
                                           args.batch)

    rf_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, max_features=0.2)
    gb_model = GradientBoostingClassifier()
    svm_model = svm.SVC(kernel='rbf', gamma=20, probability=True, max_iter=1000)

    rf_model, svm_model, gb_model = Machine_learning_Enhancement_train(train_dataloader, args, rf_model, svm_model, gb_model, device)

    folder_path = './pretrain_model/both_cold'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    joblib.dump(rf_model, folder_path + '/random_forest_model.pkl')
    joblib.dump(svm_model, folder_path + '/svm_model.pkl')
    joblib.dump(gb_model, folder_path + '/gradient_boosting_model.pkl')


    Machine_learning_Enhancement_test(test_dataloader, args, rf_model, svm_model, gb_model, device)





if __name__ == '__main__':
    # pre_train_EML()
    # pre_train_EML_drug_cold()
    # pre_train_EML_rna_cold()
    pre_train_EML_both_cold()


