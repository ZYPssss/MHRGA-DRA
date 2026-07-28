import os

from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

from model.Machine_Learning_Enhancement import SVMLoss

def train_neural_svm(model, train_data, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

def train_neural_random_forest(model, train_data, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

def train_neural_gb(model, train_data, criterion, device, epochs, args):
    model.train()
    # 初始预测训练
    print("Training initial prediction...")
    initial_optimizer = torch.optim.Adam([model.initial_prediction], lr=args.gb_lr)
    train_single_component(train_data, initial_optimizer, epochs, device)
    # 顺序训练每个弱学习器
    for i, learner in enumerate(model.weak_learners):
        # 冻结之前的学习器
        for prev_learner in model.weak_learners[:i]:
            for param in prev_learner.parameters():
                param.requires_grad = False

        # 训练当前学习器
        optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
        train_single_component(model, train_data, criterion, optimizer, epochs, device, learner_idx=i)

def train_single_component(model, train_data, criterion, optimizer, epochs, device, learner_idx=None):
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            if learner_idx is not None:
                    # 使用指定数量的学习器进行预测
                outputs = model(batch_x, num_learners=learner_idx + 1)
            else:
                outputs = model(batch_x, num_learners=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

def test_neural_svm(model, test_data, device):
    model.val()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            outputs = model(batch_x)
        auc = roc_auc_score(labels, outputs)
        aupr = average_precision_score(labels, outputs)
        print("auc:{}, aupr:{}".format(auc, aupr))

def test_neural_random_forest(model, test_data, device):
    model.val()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            outputs = model(batch_x)
            all_labels.extend(labels.cpu().numpy())

            # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(labels, outputs)
        aupr = average_precision_score(labels, outputs)
        print("auc:{}, aupr:{}".format(auc, aupr))

def test_neural_gb(model, test_data, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data, desc='Iteration')):
            RNAs_s_f, drugs_s_f, labels = data
            RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
            drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
            batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
            labels = torch.tensor(labels, dtype=torch.long).to(device)  # 改为long类型用于交叉熵

            # 使用所有弱学习器进行预测
            outputs = model(batch_x, num_learners=len(model.weak_learners))

            # 收集预测结果和真实标签
            # 取第二类的概率作为正类概率（假设类别1是正类）
            probs = outputs[:, 1].cpu().numpy()  # 正类概率
            all_predictions.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 计算评估指标

def train_machine_model(rf_model, svm_model, gb_model, train_data, train_label):
    print('train_rf_model')
    rf_model.fit(train_data, train_label)
    print('Finish train_rf_model')
    print('train_svm_model')
    svm_model.fit(train_data, train_label)
    print('Finish train_svm_model')
    print('train_rf_model')
    gb_model.fit(train_data, train_label)
    print('Finish train_gb_model')
    return rf_model, svm_model, gb_model

def test_machine_model(rf_model, svm_model, gb_model, test_data, test_label):
    Score_rf = rf_model.predict_proba(test_data)[:, 1]
    print("auc:{}, aupr:{}".format(roc_auc_score(test_label, Score_rf), average_precision_score(test_label, Score_rf)))
    Score_gb = gb_model.predict_proba(test_data)[:, 1]
    print("auc:{}, aupr:{}".format(roc_auc_score(test_label, Score_gb), average_precision_score(test_label, Score_gb)))
    Score_svm = svm_model.predict_proba(test_data)[:, 1]
    print("auc:{}, aupr:{}".format(roc_auc_score(test_label, Score_svm), average_precision_score(test_label, Score_svm)))

def Machine_learning_Enhancement_train(train_data, args, rf_model, svm_model, gb_model, device):
    for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
        RNAs_s_f, drugs_s_f, labels = data
        # RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
        # drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
        batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
        batch_x = batch_x.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        # labels = torch.tensor(labels, dtype=torch.float32).to(device)
        rf_model, svm_model, gb_model = train_machine_model(rf_model, svm_model, gb_model, batch_x, labels)

    return rf_model, svm_model, gb_model

    # train_neural_random_forest(rf_model, train_data, criterion_ce, optimizer_rf, device, args.rf_epochs)
    # train_neural_svm(svm_model, train_data, criterion_svm, optimizer_svm, device, args.svm_epochs)
    # train_neural_gb(gb_model, train_data, criterion_ce, device, args.gb_epochs, args)

def Machine_learning_Enhancement_test(test_data, args, rf_model, svm_model, gb_model, device):
    for idx, data in enumerate(tqdm(test_data, desc='Iteration')):
        RNAs_s_f, drugs_s_f, labels = data
        # RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
        # drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
        batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
        batch_x = batch_x.cpu().numpy().tolist()
        labels = labels.cpu().numpy()
        # labels = torch.tensor(labels, dtype=torch.float32).to(device)
        test_machine_model(rf_model, svm_model, gb_model, batch_x, labels)

def Add_edge_weigth(adj, add_data, args, rf_model, svm_model, gb_model, device):
    for idx, data in enumerate(tqdm(add_data, desc='Add_edge_weight')):
        RNAs_s_f, drugs_s_f, RNAs_index, drugs_index = data
        # RNAs_s_f = torch.tensor(RNAs_s_f, dtype=torch.float32).to(device)
        # drugs_s_f = torch.tensor(drugs_s_f, dtype=torch.float32).to(device)
        batch_x = torch.cat((RNAs_s_f, drugs_s_f), dim=1)
        batch_x = batch_x.cpu().numpy().tolist()
        # labels = torch.tensor(labels, dtype=torch.float32).to(device)
        P_rf = rf_model.predict_proba(batch_x)[:, 1]
        P_gb = gb_model.predict_proba(batch_x)[:, 1]
        P_svm = svm_model.predict_proba(batch_x)[:, 1]
        for i in range(len(RNAs_index)):
            adj[RNAs_index[i]][drugs_index[i]] = (P_rf[i] + P_gb[i] + P_svm[i]) / 3
    return adj

def train_model(model, train_data, edge_index, edge_weight, args, criterion, optimizer, device, z_mol, z_elem, z_drug):
    for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
        RNAs_index, drugs_index, labels = data
        RNAs_index = torch.tensor(RNAs_index,dtype=torch.long).to(device)
        drugs_index = torch.tensor(drugs_index, dtype=torch.long).to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        predicts = model(edge_index, RNAs_index, drugs_index, args, z_mol, z_elem, z_drug, edge_weight).squeeze()
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()
    return model


def test_model(model, test_data, edge_index, edge_weight, args, device, z_mol, z_elem, z_drug):
    model.eval()
    all_predictions = []
    all_labels = []
    all_indexs = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_data, desc='Iteration')):
            RNAs_index, drugs_index, labels = data
            all_indexs.extend(torch.stack((RNAs_index, drugs_index, labels), dim=1).cpu().numpy())
            RNAs_index = torch.tensor(RNAs_index,dtype=torch.long).to(device)
            drugs_index = torch.tensor(drugs_index, dtype=torch.long).to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            predicts = model(edge_index, RNAs_index, drugs_index, args, z_mol, z_elem, z_drug, edge_weight).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicts.cpu().numpy())



        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_predictions)
    aupr = average_precision_score(all_labels, all_predictions)
    print("auc:{}, aupr:{}".format(auc, aupr))
    return auc, aupr, all_labels, all_predictions, all_indexs

def train_test(model, train_data, test_data, edge_index, edge_weight, args, criterion, optimizer, epochs, device, z_mol, z_elem, z_drug, flod):
    best_auc = 0
    best_aupr = 0
    folder_path = './result/flod{}'.format(flod)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for epoch in range(epochs):
        model = train_model(model, train_data, edge_index, edge_weight, args, criterion, optimizer, device, z_mol, z_elem, z_drug)
        auc, aupr, labels, predicts, indexs = test_model(model, test_data, edge_index, edge_weight, args, device, z_mol, z_elem, z_drug)
        if((auc > best_auc) and (aupr > best_aupr)):
            best_auc = auc
            best_aupr = aupr
            np.save(folder_path + '/labels.npy', labels)
            np.save(folder_path + '/predicts.npy', predicts)
            torch.save(model, folder_path + '/model(test).pth')
            np.savetxt(folder_path + '/test_indexs.csv', np.array(indexs), delimiter=',',
                       fmt='%d')

    with open(folder_path + '/best_result.json', 'w') as f:
        json.dump({'best_result': ['best_auc:{}, best_aupr:{}'.format(best_auc, best_aupr)]}, f)
    print('best_auc:{}, best_aupr:{}'.format(best_auc, best_aupr))

def train_test_cold(model, train_data, test_data, edge_index, edge_weight, args, criterion, optimizer, epochs, device, z_mol, z_elem, z_drug, cold_type):
    best_auc = 0
    best_aupr = 0
    folder_path = './result/{}'.format(cold_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for epoch in range(epochs):
        model = train_model(model, train_data, edge_index, edge_weight, args, criterion, optimizer, device, z_mol, z_elem, z_drug)
        auc, aupr, labels, predicts, indexs = test_model(model, test_data, edge_index, edge_weight, args, device, z_mol, z_elem, z_drug)
        if((auc > best_auc) and (aupr > best_aupr)):
            best_auc = auc
            best_aupr = aupr
            np.save(folder_path + '/labels.npy', labels)
            np.save(folder_path + '/predicts.npy', predicts)
            torch.save(model, folder_path + '/model(test).pth')
            np.savetxt(folder_path + '/test_indexs.csv', np.array(indexs), delimiter=',',
                       fmt='%d')

    with open(folder_path + '/best_result.json', 'w') as f:
        json.dump({'best_result': ['best_auc:{}, best_aupr:{}'.format(best_auc, best_aupr)]}, f)
    print('best_auc:{}, best_aupr:{}'.format(best_auc, best_aupr))




