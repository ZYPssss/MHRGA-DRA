import pandas as pd
import numpy as np
import torch
import argparse

from dataloader import get_Multi_view_mol_data
import joblib
from model.DeepModel import *


random_seed = 42
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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

Multi_view_E_model = torch.load('./pretrain_model/Multi_view_E_model.pth', map_location=device)
z_mol, z_elem, z_drug = get_Multi_view_mol_data(Multi_view_E_model, device)
torch.save(z_mol, './pretrain_data/z_mol.pth')
torch.save(z_elem, './pretrain_data/z_elem.pth')
torch.save(z_drug, './pretrain_data/z_drug.pth')






