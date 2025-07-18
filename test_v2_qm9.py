import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import torch.optim as optim
from model import GNNModel
# from dataset_v2 import MoleculeACDataset_v2
# from dataset_v2_test import MoleculeACDataset_v2
from dataset_v2_test_Ha2eV import MoleculeACDataset_v2
import torch.nn as nn
import random, os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm
from datetime import datetime
import logging
from tools import calculate_metrics
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  random_split
import math

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Train a model with AdamW optimizer and learning rate scheduler.")

# 添加label_name的命令行参数
parser.add_argument('--label_name', type=str, default='None', help='The label name to be used in training')

# 解析命令行参数
args = parser.parse_args()

# v2版本加入进化path

# Set up logging
log_dir = 'logs'

# label_name in mu,alpha,homo,lumo,gap,r2,zpve,U0,U,H,G,Cv
label_name = args.label_name
os.makedirs(log_dir, exist_ok=True)
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'qm9_{label_name}_test_log_{start_time}.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# Function to log messages
def log(message, noprint=False):
    
    if noprint==False:
        print(message)  # Print to console
        
    logging.info(message)  # Save to log file

# Function to log the content of specified Python files
def log_python_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            log(f"Logging content of the file: {file_path}")
            log(content, noprint=True)
    except Exception as e:
        log(f"Error while reading the file {file_path}: {e}")

# List of Python files to log
files_to_log = [
    '/home/data1/lk/project/mol_tree/train_v2_qm9.py',
    '/home/data1/lk/project/mol_tree/tools.py',
    '/home/data1/lk/project/mol_tree/model.py',
    # '/home/data1/lk/project/mol_tree/dataset_v2.py'
    "/home/data1/lk/project/mol_tree/dataset_v2_test_Ha2eV.py"
]

# Log the content of each file
for file in files_to_log:
    log_python_file(file)
    
# Set a random seed
seed = 42
log(f'seed is {seed}')

# Python random seed
random.seed(seed)
# NumPy random seed
np.random.seed(seed)
# PyTorch random seed
torch.manual_seed(seed)

# For GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all GPUs

# If you are using cuDNN (for GPU training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import json
import random

# File paths
input_file = "/home/data1/lk/project/mol_tree/graph/evolution_graph_133885_['edit_distance', 'graph']_0.3_v2.json"
output_file = "/home/data1/lk/project/mol_tree/graph/test_smiles.txt"
dataset_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'

# Load JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

test_dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='test', label_name = label_name)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

checkpoint_path = '/home/data1/lk/project/mol_tree/checkpoint/gnn_model_epoch_20241205_083941.pth'

log(f'Check point path is :{checkpoint_path}.')

model = GNNModel(device).to(device)  # Move model to GPU

# 加载保存的模型权重
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)  # 假设 checkpoint 中的键为 'model_state_dict'



criterion = nn.MSELoss()
# scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每 10 个 epoch 将学习率缩小为原来的 0.1 倍


pth_dir = 'checkpoint'


def infer(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(dataloader):
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)

            if mode == 'test':
                all_targets.append(target.cpu().numpy())  # Store on CPU
                all_outputs.append(output.cpu().numpy())  # Store both outputs

                
    avg_loss = total_loss / len(dataloader)
    avg_loss1 = total_loss1 / len(dataloader)
    avg_loss2 = total_loss2 / len(dataloader)

    # 计算相关系数和R²
    if mode == 'test':
        all_targets = np.concatenate(all_targets)  
        all_outputs = np.concatenate(all_outputs)
        # 分别提取每个回归项的目标和输出
        target1 = all_targets[:, 0]
        target2 = all_targets[:, 1]
        target3 = all_targets[:, 2]
        
        output1 = all_outputs[:, 0]
        output2 = all_outputs[:, 1]

    # metrics1 = calculate_metrics(dataset.denormalize(target1), dataset.denormalize(output1))
    # metrics2 = calculate_metrics(dataset.denormalize(target2), dataset.denormalize(output2))
    # # if label_logarithmic_transformation:
    # #     # import pdb;pdb.set_trace()
    # #     temp_target2_pred = np.exp(output1+np.log(target3+0.00001))-0.00001
    # #     metrics3 = calculate_metrics(target2, temp_target2_pred)
    # # else:
    # # import pdb;pdb.set_trace()
    # metrics3 = calculate_metrics(dataset.denormalize(target2), dataset.denormalize( ( output1 *  (dataset.label_max_ratio - dataset.label_min_ratio) + dataset.label_min_ratio + 1 ) * target3 ))

    metrics1 = calculate_metrics(target1, output1)
    metrics2 = calculate_metrics(target2, output2 )
    metrics3 = calculate_metrics(target2, ( output1 + 1 ) * target3)
    # metrics3 = calculate_metrics(target2, output1 + target3)
    # 打印结果
    log(f"Output 1 - R² Score: {metrics1['R2']:.6f}, Pearson Correlation: {metrics1['Pearson Correlation']:.6f}, MSE: {metrics1['MSE']:.6f}, MAE: {metrics1['MAE']:.6f}, Rank Loss: {metrics1['Rank Loss']:.6f}")
    log(f"Output 2 - R² Score: {metrics2['R2']:.6f}, Pearson Correlation: {metrics2['Pearson Correlation']:.6f}, MSE: {metrics2['MSE']:.6f}, MAE: {metrics2['MAE']:.6f}, Rank Loss: {metrics2['Rank Loss']:.6f}")
    log(f"Output 3 - R² Score: {metrics3['R2']:.6f}, Pearson Correlation: {metrics3['Pearson Correlation']:.6f}, MSE: {metrics3['MSE']:.6f}, MAE: {metrics3['MAE']:.6f}, Rank Loss: {metrics3['Rank Loss']:.6f}")

    return avg_loss, avg_loss1, avg_loss2


test_loss, test_loss1, test_loss2 = infer(model, test_dataloader, mode='test')

