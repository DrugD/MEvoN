import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import torch.optim as optim
from model_gcn import GNNModel
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
log_file = os.path.join(log_dir, f'only_gcn_qm9_{label_name}_training_log_{start_time}.log')
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
    '/home/data1/lk/project/mol_tree/train_v2_qm9_only_gcn.py',
    '/home/data1/lk/project/mol_tree/tools.py',
    # '/home/data1/lk/project/mol_tree/model.py',
    '/home/data1/lk/project/mol_tree/model_gcn.py',
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

# Extract SMILES from the 'nodes' list
smiles_list = data.get('nodes', [])

# Select 5% of SMILES randomly
selected_smiles = random.sample(smiles_list, max(1, int(len(smiles_list) * 0.1)))

# Save to the output file, one SMILES per line
with open(output_file, 'w') as f:
    for smile in selected_smiles:
        f.write(smile + '\n')




# 加载数据集
dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='trainval', label_name = label_name)

label_logarithmic_transformation = dataset.label_logarithmic_transformation
log(f"Label logarithmic transformation is:{label_logarithmic_transformation}.")


# 计算训练、验证和测试集的大小
train_size = int( 8/9 * len(dataset))
val_size = len(dataset) - train_size  # 15% for validation

# 随机分割数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


test_dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='test', label_name = label_name)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)


# 初始化模型和优化器

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
model = GNNModel(device).to(device)  # Move model to GPU
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3, min_lr=1e-7)


# 0.002 for homo 8192
# 

criterion = nn.MSELoss()
# scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每 10 个 epoch 将学习率缩小为原来的 0.1 倍


pth_dir = 'checkpoint'
os.makedirs(pth_dir, exist_ok=True)
            
def predict(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for graph1, graph2, target, paths, paths_labels, paths_labels_masks in dataloader:
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)

            # 分别计算两个回归项的损失
            loss1 = criterion(output[:, 0].float(), target[:, 1].float())
            
            total_loss += loss1.item()

            # if mode == 'test':
            #     all_targets.append(target.cpu().numpy())  # Store on CPU
            #     all_outputs.append(output.cpu().numpy())  # Store both outputs

    avg_loss = total_loss / len(dataloader)

    # # 计算相关系数和R²
    # if mode == 'test':
    #     all_targets = np.concatenate(all_targets)  
    #     all_outputs = np.concatenate(all_outputs)
    #     # 分别提取每个回归项的目标和输出
    #     target1 = all_targets[:, 0]
    #     target2 = all_targets[:, 1]
    #     output1 = all_outputs[:, 0]
    #     output2 = all_outputs[:, 1]

    #     # 计算第一个输出的 R² 和 Pearson
    #     pearson_corr1 = pearsonr(target1, output1)[0]
    #     r2_1 = r2_score(target1, output1)

    #     # 计算第二个输出的 R² 和 Pearson
    #     pearson_corr2 = pearsonr(target2, output2)[0]
    #     r2_2 = r2_score(target2, output2)

    #     log(f'Output 1 - R² Score: {r2_1:.6f}, Pearson Correlation: {pearson_corr1:.6f}')
    #     log(f'Output 2 - R² Score: {r2_2:.6f}, Pearson Correlation: {pearson_corr2:.6f}')

    return avg_loss



def infer(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(dataloader):
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)
            
            if label_name == 'r2':
                output = output * dataset.label_std + dataset.label_mean
                # output[:, 1] = output[:, 1] * dataset.label_std + dataset.label_mean
                # target[:, 0] = target[:, 0] * dataset.label_std
                target[:, 1] = target[:, 1] * dataset.label_std + dataset.label_mean
                # target[:, 2] = target[:, 2] * dataset.label_std + dataset.label_mean

            # 分别计算两个回归项的损失
            loss1 = criterion(output[:, 0].float(), target[:, 1].float())

            total_loss += loss1.item()
            
            
                
            if mode == 'test':
                all_targets.append(target.cpu().numpy())  # Store on CPU
                all_outputs.append(output.cpu().numpy())  # Store both outputs

                
    avg_loss = total_loss / len(dataloader)

    # 计算相关系数和R²
    if mode == 'test':
        all_targets = np.concatenate(all_targets)  
        all_outputs = np.concatenate(all_outputs)
        # 分别提取每个回归项的目标和输出
        target1 = all_targets[:, 1]
        output1 = all_outputs[:, 0]

    metrics1 = calculate_metrics(target1, output1)

    # 打印结果
    log(f"Output 1 - R² Score: {metrics1['R2']:.6f}, Pearson Correlation: {metrics1['Pearson Correlation']:.6f}, MSE: {metrics1['MSE']:.6f}, MAE: {metrics1['MAE']:.6f}, Rank Loss: {metrics1['Rank Loss']:.6f}")

    return avg_loss

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=1000):
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs),desc=f'for {label_name}:'):
        model.train()
        total_loss = 0

        
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(train_dataloader):
            optimizer.zero_grad()
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)
            
            loss1 = criterion(output[:, 0].squeeze(), target[:, 1].float())
            
            loss1.backward()
            optimizer.step()
            
            total_loss += loss1.item()
            
            log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Iter [{iter + 1}/{len(train_dataloader)}], Loss: {loss1.item():.6f}')

            
        avg_loss = total_loss / len(train_dataloader)


        log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Loss: {avg_loss:.6f}')
        
        # Validate
        val_loss = predict(model, val_dataloader)
        log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Val Loss: {val_loss:.6f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        log(f'Epoch [{epoch + 1}/{epochs}], Current LR: {current_lr:.8e}')
        
        # Check if validation loss improved and save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss= infer(model, test_dataloader, mode='test')
            # test_loss, test_loss1, test_loss2 = predict(model, test_dataloader, mode='test')
            # log(f'Test Loss: {test_loss:.6f}, Test Loss1: {test_loss1:.6f}, Test Loss2: {test_loss2:.6f}')
            
            pth_file = os.path.join(pth_dir, f'gnn_model_epoch_{start_time}.pth')
            torch.save(model.state_dict(), pth_file)  # Save model with timestamp
            

# 训练模型
train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=800)