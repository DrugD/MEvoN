import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import torch.optim as optim
from model import GNNModel
from dataset import MoleculeDataset, MoleculeACDataset
import torch.nn as nn
import random, os

from tqdm import tqdm
from datetime import datetime
from tools import plot_target_vs_output, calculate_metrics
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np

# v2版本加入进化path


    
# Set a random seed
seed = 42
print(f'seed is {seed}')

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


from torch.utils.data import  random_split
# 加载数据集
test_dataset = MoleculeACDataset('/home/data1/lk/project/mol_tree/graph/evolution_graph_10000_graph_0.3_v1.json', '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv', test_file = '/home/data1/lk/project/mol_tree/test_smiles.txt', mode='test')
# 该数据集需要一些新分子，不在之前进化树中，没有label先验

# 创建数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=8)

# 初始化模型和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GNNModel(device).to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def infer(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    all_outputs = []

    with torch.no_grad():
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(dataloader):
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)

            # 分别计算两个回归项的损失
            loss1 = criterion(output[:, 0].float(), target[:, 0].float())
            loss2 = criterion(output[:, 1].float(), target[:, 1].float())
            loss = loss1 * 0.9 + loss2 * 0.1

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

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


    metrics1 = calculate_metrics(target1, output1)
    metrics2 = calculate_metrics(target2, output2)
    metrics3 = calculate_metrics(target2, target3 * (1 + output1))

    # 打印结果
    print(f"Output 1 - R² Score: {metrics1['R2']:.6f}, Pearson Correlation: {metrics1['Pearson Correlation']:.6f}, MSE: {metrics1['MSE']:.6f}, Rank Loss: {metrics1['Rank Loss']:.6f}")
    print(f"Output 2 - R² Score: {metrics2['R2']:.6f}, Pearson Correlation: {metrics2['Pearson Correlation']:.6f}, MSE: {metrics2['MSE']:.6f}, Rank Loss: {metrics2['Rank Loss']:.6f}")
    print(f"Output 3 - R² Score: {metrics3['R2']:.6f}, Pearson Correlation: {metrics3['Pearson Correlation']:.6f}, MSE: {metrics3['MSE']:.6f}, Rank Loss: {metrics3['Rank Loss']:.6f}")



# Define a function to load model checkpoint (only model's state_dict)
def load_checkpoint(model, checkpoint_path):
    # Load the model's state_dict from the checkpoint file
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model



# Example usage:
# checkpoint_path = '/home/data1/lk/project/mol_tree/checkpoint/gnn_model_epoch_20241106_021420.pth'  # Replace with your actual checkpoint file
checkpoint_path = '/home/data1/lk/project/mol_tree/checkpoint/gnn_model_epoch_20241107_054012.pth'
model = load_checkpoint(model, checkpoint_path)

infer(model, test_dataloader, mode='test')
