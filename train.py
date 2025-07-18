import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import torch.optim as optim
from model import GNNModel
from dataloader import MoleculeDataset, MoleculeACDataset
import torch.nn as nn
import random, os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm
from datetime import datetime
import logging

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'training_log_{start_time}.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# Function to log messages
def log(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file


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


from torch.utils.data import  random_split
# 加载数据集
dataset = MoleculeACDataset('/home/data1/lk/project/mol_tree/graph/evolution_graph_10000_graph_0.3_v1.json', '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv')

# 计算训练、验证和测试集的大小
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))  # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining for test

# 随机分割数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=0)

# 初始化模型和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GNNModel().to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# 预测函数
def predict(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for graph1, graph2, target in dataloader:
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model(graph1, graph2)
            loss = criterion(output, target.float())
            total_loss += loss.item()
            
            if mode == 'test':
                # 存储目标和输出
                all_targets.append(target.cpu().numpy())  # Store on CPU
                all_outputs.append(output.cpu().numpy())  # Store on CPU

    avg_loss = total_loss / len(dataloader)
    
    # 计算相关系数和R²
    if mode == 'test':
        all_targets = np.concatenate(all_targets).flatten()
        all_outputs = np.concatenate(all_outputs).flatten() 
        pearson_corr = pearsonr(all_targets, all_outputs)[0]
        r2 = r2_score(all_targets, all_outputs)
        log(f'R² Score: {r2:.6f}, Pearson Correlation: {pearson_corr:.6f}')

    return avg_loss

# 训练函数
def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=10):
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        for iter, (graph1, graph2, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model(graph1, graph2)
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        log(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.6f}')
        
        # Validate
        val_loss = predict(model, val_dataloader)
        log(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.6f}')
        
        # Check if validation loss improved and save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = predict(model, test_dataloader, mode='test')
            log(f'Test Loss: {test_loss:.6f}')
            
            

# 训练模型
train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=200)

pth_dir = 'checkpoint'
os.makedirs(pth_dir, exist_ok=True)
pth_file = os.path.join(pth_dir, f'gnn_model_epoch_{start_time}.pth')
torch.save(model.state_dict(), pth_file)  # Save model with timestamp
