'''
对变化的分布规律进行分析，预测-真实
''' 

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
from dataset_v2_test_infer import MoleculeACDataset_v2
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
import json
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from collections import Counter


def infer(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    all_outputs_1 = []
    all_targets_1 = []
    # all_outputs_2 = []
    all_targets_2 = []
    # all_targets_3 = []
    all_smiles1 = []
    all_smiles2 = []
    all_paths = []
    
    with torch.no_grad():
        # try:
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(tqdm(dataloader)):
            # if iter==10:
            #     break
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.get_features(graph1, graph2, paths, paths_labels, paths_labels_masks)
            # import pdb;pdb.set_trace()
            # 收集 SMILES 对、路径和属性变化信息
            all_outputs_1.append(output.float().cpu().numpy())  # 预测属性值变化
            all_targets_1.append(target[:, 0].float().cpu().numpy())  # 真实属性值变化
            all_targets_2.append(target[:, 1].float().cpu().numpy())  # 真实属性值
            # all_targets_3.append(target[:, 2].float().cpu().numpy())  # 真实属性值 路径最后一个分子
            all_smiles1.append(graph1)  # SMILES1 列表
            all_smiles2.append(graph2)  # SMILES2 列表

            paths_array = np.array(paths)  # 转为 numpy 数组
            paths_transposed = np.transpose(paths_array, (2, 0, 1))  # 转换维度顺序为 (16, 5, 9)
            all_paths.append(paths_transposed)  # 路径信息
        # except:
        #     import pdb;pdb.set_trace()

    # return all_outputs_1, all_targets_1, all_outputs_2, all_targets_2, all_targets_3, all_smiles1, all_smiles2, all_paths
    return all_outputs_1, all_targets_1, all_targets_2, all_smiles1, all_smiles2, all_paths
                  
    # metrics1 = calculate_metrics(target1, output1)
    # metrics2 = calculate_metrics(target2, output2)
    # metrics3 = calculate_metrics(target2, ( output1 + 1 ) * target3)

                
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from umap import UMAP

def plot_and_save_molecular_difference(features, targets, method='tsne', perplexity=30, n_neighbors=10, min_dist=0.3, random_state=42, save_path='molecular_difference.png', highlight_indices = [], highlight_data = []):
    """
    Visualize molecular differences by reducing dimensions and marking with property changes.
    The resulting plot will also be saved as an image.

    Parameters:
    - features (np.ndarray): Molecular feature matrix of shape (N, 512).
    - targets (np.ndarray): Corresponding target values of shape (N,).
    - method (str): Dimensionality reduction method ('tsne' or 'pca'). Default is 'tsne'.
    - perplexity (int): Perplexity parameter for t-SNE (if used). Default is 30.
    - random_state (int): Random seed for reproducibility. Default is 42.
    - save_path (str): Path to save the resulting image. Default is 'molecular_difference.png'.

    Returns:
    - None: Displays the plot and saves it to the specified location.
    """
    if features.shape[1] != 256:
        raise ValueError("Input feature matrix must have shape (N, 512).")
    
    # Perform dimensionality reduction
    try:
        if method == 'umap':
            reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=random_state)
            reduced_features = reducer.fit_transform(features)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=10)  # Avoid multiprocessing issues
            reduced_features = reducer.fit_transform(features)
        else:
            raise ValueError("Unsupported method. Choose 'umap' or 'tsne'.")
    except RuntimeError as e:
        print(f"Dimensionality reduction failed: {e}")
        return
    
    # Normalize target values for coloring
    norm = Normalize(vmin=np.min(targets), vmax=np.max(targets))

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1], 
        c=targets, cmap='nipy_spectral', norm=norm, s=5, alpha=0.3
    )
    
    # # 高亮显示特定样本
    highlighted_points = reduced_features[highlight_indices]
    # plt.scatter(
    #     highlighted_points[:, 0], highlighted_points[:, 1], 
    #     color='red', marker='x', s=20, label='Highlighted Points'
    # )

    # # 添加注释
    # for i,data_item in enumerate(highlight_data):
    #     plt.annotate(f'{data_item}', 
    #                 (reduced_features[i, 0], reduced_features[i, 1]), 
    #                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=3)
    


    plt.colorbar(scatter, label='Property Change Value')
    # plt.title(f"Molecular Difference Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.2)

    # Save the plot
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    return highlighted_points


    
label_name = 'zpve'
# Set a random seed
seed = 42

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



# 创建数据加载器
trainval_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


test_dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='test', label_name = label_name)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# 初始化模型和优化器
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = GNNModel(device)

# gap
checkpoint_path = "/home/data1/lk/project/mol_tree/checkpoint/gnn_model_epoch_20241203_112935.pth"

# 加载权重
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 加载到模型
# `strict=False` 意味着可以跳过模型中没有的权重或者没有保存的权重
model.load_state_dict(checkpoint, strict=False)

# 如果需要将模型迁移到特定设备
model = model.to(device)

# infer_reuslt = infer(model, trainval_dataloader)

infer_reuslt = infer(model, test_dataloader)

all_outputs_1, all_targets_1, all_targets_2, all_smiles1, all_smiles2, all_paths = infer_reuslt



# np.mean([calculate_metrics(torch.tensor(all_targets_2[index]), ( torch.tensor(all_outputs_1[index]) + 1 ) * torch.tensor(all_targets_3[index]))['MAE']*27 for i in range(16)])


# for i, (mol1, mol2) in enumerate(zip(all_smiles1, all_smiles2)):
#     with torch.no_grad():
#         graph_feature_1 = model.get_features(mol1.to(device))
#         graph_feature_2 = model.get_features(mol2.to(device))
        
#     graph_dif = graph_feature_1-graph_feature_2
#     features.append(graph_dif.cpu().numpy())
    
#     targets.append(all_targets_1[i])

perplexity = 42
n_neighbors=10
min_dist=0.5



# 保存为 .npy 文件
# np.save('test_all_outputs_1.npy', all_outputs_1)
# np.save('test_all_targets_1.npy', all_targets_1)

# all_outputs_1 = np.load('all_outputs_1.npy', allow_pickle=True)
# all_targets_1 = np.load('all_targets_1.npy', allow_pickle=True)


all_targets_1 = np.concatenate(all_targets_1).flatten()
all_targets_1 = all_targets_1 * np.random.uniform(0, 1, size=len(all_targets_1))


highlight_indices = np.argsort(all_targets_1)[:5]
smiles_pair = [ (all_smiles1[ i//32][ i%32].smiles, all_smiles2[ i//32][ i%32].smiles) for i in highlight_indices]
# plot_and_save_molecular_difference(np.concatenate(features, axis=0), np.concatenate(targets).flatten()*100, method='umap', perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist, save_path=f'test_difference_plot_umap_{perplexity}_{n_neighbors}_{min_dist}.png')
# highlighted_points = plot_and_save_molecular_difference(np.concatenate(all_outputs_1, axis=0), all_targets_1, method='tsne', perplexity=perplexity, save_path=f'trainval_difference_plot_tsne_perplexity_{perplexity}_hsv_uniform_top5_low5_withData_color_nipy_spectral.png', highlight_indices = highlight_indices, highlight_data = smiles_pair)
highlighted_points = plot_and_save_molecular_difference(np.concatenate(all_outputs_1, axis=0), all_targets_1, method='tsne', perplexity=perplexity, save_path=f'test_difference_plot_tsne_perplexity_{perplexity}_hsv_uniform_top5_low5_withData_color_nipy_spectral.png', highlight_indices = highlight_indices, highlight_data = smiles_pair)


import pdb;pdb.set_trace()