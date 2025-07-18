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
    all_outputs_reg = []
    with torch.no_grad():
        # try:
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(tqdm(dataloader)):
            # if iter==10:
            #     break
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output_features = model.get_features(graph1, graph2, paths, paths_labels, paths_labels_masks)
            output_reg = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)

            # import pdb;pdb.set_trace()
            # 收集 SMILES 对、路径和属性变化信息
            all_outputs_1.append(output_features.float().cpu().numpy())  # 预测属性值变化
            all_targets_1.append(target[:, 0].float().cpu().numpy())  # 真实属性值变化
            all_targets_2.append(target[:, 1].float().cpu().numpy())  # 真实属性值
            # all_targets_3.append(target[:, 2].float().cpu().numpy())  # 真实属性值 路径最后一个分子
            all_smiles1.append(graph1)  # SMILES1 列表
            all_smiles2.append(graph2)  # SMILES2 列表

            paths_array = np.array(paths)  # 转为 numpy 数组
            paths_transposed = np.transpose(paths_array, (2, 0, 1))  # 转换维度顺序为 (16, 5, 9)
            all_paths.append(paths_transposed)  # 路径信息
            
            all_outputs_reg.append(output_reg.cpu().numpy())  # Store both outputs
        # except:
        #     import pdb;pdb.set_trace()

    # return all_outputs_1, all_targets_1, all_outputs_2, all_targets_2, all_targets_3, all_smiles1, all_smiles2, all_paths
    return all_outputs_1, all_targets_1, all_targets_2, all_smiles1, all_smiles2, all_paths, all_outputs_reg
                  
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
        c=targets, cmap='hsv', norm=norm, s=5, alpha=0.8
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
    plt.title(f"Molecular Difference Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    return highlighted_points


    
label_name = 'gap'
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

'''
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


# test_dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='test', label_name = label_name)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# 初始化模型和优化器
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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

infer_reuslt = infer(model, trainval_dataloader)

# infer_reuslt = infer(model, test_dataloader)

all_outputs_1, all_targets_1, all_targets_2, all_smiles1, all_smiles2, all_paths, all_outputs_reg = infer_reuslt


# 创建一个字典来存储这些数据
data = {'smiles1': all_smiles1,'smiles2': all_smiles2,'targets': all_targets_1, 'feature_outputs': all_outputs_1, 'outputs_reg': all_outputs_reg}

# 保存为 .pth 文件
torch.save(data, '/home/data1/lk/project/mol_tree/outputs_visual/data_trainval.pth')
'''

from rdkit import Chem
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 进度条
import torch

# 读取 .pth 文件
data = torch.load('/home/data1/lk/project/mol_tree/outputs_visual/trainval/data_trainval.pth')

# 获取保存的内容
all_smiles_1 = data['smiles1']
all_smiles_2 = data['smiles2']
all_targets_1 = data['targets']
outputs_reg = data['outputs_reg']

# np.load("/home/data1/lk/project/mol_tree/all_outputs_1.npy",allow_pickle=True)


# 扁平化数据
all_targets_1 = np.concatenate(all_targets_1).flatten()  # 假设这些差值是分子从C到CC的属性变化
all_outputs_1 = np.concatenate(outputs_reg).flatten()  # 模型预测的属性变化
all_smiles1 = [x for xx in all_smiles_2 for x in xx.smiles]  # 分子C
all_smiles2 = [x for xx in all_smiles_1 for x in xx.smiles]  # 分子CC
all_outputs_1 = all_outputs_1.reshape(-1,2)[:,0]


# 统计SMILES变化类型（例如增加了C、CO等）
def extract_structure_change(smiles1, smiles2):
    # 使用RDKit解析SMILES
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return "Invalid SMILES"  # 如果SMILES无法解析
    
    # 获取分子中的原子和基团信息
    mol1_atoms = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    mol2_atoms = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    
    # 统计原子的变化情况
    atom_counts_1 = Counter(mol1_atoms)
    atom_counts_2 = Counter(mol2_atoms)

    # 找出增加的原子
    added_atoms = {atom: atom_counts_2[atom] - atom_counts_1.get(atom, 0) for atom in atom_counts_2 if atom_counts_2[atom] > atom_counts_1.get(atom, 0)}
    removed_atoms = {atom: atom_counts_1[atom] - atom_counts_2.get(atom, 0) for atom in atom_counts_1 if atom_counts_1[atom] > atom_counts_2.get(atom, 0)}
    
    added_atoms_str = ', '.join([f"{atom}: {count}" for atom, count in added_atoms.items()])
    removed_atoms_str = ', '.join([f"{atom}: {count}" for atom, count in removed_atoms.items()])
    
    # 统计添加的基团，如CO等
    if added_atoms_str:
        return f"Added atoms: {added_atoms_str}"
    elif removed_atoms_str:
        return f"Removed atoms: {removed_atoms_str}"
    
    return "No Significant Change"

# 提取每一对smiles的结构变化
structure_changes = []
for smiles1, smiles2 in tqdm(zip(all_smiles1, all_smiles2), total=len(all_smiles1), desc="Calculating Structure Changes"):
    structure_changes.append(extract_structure_change(smiles1, smiles2))

# 将数据存入 DataFrame 进行分析
change_df = pd.DataFrame({
    'Molecule_C': all_smiles1,
    'Molecule_CC': all_smiles2,
    'Structure_Change': structure_changes,
    'Target_Change': all_targets_1,
    'Predicted_Change': all_outputs_1
})

# 输出结构变化的统计
structure_change_counts = change_df['Structure_Change'].value_counts()
print("\n结构变化统计：")
print(structure_change_counts)

# 可视化：柱状图展示不同结构变化的数量
plt.figure(figsize=(12, 6))
sns.countplot(x='Structure_Change', data=change_df, palette='Set2')
plt.title('Distribution of Structure Changes')
plt.xlabel('Structure Change Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/structure_change_distribution.png')
plt.show()

# 计算预测误差
change_df['Error'] = np.abs(change_df['Target_Change'] - change_df['Predicted_Change'])

# 可视化：误差分布
plt.figure(figsize=(12, 6))
sns.histplot(change_df['Error'], kde=True, color='red', bins=30)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Absolute Difference)')
plt.ylabel('Frequency')
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/prediction_error_distribution.png')
plt.show()

# 可视化：不同结构变化类型下的预测误差
plt.figure(figsize=(12, 6))
sns.boxplot(x='Structure_Change', y='Error', data=change_df, palette='Set2')
plt.title('Prediction Error by Structure Change Type')
plt.xlabel('Structure Change Type')
plt.ylabel('Prediction Error')
plt.xticks(rotation=45)
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/error_by_structure_change.png')
plt.show()

# 针对每种结构变化类型，分析并可视化其对应的属性变化
plt.figure(figsize=(12, 6))
sns.histplot(data=change_df, x='Target_Change', hue='Structure_Change', multiple='stack', palette='Set3', bins=30)
plt.title('Property Change Distribution by Structure Change Type')
plt.xlabel('Property Change (Target_Change)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/property_change_distribution_by_structure.png')
plt.show()

# 可视化：根据每种结构变化类型绘制对应的属性变化和预测变化的对比
for change_type in structure_change_counts.index:
    plt.figure(figsize=(12, 6))
    subset = change_df[change_df['Structure_Change'] == change_type]
    
    # 计算该结构变化类型下的误差
    errors = subset['Error']
    mean_error = errors.mean()
    std_error = errors.std()
    
    # 对比真实属性变化与预测变化
    sns.histplot(subset['Target_Change'], bins=30, kde=True, color='blue', label='True Change', alpha=0.6)
    sns.histplot(subset['Predicted_Change'], bins=30, kde=True, color='green', label='Predicted Change', alpha=0.6)
    
    # 添加误差信息到标题
    plt.title(f'Property Change Distribution for {change_type} (True vs Predicted)\n'
              f'Mean Error: {mean_error:.4f}, Std Error: {std_error:.4f}')
    plt.xlabel('Property Change')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 保存图像时包括误差信息
    plt.savefig(f'/home/data1/lk/project/mol_tree/outputs_visual/property_change_{change_type.replace(" ", "_")}_true_vs_pred_error_{mean_error:.4f}_{std_error:.4f}.png')
    plt.show()






'''
精确原子变化特征分析
'''



import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.cluster import KMeans

# 筛选添加特定原子的分子数据
def filter_added_atoms(change_df, atom):
    return change_df[change_df['Structure_Change'].str.contains(f"Added atoms: {atom}: 1")]

# 获取分子的原子位置
def get_atom_position(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_positions = []
    for atom in mol.GetAtoms():
        atom_positions.append(atom.GetIdx())  # 记录原子类型和索引
    return atom_positions

# 提取并分析C原子的添加位置
added_C_df = filter_added_atoms(change_df, 'C')
added_C_df['Atom_Position'] = added_C_df['Molecule_C'].apply(get_atom_position)

# 统计添加C原子的位置分布
position_counter_C = defaultdict(int)

# 遍历所有包含原子位置的分子数据
for positions in added_C_df['Atom_Position']:
    if positions:
        # positions 是一个位置索引的列表，将其作为键，更新频率
        for pos in positions:
            position_counter_C[pos] += 1

# 可视化位置分布
plt.figure(figsize=(12, 6))
sns.barplot(x=list(position_counter_C.keys()), y=list(position_counter_C.values()), palette="Set2")
plt.title(f'C - Position Distribution')
plt.xlabel('Atom Position (Index)')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.savefig('/home/data1/lk/project/mol_tree/outputs_visual/C_position_distribution.png')
plt.show()

# 属性变化分析：C位置与属性变化的关系
def analyze_property_change_by_position(filtered_df, atom_type, position_counter):
    for position, count in position_counter.items():
        subset = filtered_df[filtered_df['Atom_Position'].apply(lambda x: position in x)]
        if len(subset) > 0:
            target_change_mean = subset['Target_Change'].mean()
            target_change_std = subset['Target_Change'].std()
            
            print(f"Analysis for position {position} and Added {atom_type}:")
            print(f"Mean of Target Change: {target_change_mean:.4f}")
            print(f"Standard Deviation of Target Change: {target_change_std:.4f}")
            
            # 可视化属性变化
            plt.figure(figsize=(12, 6))
            sns.histplot(subset['Target_Change'], kde=True, color='blue', bins=30)
            plt.title(f'Target Change Distribution for Position {position} and Added {atom_type}')
            plt.xlabel('Target Change')
            plt.ylabel('Frequency')
            plt.savefig(f'/home/data1/lk/project/mol_tree/outputs_visual/Target_Change_Position_{position}_C.png')
            plt.show()

# 对添加C原子的位置与属性变化进行分析
analyze_property_change_by_position(added_C_df, 'C', position_counter_C)

# 使用聚类分析位置与属性变化之间的关系
def cluster_positions_and_properties(df, atom_type):
    position_data = []
    for positions, change in zip(df['Atom_Position'], df['Target_Change']):
        if positions:
            for position in positions:
                position_data.append([position, change])
    
    position_data = np.array(position_data)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(position_data[:, 1].reshape(-1, 1))  # 根据属性变化进行聚类
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=position_data[:, 0], y=position_data[:, 1], hue=kmeans.labels_, palette="Set2")
    plt.title(f'{atom_type} Position and Target Change Clustering')
    plt.xlabel('Atom Position')
    plt.ylabel('Target Change')
    plt.savefig(f'/home/data1/lk/project/mol_tree/outputs_visual/{atom_type}_clustering.png')
    plt.show()

# 对C原子的位置与属性变化进行聚类分析
cluster_positions_and_properties(added_C_df, 'C')
