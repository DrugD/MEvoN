# '''
# 对变化的分布规律进行分析，路径-潜在变化趋势概率分析
# ''' 

# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from rdkit import Chem
# import torch.optim as optim
# from model import GNNModel
# # from dataset_v2 import MoleculeACDataset_v2
# # from dataset_v2_test import MoleculeACDataset_v2
# from dataset_v2_test_infer import MoleculeACDataset_v2
# import torch.nn as nn
# import random, os
# import numpy as np
# from scipy.stats import pearsonr
# from sklearn.metrics import r2_score
# from tqdm import tqdm
# from datetime import datetime
# import logging
# from tools import calculate_metrics
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import  random_split
# import math
# import json
# import random
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap
# import matplotlib.pyplot as plt
# from collections import Counter

# import pandas as pd
# import torch, json
# from torch.utils.data import Dataset
# from rdkit import Chem
# from torch_geometric.data import Data
# from tqdm import tqdm
# from tools import *
# import math

# class Graph:
#     def __init__(self):
#         self.adjacency_list = defaultdict(list)  # 存储图的邻接表
#         self.nodes = set()  # 存储所有节点
    
#     def add_node(self, label: str):
#         self.nodes.add(label)
    
#     def add_edge(self, from_node: str, to_node: str):
#         if from_node in self.nodes and to_node in self.nodes:
#             self.adjacency_list[from_node].append(to_node)

#     def get_edges(self):
#         return self.adjacency_list
    
#     def edges(self):
#         # Return a list of edges as tuples
#         return [(from_node, to_node) for from_node, to_nodes in self.adjacency_list.items() for to_node in to_nodes]

#     def __str__(self):
#         return str(dict(self.adjacency_list))


# def load_graph_from_json(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
        

#     graph = Graph()
#     # 添加节点
#     for node in data['nodes']:
#         graph.add_node(node)

#     # 添加边
#     for edge in data['edges']:
#         graph.add_edge(edge[0], edge[1])

#     return graph

# # -------------------------------------------------------------------

# import os
# import json
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from matplotlib.patches import ConnectionPatch
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import numpy as np

# def visualize_molecule(smiles):
#     """
#     生成分子的二维结构图
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if mol:
#         return Draw.MolToImage(mol)
#     return None

# def sort_by_last_homo_value(homo_values, valid_paths_with_aim_smiles):
#     """
#     按照homo_values中每个元素的最后一位值从大到小对homo_values, output_paths, valid_paths_with_aim_smiles进行排序
#     """
#     # 使用zip将homo_values, output_paths, valid_paths_with_aim_smiles绑定成元组
#     combined = list(zip(homo_values, valid_paths_with_aim_smiles))

#     # 对组合后的列表按照homo_values的最后一个元素排序，reverse=True表示降序
#     combined_sorted = sorted(combined, key=lambda x: x[0][-1], reverse=True)

#     # 解压排序后的元组
#     sorted_homo_values, sorted_valid_paths = zip(*combined_sorted)

#     # 返回排序后的三个列表
#     return list(sorted_homo_values), list(sorted_valid_paths)


# def save_property_trend_with_zoom(homo_values, output_path, valid_paths_with_aim_smiles, zoom=False, colors=None, line_styles=None, markers=None):
#     """
#     保存属性变化的折线图，并在需要时进行局部放大
#     """
#     plt.figure(figsize=(40, 25))

#     # 默认的颜色（使用RGB元组）
#     if colors is None:
#         colors = [
#             (198/255,    108/255,      97/255),
#             ( 71/255,     71/255,     105/255), 
#             (115/255,    165/255,     162/255),
#             (178/255,    182/255,     193/255)
#         ]

#     if line_styles is None:
#         line_styles = ['-', '--',]
    
#     if markers is None:
#         markers = ['o', 's', '^']

#     # import pdb;pdb.set_trace()
#     homo_values, valid_paths_with_aim_smiles = sort_by_last_homo_value(homo_values, valid_paths_with_aim_smiles)

#     # 绘制总体折线图
#     for i, homo_value in enumerate(homo_values):
#         if i>=12:
#             line_styles = ['--','-',  ]
#         color = colors[i % len(colors)]  # 循环使用颜色
#         linestyle = line_styles[i % len(line_styles)]  # 循环使用线型
#         marker = markers[i % len(markers)]  # 循环使用标记符号
#         plt.plot([0,1], homo_value[-2:],  marker=marker, markersize=15, linestyle=linestyle, color=color, label=f"{valid_paths_with_aim_smiles[i][-1]}: {homo_value[-1]}", linewidth=5)

    
#     # color_rgb = (138/255, 160/255, 183/255)
#     # plt.plot([0,1], homo_values[0][6:8],  color=color_rgb, marker='o', linestyle='-', linewidth=5)


        
#     # plt.title("Evolution Trend")
#     plt.xlabel("Steps", fontsize=45)
#     plt.ylabel("Homo value", fontsize=45)
#     plt.grid(False)
#     # plt.legend(loc="best")
#     # plt.legend(loc=9, fontsize=25, bbox_to_anchor=(0.9, 1.0))
#     plt.legend(ncol=1, loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=50)
#     plt.xticks(ticks=range(2), labels=[8,9], fontsize=45)
#     plt.yticks(fontsize=45)
#     # if zoom:
#     #     # 放大后1~2步的区域，调整xlim和ylim
#     #     plt.xlim(7, 8)  # 放大x轴范围，第n-2到第n步
        
#     #     # 强制设置Y轴范围为 [-0.28, -0.22] 之间
#     #     plt.ylim(-0.28, -0.22)  # 放大y轴范围

#     #     # 插入子坐标并绘图（放大部分）
#     #     axins = inset_axes(plt.gca(), width="30%", height="40%", loc='lower left',
#     #                        bbox_to_anchor=(0.5, 0.25, 1, 1),
#     #                        bbox_transform=plt.gca().transAxes)
#     #     for i, homo_value in enumerate(homo_values):
#     #         axins.plot(range(len(homo_value)), homo_value, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
       
#     #     axins.set_xlim(7, 8)
#     #     axins.set_ylim(-0.28, -0.22)

#         # # 添加极值点的标注
#         # axins.annotate(
#         #     "Min value",
#         #     xy=(8, homo_values[0][8]),  # 假设放大区域是第8步
#         #     xytext=(6-1, homo_values[0][8] - 0.02),
#         #     arrowprops={'arrowstyle': '->', 'color': 'red'},
#         #     fontsize=10,
#         #     color='red'
#         # )

#         # # 在主图和放大图之间添加连接线
#         # con = ConnectionPatch(xyA=(8, homo_values[0][8]), xyB=(8, homo_values[0][8]),
#         #                       coordsA="data", coordsB="data", axesA=axins, axesB=plt.gca())
#         # axins.add_artist(con)

#     # 保存折线图
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=500, format='pdf')
#     plt.close()

# def plot_all_paths_evolution(valid_paths_with_aim_smiles, labels_dict, label_name, output_dir='/home/data1/lk/project/mol_tree/outputs_visual'):
#     """
#     绘制所有路径的属性变化，展示1号元素的变化与属性的关系
#     """
#     all_homo_values = []
#     first_molecules = []  # 用来存储每条路径的第一个分子
    
#     # 收集所有路径的homo值变化
#     for path in valid_paths_with_aim_smiles:
#         homo_values = []
#         first_molecule = path[-1]  # 记录路径的第一个分子

#         for smiles in path:
#             if smiles in labels_dict:
#                 homo_values.append(labels_dict[smiles])
        
#         all_homo_values.append(homo_values)
#         first_molecules.append(first_molecule)
    
#     # 绘制总体折线图
#     plt.figure(figsize=(10, 10))

    
#     # 计算显示图片的行列数
#     num_molecules = len(first_molecules)
#     rows = (num_molecules // 7) + 1  # 每行显示最多5个分子
#     cols = min(num_molecules, 7)  # 最后一行可能不满5个
#     color_rgb = (138/255, 160/255, 183/255)
#     path_location = [
#         [0.05, 0.01, 0.15, 0.15],
#         [0.05, 0.8, 0.1, 0.1],
#         [0.225, 0.69, 0.1, 0.1],
#         [0.34, 0.74, 0.1, 0.1],
#         [0.45, 0.79, 0.1, 0.1],
#         [0.56, 0.65, 0.1, 0.1],
#         [0.68, 0.69, 0.1, 0.1],
#         [0.8, 0.73, 0.1, 0.1],
#     ]
    
#     for j, path_drug in enumerate(path[:-1]):
#         path_molecule_img = visualize_molecule(path_drug)
#         if path_molecule_img:
#             image_ax = plt.gca().inset_axes(path_location[j])  # 设置位置
#             image_ax.imshow(path_molecule_img)
#             image_ax.axis('off')
            
#             # # 添加SMILES文本到分子图片下方
#             if j!=0:
#                 image_ax.text(0.5, -0.2, path_drug, ha='center', va='center', transform=image_ax.transAxes, fontsize=8)


#     for i, homo_values in enumerate(all_homo_values):
#         # 绘制路径的折线图
        
#         plt.plot(range(len(homo_values)), homo_values,  color=color_rgb, marker='o', linestyle='-', label=f"{first_molecules[i]}", zorder=10)
        
#         # # 仅绘制路径的第一个分子
#         # first_smiles = first_molecules[i]
#         # first_molecule_img = visualize_molecule(first_smiles)
#         # if first_molecule_img:
#         #     # 在折线图上方绘制路径的第一个分子
#         #     row = i // 7  # 计算行号
#         #     col = i % 7   # 计算列号
#         #     image_ax = plt.gca().inset_axes([0.15 * (col + 1)-0.18,  (row * 0.2)-0.65, 0.12, 0.12])  # 设置位置
#         #     image_ax.imshow(first_molecule_img)
#         #     image_ax.axis('off')
            
#         #     # 添加SMILES文本到分子图片下方
#         #     image_ax.text(0.5, -0.2, first_smiles, ha='center', va='center', transform=image_ax.transAxes, fontsize=10)
   


#     # 设置标题和标签
#     plt.xlabel("Step")
#     plt.ylabel("Homo value")
#     # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
#     plt.grid(False)
#     plt.xticks(ticks=range(len(homo_values)), labels=range(1, len(homo_values) + 1))

#     # 保存总体折线图
#     overall_output_path = os.path.join(output_dir, "overall_evolution_trend_with_zoom.pdf")
#     save_property_trend_with_zoom(all_homo_values, overall_output_path, valid_paths_with_aim_smiles, zoom=True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "overall_evolution_trend.pdf"), dpi=500, format='pdf')
#     plt.close()



# # 示例使用

# # 文件路径
# graph_file = "/home/data1/lk/project/mol_tree/graph/evolution_graph_133885_['edit_distance', 'graph']_0.3_v2.json"
# # labels_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'


# # aim_smiles = "CCCC"
# # label_name = 'homo'

# # 加载图数据和CSV文件
# molecule_evo_graph = load_graph_from_json(graph_file)
# # labels_df = pd.read_csv(labels_file)

# # 创建字典以快速访问标签
# # labels_dict = labels_df.set_index('SMILES1')[label_name].to_dict()

# import pdb;pdb.set_trace()
# mols =  [x for x in molecule_evo_graph.nodes] 
# all_paths = []

# for mol in tqdm(mols):
#     aim_smiles = mol
#     # 查找有效路径
#     valid_paths = [x for x in molecule_evo_graph.edges() if x[1] == aim_smiles]
#     valid_paths_with_aim_smiles = [
#         [x for x in find_paths(molecule_evo_graph, xx[0], 'C') if x[1] == aim_smiles] 
#         for xx in valid_paths
#     ]
#     try:
#         if len(valid_paths_with_aim_smiles)>=1 and valid_paths_with_aim_smiles[0]!=[]:
#             valid_paths_with_aim_smiles = [x[0][::-1] for x in valid_paths_with_aim_smiles]
#             all_paths.extend(valid_paths_with_aim_smiles)
#     except:
#         import pdb;pdb.set_trace()
#     # 绘制所有路径的属性变化趋势图，并进行局部放大
#     # plot_all_paths_evolution(valid_paths_with_aim_smiles, labels_dict, label_name)

# import pdb;pdb.set_trace()
# import json
# with open('/home/data1/lk/project/mol_tree/graph/evolution_paths.json', 'w') as f:
#     json.dump(all_paths, f, indent=4)

# ==================================================================================================================












# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# from rdkit import Chem
# from rdkit.Chem import Draw

# from tqdm import tqdm

# # 读取保存的 JSON 文件
# with open('/home/data1/lk/project/mol_tree/graph/evolution_paths.json', 'r') as f:
#     all_paths = json.load(f)

# label_name = 'homo'

# # 读取标签数据文件
# labels_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
# labels_df = pd.read_csv(labels_file)
# labels_dict = labels_df.set_index('SMILES1')[label_name].to_dict()

# # 找到路径的标签值
# def find_path_label(path, labels_dict):
#     return [labels_dict.get(smiles, None) for smiles in path]

# # 遍历所有路径
# for idx, path in tqdm(enumerate(all_paths)):
#     # 转换每个路径中的 SMILES 为分子对象
#     molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
    
#     # 创建图形窗口：仅用于分子结构图
#     fig_molecules, axes_molecules = plt.subplots(1, len(molecules), figsize=(len(molecules)*3, 3))
    
#     if len(molecules) == 1:
#         axes_molecules = [axes_molecules]  # 确保axes是一个可迭代对象
    
#     # 绘制每个分子的结构图
#     for ax, mol, i in zip(axes_molecules, molecules, range(len(molecules))):
#         if mol:
#             # 生成分子的图像
#             img = Draw.MolToImage(mol)
#             ax.imshow(img)
#             ax.axis('off')  # 不显示坐标轴
            
#             # 添加标注文本（SMILES和序号）
#             ax.text(0.5, -0.1, f'{i+1}. {Chem.MolToSmiles(mol)}', ha='center', va='center', fontsize=12, transform=ax.transAxes)
#         else:
#             ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=12)
#             ax.axis('off')
    
#     # 保存分子结构图为PDF
#     plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9/molecule_path_{idx + 1}_molecules.pdf', format='pdf', dpi=100)
#     plt.close(fig_molecules)  # 关闭分子图

#     # 获取路径的标签值
#     path_label = find_path_label(path, labels_dict)
    
#     # 创建图形窗口：仅用于标签变化趋势的折线图
#     fig_label, ax_label = plt.subplots(figsize=(6, 3))  # 创建一个单独的图形
    
#     # 绘制标签变化趋势的折线图
#     ax_label.plot(range(1, len(path_label) + 1), path_label, marker='o', color='b', linestyle='-', linewidth=2)
#     ax_label.set_xlabel('Step', fontsize=14)
#     ax_label.set_ylabel(label_name, fontsize=14)
#     ax_label.set_xticks(range(1, len(path_label) + 1))  # 设置x轴的刻度
#     ax_label.grid(True)
    
#     # 保存标签变化趋势图为PDF
#     plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9/molecule_path_{idx + 1}_label_trend.pdf', format='pdf', dpi=100)
#     plt.close(fig_label)  # 关闭标签变化趋势图



# +===============================================================================================


import json
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from tqdm import tqdm


import shap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

    
# 读取保存的 JSON 文件
with open('/home/data1/lk/project/mol_tree/graph/evolution_paths.json', 'r') as f:
    all_paths = json.load(f)

label_name = 'gap'

# 读取标签数据文件
labels_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
labels_df = pd.read_csv(labels_file)
labels_dict = labels_df.set_index('SMILES1')[label_name].to_dict()

# 找到路径的标签值
def find_path_label(path, labels_dict):
    return [labels_dict.get(smiles, None) for smiles in path]

import numpy as np
import random

# 设置 numpy 随机种子
np.random.seed(42)

# 设置 random 随机种子
random.seed(42)


from rdkit.Chem import BondType

from rdkit import Chem
from rdkit.Chem import rdmolops

import re

from rdkit import Chem
import re
from rdkit import Chem

import torch
from torch_geometric.data import Data

# 将分子转换为图数据（节点、边和边的特征）
def mol_to_graph(mol):
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)  # 每个原子的特征
    
    data = Data(x=x, edge_index=edge_index)
    return data

from rdkit import Chem

from collections import defaultdict
from rdkit import Chem

from collections import defaultdict
from rdkit import Chem

from collections import defaultdict
from rdkit import Chem

# def compare_smiles(smiles1, smiles2):
#     # 使用defaultdict，初始化每个变化类型为0
#     changes = defaultdict(int)

#     # 初始化所有变化类型的键
#     all_changes_keys = [
#         "Add atom C", "Add atom N", "Add atom O", "Add atom S", "Add atom P", 
#         "Insert atom", "Delete atom", "Uninsert atom", "Increase bond order", 
#         "Decrease bond order", "Create ring", "Break ring", "Ring type change", 
#         "Mutate atom", "Mutate atom C to N", "Mutate atom N to C", "Mutate atom C to O", 
#         "Mutate atom O to C", "Mutate atom N to O"
#     ]

#     # 检查环的变化及类型
#     def detect_rings(mol):
#         ring_info = mol.GetRingInfo()
#         ring_sizes = {len(ring) for ring in ring_info.AtomRings()}  # 获取所有环的大小（去重）
#         return ring_sizes

#     # 检查原子数量的变化
#     def count_atoms(mol):
#         return mol.GetNumAtoms()

#     def detect_bond_changes(mol1, mol2):
#         # 将两个分子转换为图数据
#         data1 = mol_to_graph(mol1)
#         data2 = mol_to_graph(mol2)

#         # 将边索引转为元组再构造集合
#         edges_data1 = set(tuple(edge) for edge in data1.edge_index.t().numpy().tolist())
#         edges_data2 = set(tuple(edge) for edge in data2.edge_index.t().numpy().tolist())

#         # 比较边的差异
#         added_edges = edges_data2 - edges_data1
#         removed_edges = edges_data1 - edges_data2
        
#         return len(added_edges), len(removed_edges)

#     # 检查是否有原子突变
#     def detect_atom_mutations(mol1, mol2):
#         atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
#         atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
#         mutations = defaultdict(int)
#         for a1, a2 in zip(atoms1, atoms2):
#             if a1 != a2:
#                 mutations[f"Mutate atom {a1} to {a2}"] += 1
#         return mutations

#     # Convert SMILES to RDKit Mol objects
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)

#     # 判断环变化及环类型
#     rings1 = detect_rings(mol1)
#     rings2 = detect_rings(mol2)

#     # 判断环的数量变化
#     if len(rings2) > len(rings1):
#         changes["Create ring"] += 1
#         # 记录新增的环类型
#         new_rings = rings2 - rings1
#         for ring_size in new_rings:
#             changes[f"Create ring {ring_size}"] += 1
#     elif len(rings2) < len(rings1):
#         changes["Break ring"] += 1
#         # 记录删除的环类型
#         removed_rings = rings1 - rings2
#         for ring_size in removed_rings:
#             changes[f"Break ring {ring_size}"] += 1

#     # 判断环类型的变化
#     if rings1 != rings2:
#         changes["Ring type change"] += 1

#     # 判断原子数量的变化
#     atoms1 = count_atoms(mol1)
#     atoms2 = count_atoms(mol2)
#     if atoms2 > atoms1:
#         added_atoms = [atom.GetSymbol() for atom in mol2.GetAtoms()]
#         added_atoms_types = set(added_atoms)  # 只取原子类型
#         for atom_type in added_atoms_types:
#             changes[f"Add atom {atom_type}"] += added_atoms.count(atom_type)
#     elif atoms2 < atoms1:
#         changes["Delete atom"] += atoms1 - atoms2
#         deleted_atoms = [atom.GetSymbol() for atom in mol1.GetAtoms()]
#         for atom_type in deleted_atoms:
#             changes[f"Delete atom {atom_type}"] += deleted_atoms.count(atom_type)

#     # 判断键的变化
#     added_bonds, removed_bonds = detect_bond_changes(mol1, mol2)
#     if added_bonds > 0:
#         changes["Increase bond order"] += added_bonds
#     if removed_bonds > 0:
#         changes["Decrease bond order"] += removed_bonds

#     # 判断原子突变
#     atom_mutations = detect_atom_mutations(mol1, mol2)
#     for mutation, count in atom_mutations.items():
#         changes[mutation] += count

#     # 确保所有可能的变化类型都出现在返回结果中，且没有发生的用0表示
#     result = {key: changes[key] for key in all_changes_keys}
#     return result

from collections import defaultdict
from rdkit import Chem

def compare_smiles(smiles1, smiles2):
    # 使用defaultdict，初始化每个变化类型为0
    changes = defaultdict(int)

    # 初始化所有变化类型的键
    all_changes_keys = [
        "Add atom C", "Add atom N", "Add atom O", "Add atom S", "Add atom P", 
        "Insert atom", "Delete atom", "Uninsert atom", "Increase bond order", 
        "Decrease bond order", "Create 3-member ring", "Create 4-member ring", 
        "Create 5-member ring", "Create 6-member ring", "Break 3-member ring", 
        "Break 4-member ring", "Break 5-member ring", "Break 6-member ring", 
        "Ring type change"
    ]

    # 检查环的变化及类型
    def detect_rings(mol):
        ring_info = mol.GetRingInfo()
        ring_sizes = {len(ring) for ring in ring_info.AtomRings()}  # 获取所有环的大小（去重）
        return ring_sizes

    # 检查原子数量的变化
    def count_atoms(mol):
        return mol.GetNumAtoms()

    def detect_bond_changes(mol1, mol2):
        # 将两个分子转换为图数据
        data1 = mol_to_graph(mol1)
        data2 = mol_to_graph(mol2)

        # 将边索引转为元组再构造集合
        edges_data1 = set(tuple(edge) for edge in data1.edge_index.t().numpy().tolist())
        edges_data2 = set(tuple(edge) for edge in data2.edge_index.t().numpy().tolist())

        # 比较边的差异
        added_edges = edges_data2 - edges_data1
        removed_edges = edges_data1 - edges_data2
        
        return len(added_edges), len(removed_edges)

    # # 检查是否有原子突变
    # def detect_atom_mutations(mol1, mol2):
    #     atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    #     atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    #     mutations = defaultdict(int)
    #     for a1, a2 in zip(atoms1, atoms2):
    #         if a1 != a2:
    #             mutations[f"Mutate atom {a1} to {a2}"] += 1
    #     return mutations

    # Convert SMILES to RDKit Mol objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # 判断环变化及环类型
    rings1 = detect_rings(mol1)
    rings2 = detect_rings(mol2)

    # 判断环的数量变化
    for ring_size in rings2 - rings1:
        changes[f"Create {ring_size}-member ring"] += 1  # 创建新的环，并记录环的大小
    for ring_size in rings1 - rings2:
        changes[f"Break {ring_size}-member ring"] += 1  # 破坏现有环，并记录环的大小

    # 判断环类型的变化
    if rings1 != rings2:
        changes["Ring type change"] += 1

    # 判断原子数量的变化
    atoms1 = count_atoms(mol1)
    atoms2 = count_atoms(mol2)
    if atoms2 > atoms1:
        added_atoms = [atom.GetSymbol() for atom in mol2.GetAtoms()]
        added_atoms_types = set(added_atoms)  # 只取原子类型
        for atom_type in added_atoms_types:
            # 计算原子增加的数量，即mol2中的原子类型数量减去mol1中的原子类型数量
            added_count = added_atoms.count(atom_type) - [atom.GetSymbol() for atom in mol1.GetAtoms()].count(atom_type)
            if added_count > 0:
                changes[f"Add atom {atom_type}"] += added_count
    elif atoms2 < atoms1:
        changes["Delete atom"] += atoms1 - atoms2
        deleted_atoms = [atom.GetSymbol() for atom in mol1.GetAtoms()]
        for atom_type in deleted_atoms:
            # 计算删除的原子数量，即mol1中的原子类型数量减去mol2中的原子类型数量
            deleted_count = deleted_atoms.count(atom_type) - [atom.GetSymbol() for atom in mol2.GetAtoms()].count(atom_type)
            if deleted_count > 0:
                changes[f"Delete atom {atom_type}"] += deleted_count

    # 判断键的变化
    added_bonds, removed_bonds = detect_bond_changes(mol1, mol2)
    if added_bonds > 0:
        changes["Increase bond order"] += added_bonds
    if removed_bonds > 0:
        changes["Decrease bond order"] += removed_bonds

    # # 判断原子突变
    # atom_mutations = detect_atom_mutations(mol1, mol2)
    # for mutation, count in atom_mutations.items():
    #     changes[mutation] += count

    # 确保所有可能的变化类型都出现在返回结果中，且没有发生的用0表示
    result = {key: changes[key] for key in all_changes_keys}
    return result


# def compare_smiles(smiles1, smiles2):
#     changes = {
#         "Add atom": 0,
#         "Insert atom": 0,
#         "Delete atom": 0,
#         "Uninsert atom": 0,
#         "Increase bond order": 0,
#         "Create ring": 0,
#         "Break ring": 0,
#         "Mutate atom": 0,
#         "Ring type change": 0  # 新增的环类型变化记录
#     }

#     # 检查环的变化及类型
#     def detect_rings(mol):
#         ring_info = mol.GetRingInfo()
#         ring_sizes = {len(ring) for ring in ring_info.AtomRings()}  # 获取所有环的大小（去重）
#         ring_types = sorted(ring_sizes)  # 返回已排序的环类型列表
#         return ring_types

#     # 检查原子数量的变化
#     def count_atoms(mol):
#         return mol.GetNumAtoms()

#     def detect_bond_changes(mol1, mol2):
#         # 将两个分子转换为图数据
#         data1 = mol_to_graph(mol1)
#         data2 = mol_to_graph(mol2)

#         # 将边索引转为元组再构造集合
#         edges_data1 = set(tuple(edge) for edge in data1.edge_index.t().numpy().tolist())
#         edges_data2 = set(tuple(edge) for edge in data2.edge_index.t().numpy().tolist())

#         # 比较边的差异
#         added_edges = edges_data2 - edges_data1
#         removed_edges = edges_data1 - edges_data2
        
#         return len(added_edges), len(removed_edges)

#     # 检查是否有原子突变
#     def detect_atom_mutations(mol1, mol2):
#         atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
#         atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
#         mutations = 0
#         for a1, a2 in zip(atoms1, atoms2):
#             if a1 != a2:
#                 mutations += 1
#         return mutations

#     # Convert SMILES to RDKit Mol objects
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)

#     # 判断环变化及环类型
#     rings1 = detect_rings(mol1)
#     rings2 = detect_rings(mol2)

#     # 判断环的数量变化
#     if len(rings2) > len(rings1):
#         changes["Create ring"] += 1
#     elif len(rings2) < len(rings1):
#         changes["Break ring"] += 1

#     # 判断环类型的变化
#     if rings1 != rings2:
#         changes["Ring type change"] = 1  # 记录环类型变化

#     # 判断原子数量的变化
#     atoms1 = count_atoms(mol1)
#     atoms2 = count_atoms(mol2)
#     if atoms2 > atoms1:
#         changes["Add atom"] += atoms2 - atoms1
#     elif atoms2 < atoms1:
#         changes["Delete atom"] += atoms1 - atoms2

#     # 判断键的变化
#     changes["Increase bond order"], changes["Decrease bond order"] = detect_bond_changes(mol1, mol2)

#     # 判断原子突变
#     changes["Mutate atom"] = detect_atom_mutations(mol1, mol2)

#     return changes


import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# plt.rcParams['font.family'] = 'Comic Sans MS'
# plt.rcParams['font.family'] = '/home/data1/lk/project/mol_tree/outputs_visual/ComicNeue-Regular.ttf'
  
from matplotlib import font_manager
import matplotlib.colors as mcolors
# 设置自定义字体
# font_path = '/home/data1/lk/project/mol_tree/outputs_visual/ComicNeue-Regular.ttf'
# prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()
# 使用 Times New Roman 字体
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def analysis(evolution_changes, path_label, marker):
    # 定义变化类型的列表
    all_changes_keys = [
        "Add atom C", "Add atom N", "Add atom O", "Increase bond order", 
        "Decrease bond order", "Create 3-member ring", "Create 4-member ring", 
        "Create 5-member ring", "Create 6-member ring", "Break 3-member ring", 
        "Break 4-member ring", "Break 5-member ring", "Break 6-member ring", 
        "Ring type change"
    ]
    # 构建字典：index:item
    change_mapping = {index: item for index, item in enumerate(all_changes_keys)}

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame(evolution_changes)
    df.rename(columns=change_mapping, inplace=True)

    # 目标变量：path_label
    y = np.array(path_label)

    # 特征变量：evolution_changes
    X = df.to_numpy()

    # 对特征进行标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用XGBoost分类器来训练模型
    model = xgb.XGBRegressor()
    model.fit(X_scaled, y)

    # 使用SHAP解释器来分析XGBoost模型
    explainer = shap.Explainer(model, X_scaled)

    # # 计算SHAP值
    # shap_values = explainer(X_scaled)
    # # import pdb;pdb.set_trace()
    # # 绘制SHAP特征重要性图
    # shap.summary_plot(shap_values, X_scaled, feature_names=df.columns)
    # # import pdb;pdb.set_trace()
    # 计算SHAP值
    shap_values = explainer(X_scaled)

    # 绘制SHAP特征重要性图
    plt.figure(figsize=(10, 8))  # 调整图形大小
    # import pdb;pdb.set_trace()

    # shap.summary_plot(shap_values, X_scaled, feature_names=df.columns)

    # 创建自定义颜色映射（RGB）
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_rgb", [(0, (116/255, 152/255, 200/255)),  # Green
                    (1, (187/255, 91/255, 79/255))], # Red
        N=256)

    # 绘制SHAP特征重要性图，设置自定义颜色
    shap.summary_plot(shap_values, X_scaled, feature_names=df.columns, cmap=cmap)


    # 迭代所有的scatter对象并设置颜色
    for scatter in ax.collections:
        scatter.set_cmap(cmap)
        scatter.set_norm(norm)
        
    # 放大字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 设置y轴刻度显示格式为两位有效数字

    plt.xlabel('Feature Importance', fontsize=16)
    plt.ylabel('Operations', fontsize=16)
    plt.tight_layout()
    # 保存图像到指定路径
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100/shap_summary_plot_{marker}.png')


import random
results = [[],[]]
# 遍历所有路径
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# 随机抽取100个索引
# random_indices = random.sample(range(len(all_paths)), 100)
random_indices = [130871,99596]
for idx, path in enumerate(tqdm(all_paths)):
    if idx not in random_indices:
        continue
    molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
    molecules = molecules[:7]
    # 创建图形窗口：仅用于分子结构图
    fig_molecules, axes_molecules = plt.subplots(1, len(molecules), figsize=(len(molecules)*3, 3))
    
    if len(molecules) == 1:
        axes_molecules = [axes_molecules]  # 确保axes是一个可迭代对象
    
    # 绘制每个分子的结构图
    for ax, mol, i in zip(axes_molecules, molecules, range(len(molecules))):
        if mol:
            # 生成分子的图像
            img = Draw.MolToImage(mol, size=(300, 300))  # 设置更大的图像尺寸

            # img = Draw.MolToImage(mol)
            ax.imshow(img)
            ax.axis('off')  # 不显示坐标轴
            
            # 添加标注文本（SMILES和序号）
            ax.text(0.5, -0.1, f'{i+1}.{Chem.MolToSmiles(mol)}', ha='center', va='center', fontsize=21, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # 保存分子结构图为PDF
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100/molecule_path_{idx + 1}_molecules.pdf', format='pdf', dpi=1000)
    plt.close(fig_molecules)  # 关闭分子图

    # 获取路径的标签值
    path_label = find_path_label(path, labels_dict)
    path_label = path_label[:7]
    # 创建图形窗口：仅用于标签变化趋势的折线图
    fig_label, ax_label = plt.subplots(figsize=(6, 4))  # 创建一个单独的图形
    
    # 绘制标签变化趋势的折线图
    ax_label.plot(range(1, len(path_label) + 1), path_label, marker='o', color=[47/255, 82/255, 145/255], linestyle='-', linewidth=4)
    ax_label.set_xlabel('Step', fontsize=22)
    ax_label.set_ylabel("Gap", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    # 设置y轴刻度的间隔较大一些，避免过于密集
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))



    ax_label.set_xticks(range(1, len(path_label) + 1))  # 设置x轴的刻度
    ax_label.grid(False)
    # 自动调整布局，避免标签和内容重叠
    plt.tight_layout()
    
    # 保存标签变化趋势图为PDF
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100/molecule_path_{idx + 1}_label_trend.pdf', format='pdf', dpi=1000)
    plt.close(fig_label)  # 关闭标签变化趋势图
    
    for i in range(len(path) - 1):
        # mol1 = Chem.MolFromSmiles(path[i])
        # mol2 = Chem.MolFromSmiles(path[i + 1])
        # changes = compare_molecules(mol1, mol2)
        changes = compare_smiles(path[i], path[i+1])
        
        results[0].append( [x[1] for x in changes.items()][:3]+[x[1] for x in changes.items()][8:] )
        # print(path[i], '->', path[i+1], ':', changes)
        # # 累计变化统计
        # for change_type in evolution_changes:
        #     evolution_changes[change_type].append(changes[change_type])
    
    # 输出每种变化的统计结果
    # print(f"Evolution Changes for Path {idx + 1}:")
    # for change_type, count in evolution_changes.items():
    #     print(f"{change_type}: {count}")

    # results[0].extend( [x[1] for x in evolution_changes.items()])
    
    # 除去34567
    
    results[1].extend( np.diff(path_label))
    
    # path_label: [-0.3877, -0.3385, -0.323, -0.317, -0.2029, -0.2454, -0.227, -0.2252, -0.1945]
    # evolution_changes:{'Add atom': [1, 1, 1, 1, 1, 1, 1, 1], 'Insert atom': [0, 0, 0, 0, 0, 0, 0, 0], 'Delete atom': [0, 0, 0, 0, 0, 0, 0, 0], 'Uninsert atom': [0, 0, 0, 0, 0, 0, 0, 0], 'Increase bond order': [1, 1, 1, 2, 2, 1, 2, 2], 'Create ring': [0, 0, 0, 1, 0, 0, 0, 0], 'Decrease bond order': [0, 0, 0, 0, 1, 0, 1, 1], 'Break ring': [0, 0, 0, 0, 0, 0, 0, 0], 'Mutate atom': [0, 0, 0, 1, 2, 2, 3, 4]}
    # import pdb;pdb.set_trace()
    # if idx==100:
    #     break
    
    
    
# import pdb;pdb.set_trace()
# analysis(results[0], results[1],f'test_2_XGBoost_{label_name}')
# analysis(results[0], results[1],f'100_XGBoost_{label_name}')
# analysis(results[0], results[1],f'alldata_XGBoost_{label_name}')