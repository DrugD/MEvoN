'''
对变化的分布规律进行分析，路径-潜在变化趋势概率分析
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

import pandas as pd
import torch, json
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm
from tools import *
import math

class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)  # 存储图的邻接表
        self.nodes = set()  # 存储所有节点
    
    def add_node(self, label: str):
        self.nodes.add(label)
    
    def add_edge(self, from_node: str, to_node: str):
        if from_node in self.nodes and to_node in self.nodes:
            self.adjacency_list[from_node].append(to_node)

    def get_edges(self):
        return self.adjacency_list
    
    def edges(self):
        # Return a list of edges as tuples
        return [(from_node, to_node) for from_node, to_nodes in self.adjacency_list.items() for to_node in to_nodes]

    def __str__(self):
        return str(dict(self.adjacency_list))


def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        

    graph = Graph()
    # 添加节点
    for node in data['nodes']:
        graph.add_node(node)

    # 添加边
    for edge in data['edges']:
        graph.add_edge(edge[0], edge[1])

    return graph

# -------------------------------------------------------------------

import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def visualize_molecule(smiles):
    """
    生成分子的二维结构图
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol)
    return None

def sort_by_last_homo_value(homo_values, valid_paths_with_aim_smiles):
    """
    按照homo_values中每个元素的最后一位值从大到小对homo_values, output_paths, valid_paths_with_aim_smiles进行排序
    """
    # 使用zip将homo_values, output_paths, valid_paths_with_aim_smiles绑定成元组
    combined = list(zip(homo_values, valid_paths_with_aim_smiles))

    # 对组合后的列表按照homo_values的最后一个元素排序，reverse=True表示降序
    combined_sorted = sorted(combined, key=lambda x: x[0][-1], reverse=True)

    # 解压排序后的元组
    sorted_homo_values, sorted_valid_paths = zip(*combined_sorted)

    # 返回排序后的三个列表
    return list(sorted_homo_values), list(sorted_valid_paths)


def save_property_trend_with_zoom(homo_values, output_path, valid_paths_with_aim_smiles, zoom=False, colors=None, line_styles=None, markers=None):
    """
    保存属性变化的折线图，并在需要时进行局部放大
    """
    plt.figure(figsize=(40, 25))

    # 默认的颜色（使用RGB元组）
    if colors is None:
        colors = [
            (198/255,    108/255,      97/255),
            ( 71/255,     71/255,     105/255), 
            (115/255,    165/255,     162/255),
            (178/255,    182/255,     193/255)
        ]

    if line_styles is None:
        line_styles = ['-', '--',]
    
    if markers is None:
        markers = ['o', 's', '^']

    # import pdb;pdb.set_trace()
    homo_values, valid_paths_with_aim_smiles = sort_by_last_homo_value(homo_values, valid_paths_with_aim_smiles)

    # 绘制总体折线图
    for i, homo_value in enumerate(homo_values):
        if i>=12:
            line_styles = ['--','-',  ]
        color = colors[i % len(colors)]  # 循环使用颜色
        linestyle = line_styles[i % len(line_styles)]  # 循环使用线型
        marker = markers[i % len(markers)]  # 循环使用标记符号
        plt.plot([0,1], homo_value[-2:],  marker=marker, markersize=15, linestyle=linestyle, color=color, label=f"{valid_paths_with_aim_smiles[i][-1]}: {homo_value[-1]}", linewidth=5)

    
    # color_rgb = (138/255, 160/255, 183/255)
    # plt.plot([0,1], homo_values[0][6:8],  color=color_rgb, marker='o', linestyle='-', linewidth=5)


        
    # plt.title("Evolution Trend")
    plt.xlabel("Steps", fontsize=45)
    plt.ylabel("Homo value", fontsize=45)
    plt.grid(False)
    # plt.legend(loc="best")
    # plt.legend(loc=9, fontsize=25, bbox_to_anchor=(0.9, 1.0))
    plt.legend(ncol=1, loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=50)
    plt.xticks(ticks=range(2), labels=[8,9], fontsize=45)
    plt.yticks(fontsize=45)
    # if zoom:
    #     # 放大后1~2步的区域，调整xlim和ylim
    #     plt.xlim(7, 8)  # 放大x轴范围，第n-2到第n步
        
    #     # 强制设置Y轴范围为 [-0.28, -0.22] 之间
    #     plt.ylim(-0.28, -0.22)  # 放大y轴范围

    #     # 插入子坐标并绘图（放大部分）
    #     axins = inset_axes(plt.gca(), width="30%", height="40%", loc='lower left',
    #                        bbox_to_anchor=(0.5, 0.25, 1, 1),
    #                        bbox_transform=plt.gca().transAxes)
    #     for i, homo_value in enumerate(homo_values):
    #         axins.plot(range(len(homo_value)), homo_value, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
       
    #     axins.set_xlim(7, 8)
    #     axins.set_ylim(-0.28, -0.22)

        # # 添加极值点的标注
        # axins.annotate(
        #     "Min value",
        #     xy=(8, homo_values[0][8]),  # 假设放大区域是第8步
        #     xytext=(6-1, homo_values[0][8] - 0.02),
        #     arrowprops={'arrowstyle': '->', 'color': 'red'},
        #     fontsize=10,
        #     color='red'
        # )

        # # 在主图和放大图之间添加连接线
        # con = ConnectionPatch(xyA=(8, homo_values[0][8]), xyB=(8, homo_values[0][8]),
        #                       coordsA="data", coordsB="data", axesA=axins, axesB=plt.gca())
        # axins.add_artist(con)

    # 保存折线图
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, format='pdf')
    plt.close()

def plot_all_paths_evolution(valid_paths_with_aim_smiles, labels_dict, label_name, output_dir='/home/data1/lk/project/mol_tree/outputs_visual'):
    """
    绘制所有路径的属性变化，展示1号元素的变化与属性的关系
    """
    all_homo_values = []
    first_molecules = []  # 用来存储每条路径的第一个分子
    
    # 收集所有路径的homo值变化
    for path in valid_paths_with_aim_smiles:
        homo_values = []
        first_molecule = path[-1]  # 记录路径的第一个分子

        for smiles in path:
            if smiles in labels_dict:
                homo_values.append(labels_dict[smiles])
        
        all_homo_values.append(homo_values)
        first_molecules.append(first_molecule)
    
    # 绘制总体折线图
    plt.figure(figsize=(10, 10))

    
    # 计算显示图片的行列数
    num_molecules = len(first_molecules)
    rows = (num_molecules // 7) + 1  # 每行显示最多5个分子
    cols = min(num_molecules, 7)  # 最后一行可能不满5个
    color_rgb = (138/255, 160/255, 183/255)
    path_location = [
        [0.05, 0.01, 0.15, 0.15],
        [0.05, 0.8, 0.1, 0.1],
        [0.225, 0.69, 0.1, 0.1],
        [0.34, 0.74, 0.1, 0.1],
        [0.45, 0.79, 0.1, 0.1],
        [0.56, 0.65, 0.1, 0.1],
        [0.68, 0.69, 0.1, 0.1],
        [0.8, 0.73, 0.1, 0.1],
    ]
    
    for j, path_drug in enumerate(path[:-1]):
        path_molecule_img = visualize_molecule(path_drug)
        if path_molecule_img:
            image_ax = plt.gca().inset_axes(path_location[j])  # 设置位置
            image_ax.imshow(path_molecule_img)
            image_ax.axis('off')
            
            # # 添加SMILES文本到分子图片下方
            if j!=0:
                image_ax.text(0.5, -0.2, path_drug, ha='center', va='center', transform=image_ax.transAxes, fontsize=8)


    for i, homo_values in enumerate(all_homo_values):
        # 绘制路径的折线图
        
        plt.plot(range(len(homo_values)), homo_values,  color=color_rgb, marker='o', linestyle='-', label=f"{first_molecules[i]}", zorder=10)
        
        # # 仅绘制路径的第一个分子
        # first_smiles = first_molecules[i]
        # first_molecule_img = visualize_molecule(first_smiles)
        # if first_molecule_img:
        #     # 在折线图上方绘制路径的第一个分子
        #     row = i // 7  # 计算行号
        #     col = i % 7   # 计算列号
        #     image_ax = plt.gca().inset_axes([0.15 * (col + 1)-0.18,  (row * 0.2)-0.65, 0.12, 0.12])  # 设置位置
        #     image_ax.imshow(first_molecule_img)
        #     image_ax.axis('off')
            
        #     # 添加SMILES文本到分子图片下方
        #     image_ax.text(0.5, -0.2, first_smiles, ha='center', va='center', transform=image_ax.transAxes, fontsize=10)
   


    # 设置标题和标签
    plt.xlabel("Step")
    plt.ylabel("Homo value")
    # plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.grid(False)
    plt.xticks(ticks=range(len(homo_values)), labels=range(1, len(homo_values) + 1))

    # 保存总体折线图
    overall_output_path = os.path.join(output_dir, "overall_evolution_trend_with_zoom.pdf")
    save_property_trend_with_zoom(all_homo_values, overall_output_path, valid_paths_with_aim_smiles, zoom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_evolution_trend.pdf"), dpi=500, format='pdf')
    plt.close()



# 示例使用

# 文件路径
graph_file = "/home/data1/lk/project/mol_tree/graph/evolution_graph_133885_['edit_distance', 'graph']_0.3_v2.json"
labels_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'

aim_smiles = "CN1CC1"
# aim_smiles = "CCCC"
label_name = 'homo'

# 加载图数据和CSV文件
molecule_evo_graph = load_graph_from_json(graph_file)
labels_df = pd.read_csv(labels_file)

# 创建字典以快速访问标签
labels_dict = labels_df.set_index('SMILES1')[label_name].to_dict()

# 查找有效路径
valid_paths = [x for x in molecule_evo_graph.edges() if x[1] == aim_smiles]
valid_paths_with_aim_smiles = [
    [x for x in find_paths(molecule_evo_graph, xx[0], 'C') if x[1] == aim_smiles] 
    for xx in valid_paths
]

valid_paths_with_aim_smiles = [x[0][::-1] for x in valid_paths_with_aim_smiles]

import pdb;pdb.set_trace()

# 绘制所有路径的属性变化趋势图，并进行局部放大
plot_all_paths_evolution(valid_paths_with_aim_smiles, labels_dict, label_name)
print(valid_paths_with_aim_smiles)