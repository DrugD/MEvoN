

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

plt.rcParams['font.serif'] = ['Times New Roman']

# 遍历所有路径
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


import random
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from tqdm import tqdm


# 随机抽取100个索引
# random_indices = random.sample(range(len(all_paths)), 100)



random_indices = [130871,99596]


for idx, path in enumerate(tqdm(all_paths)):
    if idx not in random_indices:
        continue

    # 创建分子结构图
    molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
    molecules = molecules[:7]

    fig_molecules, axes_molecules = plt.subplots(1, len(molecules), figsize=(len(molecules) * 3, 3))

    if len(molecules) == 1:
        axes_molecules = [axes_molecules]  # 确保axes是一个可迭代对象

    for ax, mol, i in zip(axes_molecules, molecules, range(len(molecules))):
        if mol:
            img = Draw.MolToImage(mol, size=(250, 250))  # 设置更大的图像尺寸
            ax.imshow(img)
            ax.axis('off')
            if idx == 130871 and i==2:
                ax.text(0.5, -0.1, f'{i + 1}.{Chem.MolToSmiles(mol)}', ha='center', va='center', color='darkred', fontsize=22, transform=ax.transAxes)
            elif idx == 99596 and i==5:
                ax.text(0.5, -0.1, f'{i + 1}.{Chem.MolToSmiles(mol)}', ha='center', va='center', color='darkred', fontsize=22, transform=ax.transAxes)
            else:
                ax.text(0.5, -0.1, f'{i + 1}.{Chem.MolToSmiles(mol)}', ha='center', va='center', fontsize=20, transform=ax.transAxes)
        
        else:
            ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=12)
            ax.axis('off')

    # 保存分子结构图为PDF
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100_new/molecule_path_{idx + 1}_molecules.pdf', format='pdf', dpi=1000)
    plt.close(fig_molecules)

    # 获取路径的标签值
    path_label = find_path_label(path, labels_dict)
    path_label = path_label[:7]

    # 创建标签变化趋势的折线图
    fig_label, ax_label = plt.subplots(figsize=(6, 4))


    # 根据不同的索引标记折线图中的特定线段为深红色
    if idx == 130871:
        # 从1到2绘制深蓝色
        ax_label.plot(range(1, 3), path_label[0:2], marker='o', color='black', linestyle='-', linewidth=3)
        # 从2到3绘制深红色
        ax_label.plot(range(2, 4), path_label[1:3], marker='o', color='darkblue', linestyle='-', linewidth=5)
        # 从3到4绘制深红色
        ax_label.plot(range(3, 5), path_label[2:4], marker='o', color='darkred', linestyle='-', linewidth=5)
        # 从4到5绘制深蓝色
        ax_label.plot(range(4, 6), path_label[3:5], marker='o', color='black', linestyle='-', linewidth=3)
        # 从5到6绘制深蓝色
        ax_label.plot(range(5, 8), path_label[4:7], marker='o', color='black', linestyle='-', linewidth=3)
        ax_label.text(0.2, 0.2, 'Increase', ha='center', va='center', fontsize=20, color='darkblue', transform=ax_label.transAxes)
        ax_label.text(0.6, 0.4, 'Decrease', ha='center', va='center', fontsize=20, color='darkred', transform=ax_label.transAxes)



    elif idx == 99596:
        # 从1到4绘制深蓝色
        ax_label.plot(range(1, 6), path_label[0:5], marker='o', color='black', linestyle='-', linewidth=3)
        # 从5到6绘制深红色
        ax_label.plot(range(5, 7), path_label[4:6], marker='o', color='darkred', linestyle='-', linewidth=5)
        # 从6到7绘制深蓝色
        ax_label.plot(range(6, 8), path_label[5:7], marker='o', color='black', linestyle='-', linewidth=3)
        # ax_label.text(0.5, 0.1, f'Increase', ha='center', va='center', fontsize=21, transform=ax.transAxes)
        # ax_label.text(0.3, 0.1, f'Decrease', ha='center', va='center', fontsize=21, transform=ax.transAxes)
        # 调整Decrease文本的位置
        ax_label.text(0.53, 0.3, 'Decrease', ha='center', va='center', fontsize=20, color='darkred', transform=ax_label.transAxes)

    ax_label.set_xlabel('Step', fontsize=22)
    ax_label.set_ylabel("Gap", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    # 设置y轴刻度的间隔较大一些
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))

    ax_label.set_xticks(range(1, len(path_label) + 1))  # 设置x轴的刻度
    ax_label.grid(False)
    plt.tight_layout()

    # 保存标签变化趋势图为PDF
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100_new/molecule_path_{idx + 1}_label_trend.pdf', format='pdf', dpi=1000)
    plt.close(fig_label)

# import py3Dmol
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import random

# random_indices = [130871, 99596]

# for idx, path in enumerate(tqdm(all_paths)):
#     if idx not in random_indices:
#         continue

#     # 创建分子对象并生成3D坐标
#     molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
#     molecules = molecules[:7]
    
#     # 创建图形窗口：用于3D分子结构图
#     fig_molecules = plt.figure(figsize=(len(molecules) * 3, 3))

#     # 创建3D可视化窗口
#     ax3d = fig_molecules.add_subplot(111, projection='3d')
    
#     for i, mol in enumerate(molecules):
#         if mol:
#             # 生成3D坐标
#             AllChem.EmbedMolecule(mol)
#             mol_block = Chem.MolToMolBlock(mol)

#             # 使用Py3Dmol展示3D分子结构
#             viewer = py3Dmol.view(width=300, height=300)
#             viewer.addModel(mol_block, "mol")
#             viewer.setStyle({'stick': {}})
#             viewer.setBackgroundColor('white')
#             viewer.zoomTo()

#             # 将3D分子结构图显示在matplotlib上
#             viewer.render()
            
#             # 正确显示文本，确保'text'的's'参数为分子SMILES
#             ax3d.text(0.5, 0.5, f'{i + 1}.{Chem.MolToSmiles(mol)}', ha='center', va='center', fontsize=21)
#         else:
#             ax3d.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=12)
    
#     # 保存分子结构图为PDF
#     plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100_new/molecule_path_{idx + 1}_molecules_3d.pdf', format='pdf', dpi=1000)
#     plt.close(fig_molecules)

#     # 获取路径的标签值
#     path_label = find_path_label(path, labels_dict)
#     path_label = path_label[:7]

#     # 创建标签变化趋势的折线图
#     fig_label, ax_label = plt.subplots(figsize=(6, 4))

#     # 根据不同的索引标记折线图中的特定线段为深红色
#     if idx == 130871:
#         ax_label.plot(range(1, 3), path_label[0:2], marker='o', color='darkblue', linestyle='-', linewidth=4)
#         ax_label.plot(range(2, 4), path_label[1:3], marker='o', color='darkred', linestyle='-', linewidth=4)
#         ax_label.plot(range(3, 5), path_label[2:4], marker='o', color='darkred', linestyle='-', linewidth=4)
#         ax_label.plot(range(4, 6), path_label[3:5], marker='o', color='darkblue', linestyle='-', linewidth=4)
#         ax_label.plot(range(5, 8), path_label[4:7], marker='o', color='darkblue', linestyle='-', linewidth=4)
#     elif idx == 99596:
#         ax_label.plot(range(1, 6), path_label[0:5], marker='o', color='darkblue', linestyle='-', linewidth=4)
#         ax_label.plot(range(5, 7), path_label[4:6], marker='o', color='darkred', linestyle='-', linewidth=4)
#         ax_label.plot(range(6, 8), path_label[5:7], marker='o', color='darkblue', linestyle='-', linewidth=4)

#     ax_label.set_xlabel('Step', fontsize=22)
#     ax_label.set_ylabel("Gap", fontsize=22)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

#     plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))

#     ax_label.set_xticks(range(1, len(path_label) + 1))  # 设置x轴的刻度
#     ax_label.grid(False)
#     plt.tight_layout()

#     # 保存标签变化趋势图为PDF
#     plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9_random_100_new/molecule_path_{idx + 1}_label_trend.pdf', format='pdf', dpi=1000)
#     plt.close(fig_label)
