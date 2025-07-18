
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

    # 确保所有可能的变化类型都出现在返回结果中，且没有发生的用0表示
    result = {key: changes[key] for key in all_changes_keys}
    return result


import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def analysis(evolution_changes, path_label, marker):
    # 定义变化类型的列表
    all_changes_keys = [
        "Add atom C", "Add atom N", "Add atom O", "Add atom S", "Add atom P", 
        "Insert atom", "Delete atom", "Uninsert atom", "Increase bond order", 
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

    # 计算SHAP值
    shap_values = explainer(X_scaled)
    # import pdb;pdb.set_trace()
    # 绘制SHAP特征重要性图
    shap.summary_plot(shap_values, X_scaled, feature_names=df.columns)
    # import pdb;pdb.set_trace()
    
    # 保存图像到指定路径
    plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9/shap_summary_plot_{marker}.png')

    # 显示图像
    plt.show()


results = [[],[]]
# 遍历所有路径

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

def parse_arguments():
    # 解析命令行参数，获取k的值
    parser = argparse.ArgumentParser(description="Process SMILES data in chunks.")
    parser.add_argument('--k', type=int, required=True, help="The chunk index (int), starting from 0.")
    return parser.parse_args()

def save_results(results, k):
    # 保存results到指定文件
    torch.save(results, f'/home/data1/lk/project/mol_tree/graph/qm9/aaa_{k}.pth')

def process_path_chunk(k, all_paths, labels_dict, label_name):
    # 确定当前要处理的索引范围
    start_idx = k * 10000
    end_idx = min((k + 1) * 10000, len(all_paths))
    
    # 用来存储当前块的结果
    results = [[], []]  # 存储两组数据
    
    for idx in tqdm(range(start_idx, end_idx)):
        path = all_paths[idx]
        
        # 处理分子图
        molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
        fig_molecules, axes_molecules = plt.subplots(1, len(molecules), figsize=(len(molecules) * 3, 3))
        
        if len(molecules) == 1:
            axes_molecules = [axes_molecules]  # 确保axes是一个可迭代对象
        
        for ax, mol, i in zip(axes_molecules, molecules, range(len(molecules))):
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))  # 设置图像尺寸
                ax.imshow(img)
                ax.axis('off')
                ax.text(0.5, -0.1, f'{i+1}. {Chem.MolToSmiles(mol)}', ha='center', va='center', fontsize=18, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=12)
                ax.axis('off')
        
        plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9/molecule_path_{idx + 1}_molecules.pdf', format='pdf', dpi=100)
        plt.close(fig_molecules)
        
        # 获取路径的标签
        path_label = find_path_label(path, labels_dict)
        
        # 绘制标签趋势图
        fig_label, ax_label = plt.subplots(figsize=(6, 3))
        ax_label.plot(range(1, len(path_label) + 1), path_label, marker='o', color='b', linestyle='-', linewidth=2)
        ax_label.set_xlabel('Step', fontsize=14)
        ax_label.set_ylabel(label_name, fontsize=14)
        ax_label.set_xticks(range(1, len(path_label) + 1))
        ax_label.grid(True)
        
        plt.savefig(f'/home/data1/lk/project/mol_tree/graph/qm9/molecule_path_{idx + 1}_label_trend.pdf', format='pdf', dpi=100)
        plt.close(fig_label)
        
        # 分析SMILES变化并更新结果
        for i in range(len(path) - 1):
            changes = compare_smiles(path[i], path[i + 1])
            results[0].append([x[1] for x in changes.items()])
        results[1].extend(np.diff(path_label))
    
    # 保存当前k块的结果
    save_results(results, k)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    k = args.k

    # 加载所有路径和标签数据
    with open('/home/data1/lk/project/mol_tree/graph/evolution_paths.json', 'r') as f:
        all_paths = json.load(f)
    
    labels_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
    labels_df = pd.read_csv(labels_file)
    label_name = 'gap'  # 用实际的标签名替换
    labels_dict = labels_df.set_index('SMILES1')[label_name].to_dict()

    # 处理数据块并保存结果
    process_path_chunk(k, all_paths, labels_dict, label_name)

    
# import pdb;pdb.set_trace()
# analysis(results[0], results[1],f'alldata_XGBoost_{label_name}')