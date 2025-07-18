import os
import re
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import FingerprintSimilarity
from grakel import Graph, WeisfeilerLehman
from collections import defaultdict
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')  # type: ignore
from torch_geometric.utils import one_hot, scatter
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# qm9
types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

# qm7
# types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    
def find_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph.adjacency_list:
        return []
    
    paths = []
    for node in graph.adjacency_list[start]:
        if node not in path:  # 防止循环
            new_paths = find_paths(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

class SimilarityCalculator:
    def __init__(self, method="fingerprint"):
        self.method = method

    def calculate_similarity(self, mol1, mol2):
        """根据不同的方法计算分子相似度"""
        if self.method == "fingerprint":
            return self.fingerprint_similarity(mol1, mol2)
        elif self.method == "graph":
            return self.graph_similarity(mol1, mol2)
        elif self.method == "mcs":
            return self.mcs_similarity(mol1, mol2)
        elif self.method == "edit_distance":
            return self.edit_distance_similarity(mol1, mol2)
        elif self.method == "quantum":
            return self.quantum_chemical_similarity(mol1, mol2)
        elif self.method == "topological":
            return self.topological_similarity(mol1, mol2)
        else:
            raise ValueError(f"不支持的相似度方法: {self.method}")

    def fingerprint_similarity(self, smiles1, smiles2):
        """使用分子指纹计算相似度"""
        if type(smiles1)==str and  type(smiles2)==str:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
        else:
            mol1 = smiles1
            mol2 = smiles2
            
        if mol1 is None or mol2 is None:
            print(f"无法解析 SMILES: {smiles1}, {smiles2}")
            return None
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and  mol1.GetAtomWithIdx(0).GetSymbol() == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        # 尝试使用不同的指纹类型
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)  # 使用 Morgan 指纹 (ECFP)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # 计算相似度（Tanimoto coefficient）
        score = DataStructs.TanimotoSimilarity(fp1, fp2)
        print(f"{smiles1}, {smiles2}, 相似度: {score}")

        return score

    def graph_similarity(self, smiles1, smiles2):
        """使用图核计算分子图相似度"""
        # import pdb;pdb.set_trace()
        if type(smiles1)==str and  type(smiles2)==str:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
        else:
            mol1 = smiles1
            mol2 = smiles2
        
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and  mol1.GetAtomWithIdx(0).GetSymbol() == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            # print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        g1 = self.smiles_to_grakel(mol1)
        g2 = self.smiles_to_grakel(mol2)
        if g1 is None or g2 is None:
            return 0.0
        
        gk = WeisfeilerLehman(n_iter=5, normalize=True)
        K = gk.fit_transform([g1, g2])
        score = K[0, 1]
        # print(f"{smiles1}calculate_similarity,{smiles2},{score}")
        return score
    
    def mcs_similarity(self, mol1, mol2):
        """基于最大公共子结构（MCS）的相似度"""
        mcs_result = rdFMCS.FindMCS([mol1, mol2])
        common_substructure = Chem.MolFromSmarts(mcs_result.smartsString)
        if common_substructure is None:
            return 0.0
        num_common_atoms = common_substructure.GetNumAtoms()
        avg_atoms = (mol1.GetNumAtoms() + mol2.GetNumAtoms()) / 2
        return num_common_atoms / avg_atoms

    def edit_distance_similarity(self, smiles1, smiles2):
        if type(smiles1)==str and  type(smiles2)==str:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
        else:
            mol1 = smiles1
            mol2 = smiles2
            
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and  mol1.GetAtomWithIdx(0).GetSymbol() == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            # print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        """使用分子编辑距离计算相似度"""
        # 计算分子的编辑距离
        dist = self.molecular_edit_distance(mol1, mol2)
        
        # 使用原子数的最大值作为标准化因子
        max_len = max(mol1.GetNumAtoms(), mol2.GetNumAtoms())
        # 返回相似度 (1 - 归一化的编辑距离)
        score = 1 - dist / max_len
        # print(f"{smiles1}, {smiles2}, 相似度: {score}")
        return score

    def quantum_chemical_similarity(self, smiles1, smiles2):
        """使用量子化学性质计算相似度（简化为通过分子量的近似判断）"""
        if type(smiles1)==str and  type(smiles2)==str:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
        else:
            mol1 = smiles1
            mol2 = smiles2
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and  mol1.GetAtomWithIdx(0).GetSymbol() == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        weight1 = Chem.rdMolDescriptors.CalcExactMolWt(mol1)
        weight2 = Chem.rdMolDescriptors.CalcExactMolWt(mol2)
        return 1.0 - abs(weight1 - weight2) / max(weight1, weight2)

    def topological_similarity(self, smiles1, smiles2):
        """基于拓扑描述符的相似度"""
        if type(smiles1)==str and  type(smiles2)==str:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
        else:
            mol1 = smiles1
            mol2 = smiles2
            
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and  mol1.GetAtomWithIdx(0).GetSymbol() == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        # 使用简单的分子拓扑描述符如 Wiener 指数
        topo1 = Chem.rdMolDescriptors.CalcChi0n(mol1)
        topo2 = Chem.rdMolDescriptors.CalcChi0n(mol2)
        return 1.0 - abs(topo1 - topo2) / max(topo1, topo2)


    def smiles_to_grakel(self, mol):
        """将 SMILES 转化为 Grakel 支持的图表示"""
        # mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 获取原子编号作为节点标签
        atom_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        # 获取邻接矩阵
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        
        # 如果是单原子分子，特殊处理为自循环边
        if adj_matrix.shape[0] == 1:
            edges = [(0, 0)]  # 添加一个自循环边
            return Graph(edges, node_labels={0: atom_labels[0]})

        # 处理多原子分子的边
        edges = [(i, j) for i in range(adj_matrix.shape[0]) for j in range(i + 1, adj_matrix.shape[1]) if adj_matrix[i, j] == 1]

        # 将边和节点标签转换为 Grakel 支持的格式
        try:
            result = Graph(edges, node_labels={i: l for i, l in enumerate(atom_labels)})
        except:
            return None
        return result
    

    def molecular_edit_distance(self, mol1, mol2):
        # Step 1: 使用rdFMCS模块查找最大公共子结构
     
        mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=10, completeRingsOnly=False)
        
        # 最大公共子结构的SMARTS字符串
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        
        if mcs_mol is None:
            return max(mol1.GetNumAtoms(), mol2.GetNumAtoms())  # 如果没有公共子图，返回分子大小的最大值
        
        # MCS子结构的原子数量
        mcs_num_atoms = mcs_mol.GetNumAtoms()
        
        # Step 2: 分别获取两个分子的原子数量
        mol1_num_atoms = mol1.GetNumAtoms()
        mol2_num_atoms = mol2.GetNumAtoms()
        
        # Step 3: 编辑距离计算，基于最大公共子结构
        # 编辑距离 = (分子1的原子数 - MCS的原子数) + (分子2的原子数 - MCS的原子数)
        edit_distance = (mol1_num_atoms - mcs_num_atoms) + (mol2_num_atoms - mcs_num_atoms)
        
        return edit_distance
    
    
def sanitize_filename(filename):
    """移除或替换无效字符，以便于用作文件名."""
    # 定义无效字符的正则表达式
    invalid_chars = r'[<>:"/\\|?*\[\]]'
    # 使用下划线替换无效字符
    return re.sub(invalid_chars, '_', filename)

def save_smiles_as_images(smiles_list, output_dir):
    """将 SMILES 列表保存为对应名称的图像文件."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果输出目录不存在，则创建

    for smiles in smiles_list:
        # 生成分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # 确保分子有效
            # 生成文件名，使用 SMILES 的名称
            filename = sanitize_filename(smiles) + '.png'
            filepath = os.path.join(output_dir, filename)
            
            # 生成图像并保存
            img = Draw.MolToImage(mol)
            img.save(filepath)
            print(f"保存图像: {filepath}")
        else:
            print(f"无效的 SMILES: {smiles}")



def deduplicate_nodes(root, best_parent_nodes):
    """
    去重节点列表，保留相同label的最深节点。

    参数:
    best_parent_nodes: List[TreeNode] - 输入的节点列表

    返回:
    dict - 去重后的节点字典，key为节点label，value为最深的节点
    """
    unique_parent_nodes = {}
    
    for best_parent_node in best_parent_nodes:
        # 使用节点的label作为唯一标识
        node_label = best_parent_node.label
        
        # 确保分子有效
        parent_mol = Chem.MolFromSmiles(node_label)
        
        if parent_mol:
            if node_label not in unique_parent_nodes:
                unique_parent_nodes[node_label] = best_parent_node  # 记录第一个节点
                
                
            else:
                # 如果已经存在，则比较深度
                existing_node = unique_parent_nodes[node_label]
                if root.find_depth(best_parent_node) > root.find_depth(existing_node):
                    unique_parent_nodes[node_label] = best_parent_node  # 替换为深度更深的节点
    
    # if len(unique_parent_nodes.values()) > 10:
    #     import pdb; pdb.set_trace()
    return unique_parent_nodes




def read_smiles_from_file(file_path):
    """
    Reads SMILES strings from a file, one per line.
    :param file_path: Path to the file containing SMILES strings.
    :return: List of SMILES strings.
    """
    with open(file_path, 'r') as f:
        smiles_list = [line.strip().split(',')[1] for line in f if line.strip()]
    return smiles_list[1:]




def visualize_graph(graph, atom_groups, save_path):
    G = nx.DiGraph()

    # Add nodes and edges
    for node in graph.nodes:
        G.add_node(node)
    
    for from_node, to_nodes in graph.get_edges().items():
        for to_node in to_nodes:
            G.add_edge(from_node, to_node)

    # Set positions for nodes based on atom groups
    pos = {}
    y_offset = 0  # Initialize vertical offset for levels

    for atom_count, smiles_list in atom_groups.items():
        for i, smiles in enumerate(smiles_list):
            pos[smiles] = (atom_count, y_offset + i)  # X is atom_count, Y is the index in the group
        y_offset += len(smiles_list) / 500  # Increase spacing between groups

    # Extract edge positions
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Extract node positions and colors
    node_x = []
    node_y = []
    node_colors = []  # Store node colors
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        atom_count = get_smiles_atoms_num(node)
        color_value = atom_count * 50  # Color value based on atom count
        node_colors.append(color_value)

    # Create figure with larger size
    fig = go.Figure()

    # Add edges with increased width
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='grey'),  # Increase edge width
        hoverinfo='none',
        mode='lines'))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,  # Slightly increase marker size
            color=node_colors,  # Use different colors
            line_width=0.5)))  # Increase marker border width

    # Set figure title and layout with increased size
    fig.update_layout(
        title='Molecule Graph Visualization',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        width=10000,  # Increase figure width
        height=5000,  # Increase figure height
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    # Save figure
    fig.write_image(save_path, format='png')

    
    
def get_smiles_atoms_num(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()

    


def save_graph(graph: Graph, atom_groups: defaultdict, save_path: str) -> None:
    graph_data = {
        # 'nodes': list(graph.nodes),  # List of nodes
        'edges': graph.edges(),  # List of edges
        # 'atom_groups': {count: smiles for count, smiles in atom_groups.items()}  # Atom groups as dict
    }

    # Write the graph data to a file
    with open(save_path, 'w') as f:
        json.dump(graph_data, f, indent=4)  # Save as JSON with indentation for readability

def path_padding(path):
    # 最大长度设定
    max_length = 9

    # 填充路径
    padded_paths = []

    for sample in path[0]:
        # 提取子路径并限制最大长度
        sub_paths = sample[:max_length]
        sub_paths += ['PAD'] * (max_length - len(sub_paths))  # 填充至最大长度
        padded_paths.append(sub_paths)

    # 转换为Tensor格式
    padded_paths_labels = pad_sequence([torch.tensor([float(x) for x in sample])[:max_length]
                                for sample in path[1]], batch_first=True)

    return padded_paths, padded_paths_labels


def atom_features(atom):
    """Enhanced atom feature extraction."""
    # One-hot encoding for atom type (symbol)
    atom_symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])

    # One-hot encoding for atom degree (number of bonds)
    atom_degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for number of hydrogen atoms
    atom_num_hydrogens = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for implicit valence
    atom_implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # One-hot encoding for chirality
    atom_chirality = one_of_k_encoding(atom.GetChiralTag(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    
    # Aromaticity (True or False)
    atom_aromatic = [atom.GetIsAromatic()]
    
    # Atomic number of the atom
    atom_atomic_num = [atom.GetAtomicNum()]

    # Return combined feature vector
    return np.array(atom_symbol + atom_degree + atom_num_hydrogens + atom_implicit_valence + atom_chirality + atom_aromatic + atom_atomic_num)


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding for a value in an allowable set."""
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}.")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encoding with unknown values mapped to the last element of the allowable set."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom feature extraction: Include multiple features per atom
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feats = atom_features(atom)  # Use the enhanced atom feature extraction
        atom_features_list.append(atom_feats)

    atom_features_list = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    # Edge list creation for bonds
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # For undirected graph

    # If no edges are found (isolated atoms), connect each atom to itself
    if not edge_index:
        for i in range(len(atom_features_list)):
            edge_index.append((i, i))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Return a Data object with atom features and edge information
    return Data(x=atom_features_list, edge_index=edge_index, smiles=smiles)
    

def smiles_to_graph_xyz( index, suppl ):

    mol = suppl[index]

    # try:
    N = mol.GetNumAtoms()
    # except:
    #     import pdb;pdb.set_trace()
        
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)

    name = mol.GetProp('_Name')
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr,
        # y=y[i].unsqueeze(0),
        name=name,
        idx=index,
    )
    return data

def smiles_to_graph_xyz_sch(index, suppl):
    mol = suppl[index]

    try:
        N = mol.GetNumAtoms()  # 获取原子数量
    except:
        import pdb; pdb.set_trace()  # 调试点

    # 获取分子的三维坐标
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    # 获取原子的属性
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())  # 原子序数

    # 原子序数张量
    z = torch.tensor(atomic_number, dtype=torch.long)

    # 构建 Data 对象
    data = Data(
        z=z,  # 原子序数
        pos=pos,  # 原子坐标
        idx=index,  # 分子索引
    )
    return data


def smiles_to_graph_xyz_equiformer_qm7(smiles):
    # import pdb;pdb.set_trace()
    
    
    # if mol is None:
    #     print(f"Warning: RDKit could not parse SMILES: {smiles}")
    #     return None

    # # --- 核心步骤：生成3D坐标 ---
    # try:
    #     # 2. 添加氢原子，这对于生成正确的3D几何至关重要
    #     mol = Chem.AddHs(mol)
        
    #     # 3. 生成3D构象 (-1表示使用随机坐标作为种子)
    #     # embed_result = -1 表示生成失败
    #     embed_result = AllChem.EmbedMolecule(mol, randomSeed=-1)
    #     if embed_result == -1:
    #         raise ValueError("Conformer generation failed.")
            
    #     # 4. 使用UFF力场优化几何结构
    #     AllChem.UFFOptimizeMolecule(mol)
    # except Exception as e:
    #     print(f"Warning: 3D conformer generation failed for SMILES: {smiles}. Error: {e}")
    #     # 如果无法生成3D结构，可以选择返回None或一个没有pos属性的数据对象
    #     return None


    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        # 或者使用 UFF 优化
        # AllChem.UFFOptimizeMolecule(mol)

    try:
        N = mol.GetNumAtoms()  # 获取原子数量
    except:
        import pdb; pdb.set_trace()  # 调试点
        
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    # import pdb;pdb.set_trace()
    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))


    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)

    # name = mol.GetProp('_Name')
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr,
        # name=name,
    )
    return data

def smiles_to_graph_xyz_equiformer(index, suppl):
    mol = suppl[index]

    try:
        N = mol.GetNumAtoms()  # 获取原子数量
    except:
        import pdb; pdb.set_trace()  # 调试点

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)


    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)

    name = mol.GetProp('_Name')
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr,
        name=name,
    )
    return data


def plot_target_vs_output(all_targets, all_outputs, save_dir='outputs_visual'):
    """
    Plots the scatter visualizations of target vs output for two sets of targets and outputs, 
    and saves them to a file. The targets and outputs are sorted by the targets.
    
    Parameters:
    - all_targets (torch.Tensor or np.ndarray): A 2D array/tensor with shape (N, 2), where N is the number of samples.
    - all_outputs (torch.Tensor or np.ndarray): A 2D array/tensor with shape (N, 2), where N is the number of samples.
    - save_dir (str): Directory to save the plot image. Default is 'outputs_visual'.
    """
    # Ensure the save directory exists
    # os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if inputs are PyTorch tensors
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.numpy()
    if isinstance(all_outputs, torch.Tensor):
        all_outputs = all_outputs.numpy()
    
    # Sort targets and outputs based on target1 and target2
    sorted_indices1 = np.argsort(all_targets[:, 0])  # Sort based on target1
    sorted_indices2 = np.argsort(all_targets[:, 1])  # Sort based on target2

    # Apply sorting to both targets and outputs
    sorted_target1 = all_targets[sorted_indices1, 0]
    sorted_output1 = all_outputs[sorted_indices1, 0]
    sorted_target2 = all_targets[sorted_indices2, 1]
    sorted_output2 = all_outputs[sorted_indices2, 1]

    # Create a figure and axis for plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot 1: Sorted Target1 vs Sorted Output1
    axes[0].scatter(sorted_target1, sorted_target1, color='blue', label='True Target1', alpha=0.6)
    axes[0].scatter(sorted_target1, sorted_output1, color='red', label='Predicted Output1', alpha=0.6)
    axes[0].set_title('Target1 vs Output1')
    axes[0].set_xlabel('Target1')
    axes[0].set_ylabel('Output1')
    axes[0].legend()

    # Plot 2: Sorted Target2 vs Sorted Output2
    axes[1].scatter(sorted_target2, sorted_target2, color='blue', label='True Target2', alpha=0.6)
    axes[1].scatter(sorted_target2, sorted_output2, color='red', label='Predicted Output2', alpha=0.6)
    axes[1].set_title('Target2 vs Output2')
    axes[1].set_xlabel('Target2')
    axes[1].set_ylabel('Output2')
    axes[1].legend()

    # Adjust layout to avoid overlapping text
    plt.tight_layout()

    # Save the plot as an image in the specified directory
    save_path = os.path.join('outputs_visual', f'{save_dir}.png')
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")


def calculate_metrics(target, output):
    # 清理 NaN 和 inf 值
    mask = ~np.isnan(output) & ~np.isinf(output) & ~np.isnan(target) & ~np.isinf(target)
    filtered_target = target[mask]
    filtered_output = output[mask]

    # 计算各项指标
    if len(filtered_target) > 1:  # 确保有足够的数据点
        pearson_corr = pearsonr(filtered_target, filtered_output)[0]
        r2 = r2_score(filtered_target, filtered_output)
        mse = mean_squared_error(filtered_target, filtered_output)
        mae = mean_absolute_error(filtered_target, filtered_output)  # 计算 MAE
        rank_loss = np.mean(np.sign(filtered_output - filtered_target) != np.sign(np.roll(filtered_output, 1) - np.roll(filtered_target, 1)))
    else:
        pearson_corr, r2, mse, mae, rank_loss = np.nan, np.nan, np.nan, np.nan, np.nan  # 若数据不足则返回 NaN

    return {
        "R2": r2,
        "Pearson Correlation": pearson_corr,
        "MSE": mse,
        "MAE": mae,  # 返回 MAE
        "Rank Loss": rank_loss
    }


