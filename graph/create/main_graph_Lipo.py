from tree_ import Tree
from tree import Tree 
# from graph_v1 import Graph, construct_graph
from graph_v2 import Graph, construct_graph
from tools import *
import json
import pandas as pd

def read_smiles_from_file_(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保文件中包含 'smiles' 列
    if 'smiles' not in df.columns:
        raise ValueError("CSV 文件中未找到 'smiles' 列")

    # 返回 SMILES 列表
    return df['smiles'].tolist()

# 示例：从文件中读取数据
file_path = '/home/data1/lk/project/mol_property/data/lipophilicity/Lipophilicity.csv'
SmilesList = read_smiles_from_file_(file_path)

tree_construct_kenerl_method = ['edit_distance','graph']
threshold = 0.3

data_ = SmilesList


# 构建图
molecule_graph, atom_groups = construct_graph(data_, tree_construct_kenerl_method, threshold)
# 构建图


# 可视化图
# visualize_graph(molecule_graph, atom_groups, f'img/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.png')

# Save the graph to a file
save_path = f'/home/data1/lk/project/mol_tree/graph/lipo/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.json'  # Specify the path where you want to save the graph
save_graph(molecule_graph, atom_groups, save_path)

# 可视化图
visualize_graph(molecule_graph, atom_groups, f'/home/data1/lk/project/mol_tree/graph/lipo/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.png')
