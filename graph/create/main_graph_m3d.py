from tree_ import Tree
from tree import Tree 
# from graph_v1 import Graph, construct_graph
from graph_v2_1_m3d import Graph, construct_graph
from tools import *
import json

from rdkit import Chem

def read_smiles_from_file_(file_path):
    """
    Reads SMILES strings from a file, one per line.
    :param file_path: Path to the file containing SMILES strings.
    :return: List of SMILES strings.
    """
    with open(file_path, 'r') as f:
        smiles_list = [line.strip().split(',')[-1] for line in f if line.strip()]
    return smiles_list[1:]

# 读取文件路径
file_path = "/home/data1/lk/project/mol_tree/graph/m3d/m3d.csv"


SmilesList = read_smiles_from_file_(file_path)
# import pdb;pdb.set_trace()

tree_construct_kenerl_method = ['edit_distance','graph']
threshold = 0.3

data_ = SmilesList


# 构建图
molecule_graph, atom_groups = construct_graph(data_, tree_construct_kenerl_method, threshold)
# 构建图


# 可视化图
# visualize_graph(molecule_graph, atom_groups, f'img/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.png')

# Save the graph to a file
save_path = f'/home/data1/lk/project/mol_tree/graph/m3d/evolution_graph_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v2.json'  # Specify the path where you want to save the graph
save_graph(molecule_graph, atom_groups, save_path)

# 输出图的边关系
# print(molecule_graph)
