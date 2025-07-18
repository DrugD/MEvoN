# from rdkit import Chem
# import pandas as pd

# # 加载数据：假设 CSV 文件包含 'smiles' 列
# df = pd.read_csv('/home/data1/lk/project/mol_tree/graph/ani1/ani1.csv')  # 或 QM9 的 SMILES 数据
# smiles_list = df['smiles'].tolist()

# num_nodes_list = []
# degree_list = []

# for smiles in smiles_list:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         continue

#     # 添加 H 原子
#     mol = Chem.AddHs(mol)

#     num_atoms = mol.GetNumAtoms()
#     num_bonds = mol.GetNumBonds()

#     num_nodes_list.append(num_atoms)
    
#     # 平均度数 = 所有节点的度的总和 / 节点数 = 2 × 边数 / 节点数
#     avg_degree = 2 * num_bonds / num_atoms if num_atoms > 0 else 0
#     degree_list.append(avg_degree)

# # 计算统计量
# avg_num_nodes = sum(num_nodes_list) / len(num_nodes_list)
# avg_degree = sum(degree_list) / len(degree_list)

# print(f"_AVG_NUM_NODES = {avg_num_nodes}")
# print(f"_AVG_DEGREE = {avg_degree}")

# ============================================================
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import rdmolops

# # 加载 CSV 文件
# df = pd.read_csv('/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv')

# # 提取 SMILES1 列（注意剔除缺失）
# smiles_list = df['SMILES1'].dropna().tolist()

# # 初始化统计项
# num_nodes_list = []
# degree_list = []

# for smiles in smiles_list:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         continue

#     # 显式添加氢原子
#     mol = Chem.AddHs(mol)

#     # 节点数和边数
#     num_atoms = mol.GetNumAtoms()
#     num_bonds = mol.GetNumBonds()

#     if num_atoms > 0:
#         num_nodes_list.append(num_atoms)
#         avg_deg = 2 * num_bonds / num_atoms
#         degree_list.append(avg_deg)

# # 计算平均值
# avg_num_nodes = sum(num_nodes_list) / len(num_nodes_list)
# avg_degree = sum(degree_list) / len(degree_list)

# print(f"_AVG_NUM_NODES = {avg_num_nodes}")
# print(f"_AVG_DEGREE = {avg_degree}")



# ============================================================

# import pandas as pd
# from rdkit import Chem

# # 读取数据（假设 SMILES 列名为 'smiles'）
# df = pd.read_csv('/home/data1/lk/project/mol_tree/graph/ani1/ani1.csv')  # 修改为实际路径

# # 提取 SMILES 列（或替换为实际列名）
# smiles_list = df['smiles'].dropna().tolist()

# # 初始化元素集合
# atom_set = set()

# for smiles in smiles_list:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         continue

#     mol = Chem.AddHs(mol)  # 添加显式 H 原子
#     for atom in mol.GetAtoms():
#         atom_set.add(atom.GetSymbol())

# # 输出原子种类
# print("Unique atom types in ANI-1 dataset:", sorted(atom_set))



# =============================================================


from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from tqdm import tqdm
import numpy as np

# 距离阈值
CUTOFF_RADIUS = 5.0

def get_radius_graph_degree(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if success != 0:
        return None, None

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # (N, 3)
    num_atoms = len(positions)

    # 计算距离矩阵
    dist_matrix = squareform(pdist(positions))

    # 构建半径邻接矩阵
    adj_matrix = (dist_matrix < CUTOFF_RADIUS).astype(int)
    np.fill_diagonal(adj_matrix, 0)  # 排除自身连接

    # 度数：每个节点的邻居数量
    degrees = adj_matrix.sum(axis=1)
    avg_degree = degrees.mean()

    return num_atoms, avg_degree

# 读取 QM9 或其它数据（假设 CSV 第一列为 SMILES）
# df = pd.read_csv("/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv")  # 修改为你的文件路径
# all_smiles = df["SMILES1"]  # 替换为实际列名

df = pd.read_csv('/home/data1/lk/project/mol_tree/graph/ani1/ani1.csv')  # 或 QM9 的 SMILES 数据
all_smiles = df['smiles'].tolist()


num_nodes_list = []
degree_list = []

for smi in tqdm(all_smiles):
    nodes, deg = get_radius_graph_degree(smi)
    if nodes is not None and deg is not None:
        num_nodes_list.append(nodes)
        degree_list.append(deg)

# 平均统计
avg_nodes = np.mean(num_nodes_list)
avg_degree = np.mean(degree_list)

print(f"_AVG_NUM_NODES = {avg_nodes}")
print(f"_AVG_DEGREE = {avg_degree}")
