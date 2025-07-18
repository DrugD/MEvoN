

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
# with open("/home/data1/lk/project/mol_tree/graph/qm8/evolution_graph_21786_['edit_distance', 'graph']_0.3_v2.json", 'r') as f:
#     all_paths = json.load(f)

with open("/home/data1/lk/project/mol_tree/graph/qm7/evolution_graph_6834_['edit_distance', 'graph']_0.3_v2.json", 'r') as f:
    all_paths = json.load(f)
    
label_name = 'gap'

# 读取标签数据文件
labels_file = '/home/data1/lk/project/mol_tree/graph/qm7/qm7.csv'
labels_df = pd.read_csv(labels_file)

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


from collections import defaultdict
from rdkit import Chem


import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import random
results = [[],[]]
# 遍历所有路径
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from collections import Counter
import numpy as np
from rdkit import Chem
from tqdm import tqdm

mols_dict = {}
evolution_count = []

# 统计所有SMILES的种类及出现次数
smiles_counter = Counter()

# 假设all_paths已经是一个路径列表，每个路径包含多个SMILES字符串
for idx, path in enumerate(tqdm(all_paths)):
    molecules = [Chem.MolFromSmiles(smiles) for smiles in path]
    
    # 更新smiles_counter，统计每个SMILES出现的次数
    smiles_counter.update(path)

    # 计算该path中的分子间的进化关系
    # 这里的进化关系可以基于某种标准，比如分子间的某种相似度或化学变化
    # 我们假设每个路径的分子间都有某种进化关系
    # 可以根据具体需求修改此计算方式
    num_molecules = len(molecules)
    if num_molecules > 1:
        # 假设每个路径内的分子都有某种进化关系
        # 进化关系数 = (num_molecules * (num_molecules - 1)) / 2
        evolution_count.append((num_molecules * (num_molecules - 1)) // 2)

# 统计完毕后，输出结果
# smiles_counts = dict(smiles_counter)
# print("SMILES统计结果：")
# for smiles, count in smiles_counter.items():
#     print(f"SMILES: {smiles}, Count: {count}")

# 计算平均进化关系数
if evolution_count:
    avg_evolution_relations = np.mean(evolution_count)
    print(f"每个分子与其他分子之间存在进化关系的平均次数: {avg_evolution_relations:.2f}")

import pdb;pdb.set_trace()

import pandas as pd
from collections import Counter
from rdkit import Chem
from tqdm import tqdm

# 读取 CSV 文件中的 SMILES1 列
qm9_path = "/home/data1/lk/project/mol_tree/graph/qm7/qm7.csv"
df = pd.read_csv(qm9_path)

# 提取 'SMILES1' 列
smiles_from_csv = df['smiles'].tolist()

# 假设all_paths已经是一个路径列表，每个路径包含多个SMILES字符串
smiles_counter = Counter()

# 统计all_paths中所有SMILES的种类及其出现次数
for idx, path in enumerate(tqdm(all_paths)):
    smiles_counter.update(path)

# 找出不存在于smiles_counter中的分子
missing_smiles = [smiles for smiles in smiles_from_csv if smiles not in all_paths['nodes']]




# 统计 SMILES 的出现次数
smiles_counter = Counter(smiles_from_csv)

# 找出重复的 SMILES
duplicates = {smiles: count for smiles, count in smiles_counter.items() if count > 1}

import pdb;pdb.set_trace()

