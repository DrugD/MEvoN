import json
from collections import defaultdict
from rdkit import Chem
import os

# 用于计算分子含有的原子数量
def get_atoms_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mol.GetNumAtoms()
    return 0

# 初始化字典，用于存储节点、边、原子组信息
merged_data = {
    "nodes": set(),
    "edges": set(),
    "atom_groups": {str(i): [] for i in range(1, 26)}
}

# 循环读取文件
for i in range(2, 26):
    path = f"/home/data1/lk/project/mol_tree/graph/pcqm4mv2/temp_atom_count_{i}_evolution_graph_3746620_['edit_distance', 'graph']_0.3_v2.json"
    
    # 如果文件存在，读取文件内容
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)

        # 处理edges部分
        if 'edges' in data:
            for edge in data['edges']:
                merged_data["edges"].add(tuple(edge))  # 使用set去重

# 转换set为list，保证格式正确
merged_data["nodes"] = list(merged_data["edges"])
merged_data["atom_groups"] = list(merged_data["edges"])

import pdb;pdb.set_trace()

# 对atom_groups中的每个列表去重
for key, atom_group in merged_data["atom_groups"].items():
    merged_data["atom_groups"][key] = list(set(atom_group))

# 保存合并后的数据到新的JSON文件
output_path = "/home/data1/lk/project/mol_tree/graph/pcqm4mv2/merged_data_2to25.json"
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"合并完成，文件已保存至 {output_path}")
