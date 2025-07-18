# # from rdkit import Chem
# # from rdkit.Chem import Draw

# # # Dictionary with names as keys and SMILES strings as values
# # smiles_data = {
# #     "MCE-BioActive00013952": "Cc1c(O)cc2oc3c(c(=O)c2c1O)-c1ccc(O)cc1OC3O",
# #     "HY-QS00591541": "Cc1cc(C)c2nc(O)cc(C)c2c1",
# #     "HY-QS05160849": "COc1cc2c(c(O)c1OC)C(=O)C(OC)CCC2",
# #     "HY-QS06938250": "Cc1cc2oc(=O)cc(CN3CCN(Cc4ccccc4)CC3)c2cc1O",
# #     "HY-QS08610900": "COc1cc(C)c2c(c1C=O)Oc1c(c(C)c(O)c3c1C(O)OC3=O)OC2=O",
# #     "HY-QA06635504": "COc1nc(-c2cnn(C3CCC3)c2)ccc1[N+](=O)[O-]",
# #     "HY-QS07201666": "Cc1cc(O)c2c(C)ccc(C)c2n1",
# #     "HY-QS04156156": "NS(=O)(=O)c1ccc(-n2cc(C3CCC3)nn2)cc1",
# #     "HY-QS02875892": "O=C(O)c1ccc(OC2CCC2)cc1",
# #     "HY-QS01609323": "OC1(c2ccccc2)C2C3CC4C5C3C1C(O)(C5Br)C42",
# #     "HY-QS04241866": "O=C(O)c1ccc(C2CCC2)cc1",
# #     "HY-QS07177500": "COc1cc(C(c2c(O)cc(C)oc2=O)c2c(O)cc(C)oc2=O)ccc1O"
# # }

# # # Loop through the dictionary and save each molecule image
# # for name, smiles in smiles_data.items():
# #     mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to molecule
# #     if mol:
# #         # Generate the image for the molecule
# #         img = Draw.MolToImage(mol, size=(300, 300))
# #         # Save the image using the name as the filename
# #         file_name = f"{name}.png"
# #         img.save(file_name)
# #         print(f"Saved: {file_name}")
# #     else:
# #         print(f"Invalid SMILES for {name}: {smiles}")



# from rdkit import Chem
# from rdkit.Chem import rdFMCS
# import random
# from tqdm import tqdm
# import pandas as pd


# # # 示例：读取1000个分子的SMILES
# # smiles_list 读取 /home/data1/lk/project/mol_tree/graph/pcqm4mv2/pcqm4mv2.csv

# # 读取 CSV 文件
# csv_file_path = "/home/data1/lk/project/mol_tree/graph/pcqm4mv2/pcqm4mv2.csv"
# df = pd.read_csv(csv_file_path)

# # 查看 CSV 文件的列名，确保知道包含 SMILES 字符串的列名
# # print(df.columns)

# # 假设 SMILES 字符串在名为 'smiles' 的列中，提取该列
# smiles_list = df['smiles'].tolist()[:10000]


# # 解析 SMILES 为分子对象
# molecules = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list)]




# def find_mcs_for_multiple(molecules, max_results=1):
#     if len(molecules) < 2:
#         return []

#     # 初始设置：第一个分子的 MCS 作为开始
#     mcs_result = rdFMCS.FindMCS([molecules[0], molecules[1]], completeSearch=True)
    
#     if mcs_result.numAtoms == 0:
#         return []

#     mcs_results = [mcs_result]  # 存储结果

#     # 从第三个分子开始逐步查找 MCS
#     for i in range(2, len(molecules)):
#         # 找当前分子和已有的 MCS 的最大公共子结构
#         mcs_result = rdFMCS.FindMCS([Chem.MolFromSmarts(mcs_results[-1].smartsString), molecules[i]], completeSearch=True)
        
#         # 如果找到新的公共子结构，将其加入结果
#         if mcs_result.numAtoms > 0:
#             mcs_results.append(mcs_result)
        
#         # 如果达到需要的最大结果数量，停止计算
#         if len(mcs_results) >= max_results:
#             break

#     return mcs_results

# # 计算多个分子的最大公共子结构
# mcs_substructures = find_mcs_for_multiple(molecules, max_results=1)

# # 输出找到的最大公共子结构
# for i, mcs in enumerate(mcs_substructures):
#     print(f"MCS {i + 1}:")
#     print("SMARTS: ", mcs.smartsString)
#     print("Atom Count:", mcs.numAtoms)
#     print("Bond Count:", mcs.numBonds)
#     print()


# import pdb;pdb.set_trace()

# ++======================================================================

# from rdkit import Chem
# from rdkit.Chem import rdFMCS
# from rdkit import Chem
# from rdkit.Chem import rdFMCS
# import random
# from tqdm import tqdm
# import pandas as pd



# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdFMCS
# from rdkit.Chem.Fingerprints import FingerprintMols
# from rdkit.Chem import DataStructs
# from scipy.sparse import lil_matrix
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from collections import defaultdict

# # ======================
# # Helper Functions
# # ======================

# def compute_mcs_similarity(mol1, mol2):
#     """
#     计算两分子的最大公共子结构（MCS）相似性，基于 MCS 的原子比例。
#     """
#     mcs_result = rdFMCS.FindMCS(
#         [mol1, mol2],
#         completeRingsOnly=True,
#         matchValences=True,
#         ringMatchesRingOnly=True,
#         timeout=5
#     )
#     if mcs_result.canceled:
#         return 0.0  # 超时视为 0 相似性
#     mcs_smarts = mcs_result.smartsString
#     mcs_mol = Chem.MolFromSmarts(mcs_smarts)
#     if not mcs_mol:
#         return 0.0
#     return mcs_mol.GetNumAtoms() / min(mol1.GetNumAtoms(), mol2.GetNumAtoms())

# # ======================
# # Optimization Techniques
# # ======================

# def group_molecules(mols):
#     """
#     按照分子原子数对分子进行分组。
#     """
#     groups = defaultdict(list)
#     for mol in tqdm(mols):
#         num_atoms = mol.GetNumAtoms()
#         # group_id = num_atoms % num_groups  # 分组策略（可根据需要调整）
#         groups[num_atoms].append(mol)
#     return groups

# def prefilter_candidates(mols, threshold=0.4):
#     """
#     使用分子指纹的 Tanimoto 相似性进行预筛选。
#     """
#     fingerprints = [FingerprintMols.FingerprintMol(mol) for mol in mols]
#     candidates = []
#     for i in tqdm(range(len(mols))):
#         for j in range(i + 1, len(mols)):
#             similarity = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
#             if similarity >= threshold:
#                 candidates.append((i, j))
#     return candidates



# def parallel_similarity_matrix(mols, candidates):
#     """
#     并行计算分子对的相似性矩阵，并显示进度条。
#     """
#     n = len(mols)
#     sim_matrix = lil_matrix((n, n))  # 使用稀疏矩阵存储

#     def compute_similarity(pair):
#         i, j = pair
#         similarity = compute_mcs_similarity(mols[i], mols[j])
#         return i, j, similarity

#     with ThreadPoolExecutor() as executor:
#         # 使用tqdm包装candidates，显示进度条
#         results = list(tqdm(executor.map(compute_similarity, candidates), total=len(candidates), desc="Calculating similarities"))
#         for i, j, similarity in results:
#             sim_matrix[i, j] = similarity
#             sim_matrix[j, i] = similarity

#     return sim_matrix

# # ======================
# # Main Workflow
# # ======================

# def calculate_similarity(mols, prefilter_threshold=0.4):
#     """
#     主函数：计算分子集合的相似性矩阵。
#     """
#     # 1. 分组分子
#     print("Grouping molecules...")
#     groups = group_molecules(mols)
#     import pdb;pdb.set_trace()
#     # 2. 遍历组，逐组计算
#     overall_sim_matrix = lil_matrix((len(mols), len(mols)))
    
#     for group_id, group_mols in tqdm(groups.items()):
#         print(f"Processing group {group_id} with {len(group_mols)} molecules...")
#         if len(group_mols) <= 100:
#             continue
    
#         # 3. 预筛选候选对
#         print("Prefiltering candidate pairs...")
#         candidates = prefilter_candidates(group_mols, threshold=prefilter_threshold)

#         # 4. 并行计算相似性矩阵
#         print("Calculating similarity matrix...")
#         group_sim_matrix = parallel_similarity_matrix(group_mols, candidates)

#         # 5. 合并结果
#         print("Merge matrix...")
#         print("Group_mols number:", len(group_mols))
#         for i, mol1 in tqdm(enumerate(group_mols)):
#             for j, mol2 in enumerate(group_mols):
#                 overall_sim_matrix[mols.index(mol1), mols.index(mol2)] = group_sim_matrix[i, j]

#     return overall_sim_matrix

# # ======================
# # Example Usage
# # ======================

# if __name__ == "__main__":
#     # 示例分子集合
#     # 读取 CSV 文件
#     csv_file_path = "/home/data1/lk/project/mol_tree/graph/pcqm4mv2/pcqm4mv2.csv"
#     df = pd.read_csv(csv_file_path)


#     # 假设 SMILES 字符串在名为 'smiles' 的列中，提取该列
#     smiles_list = df['smiles'].tolist()
#     mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list) if Chem.MolFromSmiles(smiles)]

#     # 计算相似性矩阵
#     sim_matrix = calculate_similarity(mols, prefilter_threshold=0.5)

#     # 转为密集矩阵并打印
#     dense_matrix = sim_matrix.toarray()
#     print("Similarity Matrix:")
#     import pdb;pdb.set_trace()
#     print(dense_matrix)
    

# ++======================================================================





import plotly.graph_objects as go
import pandas as pd

# 假设我们有如下的分子关系数据（可以自定义为分子进化路径）
data = {
    'Source': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Target': ['B', 'C', 'A', 'C', 'A', 'B'],
    'Value': [10, 20, 30, 40, 50, 60]  # 关系强度或转化频率
}

# 创建一个DataFrame
df = pd.DataFrame(data)

# 创建节点名称列表
nodes = ['A', 'B', 'C']

# 创建节点索引字典
node_indices = {node: idx for idx, node in enumerate(nodes)}

# 创建Sankey图所需的源和目标索引列表
sources = [node_indices[row['Source']] for _, row in df.iterrows()]
targets = [node_indices[row['Target']] for _, row in df.iterrows()]
values = df['Value'].tolist()

# 创建Sankey图
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,  # 节点之间的间隔
        thickness=20,  # 节点的厚度
        line=dict(color="black", width=0.5),  # 节点边框的颜色和宽度
        label=nodes  # 节点标签
    ),
    link=dict(
        source=sources,  # 源节点
        target=targets,  # 目标节点
        value=values,  # 流动的值
        color=[f'rgba({116},{152},{200},{v/100})' for v in values]  # 流动颜色（渐变）
    )
))

# 更新布局设置
fig.update_layout(
    title="Molecular Evolution Pathway",
    font=dict(size=16, color="black"),
    width=800,  # 设置宽度
    height=600  # 设置高度
)

# 显示图形
fig.show()

# 保存图形为PNG格式
fig.write_image('/home/data1/lk/project/mol_tree/graph/sankey_diagram.png', format='png', width=800, height=600)
