import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, DataStructs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import MDS
from tqdm import tqdm
import numpy as np
import random
import torch,math

def calculate_mcs_similarity(mol1, mol2):
    """
    计算两分子之间的最大公共子结构（MCS）相似性。
    """
    mcs = rdFMCS.FindMCS([mol1, mol2], threshold=1.0, matchValences=True, ringMatchesRingOnly=True)
    if mcs.smartsString:
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        mcs_size = mcs_mol.GetNumAtoms() if mcs_mol else 0
        return mcs_size / max(mol1.GetNumAtoms(), mol2.GetNumAtoms())
    return 0


def smiles_to_mols(smiles_list):
    """
    将 SMILES 字符串列表转换为 RDKit 分子对象。
    """
    mols = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
    return mols

import cupy as cp
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix

def calculate_mcs_similarity_gpu(mol1, mol2, radius=2, nbits=2048):
    """
    使用 RDKit 计算两个分子的 MCS 相似度，并将计算移到 GPU 上。
    mol1 和 mol2 必须是 RDKit 的分子对象。
    """
    # 将 RDKit 分子对象转化为 Morgan Fingerprints（位向量）
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
    
    # 将位向量转化为 NumPy 数组
    fp1_array = np.array(fp1, dtype=np.float32)
    fp2_array = np.array(fp2, dtype=np.float32)
    
    # 将 NumPy 数组转移到 GPU 上
    fp1_gpu = cp.asarray(fp1_array)
    fp2_gpu = cp.asarray(fp2_array)
    
    # 计算相似度（通过计算余弦相似度）
    similarity = cp.dot(fp1_gpu, fp2_gpu) / (cp.linalg.norm(fp1_gpu) * cp.linalg.norm(fp2_gpu))
    
    # 从 GPU 转回 CPU 并返回相似度
    return similarity.get()

def construct_similarity_matrix_gpu(mols):
    """
    使用 GPU 加速构造分子之间的相似度矩阵，并仅计算上三角。
    """
    n = len(mols)
    
    # 使用 CuPy 创建 GPU 上的空稀疏矩阵
    sim_matrix = lil_matrix((n, n))  # 使用稀疏矩阵存储
    

    # 只计算上三角部分
    for i in tqdm(range(n), desc="Calculating similarity matrix on GPU"):
        for j in range(i, n):
            if i==j:
                sim_matrix[i, j] = 1.0
            else:
                # 计算相似度
                similarity = calculate_mcs_similarity_gpu(mols[i], mols[j])
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity  # 对称填充


    return sim_matrix


def select_representative_molecules(mols, n_samples=500):
    """
    随机选择一部分分子作为代表进行初步聚类。
    """
    sampled_indices = random.sample(range(len(mols)), n_samples)
    sampled_mols = [mols[i] for i in sampled_indices]
    return sampled_mols, sampled_indices

def find_cluster_centers(mols, labels, embeddings, sampled_indices):
    """
    找到每个簇的中心分子。
    mols: 所有分子对象列表
    labels: 聚类标签
    embeddings: 降维后的嵌入
    sampled_indices: 初步采样的分子在原始分子集合中的索引
    """
    cluster_centers = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # 找到属于当前簇的样本索引
        cluster_indices = np.where(labels == label)[0]
        cluster_embeddings = embeddings[cluster_indices]

        # 找到簇中最接近中心的分子
        _, closest_idx = pairwise_distances_argmin_min(cluster_embeddings, cluster_embeddings.mean(axis=0).reshape(1, -1) )

        # 对距离和索引进行排序
        sorted_indices = np.argsort(closest_idx)  # 根据距离排序
        # sorted_closest_idx = closest_idx[sorted_indices]

        # 选择距离最小的分子，找到其在原始集合中的索引
        original_index = cluster_indices[sorted_indices[0]]
        cluster_centers.append(mols[original_index])

    return cluster_centers

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def calculate_optimal_clusters(embeddings, min_clusters=2, max_clusters=15):
    """
    使用轮廓系数自动选择最佳的聚类数。
    """
    best_score = -1
    best_n_clusters = min_clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # 计算轮廓系数
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    return best_n_clusters



from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict

def calculate_mcs_similarity(mol1, mol2):
    """
    计算两个分子之间的 MCS 相似度。
    """
    # 使用 RDKit 计算分子间的最大公共子结构（MCS）相似度。
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    return AllChem.DataStructs.FingerprintSimilarity(fp1, fp2)

def assign_molecules_to_clusters(mols, cluster_centers):
    """
    基于聚类中心将分子分配到相应的聚类。
    """
    mol_cluster_map = defaultdict(list)  # 用于存储每个聚类的分子
    for mol in mols:
        # 计算分子与每个聚类中心的相似度
        similarities = [calculate_mcs_similarity(mol, center) for center in cluster_centers]
        # 找到与该分子最相似的聚类中心
        best_cluster_idx = similarities.index(max(similarities))
        mol_cluster_map[best_cluster_idx].append(mol)  # 将分子分配给相应的聚类

    return mol_cluster_map

def print_cluster_assignments(mol_cluster_map, cluster_centers):
    """
    打印每个聚类及其包含的分子（使用 SMILES 格式）。
    """
    for idx, cluster in mol_cluster_map.items():
        print(f"Cluster {idx + 1} Center SMILES: {Chem.MolToSmiles(cluster_centers[idx])}")
        print(f"Cluster {idx + 1} Members:")
        for mol in cluster:
            print(f"  - {Chem.MolToSmiles(mol)}")
            
from collections import Counter

def combine_labels_multiple_runs(labels_list, mols, center_mols):
    """
    合并多次聚类的标签结果，将每个分子在不同聚类中的标签存储起来。
    """
    # 创建一个字典存储每个分子在多次聚类中的标签
    mol_cluster_info = {}

    # 遍历每次聚类结果
    for i, labels in tqdm(enumerate(labels_list)):
        
        if mol_cluster_info.get(labels_list[i]) is None:
            mol_cluster_info[labels_list[i]] = {'mols':[],'center_mol':""}
        else:
            mol_cluster_info[labels_list[i]]['mols'].append(mols[i])
    
    for k in mol_cluster_info:
        mol_cluster_info[k]['center_mol'] = center_mols[k]
    return mol_cluster_info

from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

def assign_mols_to_clusters(all_mols, mol_cluster_info):
    """
    将全量的分子分配到聚类类别中。
    
    参数：
        - all_mols: 全量分子列表
        - mol_cluster_info: 聚类信息，包含中心分子和每个类别的分子
    
    返回：
        - cluster_assignment: 每个分子的分配类别
    """
    cluster_centers = {label: info['center_mol'] for label, info in mol_cluster_info.items()}
    cluster_assignment = {label: [] for label in cluster_centers.keys()}
    
    
    # 计算每个分子到所有聚类中心的相似度，并分配到最近的聚类中心
    for mol in tqdm(all_mols):
        if mol is None:
            continue  # 跳过无效的分子
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # 计算分子的指纹
        best_cluster = None
        best_similarity = -1
        
        for label, center_mol in cluster_centers.items():
            center_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(center_mol, radius=2, nBits=2048)
            similarity = TanimotoSimilarity(mol_fp, center_fp)  # 计算Tanimoto相似度
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = label
        
        cluster_assignment[best_cluster].append(mol)
        
        
    return cluster_assignment

def group_molecules(mols):
    """
    按照分子原子数对分子进行分组。
    """
    groups = defaultdict(list)
    for mol in tqdm(mols):
        num_atoms = mol.GetNumAtoms()
        # group_id = num_atoms % num_groups  # 分组策略（可根据需要调整）
        groups[num_atoms].append(mol)
    return groups

import argparse

if __name__ == "__main__":
    # # 创建 ArgumentParser 对象
    # parser = argparse.ArgumentParser(description="Process the group number.")
    
    # # 添加命令行参数，type指定参数类型，help指定帮助信息
    # parser.add_argument("--group_number", type=int, required=True, help="The group number")

    # # 解析命令行参数
    # args = parser.parse_args()
    # group_number_arg = args.group_number
    # # 读取 CSV 文件，假设 SMILES 在 'smiles' 列中
    # csv_file_path = "/home/data1/lk/project/mol_tree/graph/m3d/m3d.csv"
    # df = pd.read_csv(csv_file_path)
    # smiles_list = df['smiles'].tolist()  # 限制分子数量

    # mols_list = [Chem.MolFromSmiles(x) for x in tqdm(smiles_list)]
    
    # data_groups = group_molecules(mols_list)
    # torch.save(data_groups, "/home/data1/lk/project/mol_tree/graph/m3d/groups.pth")
    # import pdb;pdb.set_trace()
    data_groups = torch.load("/home/data1/lk/project/mol_tree/graph/m3d/groups.pth")
    # [len(x) for k,x in data_groups.items()]
    
    for g_num_idx in range(1,51):

        if g_num_idx in [9,10,11,12,13,14,15,16,17,18]:
            continue
        group_number = g_num_idx
        # group_number = g_num_idx
        # group_number = group_number_arg
        

        
        mols = data_groups[group_number]
        if len(mols)==0 or len(mols)<1000:
            continue
        # 转换 SMILES 为分子对象
        # mols = smiles_to_mols(smiles_list)
        
        print("group_number:",group_number, 'len(mols):',len(mols))
        
        all_labels = []
        all_mols = []
        all_cluster_centers = []

        num_runs = math.ceil(len(mols)/2400)
        n_samples = 400
        print(f"num_runs:{num_runs}, n_samples:{n_samples}")
        
        add_label_index = 0
        for run in tqdm(range(num_runs),desc=f"for {group_number}:"):
            # 随机选择样本用于初步聚类
            sampled_mols, sampled_indices = select_representative_molecules(mols, n_samples=n_samples)

            # 构造相似度矩阵
            similarity_matrix = construct_similarity_matrix_gpu(sampled_mols)

            # 将相似度矩阵转换为距离矩阵（1 - 相似度）
            dense_similarity_matrix = similarity_matrix.toarray()  # 转换为稠密矩阵

            # 然后计算距离矩阵
            distance_matrix = 1 - dense_similarity_matrix
            

            # 使用 MDS 将距离矩阵降维
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            embeddings = mds.fit_transform(distance_matrix)

            # 自动选择聚类数
            optimal_n_clusters = calculate_optimal_clusters(embeddings)

            kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            cluster_centers = find_cluster_centers(sampled_mols, labels, embeddings, sampled_indices)
            add_label_tmp = max(labels)+1
            labels = labels + add_label_index
            
            
            all_labels.extend(labels)
            all_mols.extend(sampled_mols)
            all_cluster_centers.extend(cluster_centers)
            
            add_label_index = add_label_index + add_label_tmp 
            

        # 合并多次聚类的标签结果
        mol_cluster_info = combine_labels_multiple_runs(all_labels, all_mols, all_cluster_centers)
        


        # 打印每个分子在多次聚类中的标签
        # for labels in mol_cluster_info:
        #     print(f"Cluster {labels}:")
        #     print(f"Number of molecules: {len(mol_cluster_info[labels]['mols'])}")
        #     print(f"Center Mol: {Chem.MolToSmiles(mol_cluster_info[labels]['center_mol'])}")
        
        # 打印每个聚类的中心分子和包含的所有分子
        # print_cluster_assignments(mol_cluster_map, cluster_centers)
        
        # 分配分子到聚类
        cluster_assignment = assign_mols_to_clusters(mols, mol_cluster_info)

        # import pdb;pdb.set_trace()
        # 将数据保存到一个字典中
        data_to_save = {    'mol_cluster_info': mol_cluster_info,   'mols': mols,   'cluster_assignment': cluster_assignment}
        
        # 文件保存路径
        save_path = f"/home/data1/lk/project/mol_tree/graph/m3d/cluster_results_{group_number}"
        import pickle
        # 保存数据到 .npz 文件
        # import pdb;pdb.set_trace()
        np.savez( save_path, mol_cluster_info=pickle.dumps(mol_cluster_info), mols=np.array(mols, dtype=object), cluster_assignment=pickle.dumps(cluster_assignment ))
        print(f"Data saved to {save_path}.")
        # import pdb;pdb.set_trace()
        
        # # 加载 .npz 文件
        # loaded_data = np.load(save_path, allow_pickle=True)

        # # 解压数据
        # mol_cluster_info_loaded = pickle.loads(loaded_data['mol_cluster_info'])
        # mols_loaded = loaded_data['mols']
        # cluster_assignment_loaded = pickle.loads(loaded_data['cluster_assignment'])

        # # 打印加载结果
        # print("Mol Cluster Info:", mol_cluster_info_loaded)
        # print("Mols:", mols_loaded.tolist())  # 转为列表格式
        # print("Cluster Assignment:", cluster_assignment_loaded)

# [(18, 74679),--
#  (17, 547839),-
#  (16, 636261),-
#  (15, 615160),-
#  (14, 541191),-
#  (13, 401582),--
#  (11, 224141),--
#  (12, 301161),--
#  (10, 158756),--
#  (8, 58075),
#  (9, 100641),--
#  (7, 29458),
#  (6, 13685),
#  (5, 5662),
#  (4, 2412),
#  (2, 445),
#  (1, 162),
#  (3, 961),
#  (20, 11779),
#  (19, 6260),
#  (21, 4236),
#  (23, 2254),
#  (22, 3530),
#  (24, 1735),
#  (25, 1188),
#  (26, 854),
#  (32, 98),
#  (35, 25),
#  (28, 605),
#  (34, 47),
#  (27, 623),
#  (31, 157),
#  (30, 323),
#  (37, 4),
#  (29, 443),
#  (33, 58),
#  (40, 28),
#  (39, 11),
#  (38, 3),
#  (36, 11),
#  (41, 23),
#  (43, 10),
#  (44, 12),
#  (48, 3),
#  (42, 20),
#  (51, 2),
#  (45, 3),
#  (50, 1),
#  (46, 2),
#  (49, 1)]