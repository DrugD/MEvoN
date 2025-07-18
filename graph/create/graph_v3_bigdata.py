import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from tqdm import tqdm
from grakel import Graph
from grakel.kernels import ShortestPath
import multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence

# 读取CSV文件并清洗数据
def load_and_clean_data(file_path):
    print("正在读取CSV文件...")
    data = pd.read_csv(file_path)
    print("CSV文件读取完成！")

    # 去除没有标签的样本
    print("正在去除没有标签的样本...")
    data = data.dropna(subset=['homolumogap'])  # 去除 homolumogap 为 NaN 的样本
    data = data[data['homolumogap'].notnull()]  # 去除 homolumogap 为空值的样本
    print(f"去除无效样本后，剩余样本数：{len(data)}")

    # 去除重复的 SMILES
    print("正在去除重复的 SMILES...")
    data = data.drop_duplicates(subset=['smiles'], keep='first')  # 保留第一个出现的重复样本
    print(f"去重后，剩余样本数：{len(data)}")

    return data

# 将SMILES转换为分子图
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

# 过滤无效的SMILES
def filter_invalid_smiles(data):
    print("正在转换SMILES为分子图...")
    data['mol'] = [smiles_to_graph(smiles) for smiles in tqdm(data['smiles'], desc="SMILES转换")]
    data = data.dropna(subset=['mol'])
    print(f"过滤无效 SMILES 后，剩余样本数：{len(data)}")
    return data

# 根据原子个数分组
def group_by_num_atoms(data):
    print("正在根据原子个数分组...")
    data['num_atoms'] = [mol.GetNumAtoms() for mol in tqdm(data['mol'], desc="计算原子个数")]
    grouped = data.groupby('num_atoms')
    print("分组完成！")
    return grouped

# 将RDKit分子转换为PyTorch图
def mol_to_torch_graph(mol, device='cuda'):
    """
    将 RDKit 分子转换为 PyTorch 图（邻接矩阵和节点标签）
    :param mol: RDKit 分子
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 邻接矩阵和节点标签
    """
    adj = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.float32, device=device)  # 邻接矩阵
    labels = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long, device=device)  # 节点标签
    return adj, labels

# 使用 GPU 加速的 Weisfeiler-Lehman 图核
def wl_kernel_gpu(graphs, num_iterations=3, device='cuda'):
    """
    使用 GPU 加速的 Weisfeiler-Lehman 图核计算相似度矩阵
    :param graphs: 图列表，每个图是一个元组 (adj, labels)
    :param num_iterations: WL 迭代次数
    :param device: 使用的设备（'cuda' 或 'cpu'）
    :return: 相似度矩阵
    """
    num_graphs = len(graphs)
    similarity_matrix = torch.zeros((num_graphs, num_graphs), device=device)

    # 获取所有图中标签的最大值
    max_label = max([labels.max().item() for _, labels in graphs]) + 1

    # 迭代计算 WL 核
    for it in range(num_iterations + 1):
        print(f"正在计算 WL 迭代 {it}...")
        histograms = []  # 存储每个图的直方图

        # 计算每个图的直方图
        for adj, labels in graphs:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            hist = torch.zeros(max_label, device=device)  # 统一长度的直方图
            hist[unique_labels] = counts.float()
            histograms.append(hist)

        # 计算相似度矩阵
        for i in tqdm(range(num_graphs), desc="计算相似度"):
            for j in range(i, num_graphs):
                sim = torch.dot(histograms[i], histograms[j]) / (
                    torch.norm(histograms[i]) * torch.norm(histograms[j]) + 1e-8)
                similarity_matrix[i, j] += sim
                similarity_matrix[j, i] += sim

        # 更新节点标签
        if it < num_iterations:
            for i, (adj, labels) in enumerate(graphs):
                new_labels = []
                for node in range(adj.shape[0]):
                    neighbor_labels = labels[adj[node].nonzero().squeeze()]
                    # 将 neighbor_labels 转换为一维张量
                    neighbor_labels = neighbor_labels.unsqueeze(0) if neighbor_labels.dim() == 0 else neighbor_labels
                    new_label = torch.cat([labels[node].unsqueeze(0), neighbor_labels])
                    new_label = torch.sort(new_label)[0]  # 排序以确保唯一性
                    new_labels.append(new_label)
                
                # 使用 pad_sequence 填充 new_labels
                new_labels_padded = pad_sequence(new_labels, batch_first=True)
                graphs[i] = (adj, new_labels_padded)

    # 归一化相似度矩阵
    similarity_matrix /= (num_iterations + 1)
    return similarity_matrix.cpu().numpy()

# 计算基于图的相似度矩阵（结合 WL 和 Shortest-Path 图核）
def calculate_combined_similarity(graphs_k, graphs_k_plus_1):
    """
    结合 Weisfeiler-Lehman 和 Shortest-Path 图核计算相邻组之间的相似度矩阵
    :param graphs_k: Group_k 的图列表
    :param graphs_k_plus_1: Group_{k+1} 的图列表
    :return: 相似度矩阵
    """
    print("正在计算 Weisfeiler-Lehman 图核...")
    sim_wl = wl_kernel_gpu(graphs_k + graphs_k_plus_1, num_iterations=3, device='cuda')

    print("正在计算 Shortest-Path 图核...")
    sp_kernel = ShortestPath(normalize=True)
    sp_kernel.fit(graphs_k + graphs_k_plus_1)
    sim_sp = sp_kernel.transform(graphs_k + graphs_k_plus_1)

    # 结合两种相似度矩阵
    combined_sim = 0.5 * sim_wl + 0.5 * sim_sp
    return combined_sim

# 构建进化树
def build_phylogenetic_tree(similarity_matrix):
    distance_matrix = 1 - similarity_matrix
    print("正在使用MDS降维...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    print("正在使用层次聚类构建进化树...")
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
    labels = clustering.fit_predict(coords)
    return labels, coords

# 可视化进化树
def plot_phylogenetic_tree(coords, labels):
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(f"Phylogenetic Tree (Group with {len(coords)} molecules)")
    plt.show()

# 主流程
def main():
    file_path = '/home/data1/lk/project/mol_tree/graph/pcqm4mv2/pcqm4mv2.csv'
    
    # 加载并清洗数据
    data = load_and_clean_data(file_path)
    data = data[:10000]  # 仅处理前 10000 个样本
    data = filter_invalid_smiles(data)
    grouped = group_by_num_atoms(data)

    # 将分组转换为列表并按原子个数排序
    groups = sorted([(num_atoms, group) for num_atoms, group in grouped], key=lambda x: x[0])

    # 处理每一对相邻组
    for i in tqdm(range(len(groups) - 1), desc="处理相邻组"):
        num_atoms_k, group_k = groups[i]
        num_atoms_k_plus_1, group_k_plus_1 = groups[i + 1]
        print(f"正在处理 Group{num_atoms_k} 和 Group{num_atoms_k_plus_1}...")

        # 将分子转换为 PyTorch 图
        print("正在将分子转换为 PyTorch 图...")
        graphs_k = [mol_to_torch_graph(mol) for mol in tqdm(group_k['mol'], desc=f"Group{num_atoms_k} 分子图转换")]
        graphs_k_plus_1 = [mol_to_torch_graph(mol) for mol in tqdm(group_k_plus_1['mol'], desc=f"Group{num_atoms_k_plus_1} 分子图转换")]

        # 计算相似度矩阵
        similarity_matrix = calculate_combined_similarity(graphs_k, graphs_k_plus_1)

        # 构建进化树
        labels, coords = build_phylogenetic_tree(similarity_matrix)
        plot_phylogenetic_tree(coords, labels)
        print(f"Group{num_atoms_k} 和 Group{num_atoms_k_plus_1} 处理完成！")

if __name__ == "__main__":
    main()