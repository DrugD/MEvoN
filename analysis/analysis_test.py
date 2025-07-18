import numpy as np
from collections import Counter
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def summarize_atom_differences(smiles_pairs):
    """
    汇总多个分子对的原子差异，统计总的新增和删除原子频率。
    返回每对分子的差异特征向量。
    """
    added_counter = Counter()
    removed_counter = Counter()
    
    for mol1_smiles, mol2_smiles in smiles_pairs:
        diff = compare_atoms(mol1_smiles, mol2_smiles)
        added_counter.update(diff["added"])
        removed_counter.update(diff["removed"])
    
    # 获取所有原子的种类
    all_atoms = sorted(set(added_counter.keys()).union(set(removed_counter.keys())))
    
    # 将原子类型和对应的差异频率转换为向量
    vector = []
    
    # 为每个原子生成一个差异值，新增为正值，删除为负值
    for atom in all_atoms:
        vector.append(added_counter.get(atom, 0) - removed_counter.get(atom, 0))  # 计算每个原子的差异
    
    return vector

def compare_atoms(mol1_smiles, mol2_smiles):
    """
    比较两个分子之间的原子差异，统计新增和删除的原子。
    """
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)

    if not mol1 or not mol2:
        raise ValueError("Invalid SMILES strings provided.")
    
    atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    
    count1 = Counter(atoms1)
    count2 = Counter(atoms2)
    
    added = {atom: count2[atom] - count1.get(atom, 0) for atom in count2 if count2[atom] > count1.get(atom, 0)}
    removed = {atom: count1[atom] - count2.get(atom, 0) for atom in count1 if count1[atom] > count2.get(atom, 0)}
    
    return {"added": added, "removed": removed}

def cluster_differences(all_smiles1, all_smiles2, n_clusters=5):
    """
    对大量分子对的差异进行聚类，并返回聚类标签。
    """
    feature_vectors = []
    
    # 计算每对分子的差异并向量化
    for i, (mol1, mol2) in enumerate(zip(all_smiles1, all_smiles2)):
        diff_vector = summarize_atom_differences([(mol1, mol2)])
        feature_vectors.append(diff_vector)
    
    # 填充所有向量到相同的长度
    max_length = max(len(vector) for vector in feature_vectors)
    feature_vectors_padded = [vector + [0] * (max_length - len(vector)) for vector in feature_vectors]
    
    # 将差异向量转化为 numpy 数组
    X = np.array(feature_vectors_padded)
    
    # 标准化特征数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    return labels, kmeans

def visualize_clusters(all_smiles1, all_smiles2, labels):
    """
    可视化分子对聚类的结果。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 每个点的颜色根据聚类标签来选择
    colors = [plt.cm.jet(label / max(labels)) for label in labels]

    # 可视化聚类结果（这里只是简单地将每对分子的差异特征绘制在 3D 空间中）
    for i, (mol1, mol2) in enumerate(zip(all_smiles1, all_smiles2)):
        diff_vector = summarize_atom_differences([(mol1, mol2)])

        # 处理差异向量的维度，确保使用最多三个维度进行可视化
        diff_vector = diff_vector[:3]  # 使用前三个差异特征
        if len(diff_vector) < 3:
            diff_vector = diff_vector + [0] * (3 - len(diff_vector))  # 如果少于3个维度，填充0

        x, y, z = diff_vector[0], diff_vector[1], diff_vector[2]

        ax.scatter(x, y, z, color=colors[i], label=f"Cluster {labels[i]}" if i == 0 else "")
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Clustering of Molecular Differences')
    plt.legend()
    # 保存图像
    plt.savefig("show.png", dpi=300, bbox_inches='tight')
    # print(f"Plot saved to {save_path}")

# 示例用法
all_smiles1 = ["CC", "CCO", "C1CC1", "CC=C"]
all_smiles2 = ["CCC", "CC", "CCO", "CCC"]

# 聚类
labels, kmeans = cluster_differences(all_smiles1, all_smiles2, n_clusters=3)

# 可视化
visualize_clusters(all_smiles1, all_smiles2, labels)

