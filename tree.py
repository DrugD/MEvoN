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
from tools import deduplicate_nodes

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
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            print(f"无法解析 SMILES: {smiles1}, {smiles2}")
            return None
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and smiles1 == 'C' and mol2.GetNumAtoms() == 2 and
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
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and smiles1 == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        if mol1 is None or mol2 is None:
            return None
        g1 = self.smiles_to_grakel(smiles1)
        g2 = self.smiles_to_grakel(smiles2)
        gk = WeisfeilerLehman(n_iter=5, normalize=True)
        K = gk.fit_transform([g1, g2])
        score = K[0, 1]
        # print(f"{smiles1},{smiles2},{score}")
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
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and smiles1 == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
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
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and smiles1 == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        weight1 = Chem.rdMolDescriptors.CalcExactMolWt(mol1)
        weight2 = Chem.rdMolDescriptors.CalcExactMolWt(mol2)
        return 1.0 - abs(weight1 - weight2) / max(weight1, weight2)

    def topological_similarity(self, smiles1, smiles2):
        """基于拓扑描述符的相似度"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        # 特殊情况处理：如果分子1是单个碳原子，且分子2只有两个原子，其中一个是C，直接返回相似度为1
        if (mol1.GetNumAtoms() == 1 and smiles1 == 'C' and mol2.GetNumAtoms() == 2 and
            any(atom.GetSymbol() == 'C' for atom in mol2.GetAtoms())):
            print(f"检测到根节点为单个碳原子: {smiles1}, 新增分子: {smiles2}，相似度设为 1")
            return 1.0
        
        # 使用简单的分子拓扑描述符如 Wiener 指数
        topo1 = Chem.rdMolDescriptors.CalcChi0n(mol1)
        topo2 = Chem.rdMolDescriptors.CalcChi0n(mol2)
        return 1.0 - abs(topo1 - topo2) / max(topo1, topo2)


    def smiles_to_grakel(self, smiles):
        """将 SMILES 转化为 Grakel 支持的图表示"""
        mol = Chem.MolFromSmiles(smiles)
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
        return Graph(edges, node_labels={i: l for i, l in enumerate(atom_labels)})
    

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
    
class TreeNode:
    def __init__(self, label, similarity_method="fingerprint"):
        self.label = label
        self.similarity_calculator = SimilarityCalculator(similarity_method)
        self.children = []
        self._depth = None  # 用于存储深度
        self.parent = None  # 添加父节点属性

    def get_atom_count(self, smiles):
        """计算给定 SMILES 表示的分子的原子数量"""
        mol = Chem.MolFromSmiles(smiles)  # 解析 SMILES
        if mol:
            return mol.GetNumAtoms()  # 返回原子数量
        else:
            return 0  # 如果无法解析 SMILES，则返回 0
        
    def get_all_nodes(self):
        """递归获取树中所有节点."""
        all_nodes = [self]  # 包含自身
        for child in self.children:
            all_nodes.extend(child.get_all_nodes())  # 递归收集所有子节点
        return all_nodes
    
    def add_child(self, child_node):
        child_node.parent = self  # 设置子节点的父节点
        self.children.append(child_node)

    def update_depth(self):
        if self.parent is None:
            self._depth = 0  # 根节点深度为0
        else:
            self._depth = self.parent.depth + 1  # 当前节点深度为父节点深度加1

    @property
    def depth(self):
        if self._depth is None:
            self.update_depth()  # 确保深度是最新的
        return self._depth

    def to_list(self):
        """将树的每一层转换为列表。"""
        result = []
        current_level = [self]  # 当前层，初始化为根节点

        while current_level:
            result.append([node.label for node in current_level])  # 获取当前层的标签
            next_level = []  # 下一层的节点
            for node in current_level:
                next_level.extend(node.children)  # 将当前层节点的子节点添加到下一层
            current_level = next_level  # 更新当前层为下一层

    def find_depth(self, target_node):
        """
        查找特定节点的深度。

        参数:
        target_node: TreeNode - 要查找的节点对象

        返回:
        int - 节点的深度，如果未找到则返回 -1
        """
        if self is target_node:  # 直接比较节点对象
            return self.depth
        
        for child in self.children:
            depth = child.find_depth(target_node)  # 递归查找
            if depth != -1:  # 如果找到
                return depth
        
        return -1  # 未找到目标节点


from tqdm import tqdm

class Tree:
    def __init__(self, root_label='C', similarity_method="fingerprint"):
        self.root = TreeNode(root_label, similarity_method=similarity_method)
        self.failure_count = defaultdict(int)  # 使用 defaultdict 来记录每个分子的失败次数
        self.node_num = {'num_of_mols_add':0,
                         'num_of_mols_total':0}
        
    def add_molecule(self, smiles_list, threshold=0.2, k=30):
        """添加分子到进化树中，只有相似度最高且高于阈值的分子才会被插入。
        失败次数超过 k 的分子将被丢弃。
        """
        pending_smiles = smiles_list[:]  # 将所有待插入的 SMILES 初始化到列表中
        
        self.node_num['num_of_mols_total'] = len(smiles_list)
        
        # 创建 tqdm 进度条，初始化总数为 smiles_list 长度
        pbar = tqdm(total=len(smiles_list), desc="插入分子进度", ncols=100)

        while pending_smiles:
            smiles = pending_smiles.pop(0)  # 从列表开头取出 SMILES 进行插入

            # 检查当前 SMILES 是否是单原子
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms() == 1:
                print(f"分子 {smiles} 是单原子，已被丢弃。")
                self.failure_count[smiles] = 1  # 增加失败计数
                pbar.update(1)  # 更新进度条
                continue  # 跳过此分子，进入下一个

            # 检查当前分子的失败次数
            if self.failure_count[smiles] >= k:
                pbar.update(1)  # 更新进度条
                continue  # 跳过此分子，进入下一个

            # 获取当前分子的原子数量
            current_num_atoms = mol.GetNumAtoms()

            # 记录当前分子与候选父节点的最高相似度及其对应节点
            best_similarity = -sys.maxsize - 1
            # best_parent_node = None
            best_parent_nodes = []  # 用于存储相似度最高的父节点
            
            
            # 遍历所有父节点找到最高相似度的节点
            for parent_node in self.root.get_all_nodes():
                
                parent_mol = Chem.MolFromSmiles(parent_node.label)
                parent_num_atoms = parent_mol.GetNumAtoms()

                # 检查父节点的原子数量是否小于当前分子，并且相似度是否高于阈值
                if parent_num_atoms < current_num_atoms:
                    similarity = parent_node.similarity_calculator.calculate_similarity(parent_node.label, smiles)
                    
                    if similarity > best_similarity:  # 找到更高的相似度
                        best_similarity = similarity
                        best_parent_nodes = [parent_node]  # 重置并添加新最高相似度的节点
                    elif similarity == best_similarity:  # 如果相似度相同，则也添加到列表中
                        best_parent_nodes.append(parent_node)

            # 如果找到的最佳父节点的相似度高于阈值，则添加到树中
            if best_parent_nodes and best_similarity >= threshold * pow(0.8, self.failure_count[smiles]):
                leaf_node = TreeNode(smiles, similarity_method=self.root.similarity_calculator.method)
               
        
                unique_parent_nodes = deduplicate_nodes(self.root, best_parent_nodes)
              
                for parent_node  in unique_parent_nodes.values():  # 遍历所有最佳父节点
                    parent_node.add_child(leaf_node)  # 为每个最佳父节点添加子节点
                
                pbar.update(1)  # 插入成功时，更新进度条
                
                self.node_num['num_of_mols_add'] += 1
                    
                print(f"分子 {smiles} 已插入到 {len(unique_parent_nodes.values())} 个父节点。")
                
                if len(unique_parent_nodes.values())>20:
                    import pdb;pdb.set_trace()
                
            else:
                if smiles not in pending_smiles:  # 确保分子没有重复加入待插入列表
                    pending_smiles.append(smiles)
                    # 相似度不够高，将分子放回待插入的列表末尾，并增加失败计数
                
                self.failure_count[smiles] += 1  # 增加失败计数
                if self.failure_count[smiles] >= k:
                    print(f"分子 {smiles} 失败次数超过 {k} 次，已被丢弃。")

            # 更新进度条为 pending_smiles 的剩余长度
            pbar.total = len(smiles_list)  # 重置总数为 smiles_list 的长度
            pbar.n = len(smiles_list) - len(pending_smiles)  # 更新已处理的数量
            pbar.refresh()  # 刷新进度条显示

        pbar.close()  # 完成后关闭进度条
       
        print(f"Total: {self.node_num['num_of_mols_total']}, Success:{self.node_num['num_of_mols_add']}, Faild:{len([k for x,k in self.failure_count.items() if k > 0])}")
        
        if len([x for x,k in self.failure_count.items() if k > 0])>0:
            print("Failure list:", [x for x,k in self.failure_count.items() if k > 0])
            
    def get_edges(self, node=None):
        """获取树中的所有边"""
        if node is None:
            node = self.root
        edges = []
        for child in node.children:
            edges.append((node.label, child.label))
            edges.extend(self.get_edges(child))
        return edges
    
    def draw_tree(self, save_path='tree_structure.png'):
        if self.node_num['num_of_mols_add'] == 0:
            print("No valid mols.")
            return
        
        graph = nx.DiGraph()
        edges = self.get_edges()
        graph.add_edges_from(edges)

        num_nodes = len(graph.nodes)
        
        # 动态设置图像的大小和文字大小
        fig_width = min(num_nodes * 0.5, 20)  # 限制最大宽度为 20
        fig_height = min(num_nodes * 0.5, 20)  # 限制最大高度为 20
        font_size = max(8, 12 - num_nodes // 5)  # 根据节点数量动态设置字体大小，最小为 8

        plt.figure(figsize=(fig_width, fig_height))
        pos = self.hierarchical_layout(graph)

        # 在节点位置上添加随机偏移，减少重叠
        for node in pos:
            pos[node][0] += (np.random.rand() - 0.5) * 0.5  # 随机偏移 x 坐标
            pos[node][1] += (np.random.rand() - 0.5) * 0.5  # 随机偏移 y 坐标

        # 计算每个节点的层级
        levels = self.assign_levels(graph)
        unique_levels = set(levels.values())
        
        # 为每个层级分配颜色
        color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_levels)))  # 使用Viridis调色板
        level_color_map = {level: color_map[i] for i, level in enumerate(unique_levels)}

        # 为每条边设置颜色
        edge_colors = [level_color_map[levels[u]] for u, v in graph.edges()]

        # 绘制图形
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', 
                font_size=font_size, font_weight='bold', arrows=True, edge_color=edge_colors)

        # 保存图像，设置 DPI
        plt.savefig(save_path, dpi=300)

    def hierarchical_layout(self, G):
        pos = nx.spring_layout(G, seed=42, k=0.5)  # k 参数调整弹簧的强度
        levels = self.assign_levels(G)

        for node, level in levels.items():
            pos[node][1] = -level
        return pos

    def assign_levels(self, G):
        levels = {}
        root = 'C'
        stack = [(root, 0)]
        visited = set()

        while stack:
            node, level = stack.pop()
            if node not in visited:
                visited.add(node)
                levels[node] = level
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        stack.append((neighbor, level + 1))

        return levels



# # 示例 SMILES 列表
# SmilesList = ['C', 'CC', 'CCC', 'CCCC', 'CCO', 'CCCO', 'CCCCO']

# # 创建进化树并绘制
# evolution_tree = Tree()
# evolution_tree.add_molecule(SmilesList, threshold=0.5)

# # 保存树结构为图片
# evolution_tree.draw_tree('evolution_tree.png')
