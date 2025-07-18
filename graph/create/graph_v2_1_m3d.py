from collections import defaultdict
from typing import List, Dict
from tools import  SimilarityCalculator, get_smiles_atoms_num
from tqdm import tqdm
import numpy as np
import pickle
from rdkit import Chem
from tools import *

# 增加评判的标准：相似度最高的多个分子，用其他方法再进行评价

class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)  # 存储图的邻接表
        self.nodes = set()  # 存储所有节点
    
    def add_node(self, label: str):
        self.nodes.add(label)
    
    def add_edge(self, from_node: str, to_node: str):
        if from_node in self.nodes and to_node in self.nodes:
            self.adjacency_list[from_node].append(to_node)

    def get_edges(self):
        return self.adjacency_list
    
    def edges(self):
        # Return a list of edges as tuples
        return [(from_node, to_node) for from_node, to_nodes in self.adjacency_list.items() for to_node in to_nodes]

    def __str__(self):
        return str(dict(self.adjacency_list))

def read_cluster(N):
    npz_file = f"/home/data1/lk/project/mol_tree/graph/m3d/cluster_results_{N}.npz"
    if not os.path.exists(npz_file):
        return  None, None, None
    
    # 加载 .npz 文件
    loaded_data = np.load(npz_file, allow_pickle=True)
    # import pdb;pdb.set_trace()
    # 解压数据
    mol_cluster_info_loaded = pickle.loads(loaded_data['mol_cluster_info'])
    mols_loaded = loaded_data['mols']
    cluster_assignment_loaded = pickle.loads(loaded_data['cluster_assignment'])

    # # 打印加载结果
    # print("Mol Cluster Info:", mol_cluster_info_loaded)
    # print("Mols:", mols_loaded.tolist())  # 转为列表格式
    # print("Cluster Assignment:", cluster_assignment_loaded)
    
    return  mol_cluster_info_loaded, mols_loaded.tolist(), cluster_assignment_loaded
    

def construct_graph(smiles_list: List[str], construct_graph_method: list, threshold: float) -> [Graph, defaultdict]:
    '''
    大规模数据集的进化树构建需要分而治之, 简单说就是需要分块, 减少全局的计算, 将同一种原子数量N的分子组进行聚类划区
    例如分为A B C三个区, 然后小于数量N的分子组, 有K个, 共有D E F G H 5个区, 那么首先比较大区的中心分子之间的相似度
    找到相似度最高的区, 进行相似度查找即可
    '''
    # 聚类分区的数据已经保存 /home/data1/lk/project/mol_tree/graph/pcqm4mv2/cluster_results_{N}.npz
    
    graph = Graph()
    # Group by atom count
    atom_groups = defaultdict(list)
    calculateSim = SimilarityCalculator(method=construct_graph_method[0])
    calculateSim_other = SimilarityCalculator(method=construct_graph_method[1])
    

    for smiles in tqdm(smiles_list):
        atom_count = get_smiles_atoms_num(smiles)  
        atom_groups[atom_count].append(smiles)
        graph.add_node(smiles)
    
    print(f'The length of atom_groups is {len(atom_groups)}.')
    # Build edge relationships
    # atom_groups.keys()
    # for atom_count in tqdm(range(1,5)):
    print("atom_groups.keys():",atom_groups.keys())
    for atom_count in tqdm(sorted(atom_groups.keys())):
        # 临时命令 START
        if atom_count<=16 or atom_count>=26:
            continue
        # END
        

        if atom_count==1:
            continue
        
        
        
        group_list = {}
        no_cluster_count_group = []
        
        mol_cluster_info, _, cluster_assignment = read_cluster(atom_count)
        if mol_cluster_info==None and cluster_assignment==None:
            no_cluster_count_group.append(atom_count)
            group_list[atom_count] = {'sample_group':[{"0":0}],'cluster_group':cluster_assignment}
            print(f"Count {atom_count} is no cluster.")
        else:
            group_list[atom_count] = {'sample_group':mol_cluster_info,'cluster_group':cluster_assignment}
            
        # KK = 1 if group_list[atom_count]['cluster_group'] else 2
        KK = 1
        # 前10个包括第十个，都是取前2个，例如，10对应9，8。之后只对应1个
        for count in range(atom_count, 1 if atom_count-KK<1 else atom_count-2, -1):
            mol_cluster_info, _, cluster_assignment = read_cluster(count-1)
            if mol_cluster_info==None and cluster_assignment==None:
                no_cluster_count_group.append(count-1)
                group_list[count-1] = {'sample_group':[{"0":0}],'cluster_group':cluster_assignment}
                print(f"Count {count-1} is no cluster.")
            else:
                group_list[count-1] = {'sample_group':mol_cluster_info,'cluster_group':cluster_assignment}
            
            # current_groups.extend(atom_groups[count-1])
            print("atom_count:",atom_count,"from ", count-1)
        
        
        
        
        
        # group_list[7]['sample_group'][0]['center_mol']
        
        
        # 用于存储相似度计算结果

        # 遍历 group_list 中的所有元素，提取对应的 'center_mol'
        # a是现有的,待连接的新分子,b是已有节点连接的分子
        
        for _, cluster_index in enumerate(tqdm(sorted(group_list[atom_count]['sample_group']))):
            # 获取当前 sample 中的 molecule_a (即 center_mol)

            if group_list[atom_count]['cluster_group'] != None:
                molecule_a = group_list[atom_count]['sample_group'][cluster_index]['center_mol']
                
                every_group_center_mol_similarity_results = []
                
                # 然后，和 group_list 中其他元素进行相似度计算
                for _, compare_group_atom_count in enumerate(group_list):
                    if compare_group_atom_count == atom_count:
                        continue
                    
                    if compare_group_atom_count in no_cluster_count_group:
                        # 保存结果（也可以添加其他需要的输出）
                        every_group_center_mol_similarity_results.append({
                            'molecule_a_group': atom_count,
                            'molecule_a_cluster_index': cluster_index,
                            'molecule_b_group': compare_group_atom_count,
                            'molecule_b_cluster_index': None,
                            'molecule_a': Chem.MolToSmiles(molecule_a),
                            'similarity': 1
                        })
                        continue
                    
                    for _, compare_cluster_index in enumerate(sorted(group_list[compare_group_atom_count]['sample_group'])):
    
                        molecule_b = group_list[compare_group_atom_count]['sample_group'][compare_cluster_index]['center_mol']
                        
                        # 计算相似度
                        similarity = calculateSim.calculate_similarity(molecule_a, molecule_b)
                        

                        # 保存结果（也可以添加其他需要的输出）
                        every_group_center_mol_similarity_results.append({
                            'molecule_a_group': atom_count,
                            'molecule_a_cluster_index': cluster_index,
                            'molecule_b_group': compare_group_atom_count,
                            'molecule_b_cluster_index': compare_cluster_index,
                            'molecule_a': Chem.MolToSmiles(molecule_a),
                            'molecule_b': Chem.MolToSmiles(molecule_b),
                            'similarity': similarity
                        })

                # 输出结果
                # print(every_group_center_mol_similarity_results)
                every_group_center_mol_similarity_results_sorted = sorted(every_group_center_mol_similarity_results, key=lambda x: x['similarity'], reverse=True)
            
                final_sorted_results = []
                
                # 找到最大相似度
                max_similarity_value = max(result['similarity'] for result in every_group_center_mol_similarity_results_sorted)

                # 找到所有具有最大相似度的分子
                max_similarity_group = [result for result in every_group_center_mol_similarity_results_sorted if result['similarity'] == max_similarity_value]

                current_groups = []
                
                # 不管多少个并列第一, 一律取前K个
                # 1~8 取前5个, 之后取1个
                topK = 1
                # if atom_count >= 8 and atom_count <= 18  else 5
                for group_item in [x for x in every_group_center_mol_similarity_results_sorted if x['similarity']> threshold][:topK]:
                    if group_item['molecule_b_cluster_index'] == None:
                        current_groups = current_groups + [Chem.MolFromSmiles(x) for x in atom_groups[group_item['molecule_b_group']]]
                    else:
                        current_groups = current_groups + group_list[group_item['molecule_b_group']]['cluster_group'][group_item['molecule_b_cluster_index']]

                # # 如果最大相似度有多个分子，则进行二次排序
                # if len(max_similarity_group) > 1:
                #     # 二次排序：使用 similarity_other_way 进行排序
                #     # max_similarity_group_sorted = sorted(max_similarity_group,key=lambda x: calculateSim_other.calculate_similarity(x['molecule_a'], x['molecule_b']),reverse=True)
                #     # final_sorted_results = max_similarity_group_sorted[0]
                #     import pdb;pdb.set_trace()
                    
                #     current_groups = group_list[final_sorted_results['molecule_b_group']]['cluster_group'][final_sorted_results['molecule_b_cluster_index']]

                # else:
                #     # 如果最大相似度只有一个分子，直接保留原顺序
                #     final_sorted_results = max_similarity_group[0]
                #     import pdb;pdb.set_trace()
                
                
                    
                
                # next_group = atom_groups[atom_count]
                
                next_group = group_list[atom_count]['cluster_group'][cluster_index]
            else:
                next_group = atom_groups[atom_count]
                
                current_groups = []  # List to store current groups
                for count in range(atom_count, atom_count-3, -1):
                    current_groups.extend(atom_groups[count-1])
                next_group = [Chem.MolFromSmiles(x) for x in next_group] 
                current_groups = [Chem.MolFromSmiles(x) for x in current_groups]
                
           
            # # print(final_sorted_results['molecule_b_group'],final_sorted_results['molecule_b_cluster_index'])
            # current_groups = group_list[final_sorted_results['molecule_b_group']]['cluster_group'][final_sorted_results['molecule_b_cluster_index']]
            
            # print(current_groups)
            # print(next_group)
            
            for molecule_b in next_group:
                similarities_pair = []
                for molecule_a in current_groups:  # Only the main current group
                    similarity = calculateSim.calculate_similarity(molecule_a, molecule_b)
                    # print(similarity, molecule_a, molecule_b, threshold)
                    if similarity >= threshold:  # If similarity meets the threshold
                        similarities_pair.append((molecule_a, similarity))
                        
                # print("similarities_pair:", similarities_pair)
                
                
                # Find the maximum similarity
                if similarities_pair:
                    
                    if len(similarities_pair)==1:
                        graph.add_edge(Chem.MolToSmiles(molecule_b), Chem.MolToSmiles(similarities_pair[0][0]))
                    else:    
                        similarities_other_pair = []
                        max_similarity = max(similarity for _, similarity in similarities_pair)

                        # Collect all molecules with the maximum similarity
                        max_similar_molecules = [molecule for molecule, sim in similarities_pair if sim == max_similarity]
                        
                        for max_similar_molecule in max_similar_molecules:  # Only the main current group
            
                            similarity_other = calculateSim_other.calculate_similarity(max_similar_molecule, molecule_b)
                            similarities_other_pair.append((max_similar_molecule, similarity_other))
                            
                        max_similarity = max(similarity for _, similarity in similarities_other_pair)   
                        
                        max_similar_molecules = [molecule for molecule, sim in similarities_other_pair if sim == max_similarity]
                        
                        # Add edges for all molecules with the maximum similarity
                        for similar_molecule in max_similar_molecules:
                            graph.add_edge(Chem.MolToSmiles(molecule_b), Chem.MolToSmiles(similar_molecule))
           
                else:
                    print("The ", Chem.MolToSmiles(molecule_b), "is not ok.")
                    
                    with open('/home/data1/lk/project/mol_tree/graph/m3d/bad_smiles.txt', 'a') as file:
                        file.write(Chem.MolToSmiles(molecule_b) + '\n')  # 每个 SMILES 换行写入
                        
        save_path = f'/home/data1/lk/project/mol_tree/graph/m3d/temp_atom_count_{atom_count}_evolution_graph_{len(atom_groups[atom_count])}_{construct_graph_method}_{threshold}_v2.json'  # Specify the path where you want to save the graph
        save_graph(graph, atom_groups, save_path)
        
    return graph, atom_groups
