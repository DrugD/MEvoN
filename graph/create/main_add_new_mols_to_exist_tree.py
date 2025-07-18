import json
from tqdm import tqdm
from collections import defaultdict
from graph_v2 import Graph, construct_graph
from tools import read_smiles_from_file, save_graph, SimilarityCalculator, get_smiles_atoms_num
from rdkit import Chem

def group_molecules(mols):
    """
    按照分子原子数对分子进行分组。
    """
    groups = defaultdict(list)
    for mol in tqdm(mols):
        if type(mol)==str:
            mol = Chem.MolFromSmiles(mol)
        num_atoms = mol.GetNumAtoms()
        # group_id = num_atoms % num_groups  # 分组策略（可根据需要调整）
        groups[num_atoms].append(mol)
    return groups

# 加载现有图数据的函数
def load_graph(file_path: str) -> [Graph, defaultdict]:
    if type(file_path) == str:
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif type(file_path) == dict:
        data = file_path
        
    graph = Graph()

    if data.get('nodes'):
        for node in data['nodes']:
            graph.add_node(node)
    else:
        # 添加节点
        smiles_list = [item for sublist in data.get('edges', []) for item in sublist] 
        for node in smiles_list:
            graph.add_node(node)
    
    for edge in data["edges"]:
        graph.add_edge(edge[0], edge[1])
        
    if data.get('atom_groups'):
        atom_groups = defaultdict(list, data["atom_groups"])
    else:
        atom_groups = group_molecules(smiles_list)
        
    return graph, atom_groups


# 添加新分子到现有图的函数
def add_new_molecules_to_graph(graph, atom_groups, new_smiles_list,
                               construct_graph_method, threshold):
    calculateSim = SimilarityCalculator(method=construct_graph_method[0])
    calculateSim_other = SimilarityCalculator(method=construct_graph_method[1])
    

    
    for smiles in tqdm(new_smiles_list):
        if smiles in graph.nodes:
            continue
        
        similarities_pair = []
        
        atom_count = get_smiles_atoms_num(smiles)
        molecule_b = smiles
        current_groups = []  # List to store current groups
        for count in range(atom_count, 1, -1):
            # print("xxxxxxxxxxxxxxx:",count-1, atom_count)
            
            if atom_groups.get(str(count)): 
                current_groups.extend(atom_groups[str(count)])
            
        for molecule_a in current_groups:  # Only the main current group
            similarity = calculateSim.calculate_similarity(molecule_a, molecule_b)
            # print(similarity, molecule_a, molecule_b, threshold)
            if similarity >= threshold:  # If similarity meets the threshold
                similarities_pair.append((molecule_a, similarity))
                    
            # print("similarities:", similarities)
            
            # import pdb;pdb.set_trace()
            # Find the maximum similarity
        
        
        if similarities_pair:
            
            atom_groups[str(atom_count)].append(smiles)
            graph.add_node(smiles)
            
            if len(similarities_pair)==1:
                graph.add_edge(molecule_b, similarities_pair[0][0])
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
                    graph.add_edge(molecule_b, similar_molecule)
        else:
            print("The ", molecule_b, "is not ok.")
            
    return graph, atom_groups

# 主函数
def main():
    # 加载现有的图数据
    existing_graph_path = '/home/data1/lk/project/mol_tree/graph/evolution_graph_10000_graph_0.3_v1.json'
    graph, atom_groups = load_graph(existing_graph_path)

    
    # 新分子数据路径
    new_data_file_path = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
    new_smiles_list = read_smiles_from_file(new_data_file_path)
    new_smiles_list = new_smiles_list[10000:11000]
    import pdb;pdb.set_trace()
    # 设置图构建方法和阈值
    construct_graph_method = ['edit_distance', 'graph']
    threshold = 0.3

    # 增加新的分子到图中
    molecule_graph, atom_groups = add_new_molecules_to_graph(graph, atom_groups, new_smiles_list, construct_graph_method, threshold)
    
    
    # 保存更新后的图
    updated_graph_path = f'graph/add_from10000_evolution_graph_{len(new_smiles_list)}_{construct_graph_method}_{threshold}_v2.json' 
    save_graph(graph, atom_groups, updated_graph_path)

if __name__ == "__main__":
    main()
