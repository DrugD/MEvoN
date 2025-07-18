from collections import defaultdict
from typing import List, Dict
from tools import  SimilarityCalculator, get_smiles_atoms_num
from tqdm import tqdm


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


def construct_graph(smiles_list: List[str], construct_graph_method: str, threshold: float) -> [Graph, defaultdict]:
    graph = Graph()
    # Group by atom count
    atom_groups = defaultdict(list)
    calculateSim = SimilarityCalculator(method=construct_graph_method)
    
    for smiles in tqdm(smiles_list):
        atom_count = get_smiles_atoms_num(smiles)  
        atom_groups[atom_count].append(smiles)
        graph.add_node(smiles)

    # Build edge relationships
    for atom_count in tqdm(sorted(atom_groups.keys())):
        if atom_count==1:
            continue
        
        # print("atom_count:",atom_count)
        current_groups = []  # List to store current groups
        for count in range(atom_count, 1, -1):
            # print("xxxxxxxxxxxxxxx:",count-1, atom_count)
            current_groups.extend(atom_groups[count-1])
            
        # import pdb;pdb.set_trace()
        next_group = atom_groups[atom_count]
        
        # print(current_groups)
        # print(next_group)
        
        for molecule_b in tqdm(next_group):
            similarities = []
            for molecule_a in current_groups:  # Only the main current group
                similarity = calculateSim.calculate_similarity(molecule_a, molecule_b)
                # print(similarity, molecule_a, molecule_b, threshold)
                if similarity >= threshold:  # If similarity meets the threshold
                    similarities.append((molecule_a, similarity))
                    
            # print("similarities:", similarities)
            
            # import pdb;pdb.set_trace()
            # Find the maximum similarity
            if similarities:
                max_similarity = max(similarity for _, similarity in similarities)

                # Collect all molecules with the maximum similarity
                max_similar = [molecule for molecule, sim in similarities if sim == max_similarity]

                # Add edges for all molecules with the maximum similarity
                for similar_molecule in max_similar:
                    graph.add_edge(molecule_b, similar_molecule)
            else:
                print("The ", molecule_b, "is not ok.")
    return graph, atom_groups
