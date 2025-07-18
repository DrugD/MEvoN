import pandas as pd
import torch, json
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm
from tools import *



from graph_v2 import Graph
def load_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    graph = Graph()
    # 添加节点
    for node in data['nodes']:
        graph.add_node(node)

    # 添加边
    for edge in data['edges']:
        graph.add_edge(edge[0], edge[1])

    return graph


class MoleculeACDataset_v2(Dataset):
    def __init__(self, graph_file, labels_file, test_file=None, mode='train'):
        
        self.mode = mode
        
        # Load the graph data
        with open(graph_file, 'r') as f:
            self.graph_data = json.load(f)

        # 加载图
        self.molecule_phylogenetic_graph = load_graph_from_json(graph_file)
         
        # Load the labels from the CSV file
        self.labels_df = pd.read_csv(labels_file)
        

        # Create a dictionary for quick label access
        self.labels_dict = self.labels_df.set_index('smiles')['u0_atom'].to_dict()
        
        # import pdb;pdb.set_trace()
        # 计算均值和标准差
        self.label_mean = self.labels_df['u0_atom'].mean()
        self.label_std = self.labels_df['u0_atom'].std()
            
        # import pdb;pdb.set_trace()
        self.path_data = []
        
         # 读取文件中的 SMILES
        with open(test_file, 'r') as file:
            self.test_smiles_list = [line.strip() for line in file if line.strip()]
        
        
        # The mode maybe"trainval" or "test"
        if mode=='test':
            # 输出读取的 SMILES 列表
            # 更新graph_data
            from main_add_new_mols_to_exist_tree import load_graph, add_new_molecules_to_graph
            
            # 设置图构建方法和阈值
            construct_graph_method = ['edit_distance', 'graph']
            threshold = 0.3
            
            graph, atom_groups = load_graph(self.graph_data)
            
            molecule_graph_for_test, atom_groups_for_test = add_new_molecules_to_graph(graph, atom_groups, self.test_smiles_list, construct_graph_method, threshold)
            
            self.molecule_phylogenetic_graph, self.atom_groups = molecule_graph_for_test, atom_groups_for_test
            
            # # 提前生成
            # file_mol_tree_path = "/home/data1/lk/project/mol_tree/graph/add_from10000_evolution_graph_1000_['edit_distance', 'graph']_0.3_v2.json"
            # with open(file_mol_tree_path, 'r') as f:
            #     self.graph_data = json.load(f)
            # self.molecule_phylogenetic_graph, self.atom_groups = load_graph(self.graph_data)    

        # Prepare the dataset
        self.dataset = self.construct_dataset()
        self.valid_indices = self.filter_invalid_molecules()

        
    def find_molecule_paths(self, smiles):

        all_paths = find_paths(self.molecule_phylogenetic_graph, smiles, 'C')
        if all_paths==[]:
            return None
        
        all_paths_labels = []
        for i in all_paths:
            temp = [self.labels_dict.get(j) for j in i][1:]
            # for j in i:  
            #     temp.append(self.labels_dict.get(j)['homo'])
            all_paths_labels.append(temp)

        return [[path[::-1] for path in all_paths] , [label[::-1] for label in all_paths_labels]]
        
    def construct_dataset(self):
        dataset = []
       
        for edge in self.graph_data['edges']:
            mol1 = edge[0]
            mol2 = edge[1]
            
            # if self.mode == 'test' and mol1 in self.test_smiles_list:
            #     label1 = {'homo': None, 'lumo': None}
            # else:
            #     label1 = self.labels_dict.get(mol1, {'homo': None, 'lumo': None})
                
            # if self.mode == 'test' and mol2 in self.test_smiles_list:
            #     label2 = {'homo': None, 'lumo': None}  
            # else:
            #     label2 = self.labels_dict.get(mol2, {'homo': None, 'lumo': None})
            # import pdb;pdb.set_trace()
            label1 = self.labels_dict.get(mol1, {'u0_atom': None})
            label2 = self.labels_dict.get(mol2, {'u0_atom': None})
            
            
            label1 = (label1 - self.label_mean) / self.label_std
            label2 = (label2 - self.label_mean) / self.label_std

            dataset.append({
                'molecule_1': mol1,
                'molecule_2': mol2,
                'u0_atom_1': label1,
                'u0_atom_2': label2
            })
        
        return pd.DataFrame(dataset)

    def filter_invalid_molecules(self):
        max__ = -1
        valid_indices = []
        target_length = 5
        max_length = 9

                
        def pad_or_trim(paths, labels, target_length, max_length):
            # 扩充到目标路径长度
            if len(paths) < target_length:
                pad_length = target_length - len(paths)
                paths += [['PAD'] * max_length] * pad_length  # 使用 'PAD' 填充
                
                # 用 0 填充标签
                labels = torch.cat([labels, torch.zeros((pad_length, labels.size(1)))], dim=0)  # 用 0 填充标签
            
            # 截取前 target_length 个，选择 'PAD' 最少的路径
            paths = sorted(paths, key=lambda x: x.count('PAD'))[:target_length]
            labels = labels[:target_length]  # 假设标签对应于路径的顺序

            # 创建一个新的 (target_length, max_length) 的全0张量
            new_labels = torch.zeros((target_length, max_length))
            mask = torch.zeros((target_length, max_length), dtype=torch.int) # 掩码初始化为0

    
            
            # 将有效标签值复制到新的张量中，并更新掩码
            for i in range(min(len(labels), target_length)):
                new_labels[i, :labels[i].size(0)] = labels[i]
                mask[i, :len(paths[i])] = torch.tensor([1 if x != 'PAD' else 0 for x in paths[i]], dtype=torch.int)

            return paths, new_labels, mask
        
    
        for idx in tqdm(range(len(self.dataset))):
            smiles = self.dataset.iloc[idx].molecule_1
            
            if self.mode == 'test':
                if smiles not in self.test_smiles_list:
                    continue
            elif self.mode == "trainval":
                if smiles in self.test_smiles_list:
                    continue
                
            path = self.find_molecule_paths(smiles)
            # import pdb;pdb.set_trace()
            if path == [] or path==None:
                continue
            else:
                result = ["invalid" if x in self.test_smiles_list else x for x in path[0][0][:-1]]

                if "invalid" in result:
                    continue
                else:
                    padded_paths, padded_paths_labels = path_padding(path)

                    if len(padded_paths) > max__ :
                        max__ = len(padded_paths)

                    processed_paths, processed_labels, mask = pad_or_trim(padded_paths, padded_paths_labels, target_length=5, max_length=9)
                    
                    self.path_data.append((processed_paths, processed_labels, mask))
                    valid_indices.append(idx)
        
        # import pdb;pdb.set_trace()
                    
        print(f"max path length is {max__}.")
        print(f"Total data is {len(valid_indices)}.")

        
        return valid_indices

    def is_valid_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        try:
            Chem.SanitizeMol(mol)
        except Chem.AtomValenceException:
            # here you may handle the error condition, or just ignore it
            return False

        # Check explicit valence
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > atom.GetFormalCharge() + atom.GetImplicitValence() + atom.GetTotalNumHs():
                return False
        
        return True

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the valid index
        valid_idx = self.valid_indices[idx]
        data = self.dataset.iloc[valid_idx]
        
        smiles1 = data.molecule_1
        smiles2 = data.molecule_2
        
        # # 将 SMILES 转换为图结构
        # graph1 = smiles_to_graph(smiles1)
        # graph2 = smiles_to_graph(smiles2)
        
        graph1 = smiles_to_graph_xyz_equiformer_qm7(smiles1)
        graph2 = smiles_to_graph_xyz_equiformer_qm7(smiles2)
        
        
        # 类比
        # "C#C" graph1
        # "C" graph2
        # import pdb;pdb.set_trace()
        # 这里可以加入标签或其他特征
        label_list = [x for x in data[['u0_atom_1', 'u0_atom_2']].values]
        
        target = torch.tensor([ label_list[0]-label_list[1], label_list[0], label_list[1]], dtype=torch.float64)

        # target = torch.tensor([(label_list[0]-label_list[1])/label_list[1], label_list[0], label_list[1]], dtype=torch.float64)
        padded_paths, padded_paths_labels, padded_paths_labels_masks = self.path_data[idx]

        return graph1, graph2, target, padded_paths, padded_paths_labels, padded_paths_labels_masks

