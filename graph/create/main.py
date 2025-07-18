
from tree_ import Tree
from tree import Tree 
from graph_v1 import Tree 
from read_data import read_smiles_from_file
from tools import *

# 示例：从文件中读取数据
file_path = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'
SmilesList = read_smiles_from_file(file_path)

data_ = SmilesList[:200]
# +SmilesList[10000:10100]

# 创建进化树
tree_construct_kenerl_method = 'edit_distance'
threshold = 0.0
evolution_tree = Tree(root_label='C', similarity_method=tree_construct_kenerl_method)
evolution_tree.add_molecule(data_, k=5, threshold=threshold)
# Method threshold
# edit_distance 0.0
# quantum ?
# topological ?
# conformational ?
# graph 0.3

evolution_tree.draw_tree(f'img/evolution_tree_{len(data_)}_{tree_construct_kenerl_method}_{threshold}_v1.png')

# tree_as_list = evolution_tree.root.to_list()
# output_dir = f'/home/data1/lk/project/mol_tree/{len(data_)}_img'
# save_smiles_as_images(data_, output_dir)

# import pdb;pdb.set_trace()

# # 示例 SMILES 列表
# SmilesList = ['CC', 'CCC', 'CCCC', 'CCO', 'CCCO', 'CCCCO']

# # 创建进化树并绘制
# evolution_tree = Tree()
# evolution_tree.add_molecule(SmilesList)

# # 保存树结构为图片
# evolution_tree.draw_tree('evolution_tree.png')
