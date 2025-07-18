
from tools import *


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
    
# 创建进化树
tree_construct_kenerl_method = 'edit_distance'
threshold = 0.0

METree_path = "/home/data1/lk/project/mol_tree/graph/evolution_graph_133885_['edit_distance', 'graph']_0.3_v2.json"

def load_graph_from_json(METree_path):
    with open(METree_path, 'r') as f:
        data = json.load(f)

    graph = Graph()

    # 添加节点
    for node in data['nodes']:
        graph.add_node(node)

    # 添加边
    for edge in data['edges']:
        graph.add_edge(edge[0], edge[1])

    return graph, data['atom_groups']

def filter_atom_groups(graph, atom_groups, group_ids=[1, 2, 3, 4]):
    """
    根据 atom_groups 中的组 ID 筛选出组 1 到 4 对应的分子节点，并返回筛选后的图和 atom_groups。
    
    参数：
    graph (Graph): 图对象，包含节点和边。
    atom_groups (dict): 原子组字典，键为组ID，值为节点列表。
    group_ids (list): 要筛选的组 ID 列表，默认是 [1, 2, 3, 4]。
    
    返回：
    filtered_graph (Graph): 筛选后的图对象。
    filtered_atom_groups (dict): 筛选后的 atom_groups 字典。
    """
    # 获取筛选后的节点集合
    filtered_nodes = set()
    
    # 筛选出指定组 ID 的节点
    for group_id in group_ids:
        group_key = str(group_id)  # 假设 group_id 是字符串形式
        if group_key in atom_groups:
            filtered_nodes.update(atom_groups[group_key])
            
            
    filtered_nodes = {node for node in filtered_nodes if node not in ['N', 'O']}

    # 筛选图中的边和节点
    filtered_edges = [(from_node, to_node) for from_node, to_node in graph.edges() if from_node in filtered_nodes and to_node in filtered_nodes]

    # 创建新的筛选后的图
    filtered_graph = Graph()
    for node in filtered_nodes:
        filtered_graph.add_node(node)
    
    for edge in filtered_edges:
        filtered_graph.add_edge(edge[0], edge[1])
    
    # 创建新的筛选后的 atom_groups 字典
    filtered_atom_groups = {key: value for key, value in atom_groups.items() if key in map(str, group_ids)}
    # import pdb;pdb.set_trace()
    filtered_atom_groups['1'] = ['C']
    return filtered_graph, filtered_atom_groups

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

def visualize_graph(graph, atom_groups, save_path):
    """
    可视化图，展示节点和边，图像背景透明，字体清晰，按 atom_groups 划分不同圈层并用树状结构表示。
    
    参数：
    graph (Graph): 要可视化的图对象，包含节点和边。
    atom_groups (dict): 原子组字典，键为组ID，值为节点列表。
    save_path (str): 可视化图像保存的路径。
    """
    # 使用 networkx 创建图
    G = nx.DiGraph()  # 假设是有向图，如果是无向图，可以使用 nx.Graph()

    # 添加边
    edges_to_visualize = [(from_node, to_node) for from_node, to_node in graph.edges()]
    G.add_edges_from(edges_to_visualize)

    # 创建节点的分层位置
    pos = {}
    layers = list(atom_groups.values())  # 获取按组划分的原子组列表

    # 设置环的半径和节点间距
    layer_distance = 3  # 层与层之间的纵向间距
    radius_increment = 6  # 每一层的半径增量
    layer_count = len(layers)  # 层数
    max_nodes_in_layer = max([len(group) for group in layers])  # 最大节点数

    # 每个原子组节点的y坐标逐层增加，环上均匀分布
    for i, group in enumerate(layers):
        radius = radius_increment * (i + 1)  # 每层的半径
        angle_step = 2 * np.pi / len(group)  # 计算每个节点的角度间隔
        for j, node in enumerate(group):
            angle = j * angle_step
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))  # 将节点放置在圆周上

    # 加载字体
    font_path = "/home/data1/lk/project/CLDR/CLIP_DRP/exp/spe_smiles/Arial.ttf"
    font = FontProperties(fname=font_path)  # 使用 FontProperties 加载字体

    # 可视化图
    plt.figure(figsize=(16, 16), dpi=1000)  # 增大图像尺寸，以提高可读性

    # 高亮显示每一层的节点，使用不同颜色
    layer_colors = ['#814b49', '#af9482', '#486e95', '#81b390']
    # 绘制层间连接的边，使用不同颜色
    edge_colors = ['#8d2f25', '#cb9475', '#3e608d', '#8cbf87']
    
    # 绘制所有节点
    for i, group in enumerate(layers):
        # 根据层的大小设置不同的节点大小，层数越大，节点圆圈越大
        node_size = 2000 + (i * 1000)  # 每一层节点的大小逐层增加
        nx.draw_networkx_nodes(G, pos, nodelist=group, node_size=node_size, node_color=layer_colors[i % len(layer_colors)], alpha=1)

    # 绘制带标签的节点
    for i, group in enumerate(layers):
        # 设置最后一层的字体大小为14，其他层为18
        if i == len(layers) - 1:
            font_size = 14
        elif i == 2:
            font_size = 16
        else:
            font_size = 18

        # 使用 matplotlib 添加字体标签
        for node in group:
            x, y = pos[node]
            plt.text(x, y, node, fontsize=font_size, ha='center', va='center', fontproperties=font, color='black')

    # 绘制边时按层级逆序绘制（从第 4 层开始到第 1 层）
    edges_with_color = []
    for i, group in enumerate(layers[:-1]):
        for node in group:
            # 连接到下一层的节点
            next_group = layers[i + 1]
            for target in next_group:
                # 如果边存在
                if (node, target) in G.edges() or (target, node) in G.edges():
                    edges_with_color.append((node, target, edge_colors[i % len(edge_colors)]))
    
    # 绘制具有不同颜色的边（按层级逆序）
    for index, edge_color in enumerate(reversed(edge_colors)):  # 从 4 层到 1 层绘制
        edges = [edge for edge in edges_with_color if edge[2] == edge_color]
        nx.draw_networkx_edges(G, pos, edgelist=[(e[0], e[1]) for e in edges], edge_color=edge_color, width=(index * 2 + 1), alpha=0.5 + index * 0.1)

    # 添加标题
    # plt.title("Visualization of Graph with Layered Atom Groups", fontsize=16)

    # 添加图例
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=30, label=f'Group {i+1}') 
                       for i, color in enumerate(layer_colors)]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), fontsize=25, ncol=4)

    # 去除坐标轴
    plt.axis('off')

    # 设置背景透明
    plt.savefig(save_path, format='PDF')



# 读取数据
graph, atom_groups = load_graph_from_json(METree_path)


# 筛选分子和边
graph, atom_groups = filter_atom_groups(graph, atom_groups)

# import pdb;pdb.set_trace()
# # 可视化图
visualize_graph(graph, atom_groups, f'img/test.pdf')

# # 只选择前 30 个节点
# nodes_to_visualize = list(graph.nodes)[:30]
# import pdb;pdb.set_trace()
# # 创建子图
# subgraph = graph.get_edges()
# edges_to_visualize = [(from_node, to_node) for from_node, to_node in graph.edges() if from_node in nodes_to_visualize and to_node in nodes_to_visualize]

# # 使用 networkx 创建图
# G = nx.DiGraph()  # 这里假设是有向图，如果是无向图，可以使用 nx.Graph()

# # 添加边
# G.add_edges_from(edges_to_visualize)

# # 可视化图
# plt.figure(figsize=(10, 8))

# # 设定节点的位置，使用 spring_layout 或其他布局算法
# pos = nx.spring_layout(G, seed=42)  # spring_layout 适合于大部分图，seed 保证每次结果一致

# # 绘制节点和边
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray', width=2)

# # 添加标题或其他元素
# plt.title(f"Visualization of First 30 Nodes in the Graph", fontsize=14)
# plt.savefig(f'img/test.png', format='PNG')

