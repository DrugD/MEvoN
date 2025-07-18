import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from tools import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch_geometric.data import Batch
import torch.nn.init as init

class GNNModel(nn.Module):
    def __init__(self, device):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(90, 128)
        self.conv2 = gnn.GCNConv(128, 256)
        self.conv3 = gnn.GCNConv(256, 256)
        self.conv4 = gnn.GCNConv(256, 256)
    
        self.fc1_2 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc3_2 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        
        self.device = device
        self.smiles_to_graph_cache = {}
    
    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):

        
        # 对第一个图进行卷积
        x1 = self.conv1(graph1.x, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv2(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv3(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv4(x1, graph1.edge_index)
        x1 = torch.relu(x1)
        x1 = torch.cat([gmp(x1, graph1.batch), gap(x1, graph1.batch)], dim=1)

        x_two = self.fc1_2(x1)
        x_two = torch.relu(x_two)
        x_two = self.dropout(x_two)
        x_two = self.fc2_2(x_two)
        x_two = torch.relu(x_two)
        x_two = self.fc3_2(x_two)
        return x_two
    