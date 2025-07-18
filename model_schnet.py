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

import pdb
class PathTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(PathTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 9, model_dim))  # 假设最大序列长度为 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim)  # 假设输出是一个标量（属性）

    def forward(self, paths):
        # paths: (batch_size, max_seq_len, feature_dim)
        batch_size, max_seq_len, _ = paths.size()

        # 嵌入
        embedded = self.embedding(paths)  # (batch_size, max_seq_len, model_dim)
        embedded += self.positional_encoding[:, :max_seq_len, :]  # 添加位置编码

        # 转换为 (max_seq_len, batch_size, model_dim)
        embedded = embedded.permute(1, 0, 2)

        # 使用 Transformer Encoder
        transformer_out = self.transformer_encoder(embedded)  # (max_seq_len, batch_size, model_dim) # 这里schnet r2出现nan 
        '''
        (Pdb) embedded.max()
        tensor(7.4068e+09, device='cuda:0')
        (Pdb) embedded.min()
        tensor(-7.5232e+09, device='cuda:0')
        '''

        # 取最后一个时间步的输出
        output = transformer_out[-1, :, :]  # (batch_size, model_dim)

        # 输出层
        final_output = self.fc_out(output)  # (batch_size, 1)
        return final_output
    

class LabelSeqTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(LabelSeqTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 9, model_dim))  # 假设最大序列长度为 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, model_dim)  # 假设输出是一个标量（属性）

    def forward(self, paths, mask):
        
        batch_size, max_seq_len, _ = paths.size()

        # 找到有效的行：至少有一个位置为 True
        valid_rows = mask.any(dim=1)

        # 提取有效的 paths 和 mask
        valid_paths = paths[valid_rows]  # 仅保留有效的路径
        valid_mask = mask[valid_rows]   # 仅保留对应的有效掩码

        # 替换 NaN 值为 0（如需要）
        # valid_paths = torch.nan_to_num(valid_paths, nan=0.0)

        # 嵌入
        embedded = self.embedding(valid_paths)  
        embedded += self.positional_encoding[:, :max_seq_len, :]

        # 转换为 (max_seq_len, batch_size, model_dim)
        embedded = embedded.permute(1, 0, 2)
        
        # 使用 Transformer Encoder，应用掩码
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=~valid_mask)

        # 提取最后一层的输出
        output = transformer_out[-1, :, :]
        final_output = self.fc_out(output)

        # 将结果插回到原来的形状
        result = torch.zeros(batch_size, final_output.size(1), device=final_output.device)
        result[valid_rows] = final_output

        return result

# --------------------------------schnet-----------------------------
import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor

from torch_geometric.data import Data

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, data: Data) -> Tensor:
        r"""
        Forward pass of the SchNet model.

        Args:
            data (Data): A PyG Data object containing:
                - z (torch.Tensor): Atomic number of each atom with shape `[num_atoms]`.
                - pos (torch.Tensor): Coordinates of each atom with shape `[num_atoms, 3]`.
                - batch (torch.Tensor, optional): Batch indices assigning each atom to a separate molecule with shape `[num_atoms]`. Defaults to `None`.

        Returns:
            torch.Tensor: The predicted property for each molecule with shape `[num_molecules, 1]`.
        """
        batch = torch.zeros_like(data.z) if data.batch is None else data.batch

        h = self.embedding(data.z)
        edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:  # 偶极矩
            # 计算质心
            mass = self.atomic_mass[data.z].view(-1, 1)
            M = self.sum_aggr(mass, data.batch, dim=0)
            c = (self.sum_aggr(mass * data.pos, data.batch, dim=0) / M).to(torch.float32)
            # h = h * (data.pos - c.index_select(0, data.batch))
            h = h * torch.norm(data.pos - c.index_select(0, data.batch), dim=-1, keepdim=True)
            
            # torch.norm(offsets, dim=1, keepdim=True)

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(data.z)

        out = self.readout(h, data.batch, dim=0)

        # if self.dipole:
        #     out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out
    
    def get_drug_feature(self, data: Data) -> Tensor:
        r"""
        Extracts molecular features from the SchNet model.

        Args:
            data (Data): A PyG Data object containing:
                - z (torch.Tensor): Atomic number of each atom with shape `[num_atoms]`.
                - pos (torch.Tensor): Coordinates of each atom with shape `[num_atoms, 3]`.
                - batch (torch.Tensor, optional): Batch indices assigning each atom to a separate molecule with shape `[num_atoms]`. Defaults to `None`.

        Returns:
            torch.Tensor: Molecular features of shape `[num_molecules, hidden_channels // 2]`.
        """
        # 如果未提供 batch，则默认所有原子属于同一个分子
        batch = torch.zeros_like(data.z) if data.batch is None else data.batch

        # 原子嵌入
        h = self.embedding(data.z)

        # 构建分子图
        edge_index, edge_weight = self.interaction_graph(data.pos, data.batch)
        edge_attr = self.distance_expansion(edge_weight)

        # 相互作用块
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # 输出层的第一部分
        h = self.lin1(h)
        h = self.act(h)

        # 如果需要处理偶极矩
        if self.dipole:
            # 计算质心
            mass = self.atomic_mass[data.z].view(-1, 1)
            M = self.sum_aggr(mass, data.batch, dim=0)
            c = (self.sum_aggr(mass * data.pos, data.batch, dim=0) / M).to(torch.float32)
            # h = h * (data.pos - c.index_select(0, data.batch))
            h = h * torch.norm(data.pos - c.index_select(0, data.batch), dim=-1, keepdim=True)
            

        # 如果需要归一化
        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        # 如果需要添加原子参考值
        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(data.z)

        # 聚合原子特征为分子特征
        # import pdb; pdb.set_trace()
        # features = self.readout(h, data.batch, dim=0)

        return h

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        
        # eps = 1e-1
        # edge_j, edge_i = edge_index  
        # dist = (pos[edge_i] - pos[edge_j]).pow(2).sum(dim=-1).sqrt()  
        # mask = dist > eps  
        # edge_index = edge_index[:, mask]
        
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight

class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift

# --------------------------------schnet-----------------------------
    
class GNNModel(nn.Module):
    def __init__(self, device, is_mu=False, is_r2=False):
        super(GNNModel, self).__init__()
        if is_mu:
            self.drug_model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
                max_num_neighbors=32,
                dipole=True
            ) 
        else:
            self.drug_model = SchNet(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=10.0,
                max_num_neighbors=32
            ) # 64
        
        # self.fc1_1 = nn.Linear(512*4+64+256, 512)
        self.fc1_1 = nn.Linear(128*3+256+64*5, 512)
        self.fc2_1 = nn.Linear(512, 256)
        self.fc3_1 = nn.Linear(256, 1)
        
        self.fc1_2 = nn.Linear(128, 512)
        self.fc2_2 = nn.Linear(512, 256)
        self.fc3_2 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        
        self.path_encoder = PathTransformer(input_dim=128, model_dim=256, num_heads=4, num_layers=2)
        self.path_encoder = self.path_encoder.to(device)  # Move model to GPU
        self.label_encoder = LabelSeqTransformer(input_dim=1, model_dim=64, num_heads=4, num_layers=2)
        self.label_encoder = self.label_encoder.to(device)  # Move model to GPU
        
        self.device = device
        
        import pandas as pd
        self.smiles_to_graph_cache = {}
        qm9_csv_path = "/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv"
        qm9_df = pd.read_csv(qm9_csv_path)
        self.smiles_to_index = {row['SMILES1']: row['Unnamed: 0'] for _, row in qm9_df.iterrows()}
        self.suppl = Chem.SDMolSupplier('/home/data1/lk/project/mol_property/ViSNet/dataset/raw/gdb9.sdf', removeHs=False,
                            sanitize=False)

        self.softplus = nn.Softplus()
        if is_r2:
            self.layer_norm = nn.LayerNorm(128)
            self.is_r2 = True
        else:
            self.is_r2 = False
        
    def drug_encoder(self, graph):
        '''
        # 对第一个图进行卷积
        x = self.conv1(graph.x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv3(x, graph.edge_index)
        x = torch.relu(x)
        x = self.conv4(x, graph.edge_index)
        x = torch.relu(x)
        x = torch.cat([gmp(x, graph.batch), gap(x, graph.batch)], dim=1)
        '''
        
        x = self.drug_model.get_drug_feature(graph)
        x = torch.cat([gmp(x, graph.batch), gap(x, graph.batch)], dim=1)
        return x
    
    # def forward_origin(self, graph):
    #     return self.drug_model(graph)
    
    def forward_origin(self, graph):
        x = self.drug_encoder(graph)
        x_two = self.fc1_2(x)
        x_two = torch.tanh(x_two)
        x_two = self.dropout(x_two)
        x_two = self.fc2_2(x_two)
        x_two = torch.tanh(x_two)
        x_two = self.fc3_2(x_two)
        return x_two
        
        
    def forward(self, graph1, graph2):
        x1 = self.drug_encoder(graph1)
        x2 = self.drug_encoder(graph2)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        x21 = -x1 + x2  # 使用差值进行融合
        x = torch.cat([x12,x21],dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x
    
    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):

        x1 = self.drug_encoder(graph1) # 1024 128 
        x2 = self.drug_encoder(graph2)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_two = x1

        # x_one = self.fc1_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.dropout(x_one)
        # x_one = self.fc2_1(x_one)
        # x_one = torch.tanh(x_one)
        # x_one = self.fc3_1(x_one)
        
        # x_two = self.fc1_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.dropout(x_two)
        # x_two = self.fc2_2(x_two)
        # x_two = torch.relu(x_two)
        # x_two = self.fc3_2(x_two)
        # return torch.cat([x_one,x_two],dim=1)
        
        x_one = self.fc1_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.dropout(x_one)
        x_one = self.fc2_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.fc3_1(x_one)
        
        x_two = self.fc1_2(x_two)
        x_two = torch.tanh(x_two)
        x_two = self.dropout(x_two)
        x_two = self.fc2_2(x_two)
        x_two = torch.tanh(x_two)
        x_two = self.fc3_2(x_two)
        return torch.cat([x_one,x_two],dim=1)
    
 
    
    def encode_paths(self, paths, paths_labels, paths_labels_masks):

        # paths = torch.nan_to_num(paths, nan=0.0)
        # paths_labels = torch.nan_to_num(paths_labels, nan=0.0)
        
        # 编码路径信息为特征向量
        encoded_features = []
        
        i_length, j_length, k_length = paths_labels.size()
        
        # 用于缓存已转换的图
        
        all_graphs = []  # 用于存储所有图以进行批处理
        all_indices = []  # 存储每个图对应的路径索引
        paths_list = []
        
        for i_idx in range(i_length):
        
            
            for j_idx in range(j_length):
                if paths[j_idx][0][i_idx] == 'PAD':
                    break
                
                for k_idx in range(k_length):
                    temp = paths[j_idx][k_idx][i_idx]
                    if temp == 'PAD':  # 提前结束条件
                        break
                    
                    # 检查缓存字典，避免重复计算
                    if temp in self.smiles_to_graph_cache:
                        graph = self.smiles_to_graph_cache[temp]
                    else:
                        graph = smiles_to_graph_xyz_sch(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
                        self.smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
                    all_graphs.append(graph)
                    all_indices.append((i_idx, j_idx, k_idx)) 

        # 处理所有图的批次编码
        if all_graphs:
            drug_features_list = []
            batch_size = 128  # 设置批次大小

            # 分批处理图数据
            for start_idx in range(0, len(all_graphs), batch_size):
                sub_graphs = all_graphs[start_idx:start_idx + batch_size]

                # 创建小批次并移动到指定设备
                batch = Batch.from_data_list(sub_graphs).to(self.device)

                with torch.no_grad():
                    drug_features_batch = self.drug_encoder(batch)

                # 存储当前批次结果
                drug_features_list.append(drug_features_batch)

                # # 显式释放临时变量以优化显存
                # del batch, drug_features_batch
                # torch.cuda.empty_cache()  # 手动释放未使用的显存

            # 合并所有批次的结果
            drug_features = torch.cat(drug_features_list, dim=0)
            if self.is_r2:
                drug_features = self.layer_norm(drug_features)
            # pdb.set_trace()
                                
            # 为每个路径特征分配相应的药物特征
            path_features = [[] for _ in range(i_length)]  # 创建一个长度为 i_length 的列表，用于存储每个药物的特征

            for index, feature in zip(all_indices, drug_features):
                # index[0] 是 i_idx，代表当前药物的索引
                i_idx = index[0]  # 当前药物索引
                j_idx = index[1]  # 当前路径索引

                # 确保 path_features 的双重嵌套结构
                if len(path_features[i_idx]) <= j_idx:
                    # 如果当前 i_idx 的列表长度小于 j_idx，进行填充
                    path_features[i_idx].extend([[]] * (j_idx + 1 - len(path_features[i_idx])))

                # 将特征添加到对应的 i_idx 和 j_idx
                path_features[i_idx][j_idx].append(feature)  # 将当前特征添加到对应的路径中
            

            # 计算每个药物的平均路径特征
            for path_feature in path_features:
                path_feature = [ pad_sequence(vec, batch_first=True) for vec in path_feature]
                path_vectors_for_one_drug = pad_sequence(path_feature, batch_first=True)
                path_embedding = self.path_encoder(path_vectors_for_one_drug)
                avg_path_feature = torch.mean(path_embedding, dim=0)

                # if avg_path_feature.dim()==1:
                #     avg_path_feature = avg_path_feature.expand(1,1,512)
                paths_list.append(avg_path_feature)
        

        # paths_labels_ = paths_labels.view(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        # paths_labels_masks = paths_labels_masks.view(paths_labels.size(0)*paths_labels.size(1),-1).bool()
        paths_labels_ = paths_labels.reshape(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        paths_labels_masks = paths_labels_masks.reshape(paths_labels.size(0)*paths_labels.size(1),-1).bool()

        paths_labels_features =  self.label_encoder(paths_labels_.to(self.device),paths_labels_masks.to(self.device))
     
        # paths_labels_features = torch.mean(paths_labels_features.reshape(paths_labels.size(0), paths_labels.size(1), -1),dim=1)
        paths_labels_features = paths_labels_features.reshape(paths_labels.size(0), -1)
 
        return torch.stack(paths_list).to(self.device), paths_labels_features


    def get_features(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        
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
       
        # 对第二个图进行卷积
        x2 = self.conv1(graph2.x, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv2(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv3(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = self.conv4(x2, graph2.edge_index)
        x2 = torch.relu(x2)
        x2 = torch.cat([gmp(x2, graph2.batch), gap(x2, graph2.batch)], dim=1)

        # 融合两个图的特征
        x12 = x1 - x2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        # path_features, path_labels_features = self.encode_paths(paths_slice, paths_labels_slice, paths_labels_masks_slice)

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x1, x2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_one = self.fc1_1(x_one)
        x_one = torch.tanh(x_one)
        x_one = self.dropout(x_one)
        x_one = self.fc2_1(x_one)
        return x_one
    
    
    # def encode_paths(self, paths, paths_labels):
    #     # 编码路径信息为特征向量
    #     encoded_features = []
        
    #     i_length, j_length, k_length = paths_labels.size()
        
    #     # 用于缓存已转换的图
    #     smiles_to_graph_cache = {}

    #     for i_idx in tqdm(range(i_length)):
    #         path_for_one_drug = []
            
    #         all_path_features = []
    #         graphs = []  # 用于存储所有图以进行批处理

    #         for j_idx in range(j_length):
    #             if paths[j_idx][0][i_idx] == 'PAD':
    #                 break
                
    #             path_features = []
    #             for k_idx in range(k_length):
    #                 temp = paths[j_idx][k_idx][i_idx]
    #                 if temp == 'C':  # 提前结束条件
    #                     break
                    
    #                 # 检查缓存字典，避免重复计算
    #                 if temp in smiles_to_graph_cache:
    #                     graph = smiles_to_graph_cache[temp]
    #                 else:
    #                     graph = smiles_to_graph(temp)  # 将 SMILES 转换为图
    #                     smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
    #                 graphs.append(graph)

    #             if graphs:
    #                 # 使用 Batch 将所有图组合为一个批次
    #                 batch = Batch.from_data_list(graphs).to(self.device)
    #                 drug_features = self.drug_encoder(batch)

    #                 # 将药物特征分配给路径特征
    #                 path_features = drug_features.split(1)  # 将 drug_features 分割为单独的特征
    #                 all_path_features.append(path_features)
                
    #         # 将每个子列表转换为张量，并进行填充
    #         padded_vectors = []

    #         for vectors in all_path_features:
    #             # 将1x512的向量列表转换为一个张量
    #             tensor_list = [vec.squeeze(0) for vec in vectors]  # 移除多余的维度
    #             padded_vectors.append(pad_sequence(tensor_list, batch_first=True))

    #         # 合并所有填充后的子列表
    #         padded_batch = pad_sequence(padded_vectors, batch_first=True)
    #         path_embedding = self.path_encoder(padded_batch)
    #         avg_path_feature_for_one_drug = torch.mean(path_embedding, dim=0)
    #         encoded_features.append(avg_path_feature_for_one_drug)

    #     # 最终返回编码的特征
    #     return torch.stack(encoded_features).to(self.device)

    
    # def encode_paths(self, paths, paths_labels):
    #     # 编码路径信息为特征向量
    #     encoded_features = []
        
        
    #     i_length, j_length, k_length = paths_labels.size()
        
    #     i_idx, j_idx, k_idx = 0, 0, 0
        
    #     for i_idx  in tqdm(range(i_length)):
    #         path_for_one_drug = []
    #         for j_idx in range(j_length):
    #             path_for_one_path = []
    #             path_features = []
                
    #             if paths[j_idx][0][i_idx] == 'PAD':
    #                 break
                
    #             for k_idx in range(k_length):
                    
    #                 temp = paths[j_idx][k_idx][i_idx]
    #                 path_for_one_path.append(temp)
                 
    #                 graph = smiles_to_graph(temp)  # 将 SMILES 转换为图
                    
    #                 drug_feature = self.drug_encoder(graph.to(self.device))
    #                 path_features.append(drug_feature)
                    
    #                 if temp == 'C':
    #                     break
                    
    #             path_embedding = self.path_encoder(torch.stack(path_features).permute(1,0,2).to(self.device))
    #             path_for_one_drug.append(path_embedding)
                
    #         avg_path_feature_for_one_drug = torch.mean(torch.stack(path_for_one_drug).to(self.device), dim=0).to(self.device)
    #         encoded_features.append(avg_path_feature_for_one_drug)

    #     return torch.stack(encoded_features).to(self.device)  # 返回的特征张量