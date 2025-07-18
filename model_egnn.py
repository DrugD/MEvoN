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
import pandas as pd
from visnet_block import ViSNetBlock
import visnet_output_modules


import torch
from torch import nn


from torch import nn
import torch

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)


        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr




class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh()) #we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, data):
        import pdb;pdb.set_trace()
        h0 = []
        x = data.x
        edges = data.edge_index
        edge_attr = data.edge_attr
        node_mask = []
        edge_mask = []
        n_nodes = []
        
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


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
        transformer_out = self.transformer_encoder(embedded)  # (max_seq_len, batch_size, model_dim)

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

class GNNModel(nn.Module):
    def __init__(self, device):
        super(GNNModel, self).__init__()

        self.dropout = nn.Dropout(0.3)
        
        self.path_encoder = PathTransformer(input_dim=512, model_dim=512, num_heads=4, num_layers=2)
        self.path_encoder = self.path_encoder.to(device)  # Move model to GPU
        self.label_encoder = LabelSeqTransformer(input_dim=1, model_dim=64, num_heads=4, num_layers=2)
        self.label_encoder = self.label_encoder.to(device)  # Move model to GPU
        
        self.device = device

        self.smiles_to_graph_cache = {}
        qm9_csv_path = "/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv"
        qm9_df = pd.read_csv(qm9_csv_path)
        self.smiles_to_index = {row['SMILES1']: row['Unnamed: 0'] for _, row in qm9_df.iterrows()}
        self.suppl = Chem.SDMolSupplier('/home/data1/lk/project/mol_property/ViSNet/dataset/raw/gdb9.sdf', removeHs=False,
                            sanitize=False)
        
        self.representation_model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=128, device=self.device, n_layers=7, coords_weight=1.0,
             attention=1, node_attr=0)


    
    def drug_encoder(self, graph):
        x  = self.representation_model(graph)
        return x
    
    
    def forward_with_path(self, graph1, graph2, paths, paths_labels, paths_labels_masks):
        
        x1 = self.drug_encoder(graph1)
        x2 = self.drug_encoder(graph2)
        


        x_2 = self.output_model.pre_reduce(x2, v2, graph2.z, graph2.pos, graph2.batch)
        x_2 = scatter(x_2, graph2.batch, dim=0, reduce="add")
        
        x_1 = self.output_model.pre_reduce(x1, v1, graph1.z, graph1.pos, graph1.batch)
        x_1 = scatter(x_1, graph1.batch, dim=0, reduce="add")
        
        
        
        # 融合两个图的特征
        x12 = x_1 - x_2  # 使用差值进行融合
        # x21 = -x1 + x2  # 使用差值进行融合
        # 将路径信息编码为特征向量
        path_features, path_labels_features = self.encode_paths(paths, paths_labels, paths_labels_masks)
        

       
        # x = torch.cat([x12,x21],dim=1)
        x_one = torch.cat([x12, x_1, x_2, path_features, path_labels_features], dim=1)  # 将路径特征拼接
        # x_two = torch.cat([x1, path_features, path_labels_features], dim=1)
        x_one = self.evo_reg_head(x_one)
        
        
        return torch.cat([x_one,x_1],dim=1)
    
    
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
                        graph = smiles_to_graph_xyz(self.smiles_to_index[temp], self.suppl)  # 将 SMILES 转换为图
                        self.smiles_to_graph_cache[temp] = graph  # 缓存转换后的图
                    
                    all_graphs.append(graph)
                    all_indices.append((i_idx, j_idx, k_idx))  # 记录索引

        # 处理所有图的批次编码
        if all_graphs:
            drug_features_list = []
            batch_size = 512  # 设置批次大小

            # 分批处理图数据
            for start_idx in range(0, len(all_graphs), batch_size):
                sub_graphs = all_graphs[start_idx:start_idx + batch_size]

                # 创建小批次并移动到指定设备
                batch = Batch.from_data_list(sub_graphs).to(self.device)

                with torch.no_grad():
                    drug_features_batch = self.drug_encoder(batch)[0]

                # 存储当前批次结果
                drug_features_list.append(drug_features_batch)

                # # 显式释放临时变量以优化显存
                # del batch, drug_features_batch
                # torch.cuda.empty_cache()  # 手动释放未使用的显存

            # 合并所有批次的结果
            drug_features = torch.cat(drug_features_list, dim=0)
                                
            
            
            
            
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
                path_embedding = self.path_encoder(path_vectors_for_one_drug.to(self.device))
                avg_path_feature = torch.mean(path_embedding, dim=0)

                # if avg_path_feature.dim()==1:
                #     avg_path_feature = avg_path_feature.expand(1,1,512)
                paths_list.append(avg_path_feature)
        
     
        paths_labels_ = paths_labels.view(paths_labels.size(0)*paths_labels.size(1), -1, 1)
        paths_labels_masks = paths_labels_masks.view(paths_labels.size(0)*paths_labels.size(1),-1).bool()
        paths_labels_features =  self.label_encoder(paths_labels_.to(self.device),paths_labels_masks.to(self.device))
     
        # paths_labels_features = torch.mean(paths_labels_features.reshape(paths_labels.size(0), paths_labels.size(1), -1),dim=1)
        paths_labels_features = paths_labels_features.reshape(paths_labels.size(0), -1)
 
        return torch.stack(paths_list).to(self.device), paths_labels_features


